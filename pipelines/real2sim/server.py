#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
import sys
import threading
import traceback
from collections.abc import Iterable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

# Match the notebook inference environment so pipeline imports behave the same.
os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.append(str(NOTEBOOK_DIR))

from inference import Inference  # noqa: E402
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix  # noqa: E402
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform  # noqa: E402

NEAREST_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "NEAREST")

DEFAULT_POSE_OPTIM_BACKEND = "gs"
DEFAULT_WITH_LAYOUT_POSTPROCESS = True
DEFAULT_GS_ENABLE_MANUAL_ALIGNMENT = True
DEFAULT_GS_ENABLE_SHAPE_ICP = False
DEFAULT_GS_ENABLE_OCCLUSION_CHECK = False
MODEL_TO_GLB_BASIS = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def json_default(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def normalize_vector(value: Any, expected_len: int) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != expected_len:
        raise ValueError(f"Expected vector of length {expected_len}, got shape {arr.shape}")
    return arr.tolist()


def sanitize_name(filename: str | None, fallback: str) -> str:
    stem = Path(filename or fallback).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or fallback


def pose_to_matrix(pose: dict[str, Any]) -> np.ndarray:
    if "transform_matrix" in pose:
        return np.asarray(pose["transform_matrix"], dtype=np.float64)

    rotation = normalize_vector(pose["rotation"], 4)
    translation = normalize_vector(pose["translation"], 3)
    scale = normalize_vector(pose["scale"], 3)

    quat = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0)
    trans = torch.tensor(translation, dtype=torch.float32).unsqueeze(0)
    scale_t = torch.tensor(scale, dtype=torch.float32).unsqueeze(0)

    # SAM 3D uses PyTorch3D row-vector transforms internally. Trimesh expects the
    # conventional column-vector transform, so transpose before exporting.
    transform_p3d = compose_transform(
        scale=scale_t,
        rotation=quaternion_to_matrix(quat),
        translation=trans,
    ).get_matrix()[0]
    return transform_p3d.T.cpu().numpy().astype(np.float64)


def canonical_pose_to_glb_pose(pose: dict[str, Any]) -> dict[str, Any]:
    rotation = normalize_vector(pose["rotation"], 4)
    translation = normalize_vector(pose["translation"], 3)
    scale = normalize_vector(pose["scale"], 3)

    quat = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0)
    trans = torch.tensor(translation, dtype=torch.float32).unsqueeze(0)
    scale_t = torch.tensor(scale, dtype=torch.float32).unsqueeze(0)

    canonical_transform = compose_transform(
        scale=scale_t,
        rotation=quaternion_to_matrix(quat),
        translation=trans,
    ).get_matrix()[0].T

    basis_transform = torch.eye(4, dtype=torch.float32)
    basis_transform[:3, :3] = MODEL_TO_GLB_BASIS
    glb_transform = canonical_transform @ basis_transform

    linear = glb_transform[:3, :3]
    scale_vec = torch.linalg.norm(linear, dim=0)
    safe_scale = torch.where(scale_vec > 1e-8, scale_vec, torch.ones_like(scale_vec))
    rotation_matrix = linear / safe_scale.unsqueeze(0)
    rotation_quat = matrix_to_quaternion(rotation_matrix.unsqueeze(0))[0]

    corrected_pose = {
        "rotation": rotation_quat.cpu().numpy().tolist(),
        "translation": glb_transform[:3, 3].cpu().numpy().tolist(),
        "scale": scale_vec.cpu().numpy().tolist(),
        "transform_matrix": glb_transform.cpu().numpy().tolist(),
    }
    for key in ("iou", "iou_before_optim", "optim_accepted"):
        if key in pose:
            corrected_pose[key] = pose[key]
    return corrected_pose


def export_glb_bytes(mesh_or_scene: trimesh.Trimesh | trimesh.Scene) -> bytes:
    glb_bytes = mesh_or_scene.export(file_type="glb")
    if isinstance(glb_bytes, bytes):
        return glb_bytes
    if isinstance(glb_bytes, bytearray):
        return bytes(glb_bytes)
    raise TypeError(f"Unexpected GLB export type: {type(glb_bytes)!r}")


def merge_scene(objects: list[tuple[str, trimesh.Trimesh | trimesh.Scene, dict[str, Any]]]) -> bytes:
    scene = trimesh.Scene()
    for name, mesh_or_scene, pose in objects:
        transform = pose_to_matrix(pose)
        if isinstance(mesh_or_scene, trimesh.Scene):
            for geom_name, geom in mesh_or_scene.geometry.items():
                scene.add_geometry(
                    geom.copy(),
                    geom_name=f"{name}_{geom_name}",
                    node_name=f"{name}_{geom_name}",
                    transform=transform,
                )
        else:
            scene.add_geometry(
                mesh_or_scene.copy(),
                geom_name=name,
                node_name=name,
                transform=transform,
            )
    return export_glb_bytes(scene)


def encode_frame(meta: dict[str, Any], payload: bytes = b"") -> bytes:
    header = json.dumps(meta, ensure_ascii=False, default=json_default).encode("utf-8")
    return struct.pack(">I", len(header)) + header + struct.pack(">Q", len(payload)) + payload


async def read_upload_image(file: UploadFile, *, mode: str) -> np.ndarray:
    raw = await file.read()
    if not raw:
        raise ValueError(f"Uploaded file {file.filename or '<unknown>'} is empty")
    image = Image.open(io.BytesIO(raw))
    image.load()
    return np.array(image.convert(mode), dtype=np.uint8)


async def read_upload_mask(file: UploadFile) -> np.ndarray:
    raw = await file.read()
    if not raw:
        raise ValueError(f"Uploaded mask {file.filename or '<unknown>'} is empty")

    image = Image.open(io.BytesIO(raw))
    image.load()
    if "A" in image.getbands():
        alpha = np.array(image.getchannel("A"), dtype=np.uint8)
        if np.any(alpha > 0):
            return alpha
    return np.array(image.convert("L"), dtype=np.uint8)


@dataclass
class PredictRequest:
    image: np.ndarray
    masks: list[tuple[str, np.ndarray]]
    seed: int
    texture_baking: bool
    pose_optim_backend: str
    with_layout_postprocess: bool
    gs_enable_manual_alignment: bool
    gs_enable_shape_icp: bool
    gs_enable_occlusion_check: bool
    allow_mask_resize: bool

class ModelRunner:
    def __init__(self, config_path: Path, *, compile_model: bool = False):
        self.config_path = str(config_path)
        self.compile_model = compile_model
        self._inference: Inference | None = None
        self._init_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def load(self) -> None:
        self._get_inference()

    def is_loaded(self) -> bool:
        return self._inference is not None

    def _get_inference(self) -> Inference:
        if self._inference is None:
            with self._init_lock:
                if self._inference is None:
                    print(f"[INIT] loading model from {self.config_path}", flush=True)
                    self._inference = Inference(self.config_path, compile=self.compile_model)
                    print("[INIT] model ready", flush=True)
        return self._inference

    def _predict_one(
        self,
        *,
        request: PredictRequest,
        index: int,
        name: str,
        mask: np.ndarray,
    ) -> tuple[str, dict[str, Any], trimesh.Trimesh | trimesh.Scene]:
        inference = self._get_inference()
        output = inference(
            request.image,
            mask,
            seed=request.seed,
            with_layout_postprocess=request.with_layout_postprocess,
            with_mesh_postprocess=True,
            with_texture_baking=request.texture_baking,
            use_vertex_color=not request.texture_baking,
        )
        mesh = output.get("glb")
        if mesh is None:
            raise RuntimeError(f"Pipeline did not return a GLB for mask '{name}'")

        pose = {
            "rotation": normalize_vector(output["rotation"], 4),
            "translation": normalize_vector(output["translation"], 3),
            "scale": normalize_vector(output["scale"], 3),
        }
        if "iou" in output:
            pose["iou"] = float(np.asarray(json_default(output["iou"])).reshape(-1)[0])
        if "iou_before_optim" in output:
            pose["iou_before_optim"] = float(
                np.asarray(json_default(output["iou_before_optim"])).reshape(-1)[0]
            )
        if "optim_accepted" in output:
            pose["optim_accepted"] = bool(np.asarray(json_default(output["optim_accepted"])).reshape(-1)[0])
        pose = canonical_pose_to_glb_pose(pose)
        return name, pose, mesh

    def iter_predict(
        self, request: PredictRequest
    ) -> Iterator[tuple[str, dict[str, Any], trimesh.Trimesh | trimesh.Scene]]:
        with self._run_lock:
            for index, (name, mask) in enumerate(request.masks):
                yield self._predict_one(
                    request=request,
                    index=index,
                    name=name,
                    mask=mask,
                )

    def predict(self, request: PredictRequest) -> list[tuple[str, dict[str, Any], trimesh.Trimesh | trimesh.Scene]]:
        return list(self.iter_predict(request))


async def build_predict_request(
    *,
    image: UploadFile,
    masks: list[UploadFile],
    seed: int,
    texture_baking: str,
    pose_optim_backend: str,
    with_layout_postprocess: str,
    gs_enable_manual_alignment: str,
    gs_enable_shape_icp: str,
    gs_enable_occlusion_check: str,
    allow_mask_resize: str,
) -> PredictRequest:
    if pose_optim_backend not in {"gs", "mesh"}:
        raise ValueError("pose_optim_backend must be one of: gs, mesh")

    image_np = await read_upload_image(image, mode="RGBA")
    image_hw = image_np.shape[:2]
    allow_resize = parse_bool(allow_mask_resize)

    if not masks:
        raise ValueError("At least one 'masks' upload is required")

    seen_names: dict[str, int] = {}
    decoded_masks: list[tuple[str, np.ndarray]] = []
    for index, item in enumerate(masks):
        raw_mask = await read_upload_mask(item)
        if raw_mask.shape[:2] != image_hw:
            if not allow_resize:
                raise ValueError(
                    f"Mask '{item.filename or index}' has shape {raw_mask.shape[:2]}, expected {image_hw}. "
                    "Pass allow_mask_resize=true to resize with nearest-neighbor."
                )
            raw_mask = np.array(
                Image.fromarray(raw_mask, mode="L").resize(
                    (image_hw[1], image_hw[0]), resample=NEAREST_RESAMPLE
                ),
                dtype=np.uint8,
            )

        mask = raw_mask > 0
        if not np.any(mask):
            raise ValueError(
                f"Mask '{item.filename or index}' is empty after decoding. "
                "Please verify the uploaded mask content."
            )

        name = sanitize_name(item.filename, f"obj_{index:02d}")
        count = seen_names.get(name, 0)
        seen_names[name] = count + 1
        if count:
            name = f"{name}_{count}"
        decoded_masks.append((name, mask))

    request = PredictRequest(
        image=image_np,
        masks=decoded_masks,
        seed=seed,
        texture_baking=parse_bool(texture_baking),
        pose_optim_backend=pose_optim_backend,
        with_layout_postprocess=parse_bool(with_layout_postprocess),
        gs_enable_manual_alignment=parse_bool(gs_enable_manual_alignment),
        gs_enable_shape_icp=parse_bool(gs_enable_shape_icp),
        gs_enable_occlusion_check=parse_bool(gs_enable_occlusion_check),
        allow_mask_resize=allow_resize,
    )
    return request


def encode_predict_stream(
    request: PredictRequest,
    objects: Iterable[tuple[str, dict[str, Any], trimesh.Trimesh | trimesh.Scene]]
) -> Iterator[bytes]:
    poses: dict[str, Any] = {}
    merged_inputs: list[tuple[str, trimesh.Trimesh | trimesh.Scene, dict[str, Any]]] = []

    for index, (name, pose, mesh) in enumerate(objects):
        glb_bytes = export_glb_bytes(mesh)
        yield encode_frame(
            {
                "type": "object_glb",
                "index": index,
                "obj_name": name,
                "glb_filename": f"{name}.glb",
                "pose": pose,
            },
            glb_bytes,
        )
        poses[name] = pose
        merged_inputs.append((name, mesh, pose))

    yield encode_frame(
        {"type": "scene_glb", "scene_glb_filename": "scene_merged.glb"},
        merge_scene(merged_inputs),
    )
    yield encode_frame(
        {"type": "poses_json", "poses_filename": "poses.json"},
        json.dumps(
            {
                "_meta": {
                    "pose_estimation": {
                        "backend": request.pose_optim_backend,
                        "with_layout_postprocess": request.with_layout_postprocess,
                        "gs_enable_manual_alignment": request.gs_enable_manual_alignment,
                        "gs_enable_shape_icp": request.gs_enable_shape_icp,
                        "gs_enable_occlusion_check": request.gs_enable_occlusion_check,
                        "manual_alignment_outlier_filter": {
                            "enabled": True,
                            "method": "open3d.remove_statistical_outlier",
                            "nb_neighbors": 20,
                            "std_ratio": 2.0,
                            "min_points_after_filter": 64,
                        },
                    }
                },
                **poses,
            },
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8"),
    )
    yield encode_frame({"type": "done"})


def stream_predict_response(
    runner: ModelRunner, request: PredictRequest
) -> Iterator[bytes]:
    try:
        yield from encode_predict_stream(request, runner.iter_predict(request))
    except Exception:
        print("[ERROR] predict_stream failed", flush=True)
        traceback.print_exc()
        raise


def create_app(*, config_path: Path, compile_model: bool = False) -> FastAPI:
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    runner = ModelRunner(config_path, compile_model=compile_model)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"[START] preloading model from {config_path}", flush=True)
        await run_in_threadpool(runner.load)
        app.state.runner = runner
        print("[START] model preloaded; accepting requests", flush=True)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return JSONResponse({"ok": True, "model_loaded": runner.is_loaded()})

    @app.post("/predict_stream")
    async def predict_stream(
        image: UploadFile = File(...),
        masks: list[UploadFile] = File(...),
        seed: int = Form(42),
        texture_baking: str = Form("true"),
        pose_optim_backend: str = Form(DEFAULT_POSE_OPTIM_BACKEND),
        with_layout_postprocess: str = Form(
            "true" if DEFAULT_WITH_LAYOUT_POSTPROCESS else "false"
        ),
        gs_enable_manual_alignment: str = Form(
            "true" if DEFAULT_GS_ENABLE_MANUAL_ALIGNMENT else "false"
        ),
        gs_enable_shape_icp: str = Form(
            "true" if DEFAULT_GS_ENABLE_SHAPE_ICP else "false"
        ),
        gs_enable_occlusion_check: str = Form(
            "true" if DEFAULT_GS_ENABLE_OCCLUSION_CHECK else "false"
        ),
        allow_mask_resize: str = Form("false"),
    ):
        try:
            request = await build_predict_request(
                image=image,
                masks=masks,
                seed=seed,
                texture_baking=texture_baking,
                pose_optim_backend=pose_optim_backend,
                with_layout_postprocess=with_layout_postprocess,
                gs_enable_manual_alignment=gs_enable_manual_alignment,
                gs_enable_shape_icp=gs_enable_shape_icp,
                gs_enable_occlusion_check=gs_enable_occlusion_check,
                allow_mask_resize=allow_mask_resize,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return StreamingResponse(
            stream_predict_response(runner, request),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FastAPI binary streaming server for SAM 3D real2sim inference.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "hf" / "pipeline.yaml",
        help="Path to the SAM 3D pipeline config.",
    )
    parser.add_argument(
        "--compile",
        type=parse_bool,
        default=False,
        help="Whether to torch.compile the pipeline on startup.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(config_path=args.config, compile_model=args.compile)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
