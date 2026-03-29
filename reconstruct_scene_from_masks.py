#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "notebook"))

from inference import (  # noqa: E402
    Inference,
    load_image,
    load_masks,
)
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix  # noqa: E402
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform  # noqa: E402


MODEL_TO_GLB_BASIS = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)


def parse_args() -> argparse.Namespace:
    default_input_dir = ROOT / "masks"
    default_config = ROOT / "checkpoints" / "hf" / "pipeline.yaml"

    parser = argparse.ArgumentParser(
        description=(
            "Run multi-mask SAM 3D reconstruction with Gaussian post-optimization "
            "and export a complete textured scene GLB."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Directory containing image.png and mask files named 0.png, 1.png, ...",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional explicit image path. Defaults to <input-dir>/image.png.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to the SAM 3D pipeline config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input-dir>/outputs.",
    )
    parser.add_argument(
        "--mask-extension",
        type=str,
        default=".png",
        help="Mask file extension. Default: .png",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for each object reconstruction.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the inference pipeline.",
    )
    parser.add_argument(
        "--save-per-object",
        action="store_true",
        help="Also export each reconstructed object textured GLB.",
    )
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_dir = args.input_dir.resolve()
    image_path = (args.image_path or (input_dir / "image.png")).resolve()
    config_path = args.config.resolve()
    output_dir = (args.output_dir or (input_dir / "outputs")).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, image_path, config_path, output_dir


def normalize_vector(value, expected_len: int) -> list[float]:
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


def canonical_pose_to_glb_pose(output: dict) -> dict:
    rotation = normalize_vector(output["rotation"], 4)
    translation = normalize_vector(output["translation"], 3)
    scale = normalize_vector(output["scale"], 3)

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

    return {
        "rotation": rotation_quat.cpu().numpy().tolist(),
        "translation": glb_transform[:3, 3].cpu().numpy().tolist(),
        "scale": scale_vec.cpu().numpy().tolist(),
        "transform_matrix": glb_transform.cpu().numpy().tolist(),
    }


def pose_to_matrix(pose: dict) -> np.ndarray:
    if "transform_matrix" in pose:
        return np.asarray(pose["transform_matrix"], dtype=np.float64)

    rotation = normalize_vector(pose["rotation"], 4)
    translation = normalize_vector(pose["translation"], 3)
    scale = normalize_vector(pose["scale"], 3)

    quat = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0)
    trans = torch.tensor(translation, dtype=torch.float32).unsqueeze(0)
    scale_t = torch.tensor(scale, dtype=torch.float32).unsqueeze(0)

    transform_p3d = compose_transform(
        scale=scale_t,
        rotation=quaternion_to_matrix(quat),
        translation=trans,
    ).get_matrix()[0]
    return transform_p3d.T.cpu().numpy().astype(np.float64)


def merge_scene_glb(
    objects: list[tuple[str, trimesh.Trimesh | trimesh.Scene, dict]]
) -> trimesh.Scene:
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
    return scene


def build_metadata(
    image_path: Path,
    mask_count: int,
    seed: int,
    outputs: list[dict],
    scene_glb_path: Path,
) -> dict:
    object_summaries = []
    for idx, output in enumerate(outputs):
        object_summaries.append(
            {
                "mask_index": idx,
                "iou": float(output.get("iou", 0.0)),
                "iou_before_optim": float(output.get("iou_before_optim", 0.0)),
                "optim_accepted": bool(output.get("optim_accepted", False)),
            }
        )

    return {
        "image_path": str(image_path),
        "num_masks": mask_count,
        "seed": seed,
        "layout_post_optimization": True,
        "scene_glb": str(scene_glb_path),
        "objects": object_summaries,
    }


def main() -> None:
    args = parse_args()
    input_dir, image_path, config_path, output_dir = validate_paths(args)

    image = load_image(str(image_path))
    masks = load_masks(str(input_dir), extension=args.mask_extension)
    if len(masks) == 0:
        raise RuntimeError(
            f"No masks found in {input_dir} with extension {args.mask_extension}"
        )

    inference = Inference(str(config_path), compile=args.compile)

    outputs = []
    scene_objects = []
    image_stem = image_path.stem
    for idx, mask in enumerate(masks):
        print(f"[{idx + 1}/{len(masks)}] reconstructing mask {idx} with post-optimization")
        output = inference(
            image,
            mask,
            seed=args.seed,
            with_layout_postprocess=True,
            with_mesh_postprocess=True,
            with_texture_baking=True,
            use_vertex_color=False,
        )
        outputs.append(output)

        mesh = output.get("glb")
        if mesh is None:
            raise RuntimeError(f"Pipeline did not return a textured GLB mesh for mask {idx}")

        pose = canonical_pose_to_glb_pose(output)
        object_name = f"{image_stem}_object_{idx:02d}"
        scene_objects.append((object_name, mesh, pose))

        if args.save_per_object:
            object_path = output_dir / f"{object_name}.glb"
            mesh.export(object_path)

    scene_glb = merge_scene_glb(scene_objects)
    scene_glb_path = output_dir / f"{image_stem}_scene_post_optimized.glb"
    scene_glb.export(scene_glb_path)

    metadata = build_metadata(
        image_path=image_path,
        mask_count=len(masks),
        seed=args.seed,
        outputs=outputs,
        scene_glb_path=scene_glb_path,
    )
    metadata_path = output_dir / f"{image_stem}_scene_post_optimized.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved textured scene GLB to: {scene_glb_path}")
    print(f"Saved reconstruction metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
