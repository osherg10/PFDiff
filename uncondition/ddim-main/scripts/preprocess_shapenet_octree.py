"""Preprocess ShapeNet meshes/point clouds into octree token .npz files.

Example usage:
    python scripts/preprocess_shapenet_octree.py \
        --input-root /path/to/shapenet \
        --output-root /path/to/shapenet_octree \
        --category chair \
        --split train \
        --extension .obj \
        --num-points 4096 \
        --max-depth 6

Input expectations:
    <input-root>/<category>/<split>/<shape_id><extension>

Output layout:
    <output-root>/<category>/<split>/<shape_id>.npz
    <output-root>/<category>/<split>.txt  (manifest of saved files)
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Tuple

import numpy as np


def _load_points(path: str, num_points: int) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        if "points" in data:
            points = data["points"]
        elif "pointcloud" in data:
            points = data["pointcloud"]
        else:
            raise KeyError(f"No 'points' or 'pointcloud' array in {path}")
        return points

    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "trimesh is required to load mesh files; install trimesh or use .npz point clouds"
        ) from exc

    mesh = trimesh.load(path, force="mesh")
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {path}")
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def _normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    scale = np.max(max_corner - min_corner)
    if scale <= 0:
        raise ValueError("Point cloud has zero extent.")
    normalized = (points - min_corner) / scale
    return np.clip(normalized, 0.0, 1.0)


def _octree_children(points: np.ndarray, center: np.ndarray) -> List[np.ndarray]:
    children = [None] * 8
    greater = points > center
    idx = (
        (greater[:, 0].astype(np.int64) << 2)
        | (greater[:, 1].astype(np.int64) << 1)
        | greater[:, 2].astype(np.int64)
    )
    for i in range(8):
        children[i] = points[idx == i]
    return children


def build_octree_tokens(points: np.ndarray, max_depth: int) -> List[int]:
    """Build a BFS-ordered octree token list."""
    tokens: List[int] = []
    queue: List[Tuple[np.ndarray, int, np.ndarray, np.ndarray]] = [
        (points, 0, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    ]

    while queue:
        node_points, depth, bounds_min, bounds_max = queue.pop(0)
        if node_points.size == 0:
            tokens.append(0)
            continue

        if depth >= max_depth:
            tokens.append(1)
            continue

        center = (bounds_min + bounds_max) * 0.5
        children = _octree_children(node_points, center)
        mask = 0
        for idx, child_points in enumerate(children):
            if child_points.size > 0:
                mask |= 1 << idx
        tokens.append(mask)

        for idx, child_points in enumerate(children):
            if child_points.size == 0:
                continue
            offset = np.array(
                [(idx >> 2) & 1, (idx >> 1) & 1, idx & 1], dtype=np.float32
            )
            child_min = bounds_min + (bounds_max - bounds_min) * 0.5 * offset
            child_max = child_min + (bounds_max - bounds_min) * 0.5
            queue.append((child_points, depth + 1, child_min, child_max))

    return tokens


def _iter_shapes(input_dir: str, extension: str) -> Iterable[str]:
    for name in sorted(os.listdir(input_dir)):
        if name.endswith(extension):
            yield os.path.join(input_dir, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--extension", default=".obj")
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--category-id", type=int, default=0)
    parser.add_argument("--shape-list", default=None)
    args = parser.parse_args()

    input_dir = os.path.join(args.input_root, args.category, args.split)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Missing input directory: {input_dir}")

    if args.max_nodes is None:
        args.max_nodes = (8 ** (args.max_depth + 1) - 1) // 7

    output_dir = os.path.join(args.output_root, args.category, args.split)
    os.makedirs(output_dir, exist_ok=True)

    if args.shape_list:
        with open(args.shape_list, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]
        shape_paths = [os.path.join(input_dir, name) for name in names]
    else:
        shape_paths = list(_iter_shapes(input_dir, args.extension))

    manifest_entries: List[str] = []
    for path in shape_paths:
        shape_id = os.path.splitext(os.path.basename(path))[0]
        points = _load_points(path, args.num_points)
        points = _normalize_points(points)
        tokens = build_octree_tokens(points, args.max_depth)
        if len(tokens) > args.max_nodes:
            raise ValueError(
                f"Octree token count {len(tokens)} exceeds max_nodes {args.max_nodes}"
            )

        padded_tokens = np.full(args.max_nodes, 0, dtype=np.int16)
        mask = np.zeros(args.max_nodes, dtype=np.uint8)
        padded_tokens[: len(tokens)] = np.asarray(tokens, dtype=np.int16)
        mask[: len(tokens)] = 1

        out_path = os.path.join(output_dir, f"{shape_id}.npz")
        np.savez_compressed(
            out_path,
            tokens=padded_tokens,
            mask=mask,
            category=np.asarray(args.category_id, dtype=np.int64),
            shape_id=np.asarray(shape_id),
        )
        manifest_entries.append(f"{shape_id}.npz")

    manifest_path = os.path.join(args.output_root, args.category, f"{args.split}.txt")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(manifest_entries))


if __name__ == "__main__":
    main()
