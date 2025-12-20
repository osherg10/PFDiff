"""ShapeNet octree token dataset.

On-disk format
--------------
Each sample is stored as a ``.npz`` file with the following arrays:

* ``tokens``: int array of shape ``(max_nodes,)`` containing octree tokens.
  Internal nodes store an 8-bit occupancy mask (0-255). Leaf nodes store
  ``1`` if occupied, ``0`` otherwise.
* ``mask``: uint8/bool array of shape ``(max_nodes,)`` where 1 marks valid
  tokens and 0 marks padding.
* ``category``: scalar int with the category ID (user-defined).
* ``shape_id``: string array with the ShapeNet identifier.

Expected folder layout:

root/
  <category>/
    train/
      <shape_id>.npz
    val/
    test/
    train.txt  # optional manifest listing .npz file names
    val.txt
    test.txt
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetOctreeDataset(Dataset):
    """Dataset for octree tokens derived from ShapeNet shapes."""

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        *,
        return_mask: bool = True,
        manifest: Optional[str] = None,
    ) -> None:
        self.root = root
        self.category = category
        self.split = split
        self.return_mask = return_mask
        category_dir = os.path.join(root, category, split)
        if manifest is None:
            manifest = os.path.join(root, category, f"{split}.txt")

        if os.path.isfile(manifest):
            with open(manifest, "r", encoding="utf-8") as handle:
                rel_paths = [line.strip() for line in handle if line.strip()]
            self.samples = [
                os.path.join(category_dir, path) if not os.path.isabs(path) else path
                for path in rel_paths
            ]
        else:
            if not os.path.isdir(category_dir):
                raise FileNotFoundError(
                    f"Could not find split directory: {category_dir}"
                )
            self.samples = [
                os.path.join(category_dir, name)
                for name in sorted(os.listdir(category_dir))
                if name.endswith(".npz")
            ]

        if not self.samples:
            raise FileNotFoundError(
                f"No .npz files found for category '{category}' split '{split}'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, object]]:
        path = self.samples[index]
        data = np.load(path, allow_pickle=False)
        tokens = torch.from_numpy(data["tokens"]).long()
        shape_id = str(data["shape_id"]) if "shape_id" in data else os.path.basename(path)
        category = int(data["category"]) if "category" in data else -1
        mask_tensor: Optional[torch.Tensor] = None
        if self.return_mask and "mask" in data:
            mask_tensor = torch.from_numpy(data["mask"]).bool()

        meta: Dict[str, object] = {
            "category": torch.tensor(category, dtype=torch.long),
        }
        if mask_tensor is not None:
            meta["mask"] = mask_tensor
        meta["shape_id"] = shape_id
        return tokens, meta
