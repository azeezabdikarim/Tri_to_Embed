#!/usr/bin/env python3
"""split_transforms.py
----------------------------------
Walk every sub‑directory under a given *root*; whenever we find a
single‑file `transforms.json` (Blender NeRF style) **and** there is no
`transforms_train.json`, generate three split files:

    transforms_train.json   ~80 %
    transforms_val.json     ~10 %
    transforms_test.json    ~10 %

The ratios are user‑configurable.

Example
~~~~~~~
```bash
python split_transforms.py \
    --root /path/to/nerf_training_images_datasets/rotation/favs \
    --train_ratio 0.8 --val_ratio 0.1 --seed 0
```
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _load_transforms(json_path: Path) -> dict:
    """Read a JSON file and return its dict."""
    with open(json_path, "r", encoding="utf‑8") as fp:
        return json.load(fp)


def _write_transforms(data: dict, dest: Path) -> None:
    dest.write_text(json.dumps(data, indent=2), encoding="utf‑8")


def _split_indices(n: int, train_r: float, val_r: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


# -----------------------------------------------------------------------------
# main routine per variation folder
# -----------------------------------------------------------------------------

def _process_variation(var_dir: Path, ratios: Tuple[float, float, float], seed: int):
    generic = var_dir / "transforms.json"

    train_f = var_dir / "transforms_train.json"
    val_f   = var_dir / "transforms_val.json"
    test_f  = var_dir / "transforms_test.json"

    if not generic.exists():
        return  # nothing to do
    if train_f.exists() and val_f.exists() and test_f.exists():
        return  # already split

    data = _load_transforms(generic)
    frames = data.get("frames")
    if not frames:
        print(f"⚠️  {generic} has no 'frames' list – skipping")
        return

    train_r, val_r, test_r = ratios
    if abs((train_r + val_r + test_r) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    tr_idx, va_idx, te_idx = _split_indices(len(frames), train_r, val_r, seed)

    def _subset(idxs: List[int]) -> dict:
        subset = {k: v for k, v in data.items() if k != "frames"}
        subset["frames"] = [frames[i] for i in idxs]
        return subset

    _write_transforms(_subset(tr_idx), train_f)
    _write_transforms(_subset(va_idx), val_f)
    _write_transforms(_subset(te_idx), test_f)

    print(f"✅  Split {generic.relative_to(root)} → train:{len(tr_idx)} val:{len(va_idx)} test:{len(te_idx)}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("create Tri‑MipRF style train/val/test json splits")
    p.add_argument("--root", required=True, type=str, help="root directory to recurse through")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="PRNG seed for deterministic splits")
    opt = p.parse_args()

    global root  # only for pretty printing inside helper
    root = Path(opt.root).expanduser().resolve()

    test_ratio = 1.0 - opt.train_ratio - opt.val_ratio
    ratios = (opt.train_ratio, opt.val_ratio, test_ratio)
    if min(ratios) < 0:
        raise ValueError("ratios produce negative test share – adjust your numbers")

    # walk
    for path in root.rglob("transforms.json"):
        _process_variation(path.parent, ratios, opt.seed)


if __name__ == "__main__":
    main()
