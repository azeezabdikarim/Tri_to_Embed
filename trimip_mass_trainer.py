#!/usr/bin/env python3
"""
train_trimipnerf.py  ―  Tri-MipRF trainer using Gin configuration

Replaces CLI flags for hyperparameters with a single Gin file (`--ginc`)
plus optional Gin bindings (`--ginb`). No manual GPU flags; uses
`torch.cuda.is_available()` to pick `cuda:0` or CPU.

Usage:
```bash
python train_trimipnerf.py \
  --ginc config/train_trimipnerf.gin \
  --ginb "Trainer.exp_name=None"
```
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime as dt
from pathlib import Path

import gin
import torch
from torch.utils.data import DataLoader

from dataset.ray_dataset import RayDataset, ray_collate
from neural_field.model import get_model
from trainer import Trainer


@gin.configurable()
def main(
    input_dir: str,
    output_dir: str,
    scene_type: str,
    train_split: str,
    model_name: str,
    iters: int,
    eval_interval: int,
    log_step: int,
    batch_size: int,
    num_rays: int,
    num_workers: int,
):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # Paths
    in_root = Path(input_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Log used Gin config
    (out_root / "config_used.gin").write_text(gin.operative_config_str(), encoding="utf-8")

    # Walk objects
    objects = [d for d in in_root.iterdir() if d.is_dir()]
    random.shuffle(objects)

    stamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    for obj in objects:
        vars_ = [
                p for p in obj.iterdir()
                if p.is_dir()
                and not p.name.lower().endswith("validation")
            ]
        for var in sorted(vars_):
            # Ensure splits exist
            generic = var / "transforms.json"
            if generic.exists() and not (var / "transforms_train.json").exists():
                for split in ["train", "val", "test"]:
                    tgt = var / f"transforms_{split}.json"
                    if not tgt.exists():
                        tgt.symlink_to(generic)

            print(f"⇒ {obj.name}/{var.name}")

            # Datasets
            train_ds = RayDataset(
                base_path=str(var.parent),
                scene=var.name,
                scene_type=scene_type,
                split=train_split,
                num_rays=num_rays,
                render_bkgd = "black"
            )
            test_ds = RayDataset(
                base_path=str(var.parent),
                scene=var.name,
                scene_type=scene_type,
                split="test",
                num_rays=num_rays * 10,
                render_bkgd = "black"
            )

            pin_dev = "cuda" if device.type == "cuda" else "cpu"
            train_ld = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=ray_collate,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                pin_memory_device=pin_dev,
            )
            test_ld = DataLoader(
                test_ds,
                batch_size=None,
                shuffle=False,
                num_workers=1,
                pin_memory=device.type == "cuda",
                pin_memory_device=pin_dev,
            )

            # Model
            model_cls = get_model(model_name)
            model = model_cls(aabb=train_ds.aabb)

            # Experiment name
            # stamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
            # exp_name = gin.query_parameter("Trainer.exp_name")
            # if exp_name is None:
            #     exp_name = f"{scene_type}/{obj.name}/{var.name}/{model_name}/{stamp}"
            #     gin.bind_parameter("Trainer.exp_name", exp_name)

            
            exp_name = f"{stamp}/{obj.name}/{var.name}"
            gin.bind_parameter("Trainer.exp_name", exp_name)

            # Trainer
            trainer = Trainer(
                model=model,
                train_loader=train_ld,
                eval_loader=test_ld,
                base_exp_dir=output_dir,
                exp_name=exp_name,
                max_steps=iters,
                log_step=log_step,
                eval_step=eval_interval,
                num_rays=num_rays,
            )

            trainer.fit()
            trainer.eval(save_results=True, rendering_channels=["rgb", "depth"])

    print("✅ Finished all variations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tri-MipRF trainer (Gin-powered)"
    )
    parser.add_argument(
        "--ginc", action="append", help="Gin config file"
    )
    parser.add_argument(
        "--ginb", action="append", help="Gin bindings"
    )
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(
        args.ginc or [],
        args.ginb or [],
        finalize_config=False,
    )
    main()
