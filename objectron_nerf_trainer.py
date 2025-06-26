#!/usr/bin/env python3
"""
Objectron NeRF Trainer - Train Tri-MipRF on processed Objectron scenes

This script trains NeRFs on individual Objectron scenes that have been processed 
by the objectron_nerf_processor. Each scene is a complete NeRF dataset with
multiple camera poses and images.

Usage:
    python objectron_nerf_trainer.py --ginc config/objectron_nerf_config.gin
    python objectron_nerf_trainer.py --ginc config/objectron_nerf_config.gin --scene_filter bike
    python objectron_nerf_trainer.py --ginc config/objectron_nerf_config.gin --max_scenes 10
"""

import argparse
import json
import random
from datetime import datetime as dt
from pathlib import Path
import logging
import time

import gin
import torch
from torch.utils.data import DataLoader

from dataset.ray_dataset import RayDataset, ray_collate
from neural_field.model import get_model
from trainer import Trainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    seed: int = 42,
):
    """Main training function for Objectron NeRF scenes."""
    from utils.common import set_random_seed
    set_random_seed(seed)
    logger.info(f"Set random seed to: {seed}")

    # Setup paths
    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Log used Gin config
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    config_save_path = output_root / f"config_used_{timestamp}.gin"
    config_save_path.write_text(gin.operative_config_str(), encoding="utf-8")
    logger.info(f"Saved config to: {config_save_path}")

    # Find all valid scene directories
    scene_dirs = find_valid_scenes(input_root)
    logger.info(f"Found {len(scene_dirs)} valid Objectron scenes")

    if not scene_dirs:
        logger.error("No valid scenes found!")
        return

    # Apply filtering if requested
    args = get_args()  # Get command line args
    filtered_scenes = apply_scene_filtering(scene_dirs, args)
    
    logger.info(f"Processing {len(filtered_scenes)} scenes after filtering")

    # Process each scene
    success_count = 0
    failure_count = 0
    
    for i, scene_dir in enumerate(filtered_scenes, 1):
        scene_name = scene_dir.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing scene {i}/{len(filtered_scenes)}: {scene_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = train_single_scene(
                scene_dir=scene_dir,
                output_root=output_root,
                scene_type=scene_type,
                train_split=train_split,
                model_name=model_name,
                iters=iters,
                eval_interval=eval_interval,
                log_step=log_step,
                batch_size=batch_size,
                num_rays=num_rays,
                num_workers=num_workers,
                device=device,
                timestamp=timestamp
            )
            
            if success:
                success_count += 1
                logger.info(f"✅ Successfully trained {scene_name}")
            else:
                failure_count += 1
                logger.error(f"❌ Failed to train {scene_name}")
                
        except Exception as e:
            failure_count += 1
            logger.error(f"❌ Exception training {scene_name}: {e}")
            continue

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Successfully trained: {success_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info(f"Total scenes: {len(filtered_scenes)}")
    logger.info(f"Success rate: {success_count/len(filtered_scenes)*100:.1f}%")
    logger.info(f"Output directory: {output_root}")


def find_valid_scenes(input_root: Path) -> list[Path]:
    """Find all valid Objectron scene directories."""
    scene_dirs = []
    
    for scene_dir in input_root.iterdir():
        if not scene_dir.is_dir():
            continue
            
        # Check for required files
        transforms_file = scene_dir / "transforms.json"
        images_dir = scene_dir / "images"
        metadata_file = scene_dir / "scene_metadata.json"
        
        if not transforms_file.exists():
            logger.debug(f"Skipping {scene_dir.name}: missing transforms.json")
            continue
            
        if not images_dir.exists():
            logger.debug(f"Skipping {scene_dir.name}: missing images directory")
            continue
            
        # Check minimum number of images
        image_files = list(images_dir.glob("*.png"))
        if len(image_files) < 20:  # Minimum viable scene
            logger.debug(f"Skipping {scene_dir.name}: only {len(image_files)} images")
            continue
            
        # Validate transforms.json structure
        try:
            with open(transforms_file, 'r') as f:
                transforms = json.load(f)
            
            if 'frames' not in transforms or len(transforms['frames']) < 10:
                logger.debug(f"Skipping {scene_dir.name}: insufficient camera poses")
                continue
                
        except Exception as e:
            logger.debug(f"Skipping {scene_dir.name}: invalid transforms.json - {e}")
            continue
        
        scene_dirs.append(scene_dir)
    
    return sorted(scene_dirs)


def apply_scene_filtering(scene_dirs: list[Path], args) -> list[Path]:
    """Apply command line filtering to scene list."""
    filtered = scene_dirs.copy()
    
    # Filter by category if requested
    if args.scene_filter:
        category_filter = args.scene_filter.lower()
        filtered = [s for s in filtered if category_filter in s.name.lower()]
        logger.info(f"Filtered to {len(filtered)} scenes containing '{category_filter}'")
    
    # Limit number of scenes if requested
    if args.max_scenes and len(filtered) > args.max_scenes:
        # Random sample for diversity
        random.shuffle(filtered)
        filtered = filtered[:args.max_scenes]
        logger.info(f"Limited to {args.max_scenes} random scenes")
    
    return filtered


def train_single_scene(
    scene_dir: Path,
    output_root: Path,
    scene_type: str,
    train_split: str,
    model_name: str,
    iters: int,
    eval_interval: int,
    log_step: int,
    batch_size: int,
    num_rays: int,
    num_workers: int,
    device: torch.device,
    timestamp: str
) -> bool:
    """Train a single NeRF on one Objectron scene."""
    
    scene_name = scene_dir.name
    start_time = time.time()
    
    try:
        # Create output directory for this scene
        scene_output_dir = output_root / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already trained
        checkpoint_path = scene_output_dir / "model.ckpt"
        if checkpoint_path.exists():
            logger.info(f"Scene {scene_name} already trained, skipping...")
            return True
        
        logger.info(f"Creating datasets for scene: {scene_name}")
        
        # Create train dataset
        train_dataset = RayDataset(
            base_path=str(scene_dir.parent),
            scene=scene_dir.name,
            scene_type=scene_type,
            split=train_split,
            num_rays=num_rays,
            render_bkgd="black"
        )
        
        # Create test dataset for evaluation
        test_dataset = RayDataset(
            base_path=str(scene_dir.parent),
            scene=scene_dir.name,
            scene_type=scene_type,
            split="test",
            num_rays=num_rays * 10,
            render_bkgd="black"
        )
        
        # Create data loaders
        pin_dev = "cuda" if device.type == "cuda" else "cpu"
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ray_collate,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            pin_memory_device=pin_dev,
            prefetch_factor=2,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=1,
            pin_memory=(device.type == "cuda"),
            pin_memory_device=pin_dev,
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Test dataset: {len(test_dataset)} samples")
        
        # Create model
        model_cls = get_model(model_name)
        model = model_cls(aabb=train_dataset.aabb)
        model = model.to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        # Create experiment name (similar to Objaverse script)
        exp_name = f"objectron/{timestamp}/{scene_name}"
        
        # Create trainer with ALL necessary parameters (like working Objaverse script)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            eval_loader=test_loader,
            base_exp_dir=str(output_root),      # CRITICAL: Base directory for experiments
            exp_name=exp_name,                  # CRITICAL: Experiment name for logging
            max_steps=iters,                    # CRITICAL: Number of training steps
            log_step=log_step,                  # CRITICAL: Logging frequency
            eval_step=eval_interval,            # CRITICAL: Evaluation frequency (enables image generation!)
            num_rays=num_rays,                  # Number of rays per batch
        )
        
        # Train the model
        logger.info(f"Starting training for {iters} iterations...")
        logger.info(f"Images will be generated every {eval_interval} steps")
        logger.info(f"Logging every {log_step} steps")
        
        # This will now generate images during training at eval_interval steps
        trainer.fit()
        
        # Final evaluation with image generation
        logger.info("Running final evaluation with image generation...")
        trainer.eval(save_results=True, rendering_channels=["rgb", "depth"])
        
        # Save scene metadata (lightweight)
        save_scene_info(scene_dir, trainer.exp_dir)
        
        training_time = time.time() - start_time
        logger.info(f"Completed training {scene_name} in {training_time:.1f} seconds")
        logger.info(f"Results saved to: {trainer.exp_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to train scene {scene_name}: {e}")
        return False


def save_scene_info(scene_dir: Path, output_dir: Path):
    """Save scene metadata alongside the trained model."""
    try:
        # Copy scene metadata if it exists
        metadata_src = scene_dir / "scene_metadata.json"
        metadata_dst = output_dir / "scene_metadata.json"
        
        if metadata_src.exists():
            import shutil
            shutil.copy2(metadata_src, metadata_dst)
            logger.debug(f"Copied scene metadata to {metadata_dst}")
        
        # Save training info
        training_info = {
            "scene_name": scene_dir.name,
            "scene_path": str(scene_dir),
            "training_completed": dt.now().isoformat(),
            "model_type": "Tri-MipRF",
            "data_source": "objectron"
        }
        
        info_path = output_dir / "training_info.json"
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
            
    except Exception as e:
        logger.warning(f"Failed to save scene info: {e}")


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description="Train Tri-MipRF on Objectron scenes")
    parser.add_argument("--ginc", action="append", help="Gin config file")
    parser.add_argument("--ginb", action="append", help="Gin bindings")
    parser.add_argument("--scene_filter", type=str, help="Filter scenes by category (e.g., 'bike', 'chair')")
    parser.add_argument("--max_scenes", type=int, help="Maximum number of scenes to process")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Parse gin configuration
    gin.parse_config_files_and_bindings(
        args.ginc or [],
        args.ginb or [],
        finalize_config=False,
    )
    
    # Run training
    main()