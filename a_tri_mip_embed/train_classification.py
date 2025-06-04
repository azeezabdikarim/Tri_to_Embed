#!/usr/bin/env python3
"""
Main training script for Week 1: Rotation Classification from Feature Planes
"""

import sys
sys.path.append('.')  # Add current directory to path

import torch
import yaml
import argparse
from pathlib import Path
import random
import numpy as np

from data.dataset import create_dataloaders
from models.cnn_classification.resnet_encoder import create_resnet_classifier
from trainers.classification_trainer import ClassificationTrainer
from evaluation.analyze_embeddings import run_full_embedding_analysis
from training_utils.logging import save_config, log_model_info

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train rotation classification model')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root from config')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation on existing checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--use_fallback', action='store_true',
                        help='Enable fallback to raw checkpoints (overrides config)')
    parser.add_argument('--preprocessed_only', action='store_true',
                        help='Use only preprocessed features, no fallback (overrides config)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override data root if provided
    if args.data_root:
        config['data']['data_root'] = args.data_root
    
    # Override fallback setting if provided via command line
    if args.use_fallback:
        config['data']['use_raw_fallback'] = True
        print("Command line override: Enabled raw checkpoint fallback")
    elif args.preprocessed_only:
        config['data']['use_raw_fallback'] = False
        print("Command line override: Disabled raw checkpoint fallback (preprocessed only)")
    
    # Print data mode
    if config['data'].get('use_raw_fallback', False):
        print("Data mode: Using preprocessed features with raw checkpoint fallback")
    else:
        print("Data mode: Using preprocessed features only (no fallback)")
    
    # Set seed
    set_seed(config['seed'])
    
    # Create data loaders
    print("\nCreating data loaders...")
    try:
        train_loader, val_loader = create_dataloaders(config)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    except FileNotFoundError as e:
        print(f"Error creating data loaders: {e}")
        print("\nSuggestions:")
        print("1. Run preprocessing script: python scripts/preprocess_data.py --config configs/base_config.yaml")
        print("2. Use --use_fallback flag to enable raw checkpoint fallback")
        print("3. Check that preprocessed_root path in config is correct")
        return 1
    
    # Create model
    print("\nCreating model...")
    model = create_resnet_classifier(config)
    print(f"Model type: {config['model']['encoder_type']}")
    print(f"Plane fusion: {config['model']['plane_fusion']}")
    print(f"Pair combination: {config['model']['pair_combination']}")
    
    # Move to device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if args.eval_only:
        # Evaluation only mode
        if args.checkpoint:
            print(f"\nLoading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("--checkpoint must be provided in eval_only mode")
        
        # Run evaluation
        print("\nRunning evaluation...")
        save_dir = Path(config['logging']['log_dir']) / 'eval_only'
        run_full_embedding_analysis(model, val_loader, save_dir, device)
    else:
        # Training mode
        print("\nCreating trainer...")
        trainer = ClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=args.device
        )
        
        # Save config to the timestamped directory
        save_config(config, trainer.timestamped_log_dir)
        
        # Log model info
        log_model_info(model, trainer.writer)
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Final evaluation using the same timestamped directory
        print("\nRunning final evaluation...")
        run_full_embedding_analysis(
            model,
            val_loader,
            trainer.timestamped_log_dir,
            device
        )

if __name__ == '__main__':
    main()