#!/usr/bin/env python3
"""
Main training script for NeRF Rotation Classification

Supports both planes-only and hybrid (planes + MLP) models with automatic
experiment management and organization.
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
from models import create_model
from trainers.classification_trainer import ClassificationTrainer
from evaluation.analyze_embeddings import run_full_embedding_analysis
from training_utils.logging import save_config, log_model_info
from training_utils.experiment_manager import ExperimentManager

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_config(config: dict):
    model_config = config.get('model', {})
    use_planes = model_config.get('use_planes', True)
    use_mlp_features = model_config.get('use_mlp_features', False)
    
    if not use_planes and not use_mlp_features:
        raise ValueError("At least one of 'use_planes' or 'use_mlp_features' must be True")
    
    if use_mlp_features and not config['data'].get('load_mlp_features', False):
        config['data']['load_mlp_features'] = True
        print("Auto-enabled MLP feature loading in data config")

def main():
    parser = argparse.ArgumentParser(description='Train rotation classification model')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--use_fallback', action='store_true')
    parser.add_argument('--preprocessed_only', action='store_true')
    parser.add_argument('--experiment_name', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.data_root:
        config['data']['data_root'] = args.data_root
    
    if args.use_fallback:
        config['data']['use_raw_fallback'] = True
        print("Command line override: Enabled raw checkpoint fallback")
    elif args.preprocessed_only:
        config['data']['use_raw_fallback'] = False
        print("Command line override: Disabled raw checkpoint fallback")
    
    validate_config(config)
    
    if config['data'].get('use_raw_fallback', False):
        print("Data mode: Using preprocessed features with raw checkpoint fallback")
    else:
        print("Data mode: Using preprocessed features only")
    
    set_seed(config['seed'])
    
    exp_manager = ExperimentManager(config['logging']['log_dir'])
    
    print("\nCreating data loaders...")
    try:
        train_loader, val_loader = create_dataloaders(config)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    except FileNotFoundError as e:
        print(f"Error creating data loaders: {e}")
        print("\nSuggestions:")
        print("1. Run preprocessing script")
        print("2. Use --use_fallback flag")
        print("3. Check preprocessed_root path")
        return 1
    
    print("\nCreating model...")
    model = create_model(config)
    
    model_config = config['model']
    print(f"Model type: {'Hybrid' if model_config.get('use_mlp_features', False) else 'Planes-only'}")
    if model_config.get('use_planes', True):
        print(f"Plane encoder: {model_config.get('encoder_type', 'resnet18')}")
        print(f"Plane fusion: {model_config.get('plane_fusion', 'concat')}")
    if model_config.get('use_mlp_features', False):
        print(f"MLP features enabled")
        mlp_config = model_config.get('mlp_encoders', {})
        print(f"Pooling: {mlp_config.get('pooling_strategy', 'max')}")
        print(f"Layer fusion: {model_config.get('fusion', {}).get('layer_fusion', 'concat')}")
    print(f"Pair combination: {model_config.get('pair_combination', 'subtract')}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if args.eval_only:
        if args.checkpoint:
            print(f"\nLoading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("--checkpoint must be provided in eval_only mode")
        
        print("\nRunning evaluation...")
        save_dir = Path(config['logging']['log_dir']) / 'eval_only'
        run_full_embedding_analysis(model, val_loader, save_dir, device)
    else:
        experiment_dir = exp_manager.create_experiment_directory(
            config, 
            experiment_name=args.experiment_name
        )
        
        print(f"\nExperiment directory: {experiment_dir}")
        
        config['logging']['log_dir'] = str(experiment_dir)
        
        print("\nCreating trainer...")
        trainer = ClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=args.device
        )
        
        save_config(config, trainer.timestamped_log_dir)
        log_model_info(model, trainer.writer)
        
        print("\nStarting training...")
        trainer.train()
        
        print("\nRunning final evaluation...")
        run_full_embedding_analysis(
            model,
            val_loader,
            trainer.timestamped_log_dir,
            device
        )

if __name__ == '__main__':
    main()