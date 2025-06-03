import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random

from .utils import (
    extract_planes_from_checkpoint,
    get_rotation_label,
    find_all_checkpoints
)

class NeRFPairDataset(Dataset):
    """
    Dataset that loads pairs of NeRFs (base + augmented) for rotation classification.
    
    Always returns:
    - base_planes: The base (non-rotated) model planes
    - aug_planes: The rotated model planes
    - rotation_label: Integer label (0-4) for the rotation
    - metadata: Additional info (object_id, rotation_name)
    """
    
    def __init__(self, data_root: str, split: str = 'train', train_split: float = 0.8):
        """
        Args:
            data_root: Root directory containing all checkpoints
            split: 'train' or 'val'
            train_split: Fraction of objects to use for training
        """
        self.data_root = Path(data_root)
        self.split = split
        
        # Find all checkpoints
        all_checkpoints = find_all_checkpoints(data_root)
        
        # Filter to only include objects that have all rotations
        self.valid_objects = []
        required_rotations = [
            'base_000_000_000', 
            'x_180_000_000', 
            'y_000_180_000', 
            'z_000_000_120', 
            'z_000_000_240',
            'compound_090_000_090'
        ]
        
        for obj_id, rotations in all_checkpoints.items():
            if all(rot in rotations for rot in required_rotations):
                self.valid_objects.append(obj_id)
        
        print(f"Found {len(self.valid_objects)} objects with all rotations")
        
        # Split into train/val
        n_train = int(len(self.valid_objects) * train_split)
        
        # Use a fixed seed for reproducible splits
        random.Random(42).shuffle(self.valid_objects)
        
        if split == 'train':
            self.object_ids = self.valid_objects[:n_train]
        else:
            self.object_ids = self.valid_objects[n_train:]
        
        print(f"{split} set: {len(self.object_ids)} objects")
        
        # Store checkpoint mapping
        self.checkpoints = all_checkpoints
        
        # Create all valid pairs (base + each rotation except base)
        self.pairs = []
        rotation_options = [
            'x_180_000_000', 
            'y_000_180_000', 
            'z_000_000_120', 
            'z_000_000_240',
            'compound_090_000_090'
        ]
        
        for obj_id in self.object_ids:
            for rotation in rotation_options:
                self.pairs.append((obj_id, rotation))
        
        print(f"Total {split} pairs: {len(self.pairs)}")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obj_id, rotation = self.pairs[idx]
        
        # Load base model planes
        base_checkpoint = self.checkpoints[obj_id]['base_000_000_000']
        base_planes = extract_planes_from_checkpoint(base_checkpoint)
        
        # Load augmented model planes
        aug_checkpoint = self.checkpoints[obj_id][rotation]
        aug_planes = extract_planes_from_checkpoint(aug_checkpoint)
        
        # Get rotation label
        rotation_label = get_rotation_label(rotation)
        
        return {
            'base_planes': base_planes.float(),
            'aug_planes': aug_planes.float(),
            'rotation_label': torch.tensor(rotation_label, dtype=torch.long),
            'object_id': obj_id,
            'rotation_name': rotation
        }

def create_dataloaders(config: dict):
    """
    Create train and validation dataloaders from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = NeRFPairDataset(
        data_root=config['data']['data_root'],
        split='train'
    )
    
    val_dataset = NeRFPairDataset(
        data_root=config['data']['data_root'],
        split='val'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=config['data']['shuffle'],
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader