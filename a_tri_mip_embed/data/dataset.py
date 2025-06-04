import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random

from .utils import (
    load_preprocessed_planes,
    get_rotation_label,
    find_all_checkpoints
)

class NeRFPairDataset(Dataset):
    """
    Dataset that loads pairs of NeRFs (base + augmented) for rotation classification.
    
    Now uses preprocessed feature planes for faster loading.
    
    Always returns:
    - base_planes: The base (non-rotated) model planes
    - aug_planes: The rotated model planes
    - rotation_label: Integer label (0-4) for the rotation
    - metadata: Additional info (object_id, rotation_name)
    """
    
    def __init__(
        self, 
        data_root: str, 
        split: str = 'train', 
        train_split: float = 0.8,
        preprocessed_root: str = None,
        use_fallback: bool = False
    ):
        """
        Args:
            data_root: Root directory containing all checkpoints (for fallback)
            split: 'train' or 'val'
            train_split: Fraction of objects to use for training
            preprocessed_root: Root directory for preprocessed features
            use_fallback: Whether to fallback to raw checkpoints if preprocessed missing
        """
        self.data_root = Path(data_root)
        self.split = split
        self.use_fallback = use_fallback
        
        # Set preprocessed root
        if preprocessed_root is None:
            self.preprocessed_root = Path("./preprocessed_features")
        else:
            self.preprocessed_root = Path(preprocessed_root)
        
        # Check if preprocessed directory exists
        if not self.preprocessed_root.exists():
            if use_fallback:
                print(f"Warning: Preprocessed features directory not found: {self.preprocessed_root}")
                print("Will use raw checkpoint fallback")
            else:
                raise FileNotFoundError(
                    f"Preprocessed features directory not found: {self.preprocessed_root}\n"
                    f"Please run preprocessing script first:\n"
                    f"python scripts/preprocess_data.py --config configs/base_config.yaml\n"
                    f"Or set use_fallback=True to use raw checkpoints"
                )
        
        # Find all checkpoints (used for fallback and validation)
        if use_fallback:
            all_checkpoints = find_all_checkpoints(data_root)
        else:
            all_checkpoints = {}  # Don't scan raw data if not using fallback

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
        
        if use_fallback and all_checkpoints:
            # Use raw checkpoint validation
            for obj_id, rotations in all_checkpoints.items():
                if all(rot in rotations for rot in required_rotations):
                    self.valid_objects.append(obj_id)
        else:
            # Use preprocessed file validation
            self.valid_objects = self._find_complete_preprocessed_objects(required_rotations)
        
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
        
        # Store checkpoint mapping for fallback
        self.checkpoints = all_checkpoints
        
        # Create all valid pairs (base + each rotation except base)
        rotation_options = [
            'x_180_000_000', 
            'y_000_180_000', 
            'z_000_000_120', 
            'z_000_000_240',
            'compound_090_000_090'
        ]
        
        initial_pairs = []
        for obj_id in self.object_ids:
            for rotation in rotation_options:
                initial_pairs.append((obj_id, rotation))
        
        print(f"Initial {split} pairs: {len(initial_pairs)}")
        
        # Pre-scan for corrupted files and filter them out
        print(f"Scanning for corrupted preprocessed files...")
        self.pairs = []
        corrupted_count = 0
        
        for obj_id, rotation in initial_pairs:
            try:
                # Check base planes
                base_path = self.preprocessed_root / "planes" / f"{obj_id}_base_000_000_000.pt"
                aug_path = self.preprocessed_root / "planes" / f"{obj_id}_{rotation}.pt"
                
                is_corrupted = False
                
                # Check base file
                if base_path.exists():
                    try:
                        base_planes = torch.load(base_path, map_location='cpu')
                        if base_planes.isnan().any() or base_planes.isinf().any():
                            print(f"  Corrupted base planes: {obj_id}")
                            is_corrupted = True
                    except Exception as e:
                        print(f"  Error loading base {obj_id}: {e}")
                        is_corrupted = True
                elif not use_fallback:
                    # If not using fallback and file doesn't exist, skip
                    is_corrupted = True
                
                # Check augmented file
                if not is_corrupted and aug_path.exists():
                    try:
                        aug_planes = torch.load(aug_path, map_location='cpu')
                        if aug_planes.isnan().any() or aug_planes.isinf().any():
                            print(f"  Corrupted aug planes: {obj_id}/{rotation}")
                            is_corrupted = True
                    except Exception as e:
                        print(f"  Error loading aug {obj_id}/{rotation}: {e}")
                        is_corrupted = True
                elif not is_corrupted and not use_fallback:
                    # If not using fallback and file doesn't exist, skip
                    is_corrupted = True
                
                # Add to clean pairs if not corrupted
                if not is_corrupted:
                    self.pairs.append((obj_id, rotation))
                else:
                    corrupted_count += 1
                    
            except Exception as e:
                print(f"  Unexpected error checking {obj_id}/{rotation}: {e}")
                corrupted_count += 1
        
        print(f"Excluded {corrupted_count} corrupted pairs")
        print(f"Final {split} pairs: {len(self.pairs)}")
        
        if len(self.pairs) == 0:
            raise ValueError(f"No valid pairs found for {split} split! All files may be corrupted.")

    def _find_complete_preprocessed_objects(self, required_rotations: List[str]) -> List[str]:
        """Find objects that have all required preprocessed files."""
        complete_objects = []
        
        if not (self.preprocessed_root / "planes").exists():
            print(f"Warning: Preprocessed planes directory not found: {self.preprocessed_root / 'planes'}")
            return complete_objects
        
        # Get all preprocessed plane files
        plane_files = list((self.preprocessed_root / "planes").glob("*.pt"))
        
        # Group by object ID
        object_files = {}
        for file_path in plane_files:
            # Parse filename: {obj_id}_{rotation}.pt
            filename = file_path.stem
            parts = filename.split('_')
            
            if len(parts) >= 4:  # Ensure we have obj_id + rotation parts
                # Find where rotation part starts (look for known rotation patterns)
                for i in range(1, len(parts) - 2):
                    potential_rotation = '_'.join(parts[i:])
                    if potential_rotation in required_rotations:
                        obj_id = '_'.join(parts[:i])
                        if obj_id not in object_files:
                            object_files[obj_id] = []
                        object_files[obj_id].append(potential_rotation)
                        break
        
        # Check which objects have all required rotations
        for obj_id, rotations in object_files.items():
            if all(rot in rotations for rot in required_rotations):
                complete_objects.append(obj_id)
        
        print(f"Found {len(complete_objects)} objects with complete preprocessed features")
        return complete_objects

    def _verify_all_preprocessed_files(self):
        """Verify that all preprocessed files exist for the objects in this split."""
        missing_files = []
        required_rotations = ['base_000_000_000', 'x_180_000_000', 'y_000_180_000', 
                            'z_000_000_120', 'z_000_000_240', 'compound_090_000_090']
        
        for obj_id in self.object_ids:
            for rotation in required_rotations:
                preprocessed_path = self.preprocessed_root / "planes" / f"{obj_id}_{rotation}.pt"
                if not preprocessed_path.exists():
                    missing_files.append(str(preprocessed_path))
        
        if missing_files:
            print(f"Error: {len(missing_files)} preprocessed files are missing:")
            for i, file_path in enumerate(missing_files[:10]):  # Show first 10
                print(f"  {file_path}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")
            
            raise FileNotFoundError(
                f"Missing {len(missing_files)} preprocessed files. "
                f"Please run preprocessing script or set use_fallback=True"
            )
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obj_id, rotation = self.pairs[idx]
        
        # Get fallback checkpoint paths if using fallback
        base_checkpoint = None
        aug_checkpoint = None
        if self.use_fallback and obj_id in self.checkpoints:
            base_checkpoint = self.checkpoints[obj_id].get('base_000_000_000')
            aug_checkpoint = self.checkpoints[obj_id].get(rotation)
        
        # Load preprocessed planes (with optional fallback)
        base_planes = load_preprocessed_planes(
            obj_id=obj_id,
            rotation='base_000_000_000',
            preprocessed_root=self.preprocessed_root,
            fallback_checkpoint_path=base_checkpoint,
            allow_fallback=self.use_fallback
        )
        
        aug_planes = load_preprocessed_planes(
            obj_id=obj_id,
            rotation=rotation,
            preprocessed_root=self.preprocessed_root,
            fallback_checkpoint_path=aug_checkpoint,
            allow_fallback=self.use_fallback
        )
        
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
    # Get fallback setting from config
    use_fallback = config['data'].get('use_raw_fallback', False)
    
    train_dataset = NeRFPairDataset(
        data_root=config['data']['data_root'],
        split='train',
        preprocessed_root=config['data'].get('preprocessed_root', None),
        use_fallback=use_fallback
    )
    
    val_dataset = NeRFPairDataset(
        data_root=config['data']['data_root'],
        split='val',
        preprocessed_root=config['data'].get('preprocessed_root', None),
        use_fallback=use_fallback
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