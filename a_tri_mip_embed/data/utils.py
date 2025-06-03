import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

def extract_planes_from_checkpoint(checkpoint_path: str) -> torch.Tensor:
    """
    Extract feature planes from TriMipRF checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        planes: Tensor of shape (3, 512, 512, 16)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
    
    # TriMipRF checkpoints store the field encoding as 'field.encoding.fm'
    # This is the feature map tensor containing all three planes
    fm_key = 'field.encoding.fm'
    
    if fm_key in checkpoint:
        # The feature map tensor has shape (3, 512, 512, 16)
        planes = checkpoint[fm_key]
        
        # Verify shape
        if planes.shape != (3, 512, 512, 16):
            raise ValueError(f"Unexpected feature map shape: {planes.shape}")
        
        return planes
    else:
        raise KeyError(f"Could not find feature maps at key '{fm_key}' in checkpoint. "
                      f"Available keys: {list(checkpoint.keys())}")

def get_rotation_label(rotation_name: str) -> int:
    """
    Convert rotation string to class index.
    
    Args:
        rotation_name: Directory name like 'x_180_000_000', 'y_000_180_000', etc.
        
    Returns:
        Class index (0-4)
    """
    rotation_to_idx = {
        'x_180_000_000': 0,    # x180
        'y_000_180_000': 1,    # y180
        'z_000_000_120': 2,    # z120
        'z_000_000_240': 3,    # z240
        'compound_090_000_090': 4  # compound rotation
    }
    
    if rotation_name not in rotation_to_idx:
        raise ValueError(f"Unknown rotation: {rotation_name}. "
                        f"Valid rotations: {list(rotation_to_idx.keys())}")
    
    return rotation_to_idx[rotation_name]

def parse_checkpoint_path(path: Path) -> Tuple[str, str]:
    """
    Parse object ID and rotation from checkpoint path.
    
    Expected format: /path/to/data/object_id/rotation/checkpoint.pth
    
    Args:
        path: Path to checkpoint
        
    Returns:
        object_id: String identifier for the object
        rotation: Rotation string (base, x90, x180, etc.)
    """
    parts = path.parts
    
    # Assuming structure like: .../object_001/x90/checkpoint.pth
    # Adjust based on your actual structure
    if len(parts) >= 3:
        object_id = parts[-3]
        rotation = parts[-2]
        return object_id, rotation
    else:
        raise ValueError(f"Cannot parse path: {path}")

def find_all_checkpoints(data_root: str) -> Dict[str, Dict[str, Path]]:
    """
    Find all checkpoints organized by object and rotation.
    
    Args:
        data_root: Root directory containing all checkpoints
        
    Returns:
        Dictionary mapping object_id -> rotation -> checkpoint_path
    """
    data_root = Path(data_root)
    checkpoints = {}
    
    # Iterate through all object directories
    for obj_dir in sorted(data_root.iterdir()):
        if not obj_dir.is_dir():
            continue
            
        obj_id = obj_dir.name
        checkpoints[obj_id] = {}
        
        # Look for variation directories
        for var_dir in sorted(obj_dir.iterdir()):
            if not var_dir.is_dir():
                continue
                
            variation_name = var_dir.name
            
            # Look for model.ckpt file
            ckpt_path = var_dir / "model.ckpt"
            if ckpt_path.exists():
                checkpoints[obj_id][variation_name] = ckpt_path
    
    return checkpoints