import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
from tqdm import tqdm

class FeatureExtractor:
    """
    Handles extraction and saving of features from NeRF checkpoints.
    """
    
    def __init__(self, raw_data_root: str, preprocessed_root: Optional[str] = None):
        """
        Initialize the feature extractor.
        
        Args:
            raw_data_root: Path to raw checkpoint data
            preprocessed_root: Path to save preprocessed features (default: ./preprocessed_features/)
        """
        self.raw_data_root = Path(raw_data_root)
        
        if preprocessed_root is None:
            self.preprocessed_root = Path("./preprocessed_features")
        else:
            self.preprocessed_root = Path(preprocessed_root)
        
        # Create directories
        self.planes_dir = self.preprocessed_root / "planes"
        self.planes_dir.mkdir(parents=True, exist_ok=True)
        
        # Future directories for MLP features
        self.geo_mlp_dir = self.preprocessed_root / "geo_mlp"
        self.color_mlp_dir = self.preprocessed_root / "color_mlp"
    
    def extract_planes_from_checkpoint(self, checkpoint_path: Path) -> torch.Tensor:
        """
        Extract feature planes from TriMipRF checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            planes: Tensor of shape (3, 512, 512, 16)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
        
        # TriMipRF checkpoints store the field encoding as 'field.encoding.fm'
        fm_key = 'field.encoding.fm'
        
        if fm_key in checkpoint:
            planes = checkpoint[fm_key]
            
            # Verify shape
            if planes.shape != (3, 512, 512, 16):
                raise ValueError(f"Unexpected feature map shape: {planes.shape}")
            
            return planes
        else:
            raise KeyError(f"Could not find feature maps at key '{fm_key}' in checkpoint. "
                          f"Available keys: {list(checkpoint.keys())}")
    
    def get_preprocessed_path(self, obj_id: str, rotation: str, feature_type: str = "planes") -> Path:
        """
        Get the path where preprocessed features should be saved.
        
        Args:
            obj_id: Object identifier
            rotation: Rotation string
            feature_type: Type of feature ("planes", "geo_mlp", "color_mlp")
            
        Returns:
            Path to preprocessed feature file
        """
        filename = f"{obj_id}_{rotation}.pt"
        
        if feature_type == "planes":
            return self.planes_dir / filename
        elif feature_type == "geo_mlp":
            return self.geo_mlp_dir / filename
        elif feature_type == "color_mlp":
            return self.color_mlp_dir / filename
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def needs_extraction(self, checkpoint_path: Path, preprocessed_path: Path) -> bool:
        """
        Check if extraction is needed (preprocessed file missing or older than checkpoint).
        
        Args:
            checkpoint_path: Path to raw checkpoint
            preprocessed_path: Path to preprocessed feature file
            
        Returns:
            True if extraction is needed
        """
        if not preprocessed_path.exists():
            return True
        
        # Check if checkpoint is newer than preprocessed file
        checkpoint_mtime = checkpoint_path.stat().st_mtime
        preprocessed_mtime = preprocessed_path.stat().st_mtime
        
        return checkpoint_mtime > preprocessed_mtime
    
    def extract_and_save_planes(self, obj_id: str, rotation: str, checkpoint_path: Path) -> bool:
        """
        Extract planes from checkpoint and save if needed.
        
        Args:
            obj_id: Object identifier
            rotation: Rotation string
            checkpoint_path: Path to checkpoint
            
        Returns:
            True if extraction was performed, False if skipped
        """
        preprocessed_path = self.get_preprocessed_path(obj_id, rotation, "planes")
        
        # Check if extraction is needed
        if not self.needs_extraction(checkpoint_path, preprocessed_path):
            return False
        
        try:
            # Extract planes
            planes = self.extract_planes_from_checkpoint(checkpoint_path)
            
            # Save to preprocessed location
            torch.save(planes, preprocessed_path)
            
            return True
            
        except Exception as e:
            print(f"Error extracting from {checkpoint_path}: {e}")
            raise
    
    def find_all_checkpoints(self) -> Dict[str, Dict[str, Path]]:
        """
        Find all checkpoints organized by object and rotation.
        
        Returns:
            Dictionary mapping object_id -> rotation -> checkpoint_path
        """
        checkpoints = {}
        
        # Iterate through all object directories
        for obj_dir in sorted(self.raw_data_root.iterdir()):
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
    
    def process_all_objects(self, feature_types: List[str] = ["planes"]) -> Dict[str, int]:
        """
        Process all objects and extract specified features.
        
        Args:
            feature_types: List of feature types to extract
            
        Returns:
            Dictionary with extraction statistics
        """
        checkpoints = self.find_all_checkpoints()
        
        stats = {
            "total_objects": len(checkpoints),
            "total_variations": 0,
            "extracted": 0,
            "skipped": 0,
            "errors": 0
        }
        
        print(f"Found {stats['total_objects']} objects")
        
        # Create progress bar for objects
        obj_pbar = tqdm(checkpoints.items(), desc="Processing objects")
        
        for obj_id, rotations in obj_pbar:
            obj_pbar.set_postfix({"current": obj_id})
            
            # Create progress bar for rotations within this object
            rot_pbar = tqdm(rotations.items(), desc=f"  {obj_id} rotations", leave=False)
            
            for rotation, checkpoint_path in rot_pbar:
                stats["total_variations"] += 1
                rot_pbar.set_postfix({"rotation": rotation})
                
                try:
                    # Extract planes (only supported feature type for now)
                    if "planes" in feature_types:
                        extracted = self.extract_and_save_planes(obj_id, rotation, checkpoint_path)
                        
                        if extracted:
                            stats["extracted"] += 1
                        else:
                            stats["skipped"] += 1
                            
                except Exception as e:
                    print(f"\nError processing {obj_id}/{rotation}: {e}")
                    stats["errors"] += 1
        
        return stats
    
    def save_manifest(self, stats: Dict[str, int]):
        """
        Save a manifest file with processing statistics.
        
        Args:
            stats: Processing statistics
        """
        manifest = {
            "preprocessing_stats": stats,
            "raw_data_root": str(self.raw_data_root),
            "preprocessed_root": str(self.preprocessed_root),
            "feature_types": ["planes"]  # Will expand later
        }
        
        manifest_path = self.preprocessed_root / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Saved manifest to {manifest_path}")