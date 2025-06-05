import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    
    def extract_mlp_weights_from_checkpoint(self, checkpoint_path: Path) -> Tuple[Dict, Dict]:
        """
        Extract and structure MLP weights from TriMipRF checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            mlp_base_dict: Structured weights for base MLP
            mlp_head_dict: Structured weights for head MLP
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
        
        # Extract raw parameter tensors
        mlp_base_params = checkpoint['field.mlp_base.params']  # (either 24576 or 24848)
        mlp_head_params = checkpoint['field.mlp_head.params']  # (typically 55296)
        
        # Structure MLP Base
        mlp_base_dict = self._structure_mlp_base(mlp_base_params)
        
        # Structure MLP Head (4 hidden layers, 128 width)  
        # Input: 31 (16 SH + 15 geo_feat), Output: 3 (RGB)
        mlp_head_dict = self._structure_mlp_head(mlp_head_params)
        
        return mlp_base_dict, mlp_head_dict
    
    def _structure_mlp_base(self, params: torch.Tensor) -> Dict:
        """Structure the flattened MLP base parameters."""
        param_count = params.shape[0]
        
        if param_count == 24576:
            # Weights-only export: no bias vectors in the flat tensor
            return self._structure_mlp_base_24576_WEIGHTS_ONLY(params)
        elif param_count == 24848:
            # Weights + explicit bias vectors
            return self._structure_mlp_base_24848_WITH_BIAS(params)
        else:
            raise ValueError(f"Unexpected MLP base parameter count: {param_count}")
    
    def _structure_mlp_base_24576_WEIGHTS_ONLY(self, params: torch.Tensor) -> Dict:
        """
        Structure the MLP base when `.params` contains only weights (24 576 values:
        6 144 + 16 384 + 2 048). We create zero biases so that downstream code can
        always find a bias tensor.
        Architecture: 48 → 128 → 128 → 16
        """
        idx = 0
        layers = {}
        
        # Layer 0: (48 → 128)
        w0_size = 128 * 48  # 6 144
        W0 = params[idx:idx + w0_size].reshape(128, 48)
        layers['layer_0'] = {
            'weight': W0,
            'bias': torch.zeros(128)
        }
        idx += w0_size
        
        # Layer 1: (128 → 128)
        w1_size = 128 * 128  # 16 384
        W1 = params[idx:idx + w1_size].reshape(128, 128)
        layers['layer_1'] = {
            'weight': W1,
            'bias': torch.zeros(128)
        }
        idx += w1_size
        
        # Layer 2: (128 → 16)
        w2_size = 16 * 128  # 2 048
        W2 = params[idx:idx + w2_size].reshape(16, 128)
        layers['layer_2'] = {
            'weight': W2,
            'bias': torch.zeros(16)
        }
        idx += w2_size
        
        assert idx == 24576, f"Parameter count mismatch: {idx} vs 24576"
        
        return {
            **layers,
            'metadata': {
                'total_params': 24576,
                'architecture': 'FullyFusedMLP',
                'depth': 2,
                'width': 128,
                'input_dim': 48,
                'output_dim': 16,
                'bias_mode': 'absent-in-params',
            }
        }
    
    def _structure_mlp_base_24848_WITH_BIAS(self, params: torch.Tensor) -> Dict:
        """
        Structure the MLP base when `.params` contains weights **and** explicit bias vectors
        (24 848 values: 6 144 + 16 384 + 2 048 + 128 + 128 + 16).
        Architecture: 48 → 128 → 128 → 16
        """
        idx = 0
        layers = {}
        
        # Layer 0: (48 → 128)
        w0_size = 128 * 48  # 6 144
        layers['layer_0'] = {
            'weight': params[idx:idx + w0_size].reshape(128, 48)
        }
        idx += w0_size
        
        # Layer 1: (128 → 128)
        w1_size = 128 * 128  # 16 384
        layers['layer_1'] = {
            'weight': params[idx:idx + w1_size].reshape(128, 128)
        }
        idx += w1_size
        
        # Layer 2: (128 → 16)
        w2_size = 16 * 128  # 2 048
        layers['layer_2'] = {
            'weight': params[idx:idx + w2_size].reshape(16, 128)
        }
        idx += w2_size
        
        # Now biases
        layers['layer_0']['bias'] = params[idx:idx + 128]
        idx += 128
        layers['layer_1']['bias'] = params[idx:idx + 128]
        idx += 128
        layers['layer_2']['bias'] = params[idx:idx + 16]
        idx += 16
        
        assert idx == 24848, f"Parameter count mismatch: {idx} vs 24848"
        
        return {
            **layers,
            'metadata': {
                'total_params': 24848,
                'architecture': 'FullyFusedMLP',
                'depth': 2,
                'width': 128,
                'input_dim': 48,
                'output_dim': 16,
                'bias_mode': 'explicit_vectors',
            }
        }
    
    
    def _structure_mlp_head(self, params: torch.Tensor) -> Dict:
            """Structure the flattened MLP head parameters with auto-detection."""
            param_count = params.shape[0]
            
            if param_count == 55296:
                # Your architecture: 40 → 128 → 128 → 128 → 128 → 3 + 125 extra params
                return self._structure_mlp_head_55296_with_extras(params)
            elif param_count == 55171:
                # Pure architecture without extras: 40 → 128 → 128 → 128 → 128 → 3
                return self._structure_mlp_head_55171(params)
            elif param_count == 54019:
                # Original architecture: 31 → 128 → 128 → 128 → 128 → 3
                return self._structure_mlp_head_54019(params)
            else:
                # Unexpected size: try to infer or store raw
                print(f"Warning: Unexpected MLP head parameter count: {param_count}")
                print("Attempting to infer architecture...")
                
                # Try to infer the architecture
                inferred = self._infer_mlp_head_architecture(params)
                if inferred is not None:
                    return inferred
                
                # Fallback: store raw
                return {
                    'raw_params': params,
                    'metadata': {
                        'total_params': param_count,
                        'architecture': 'FullyFusedMLP',
                        'note': 'Unparsed head – stored raw'
                    }
                }
            
    def _structure_mlp_head_55296_with_extras(self, params: torch.Tensor) -> Dict:
        """Structure the 55,296-parameter MLP head with SH degree 4 and extra parameters."""
        idx = 0
        layers = {}
        
        # Architecture: 40 → 128 → 128 → 128 → 128 → 3
        # Input: 25 (SH degree 4) + 15 (geo_feat) = 40
        
        # Layer 0: (40 → 128)
        w0_size = 128 * 40  # 5120
        layers['layer_0'] = {
            'weight': params[idx:idx + w0_size].reshape(128, 40)
        }
        idx += w0_size
        
        # Layers 1–3: (128 → 128)
        for i in range(1, 4):
            w_size = 128 * 128  # 16384 each
            layers[f'layer_{i}'] = {
                'weight': params[idx:idx + w_size].reshape(128, 128)
            }
            idx += w_size
        
        # Layer 4: (128 → 3)
        w4_size = 3 * 128  # 384
        layers['layer_4'] = {
            'weight': params[idx:idx + w4_size].reshape(3, 128)
        }
        idx += w4_size
        
        # Now biases
        layers['layer_0']['bias'] = params[idx:idx + 128]
        idx += 128
        for i in range(1, 4):
            layers[f'layer_{i}']['bias'] = params[idx:idx + 128]
            idx += 128
        layers['layer_4']['bias'] = params[idx:idx + 3]
        idx += 3
        
        # Handle the extra 125 parameters
        # These might be padding, normalization params, or tinycudann-specific data
        extra_params = params[idx:idx + 125]
        layers['extra_params'] = extra_params
        idx += 125
        
        assert idx == 55296, f"Parameter count mismatch: {idx} vs 55296"
        
        return {
            **layers,
            'metadata': {
                'total_params': 55296,
                'architecture': 'FullyFusedMLP',
                'depth': 4,
                'width': 128,
                'input_dim': 40,  # 25 SH degree 4 + 15 geo_feat
                'output_dim': 3,
                'sh_degree': 4,
                'geo_feat_dim': 15,
                'extra_params_count': 125,
                'note': 'tinycudann MLP with extra parameters (possibly padding)'
            }
        }

    def _structure_mlp_head_55171(self, params: torch.Tensor) -> Dict:
        """Structure the 55,171-parameter MLP head (pure architecture without extras)."""
        idx = 0
        layers = {}
        
        # Architecture: 40 → 128 → 128 → 128 → 128 → 3
        
        # Layer 0: (40 → 128)
        w0_size = 128 * 40  # 5120
        layers['layer_0'] = {
            'weight': params[idx:idx + w0_size].reshape(128, 40)
        }
        idx += w0_size
        
        # Layers 1–3: (128 → 128)
        for i in range(1, 4):
            w_size = 128 * 128  # 16384 each
            layers[f'layer_{i}'] = {
                'weight': params[idx:idx + w_size].reshape(128, 128)
            }
            idx += w_size
        
        # Layer 4: (128 → 3)
        w4_size = 3 * 128  # 384
        layers['layer_4'] = {
            'weight': params[idx:idx + w4_size].reshape(3, 128)
        }
        idx += w4_size
        
        # Now biases
        layers['layer_0']['bias'] = params[idx:idx + 128]
        idx += 128
        for i in range(1, 4):
            layers[f'layer_{i}']['bias'] = params[idx:idx + 128]
            idx += 128
        layers['layer_4']['bias'] = params[idx:idx + 3]
        idx += 3
        
        assert idx == 55171, f"Parameter count mismatch: {idx} vs 55171"
        
        return {
            **layers,
            'metadata': {
                'total_params': 55171,
                'architecture': 'FullyFusedMLP',
                'depth': 4,
                'width': 128,
                'input_dim': 40,
                'output_dim': 3,
                'sh_degree': 4,
                'geo_feat_dim': 15
            }
        }

    def _structure_mlp_head_54019(self, params: torch.Tensor) -> Dict:
        """Structure the original 54,019-parameter MLP head (31 input dims)."""
        idx = 0
        layers = {}
        
        # Architecture: 31 → 128 → 128 → 128 → 128 → 3
        
        # Layer 0: (31 → 128)
        w0_size = 128 * 31  # 3968
        layers['layer_0'] = {
            'weight': params[idx:idx + w0_size].reshape(128, 31)
        }
        idx += w0_size
        
        # Layers 1–3: (128 → 128)
        for i in range(1, 4):
            w_size = 128 * 128  # 16384 each
            layers[f'layer_{i}'] = {
                'weight': params[idx:idx + w_size].reshape(128, 128)
            }
            idx += w_size
        
        # Layer 4: (128 → 3)
        w4_size = 3 * 128  # 384
        layers['layer_4'] = {
            'weight': params[idx:idx + w4_size].reshape(3, 128)
        }
        idx += w4_size
        
        # Now biases
        layers['layer_0']['bias'] = params[idx:idx + 128]
        idx += 128
        for i in range(1, 4):
            layers[f'layer_{i}']['bias'] = params[idx:idx + 128]
            idx += 128
        layers['layer_4']['bias'] = params[idx:idx + 3]
        idx += 3
        
        assert idx == 54019, f"Parameter count mismatch: {idx} vs 54019"
        
        return {
            **layers,
            'metadata': {
                'total_params': 54019,
                'architecture': 'FullyFusedMLP',
                'depth': 4,
                'width': 128,
                'input_dim': 31,  # 16 SH degree 3 + 15 geo_feat
                'output_dim': 3,
                'sh_degree': 3,
                'geo_feat_dim': 15
            }
        }

    
    def get_preprocessed_path(self, obj_id: str, rotation: str, feature_type: str = "planes") -> Path:
        """
        Get the path where preprocessed features should be saved.
        
        Args:
            obj_id: Object identifier
            rotation: Rotation string
            feature_type: Type of feature ("planes", "mlp_base", "mlp_head")
            
        Returns:
            Path to preprocessed feature file
        """
        # Create object directory
        obj_dir = self.preprocessed_root / obj_id
        
        # Create feature type subdirectory
        feature_dir = obj_dir / feature_type
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Return path to specific file
        filename = f"{rotation}.pt"
        return feature_dir / filename
    
    def needs_extraction(self, checkpoint_path: Path, preprocessed_paths: List[Path]) -> bool:
        """
        Check if extraction is needed for any of the preprocessed paths.
        
        Args:
            checkpoint_path: Path to raw checkpoint
            preprocessed_paths: List of paths to preprocessed feature files
            
        Returns:
            True if extraction is needed for any feature
        """
        if not all(p.exists() for p in preprocessed_paths):
            return True
        
        # Check if checkpoint is newer than any preprocessed file
        checkpoint_mtime = checkpoint_path.stat().st_mtime
        
        for preprocessed_path in preprocessed_paths:
            if preprocessed_path.exists():
                preprocessed_mtime = preprocessed_path.stat().st_mtime
                if checkpoint_mtime > preprocessed_mtime:
                    return True
        
        return False
    
    def extract_and_save_all_features(self, obj_id: str, rotation: str, checkpoint_path: Path) -> Dict[str, bool]:
        """
        Extract all features (planes, mlp_base, mlp_head) from checkpoint and save if needed.
        
        Args:
            obj_id: Object identifier
            rotation: Rotation string
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dictionary indicating which features were extracted
        """
        # Get all preprocessed paths
        planes_path = self.get_preprocessed_path(obj_id, rotation, "planes")
        mlp_base_path = self.get_preprocessed_path(obj_id, rotation, "mlp_base")
        mlp_head_path = self.get_preprocessed_path(obj_id, rotation, "mlp_head")
        
        all_paths = [planes_path, mlp_base_path, mlp_head_path]
        
        # Check if extraction is needed
        if not self.needs_extraction(checkpoint_path, all_paths):
            return {'planes': False, 'mlp_base': False, 'mlp_head': False}
        
        try:
            # Load checkpoint once
            checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
            
            extracted = {}
            
            # Extract and save planes
            if not planes_path.exists() or self.needs_extraction(checkpoint_path, [planes_path]):
                planes = checkpoint['field.encoding.fm']
                if planes.shape != (3, 512, 512, 16):
                    raise ValueError(f"Unexpected feature map shape: {planes.shape}")
                torch.save(planes, planes_path)
                extracted['planes'] = True
            else:
                extracted['planes'] = False
            
            # Extract and save structured MLP weights
            if (not mlp_base_path.exists() or not mlp_head_path.exists() or 
                self.needs_extraction(checkpoint_path, [mlp_base_path, mlp_head_path])):
                
                # Extract MLPs
                mlp_base_params = checkpoint['field.mlp_base.params']
                mlp_head_params = checkpoint['field.mlp_head.params']
                
                # Structure and save
                mlp_base_dict = self._structure_mlp_base(mlp_base_params)
                mlp_head_dict = self._structure_mlp_head(mlp_head_params)
                
                torch.save(mlp_base_dict, mlp_base_path)
                torch.save(mlp_head_dict, mlp_head_path)
                
                extracted['mlp_base'] = True
                extracted['mlp_head'] = True
            else:
                extracted['mlp_base'] = False
                extracted['mlp_head'] = False
            
            return extracted
            
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
    
    def process_all_objects(self, feature_types: List[str] = ["planes", "mlp_base", "mlp_head"], max_samples: Optional[int] = None) -> Dict[str, int]:
        """
        Process all objects and extract specified features.
        
        Args:
            feature_types: List of feature types to extract
            max_samples: Maximum number of samples to process (default: process all)
            
        Returns:
            Dictionary with extraction statistics
        """
        checkpoints = self.find_all_checkpoints()
        
        stats = {
            "total_objects": len(checkpoints),
            "total_variations": 0,
            "extracted": {ft: 0 for ft in feature_types},
            "skipped": {ft: 0 for ft in feature_types},
            "errors": 0
        }
        
        print(f"Found {stats['total_objects']} objects")
        if max_samples is not None:
            print(f"Processing limited to {max_samples} samples")
        
        # Create progress bar for objects
        obj_pbar = tqdm(checkpoints.items(), desc="Processing objects")
        sample_count = 0

        for obj_id, rotations in obj_pbar:
            obj_pbar.set_postfix({"current": obj_id})
            
            # Create progress bar for rotations within this object
            rot_pbar = tqdm(rotations.items(), desc=f"  {obj_id} rotations", leave=False)
            
            for rotation, checkpoint_path in rot_pbar:
                # Check if we've reached the sample limit
                if max_samples is not None and sample_count >= max_samples:
                    break
                    
                stats["total_variations"] += 1
                rot_pbar.set_postfix({"rotation": rotation})
                
                try:
                    # Extract all features
                    extracted = self.extract_and_save_all_features(obj_id, rotation, checkpoint_path)
                    
                    # Update stats
                    for ft in feature_types:
                        if ft in extracted:
                            if extracted[ft]:
                                stats["extracted"][ft] += 1
                            else:
                                stats["skipped"][ft] += 1
                    
                    # Increment sample count after successful processing
                    sample_count += 1
                        
                except Exception as e:
                    print(f"\nError processing {obj_id}/{rotation}: {e}")
                    stats["errors"] += 1
            
            # Break out of outer loop if we've reached the limit
            if max_samples is not None and sample_count >= max_samples:
                break
        
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
            "feature_types": ["planes", "mlp_base", "mlp_head"],
            "directory_structure": "hierarchical (object/feature_type/rotation.pt)"
        }
        
        manifest_path = self.preprocessed_root / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Saved manifest to {manifest_path}")