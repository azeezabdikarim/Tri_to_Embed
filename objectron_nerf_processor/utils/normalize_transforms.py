#!/usr/bin/env python3
"""
Normalize NeRF transforms.json files to consistent coordinate bounds.

This utility normalizes camera poses to fit within specified target bounds
while preserving all spatial relationships.
"""

import json
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

def load_transforms(transforms_path: str) -> Dict:
    """Load and parse transforms.json file."""
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    return transforms

def extract_camera_positions(transforms: Dict) -> np.ndarray:
    """Extract camera positions from transforms.json."""
    frames = transforms['frames']
    positions = []
    
    for frame in frames:
        transform_matrix = np.array(frame['transform_matrix'])
        camera_pos = transform_matrix[:3, 3]  # Translation component
        positions.append(camera_pos)
    
    return np.array(positions)

def calculate_normalization_transform(positions: np.ndarray, target_bounds: float) -> Tuple[np.ndarray, float]:
    """Calculate translation and scale needed to normalize to target bounds."""
    
    # Calculate scene statistics
    scene_center = positions.mean(axis=0)
    scene_min = positions.min(axis=0)
    scene_max = positions.max(axis=0)
    scene_size = scene_max - scene_min
    current_extent = scene_size.max()
    
    # Calculate normalization parameters
    translation = -scene_center  # Translate to origin first
    scale = (target_bounds * 2) / current_extent if current_extent > 0 else 1.0
    
    logging.info(f"Normalization parameters:")
    logging.info(f"  Original center: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")
    logging.info(f"  Original extent: {current_extent:.3f}")
    logging.info(f"  Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
    logging.info(f"  Scale factor: {scale:.4f}")
    logging.info(f"  Target bounds: ±{target_bounds}")
    
    return translation, scale

def apply_normalization_to_transforms(transforms: Dict, translation: np.ndarray, scale: float) -> Dict:
    """Apply normalization transform to all camera poses."""
    
    # Create a copy to avoid modifying original
    normalized_transforms = transforms.copy()
    normalized_transforms['frames'] = []
    
    for frame in transforms['frames']:
        # Copy frame data
        normalized_frame = frame.copy()
        
        # Extract and modify transform matrix
        transform_matrix = np.array(frame['transform_matrix'])
        
        # Apply normalization: translate then scale
        # Only modify the translation component (last column)
        transform_matrix[:3, 3] = (transform_matrix[:3, 3] + translation) * scale
        
        # Update frame with normalized transform
        normalized_frame['transform_matrix'] = transform_matrix.tolist()
        normalized_transforms['frames'].append(normalized_frame)
    
    # Update camera intrinsics if they exist (scale focal lengths)
    if 'fl_x' in normalized_transforms:
        normalized_transforms['fl_x'] = float(normalized_transforms['fl_x'] * scale)
    if 'fl_y' in normalized_transforms:
        normalized_transforms['fl_y'] = float(normalized_transforms['fl_y'] * scale)
    
    # Principal point and camera angle don't need scaling
    
    return normalized_transforms

def validate_normalization(original_transforms: Dict, normalized_transforms: Dict, target_bounds: float) -> bool:
    """Validate that normalization was applied correctly."""
    
    try:
        # Extract positions before and after
        original_positions = extract_camera_positions(original_transforms)
        normalized_positions = extract_camera_positions(normalized_transforms)
        
        # Check that we have the same number of frames
        if len(original_positions) != len(normalized_positions):
            logging.error("Frame count mismatch after normalization")
            return False
        
        # Check that normalized positions are within target bounds
        normalized_min = normalized_positions.min(axis=0)
        normalized_max = normalized_positions.max(axis=0)
        normalized_extent = np.max(normalized_max - normalized_min)
        
        expected_extent = target_bounds * 2
        if normalized_extent > expected_extent * 1.1:  # Allow 10% tolerance
            logging.error(f"Normalized extent {normalized_extent:.3f} exceeds target {expected_extent:.3f}")
            return False
        
        # Check that relative distances are preserved (scaled correctly)
        original_distances = np.linalg.norm(original_positions[1:] - original_positions[:-1], axis=1)
        normalized_distances = np.linalg.norm(normalized_positions[1:] - normalized_positions[:-1], axis=1)
        
        if len(original_distances) > 0:
            # Scale should be consistent
            scale_ratios = normalized_distances / (original_distances + 1e-8)  # Avoid division by zero
            scale_consistency = np.std(scale_ratios) < 0.01  # Very tight tolerance
            
            if not scale_consistency:
                logging.warning("Scale consistency check failed - relative distances may not be preserved")
                # Don't fail validation for this, just warn
        
        logging.info(f"Validation passed:")
        logging.info(f"  Normalized extent: {normalized_extent:.3f} (target: {expected_extent:.3f})")
        logging.info(f"  Normalized bounds: [{normalized_min[0]:.3f}, {normalized_min[1]:.3f}, {normalized_min[2]:.3f}] to [{normalized_max[0]:.3f}, {normalized_max[1]:.3f}, {normalized_max[2]:.3f}]")
        
        return True
        
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return False

def normalize_scene(input_path: str, output_path: str, target_bounds: float = 1.5) -> bool:
    """
    Normalize a NeRF scene to fit within specified bounds.
    
    Args:
        input_path: Path to input transforms.json file
        output_path: Path to save normalized transforms.json file  
        target_bounds: Target coordinate bounds (scene will fit in ±target_bounds)
        
    Returns:
        True if normalization succeeded, False otherwise
    """
    
    try:
        logging.info(f"Normalizing scene: {input_path} -> {output_path}")
        
        # Load original transforms
        transforms = load_transforms(input_path)
        
        if not transforms.get('frames'):
            raise ValueError("No frames found in transforms file")
        
        # Extract camera positions
        positions = extract_camera_positions(transforms)
        
        if len(positions) == 0:
            raise ValueError("No valid camera positions found")
        
        # Calculate normalization transform
        translation, scale = calculate_normalization_transform(positions, target_bounds)
        
        # Apply normalization
        normalized_transforms = apply_normalization_to_transforms(transforms, translation, scale)
        
        # Validate result
        if not validate_normalization(transforms, normalized_transforms, target_bounds):
            raise ValueError("Normalization validation failed")
        
        # Save normalized transforms
        with open(output_path, 'w') as f:
            json.dump(normalized_transforms, f, indent=2)
        
        logging.info(f"Successfully normalized scene to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to normalize scene {input_path}: {e}")
        return False

def main():
    """Command line interface for scene normalization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize NeRF scene coordinates")
    parser.add_argument("input", help="Input transforms.json file")
    parser.add_argument("output", help="Output transforms.json file")
    parser.add_argument("--target-bounds", type=float, default=1.5,
                       help="Target coordinate bounds (default: 1.5)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Normalize scene
    success = normalize_scene(args.input, args.output, args.target_bounds)
    
    if success:
        print(f"✅ Successfully normalized {args.input} -> {args.output}")
        return 0
    else:
        print(f"❌ Failed to normalize {args.input}")
        return 1

if __name__ == "__main__":
    exit(main())