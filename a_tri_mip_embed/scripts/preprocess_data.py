#!/usr/bin/env python3
"""
Preprocess all NeRF checkpoints to extract features.

This script extracts feature planes and MLP weights from TriMipRF checkpoints and saves them
as lightweight .pt files for faster loading during training.

Usage:
    python scripts/preprocess_data.py --config configs/base_config.yaml
    python scripts/preprocess_data.py --config configs/base_config.yaml --force
    python scripts/preprocess_data.py --config configs/base_config.yaml --feature-types planes mlp_base mlp_head
"""

import sys
import argparse
import yaml
import time
from pathlib import Path

# Add project root to path
sys.path.append('.')

from data.preprocessing import FeatureExtractor

def main():
    parser = argparse.ArgumentParser(description='Preprocess NeRF checkpoint data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--force', action='store_true',
                        help='Force rebuild all features (ignore existing files)')
    parser.add_argument('--feature-types', nargs='+', default=['planes', 'mlp_base', 'mlp_head'],
                        choices=['planes', 'mlp_base', 'mlp_head'],
                        help='Types of features to extract (default: all)')
    parser.add_argument('--n', type=int, default=None,
                    help='Maximum number of samples to preprocess (default: process all)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config
    raw_data_root = config['data']['data_root']
    preprocessed_root = config['data'].get('preprocessed_root', None)
    
    print("="*60)
    print("NeRF Feature Preprocessing")
    print("="*60)
    print(f"Raw data root: {raw_data_root}")
    print(f"Preprocessed root: {preprocessed_root or './preprocessed_features/'}")
    print(f"Feature types: {args.feature_types}")
    print(f"Force rebuild: {args.force}")
    print()
    
    # Verify raw data exists
    if not Path(raw_data_root).exists():
        print(f"ERROR: Raw data root does not exist: {raw_data_root}")
        return 1
    
    # Create feature extractor
    extractor = FeatureExtractor(
        raw_data_root=raw_data_root,
        preprocessed_root=preprocessed_root
    )
    
    # If force rebuild, remove existing preprocessed files
    if args.force:
        print("Force rebuild enabled - removing existing preprocessed files...")
        
        # Get all object directories
        if extractor.preprocessed_root.exists():
            import shutil
            
            # Remove old flat structure if it exists
            old_planes_dir = extractor.preprocessed_root / "planes"
            if old_planes_dir.exists() and old_planes_dir.is_dir():
                print(f"Removing old flat structure: {old_planes_dir}")
                shutil.rmtree(old_planes_dir)
            
            # Remove hierarchical structure directories
            for obj_dir in extractor.preprocessed_root.iterdir():
                if obj_dir.is_dir() and not obj_dir.name.startswith('.'):
                    # Check if it's an object directory (has feature subdirs)
                    has_feature_dirs = any(
                        (obj_dir / ft).exists() 
                        for ft in ['planes', 'mlp_base', 'mlp_head']
                    )
                    if has_feature_dirs:
                        print(f"Removing object directory: {obj_dir}")
                        shutil.rmtree(obj_dir)
        
        print("Existing files removed.\n")
    
    # Check if any checkpoints exist
    checkpoints = extractor.find_all_checkpoints()
    if not checkpoints:
        print("ERROR: No checkpoints found in raw data directory")
        return 1
    
    print(f"Found {len(checkpoints)} objects with checkpoints")
    
    # Calculate total variations for progress tracking
    total_variations = sum(len(rotations) for rotations in checkpoints.values())
    print(f"Total checkpoint variations: {total_variations}")
    print()
    
    # Start processing
    start_time = time.time()
    
    try:
        stats = extractor.process_all_objects(feature_types=args.feature_types, max_samples=args.n)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print results
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Total objects: {stats['total_objects']}")
    print(f"Total variations: {stats['total_variations']}")
    
    # Print per-feature-type stats
    total_extracted = 0
    total_skipped = 0
    for ft in args.feature_types:
        extracted = stats['extracted'].get(ft, 0)
        skipped = stats['skipped'].get(ft, 0)
        total_extracted += extracted
        total_skipped += skipped
        print(f"{ft} - Extracted: {extracted}, Skipped: {skipped}")
    
    print(f"\nTotal features extracted: {total_extracted}")
    print(f"Total files skipped (already exist): {total_skipped}")
    print(f"Errors encountered: {stats['errors']}")
    print(f"Processing time: {processing_time:.1f} seconds")
    
    if total_extracted > 0:
        avg_time = processing_time / total_extracted
        print(f"Average time per extraction: {avg_time:.2f} seconds")
    
    print()
    
    # Save manifest
    extractor.save_manifest(stats)
    
    # Print next steps
    print("Next steps:")
    print("1. Verify preprocessing completed successfully")
    print("2. Run your training script - it will automatically use preprocessed features")
    print(f"3. Preprocessed features saved to: {extractor.preprocessed_root}")
    
    if 'mlp_base' in args.feature_types or 'mlp_head' in args.feature_types:
        print("\nTo use MLP features in training, add to your config:")
        print("  data:")
        print("    load_mlp_features: true")
    
    if stats['errors'] > 0:
        print(f"\nWARNING: {stats['errors']} errors encountered during processing")
        print("Check the error messages above and consider re-running with --force")
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)