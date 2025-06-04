#!/usr/bin/env python3
"""
Preprocess all NeRF checkpoints to extract features.

This script extracts feature planes from TriMipRF checkpoints and saves them
as lightweight .pt files for faster loading during training.

Usage:
    python scripts/preprocess_data.py --config configs/base_config.yaml
    python scripts/preprocess_data.py --config configs/base_config.yaml --force
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
    parser.add_argument('--feature-types', nargs='+', default=['planes'],
                        choices=['planes', 'geo_mlp', 'color_mlp'],
                        help='Types of features to extract')
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
        
        # Remove planes directory if it exists
        if extractor.planes_dir.exists():
            import shutil
            shutil.rmtree(extractor.planes_dir)
            extractor.planes_dir.mkdir(parents=True, exist_ok=True)
        
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
        stats = extractor.process_all_objects(feature_types=args.feature_types)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nERROR during processing: {e}")
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
    print(f"Features extracted: {stats['extracted']}")
    print(f"Files skipped (already exist): {stats['skipped']}")
    print(f"Errors encountered: {stats['errors']}")
    print(f"Processing time: {processing_time:.1f} seconds")
    
    if stats['extracted'] > 0:
        avg_time = processing_time / stats['extracted']
        print(f"Average time per extraction: {avg_time:.2f} seconds")
    
    print()
    
    # Save manifest
    extractor.save_manifest(stats)
    
    # Print next steps
    print("Next steps:")
    print("1. Verify preprocessing completed successfully")
    print("2. Run your training script - it will automatically use preprocessed features")
    print(f"3. Preprocessed features saved to: {extractor.preprocessed_root}")
    
    if stats['errors'] > 0:
        print(f"\nWARNING: {stats['errors']} errors encountered during processing")
        print("Check the error messages above and consider re-running with --force")
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)