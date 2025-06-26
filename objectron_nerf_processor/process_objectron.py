#!/usr/bin/env python3
"""
Objectron NeRF Dataset Processor
Single-command processor to convert Objectron dataset to NeRF-ready scenes.
"""

import os
import sys
import json
import csv
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from utils.normalize_transforms import normalize_scene
import yaml
from datetime import datetime

# Import our utility modules
from utils.downloader import ObjectronDownloader
from utils.frame_extractor import FrameExtractor
from utils.pose_converter import PoseConverter
from utils.annotation_parser import AnnotationParser
from utils.scene_builder import SceneBuilder

class ObjectronProcessor:
    def __init__(self, config_path: str, resume_csv: Optional[str] = None):
        """Initialize the Objectron processor with configuration."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize components
        self.downloader = ObjectronDownloader(self.config)
        self.frame_extractor = FrameExtractor(self.config)
        self.pose_converter = PoseConverter(self.config)
        self.annotation_parser = AnnotationParser(self.config)
        self.scene_builder = SceneBuilder(self.config)
        
        # Setup tracking
        self.tracking_file = Path(self.config['paths']['tracking_dir']) / 'processing_status.csv'
        self.processed_samples = self._load_existing_progress(resume_csv)
        self.stats = {'completed': 0, 'failed': 0, 'total': 0}
        
        # Create directories
        self._create_directories()
        
        logging.info(f"Objectron Processor initialized. Target: {self.config['target_samples']} samples")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = Path(self.config['paths']['tracking_dir']).parent / 'logs' / 'processing.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _create_directories(self):
        """Create necessary directories."""
        for path_key in ['output_dir', 'temp_dir', 'tracking_dir']:
            Path(self.config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
        
        # Create temp subdirectories
        temp_dir = Path(self.config['paths']['temp_dir'])
        for subdir in ['videos', 'geometry', 'annotations']:
            (temp_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _load_existing_progress(self, resume_csv: Optional[str]) -> set:
        """Load previously processed samples."""
        processed = set()
        
        if resume_csv and os.path.exists(resume_csv):
            with open(resume_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['status'] == 'completed':
                        processed.add(row['sample_id'])
                        self.stats['completed'] += 1
                    elif row['status'] == 'failed':
                        self.stats['failed'] += 1
        
        logging.info(f"Loaded {len(processed)} previously processed samples")
        return processed

    def generate_sample_list(self) -> List[Tuple[str, str]]:
        """Generate balanced sample list across all categories."""
        categories = self.config['categories']
        target_total = self.config['target_samples']
        
        # Get available samples per category from Objectron indices
        category_samples = {}
        for category in categories:
            samples = self.downloader.get_category_samples(category)
            category_samples[category] = samples
            logging.info(f"{category}: {len(samples)} available samples")
        
        # Calculate balanced distribution
        samples_per_category = target_total // len(categories)
        selected_samples = []
        
        random.seed(42)  # Deterministic sampling
        
        for category in categories:
            available = category_samples[category]
            if len(available) <= samples_per_category:
                # Take all available samples
                selected_samples.extend([(sample, category) for sample in available])
                logging.info(f"{category}: taking all {len(available)} samples")
            else:
                # Random sample
                sampled = random.sample(available, samples_per_category)
                selected_samples.extend([(sample, category) for sample in sampled])
                logging.info(f"{category}: sampled {len(sampled)} from {len(available)}")
        
        # If we need more samples, redistribute from remaining
        if len(selected_samples) < target_total:
            remaining_needed = target_total - len(selected_samples)
            logging.info(f"Need {remaining_needed} more samples, redistributing...")
            
            # Add more from categories with surplus
            for category in categories:
                if remaining_needed <= 0:
                    break
                    
                available = category_samples[category]
                already_selected = len([s for s, c in selected_samples if c == category])
                surplus = len(available) - already_selected
                
                if surplus > 0:
                    additional_needed = min(surplus, remaining_needed)
                    remaining_samples = [s for s in available if (s, category) not in selected_samples]
                    additional = random.sample(remaining_samples, additional_needed)
                    selected_samples.extend([(sample, category) for sample in additional])
                    remaining_needed -= additional_needed
                    logging.info(f"{category}: added {len(additional)} more samples")
        
        self.stats['total'] = len(selected_samples)
        logging.info(f"Generated sample list: {len(selected_samples)} total samples")
        return selected_samples

    def process_single_sample(self, sample_id: str, category: str) -> Tuple[bool, Optional[str]]:
        """Process a single Objectron sample end-to-end."""
        start_time = time.time()
        
        try:
            logging.info(f"Processing {sample_id}")
            
            # Check if already processed
            if sample_id in self.processed_samples:
                # Verify scene exists and is valid
                scene_dir = self._get_scene_directory(sample_id, category)
                if self._validate_existing_scene(scene_dir):
                    logging.info(f"Skipping {sample_id} - already processed and valid")
                    return True, None
                else:
                    logging.warning(f"Re-processing {sample_id} - existing scene invalid")
            
            # Step 1: Download files
            video_path, geometry_path, annotation_path = self.downloader.download_sample(sample_id)
            
            # Step 2: Extract frames
            frames_info = self.frame_extractor.extract_frames(video_path, sample_id)
            if not frames_info or len(frames_info['frame_paths']) < self.config['frame_extraction']['min_frames']:
                raise ValueError(f"Insufficient frames: {len(frames_info['frame_paths']) if frames_info else 0}")
            
            # Step 3: Convert poses
            transforms_data = self.pose_converter.convert_poses(geometry_path, frames_info)
            
            # Step 4: Parse annotations
            try:
                scene_metadata = self.annotation_parser.parse_annotations(annotation_path, transforms_data)
            except Exception as e:
                logging.warning(f"Annotation parsing failed for {sample_id}, creating minimal metadata: {e}")
                # Create minimal scene metadata without bounding boxes
                scene_metadata = {
                    'scene_type': 'objectron',
                    'total_objects': 0,
                    'object_categories': [],
                    'objects': [],
                    'annotation_source': 'objectron_failed_parsing',
                    'coordinate_system': 'camera_centered',
                    'parsing_error': str(e)
                }
            
            # Step 5: Build final scene
            scene_dir = self.scene_builder.build_scene(
                    sample_id, category, frames_info, transforms_data, scene_metadata
                )
            self._normalize_scene_transforms(scene_dir)

            # Step 6: Cleanup temporary files
            if self.config['cleanup']['delete_temp_files']:
                self._cleanup_temp_files([video_path, geometry_path, annotation_path])
                # Also cleanup temporary frame files if they were moved
                for frame_path in frames_info['frame_paths']:
                    if frame_path.exists():
                        frame_path.unlink()
            
            # Log success
            processing_time = time.time() - start_time
            self._log_sample_completion(sample_id, category, 'completed', 
                                      len(frames_info['frame_paths']), 
                                      len(scene_metadata.get('objects', [])), 
                                      processing_time)
            
            self.stats['completed'] += 1
            logging.info(f"Successfully processed {sample_id} in {processing_time:.1f}s")
            return True, None
            
        except Exception as e:
            # Log failure
            processing_time = time.time() - start_time
            error_msg = str(e)
            self._log_sample_completion(sample_id, category, 'failed', 0, 0, processing_time, error_msg)
            
            self.stats['failed'] += 1
            logging.error(f"Failed to process {sample_id}: {error_msg}")
            return False, error_msg

    def _get_scene_directory(self, sample_id: str, category: str) -> Path:
        """Get the scene directory path for a sample."""
        scene_name = sample_id.replace('/', '_').replace('-', '_')
        return Path(self.config['paths']['output_dir']) / scene_name

    def _validate_existing_scene(self, scene_dir: Path) -> bool:
        """Validate that an existing scene is complete and valid."""
        if not scene_dir.exists():
            return False
        
        required_files = ['transforms.json', 'scene_metadata.json']
        for file in required_files:
            if not (scene_dir / file).exists():
                return False
        
        # Check images directory
        images_dir = scene_dir / 'images'
        if not images_dir.exists():
            return False
        
        # Check minimum number of images
        image_files = list(images_dir.glob('*.png'))
        if len(image_files) < self.config['frame_extraction']['min_frames']:
            return False
        
        return True

    def _cleanup_temp_files(self, file_paths: List[Path]):
        """Clean up temporary downloaded files."""
        for file_path in file_paths:
            if file_path and file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete {file_path}: {e}")

    def _log_sample_completion(self, sample_id: str, category: str, status: str, 
                             frame_count: int, object_count: int, processing_time: float, 
                             error_msg: str = ""):
        """Log sample completion to CSV tracking file."""
        # Ensure tracking file has header
        if not self.tracking_file.exists():
            with open(self.tracking_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'category', 'status', 'total_frames', 
                               'objects_found', 'processing_time', 'error_msg', 'timestamp'])
        
        # Append result
        with open(self.tracking_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                sample_id, category, status, frame_count, object_count, 
                f"{processing_time:.2f}", error_msg, datetime.now().isoformat()
            ])

    def run(self, categories: Optional[List[str]] = None):
        """Run the complete processing pipeline."""
        logging.info("Starting Objectron dataset processing")
        
        # Generate sample list
        if categories:
            # Filter to specific categories
            all_samples = self.generate_sample_list()
            sample_list = [(s, c) for s, c in all_samples if c in categories]
            logging.info(f"Filtered to {len(sample_list)} samples for categories: {categories}")
        else:
            sample_list = self.generate_sample_list()
        
        # Process samples
        for i, (sample_id, category) in enumerate(sample_list, 1):
            if sample_id in self.processed_samples:
                continue  # Skip already processed
                
            logging.info(f"Progress: {i}/{len(sample_list)} ({i/len(sample_list)*100:.1f}%)")
            
            success, error = self.process_single_sample(sample_id, category)
            
            # Progress update
            if i % 10 == 0:
                self._print_progress_summary()
        
        # Final summary
        self._print_final_summary()

    def _print_progress_summary(self):
        """Print current progress summary."""
        total_processed = self.stats['completed'] + self.stats['failed']
        success_rate = (self.stats['completed'] / total_processed * 100) if total_processed > 0 else 0
        
        logging.info(f"Progress Summary: {self.stats['completed']} completed, "
                    f"{self.stats['failed']} failed, {success_rate:.1f}% success rate")
        
    def _normalize_scene_transforms(self, scene_dir: Path):
        """Normalize scene transforms for consistent training bounds."""
        transforms_file = scene_dir / 'transforms.json'
        transforms_raw_file = scene_dir / 'transforms_raw.json'
        
        if not transforms_file.exists():
            logging.warning(f"No transforms.json found in {scene_dir}, skipping normalization")
            return
        
        try:
            # Save raw version
            import shutil
            shutil.copy2(transforms_file, transforms_raw_file)
            logging.debug(f"Saved raw transforms to {transforms_raw_file}")
            
            # Normalize for training (target bounds [-1.5, 1.5])
            normalize_scene(str(transforms_raw_file), str(transforms_file), target_bounds=1.5)
            logging.debug(f"Normalized transforms saved to {transforms_file}")
            
        except Exception as e:
            logging.error(f"Failed to normalize transforms for {scene_dir}: {e}")
            # If normalization fails, restore raw version
            if transforms_raw_file.exists():
                shutil.copy2(transforms_raw_file, transforms_file)
            raise

    def _print_final_summary(self):
        """Print final processing summary."""
        total_processed = self.stats['completed'] + self.stats['failed']
        success_rate = (self.stats['completed'] / total_processed * 100) if total_processed > 0 else 0
        
        logging.info("="*60)
        logging.info("PROCESSING COMPLETE")
        logging.info("="*60)
        logging.info(f"Target samples: {self.stats['total']}")
        logging.info(f"Successfully processed: {self.stats['completed']}")
        logging.info(f"Failed: {self.stats['failed']}")
        logging.info(f"Success rate: {success_rate:.1f}%")
        logging.info(f"Output directory: {self.config['paths']['output_dir']}")
        logging.info(f"Tracking file: {self.tracking_file}")


def main():
    parser = argparse.ArgumentParser(description='Process Objectron dataset for NeRF training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, 
                       help='Path to existing processing status CSV to resume from')
    parser.add_argument('--categories', nargs='+', 
                       help='Specific categories to process (default: all)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ObjectronProcessor(args.config, args.resume)
    
    # Run processing
    try:
        processor.run(args.categories)
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Processing failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()