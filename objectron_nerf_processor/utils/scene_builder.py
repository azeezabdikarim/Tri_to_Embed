"""
Build final NeRF scene structure from processed components
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging
import numpy as np

class SceneBuilder:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config['paths']['output_dir'])
        self.cleanup_config = config.get('cleanup', {})

    def build_scene(self, sample_id: str, category: str, frames_info: Dict, 
                   transforms_data: Dict, scene_metadata: Dict) -> Path:
        """Build complete NeRF scene from processed components."""
        
        # Create scene directory
        scene_name = self._generate_scene_name(sample_id, category)
        scene_dir = self.output_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Building scene: {scene_dir}")
        
        try:
            # Create images directory and copy frames
            self._setup_images(scene_dir, frames_info)
            
            # Save transforms.json
            self._save_transforms_json(scene_dir, transforms_data)
            
            # Save scene metadata
            self._save_scene_metadata(scene_dir, scene_metadata, sample_id, category)
            
            # Validate final scene
            self._validate_scene(scene_dir)
            
            logging.info(f"Successfully built scene: {scene_dir}")
            return scene_dir
            
        except Exception as e:
            # Cleanup on failure
            if scene_dir.exists():
                shutil.rmtree(scene_dir, ignore_errors=True)
            raise Exception(f"Failed to build scene {scene_name}: {e}")

    def _generate_scene_name(self, sample_id: str, category: str) -> str:
        """Generate scene directory name from sample ID."""
        # Convert sample_id to filesystem-safe name
        # Example: "chair/batch-24/33" -> "chair_batch-24_33"
        scene_name = sample_id.replace('/', '_').replace('-', '_')
        return scene_name

    def _setup_images(self, scene_dir: Path, frames_info: Dict):
        """Copy/move frame images to scene images directory."""
        images_dir = scene_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        frame_paths = frames_info['frame_paths']
        
        logging.debug(f"Copying {len(frame_paths)} images to {images_dir}")
        
        # Copy frames with sequential naming
        for i, frame_path in enumerate(frame_paths):
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame file not found: {frame_path}")
            
            # Generate sequential filename
            target_name = f"{i:06d}.png"
            target_path = images_dir / target_name
            
            # Copy frame to scene directory
            shutil.copy2(frame_path, target_path)
            
            if not target_path.exists():
                raise RuntimeError(f"Failed to copy frame: {frame_path} -> {target_path}")
        
        logging.debug(f"Copied {len(frame_paths)} images successfully")

    def _save_transforms_json(self, scene_dir: Path, transforms_data: Dict):
        """Save NeRF transforms.json file."""
        transforms_path = scene_dir / 'transforms.json'
        
        # Update file paths to match copied images
        updated_transforms = transforms_data.copy()
        frames = updated_transforms['frames']
        
        # Update frame paths to sequential numbering
        for i, frame in enumerate(frames):
            frame['file_path'] = f"./images/{i:06d}.png"
        
        # Validate transforms data
        self._validate_transforms_data(updated_transforms)
        
        # Save to file
        with open(transforms_path, 'w') as f:
            json.dump(updated_transforms, f, indent=2)
        
        logging.debug(f"Saved transforms.json with {len(frames)} frames")

    def _save_scene_metadata(self, scene_dir: Path, scene_metadata: Dict, 
                           sample_id: str, category: str):
        """Save comprehensive scene metadata including bounding boxes."""
        metadata_path = scene_dir / 'scene_metadata.json'
        
        # Enhance metadata with scene information
        enhanced_metadata = scene_metadata.copy()
        enhanced_metadata.update({
            'scene_id': self._generate_scene_name(sample_id, category),
            'original_sample_id': sample_id,
            'primary_category': category,
            'data_source': 'objectron',
            'processing_version': '1.0',
            'coordinate_system': self.config['pose_conversion']['coordinate_system'],
            'scene_bounds': self._calculate_scene_bounds(scene_metadata),
            'frame_info': {
                'total_frames': len(scene_metadata.get('frames', [])),
                'frame_format': 'png',
                'frame_naming': 'sequential_6digit'
            }
        })
        
        # Add summary statistics
        enhanced_metadata['statistics'] = self._calculate_scene_statistics(enhanced_metadata)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        logging.debug(f"Saved scene metadata for {len(scene_metadata.get('objects', []))} objects")

    def _validate_transforms_data(self, transforms_data: Dict):
        """Validate transforms.json structure."""
        required_fields = ['camera_angle_x', 'frames']
        for field in required_fields:
            if field not in transforms_data:
                raise ValueError(f"Missing required field in transforms: {field}")
        
        frames = transforms_data['frames']
        if not frames:
            raise ValueError("No frames in transforms data")
        
        # Validate each frame
        for i, frame in enumerate(frames):
            if 'file_path' not in frame:
                raise ValueError(f"Frame {i} missing file_path")
            if 'transform_matrix' not in frame:
                raise ValueError(f"Frame {i} missing transform_matrix")
            
            # Validate transform matrix
            transform = np.array(frame['transform_matrix'])
            if transform.shape != (4, 4):
                raise ValueError(f"Frame {i} transform matrix has invalid shape: {transform.shape}")

    def _calculate_scene_bounds(self, scene_metadata: Dict) -> Dict:
        """Calculate scene bounding information."""
        objects = scene_metadata.get('objects', [])
        
        if not objects:
            return {'type': 'empty_scene'}
        
        # Collect all object centers and sizes
        centers = []
        sizes = []
        
        for obj in objects:
            bbox = obj.get('bbox_3d')
            if bbox and 'center' in bbox:
                centers.append(bbox['center'])
                if 'size' in bbox:
                    sizes.append(bbox['size'])
        
        if not centers:
            return {'type': 'no_valid_bboxes'}
        
        centers = np.array(centers)
        
        # Calculate scene bounds
        scene_bounds = {
            'type': 'object_based',
            'scene_center': np.mean(centers, axis=0).tolist(),
            'scene_min': np.min(centers, axis=0).tolist(),
            'scene_max': np.max(centers, axis=0).tolist(),
            'scene_size': (np.max(centers, axis=0) - np.min(centers, axis=0)).tolist()
        }
        
        if sizes:
            sizes = np.array(sizes)
            scene_bounds['average_object_size'] = np.mean(sizes, axis=0).tolist()
            scene_bounds['max_object_size'] = np.max(sizes, axis=0).tolist()
        
        return scene_bounds

    def _calculate_scene_statistics(self, metadata: Dict) -> Dict:
        """Calculate scene statistics for analysis."""
        objects = metadata.get('objects', [])
        
        stats = {
            'total_objects': len(objects),
            'categories_present': list(set(obj['category'] for obj in objects)),
            'objects_per_category': {},
            'has_multiple_objects': len(objects) > 1,
            'has_keypoints_3d': sum(1 for obj in objects if obj.get('keypoints_3d')),
            'has_keypoints_2d': sum(1 for obj in objects if obj.get('keypoints_2d')),
            'has_bbox_3d': sum(1 for obj in objects if obj.get('bbox_3d'))
        }
        
        # Count objects per category
        for obj in objects:
            category = obj['category']
            stats['objects_per_category'][category] = stats['objects_per_category'].get(category, 0) + 1
        
        # Calculate bounding box statistics
        if objects:
            bbox_volumes = []
            for obj in objects:
                bbox = obj.get('bbox_3d')
                if bbox and 'volume' in bbox:
                    bbox_volumes.append(bbox['volume'])
            
            if bbox_volumes:
                stats['bbox_volume_stats'] = {
                    'mean': float(np.mean(bbox_volumes)),
                    'std': float(np.std(bbox_volumes)),
                    'min': float(np.min(bbox_volumes)),
                    'max': float(np.max(bbox_volumes))
                }
        
        return stats

    def _validate_scene(self, scene_dir: Path):
        """Validate that the built scene is complete and valid."""
        logging.debug(f"Validating scene: {scene_dir}")
        
        # Check required files exist
        required_files = ['transforms.json', 'scene_metadata.json']
        for filename in required_files:
            file_path = scene_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Missing required file: {filename}")
        
        # Check images directory
        images_dir = scene_dir / 'images'
        if not images_dir.exists():
            raise FileNotFoundError("Missing images directory")
        
        # Check image files
        image_files = list(images_dir.glob('*.png'))
        if not image_files:
            raise ValueError("No image files found in images directory")
        
        # Validate transforms.json
        with open(scene_dir / 'transforms.json', 'r') as f:
            transforms = json.load(f)
        
        if len(transforms['frames']) != len(image_files):
            raise ValueError(f"Frame count mismatch: {len(transforms['frames'])} transforms vs {len(image_files)} images")
        
        # Validate scene_metadata.json
        with open(scene_dir / 'scene_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        if not metadata.get('objects'):
            logging.warning(f"Scene {scene_dir.name} has no objects - this may be intentional")
        
        # Check image file naming consistency
        expected_names = {f"{i:06d}.png" for i in range(len(image_files))}
        actual_names = {f.name for f in image_files}
        
        if expected_names != actual_names:
            raise ValueError("Image file naming is not sequential")
        
        logging.debug(f"Scene validation passed: {len(image_files)} images, {len(transforms['frames'])} poses")

    def cleanup_temp_files(self, temp_files: List[Path]):
        """Clean up temporary files if configured to do so."""
        if not self.cleanup_config.get('delete_temp_files', True):
            return
        
        for file_path in temp_files:
            if file_path and file_path.exists():
                try:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                    logging.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to cleanup {file_path}: {e}")