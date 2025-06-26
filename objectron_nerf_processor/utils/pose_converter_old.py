"""
Convert ARKit camera poses from Objectron to NeRF transforms.json format
"""

import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

# Import protobuf schema - you'll need to install objectron package or copy the proto files
# For now, I'll provide a minimal protobuf parser
class ARFrameParser:
    """Minimal parser for AR metadata protobuf files."""
    
    @staticmethod
    def parse_geometry_file(file_path: Path) -> List[Dict]:
        """Parse geometry.pbdata file and extract camera poses."""
        frames = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                
            # Parse protobuf messages
            offset = 0
            while offset < len(data):
                try:
                    # Read message length (first 4 bytes, little endian)
                    if offset + 4 > len(data):
                        break
                    msg_len = struct.unpack('<I', data[offset:offset + 4])[0]
                    offset += 4
                    
                    if offset + msg_len > len(data):
                        break
                    
                    # Extract message data
                    message_data = data[offset:offset + msg_len]
                    offset += msg_len
                    
                    # Parse camera data from message
                    # This is a simplified parser - in practice you'd use the full protobuf
                    frame_data = ARFrameParser._parse_frame_message(message_data)
                    if frame_data:
                        frames.append(frame_data)
                        
                except Exception as e:
                    logging.warning(f"Failed to parse frame at offset {offset}: {e}")
                    break
            
            logging.info(f"Parsed {len(frames)} camera frames from {file_path}")
            return frames
            
        except Exception as e:
            logging.error(f"Failed to parse geometry file {file_path}: {e}")
            return []

    @staticmethod
    def _parse_frame_message(data: bytes) -> Dict:
        """Parse individual frame message - simplified implementation."""
        # This is a very simplified parser
        # In practice, you'd use the actual protobuf schema
        # For now, we'll extract basic camera transform assuming standard format
        
        try:
            # Look for camera transform matrix (16 float32 values)
            # This is a heuristic - real implementation would use protobuf schema
            frame_data = {}
            
            # Find camera transform pattern (16 consecutive floats)
            for i in range(0, len(data) - 64, 4):
                try:
                    # Try to extract 16 floats
                    transform = struct.unpack('<16f', data[i:i + 64])
                    
                    # Basic validation - check if this looks like a transformation matrix
                    matrix = np.array(transform).reshape(4, 4)
                    
                    # Check if bottom row is [0, 0, 0, 1] (standard homogeneous transform)
                    if np.allclose(matrix[3, :], [0, 0, 0, 1], atol=1e-3):
                        frame_data['camera_transform'] = matrix
                        break
                        
                except:
                    continue
            
            # Look for camera intrinsics (4 floats: fx, fy, cx, cy)
            for i in range(0, len(data) - 16, 4):
                try:
                    intrinsics = struct.unpack('<4f', data[i:i + 16])
                    # Basic validation - focal lengths should be positive
                    if intrinsics[0] > 0 and intrinsics[1] > 0:
                        frame_data['intrinsics'] = {
                            'fx': intrinsics[0],
                            'fy': intrinsics[1], 
                            'cx': intrinsics[2],
                            'cy': intrinsics[3]
                        }
                        break
                except:
                    continue
            
            # Look for image resolution
            for i in range(0, len(data) - 8, 4):
                try:
                    width, height = struct.unpack('<2I', data[i:i + 8])
                    if 100 < width < 5000 and 100 < height < 5000:  # Reasonable image dimensions
                        frame_data['image_resolution'] = {'width': width, 'height': height}
                        break
                except:
                    continue
            
            return frame_data if 'camera_transform' in frame_data else None
            
        except Exception as e:
            logging.debug(f"Failed to parse frame message: {e}")
            return None


class PoseConverter:
    def __init__(self, config: dict):
        self.config = config
        self.pose_config = config['pose_conversion']

    def convert_poses(self, geometry_path: Path, frames_info: Dict) -> Dict:
        """Convert ARKit poses to NeRF transforms.json format."""
        logging.info(f"Converting poses from {geometry_path}")
        
        # Parse AR camera data
        ar_frames = ARFrameParser.parse_geometry_file(geometry_path)
        
        if not ar_frames:
            raise ValueError("No valid camera frames found in geometry file")
        
        # Match AR frames to extracted video frames
        matched_frames = self._match_frames_to_poses(ar_frames, frames_info)
        
        # Convert to NeRF format
        transforms_data = self._build_transforms_json(matched_frames)
        
        logging.info(f"Converted {len(matched_frames)} camera poses")
        return transforms_data

    def _match_frames_to_poses(self, ar_frames: List[Dict], frames_info: Dict) -> List[Dict]:
        """Match extracted video frames to AR camera poses - improved matching."""
        extracted_frames = frames_info['frame_paths']
        frame_indices = frames_info['frame_indices']
        
        matched = []
        
        # More flexible matching strategy
        total_ar_frames = len(ar_frames)
        total_extracted = len(extracted_frames)
        
        logging.info(f"Matching {total_extracted} extracted frames to {total_ar_frames} AR poses")
        
        for i, frame_path in enumerate(extracted_frames):
            original_idx = frame_indices[i]
            
            # Try multiple matching strategies
            ar_frame = None
            
            # Strategy 1: Direct index mapping
            if original_idx < total_ar_frames:
                candidate = ar_frames[original_idx]
                if candidate and 'camera_transform' in candidate:
                    ar_frame = candidate
            
            # Strategy 2: Proportional mapping if direct fails
            if not ar_frame:
                proportional_idx = int((original_idx / max(frame_indices)) * (total_ar_frames - 1))
                if 0 <= proportional_idx < total_ar_frames:
                    candidate = ar_frames[proportional_idx]
                    if candidate and 'camera_transform' in candidate:
                        ar_frame = candidate
            
            # Strategy 3: Find nearest valid frame
            if not ar_frame:
                for offset in range(min(10, total_ar_frames)):
                    for direction in [1, -1]:
                        test_idx = original_idx + (direction * offset)
                        if 0 <= test_idx < total_ar_frames:
                            candidate = ar_frames[test_idx]
                            if candidate and 'camera_transform' in candidate:
                                ar_frame = candidate
                                break
                    if ar_frame:
                        break
            
            if ar_frame:
                matched.append({
                    'frame_path': frame_path,
                    'frame_index': original_idx,
                    'ar_data': ar_frame
                })
        
        match_rate = len(matched) / len(extracted_frames) if extracted_frames else 0
        logging.info(f"Matched {len(matched)}/{len(extracted_frames)} frames ({match_rate:.1%})")
        
        return matched

    def _build_transforms_json(self, matched_frames: List[Dict]) -> Dict:
        """Build NeRF-format transforms.json from matched frames."""
        frames = []
        
        # Get camera intrinsics from first frame
        intrinsics = None
        image_res = None
        for frame in matched_frames:
            if 'intrinsics' in frame['ar_data']:
                intrinsics = frame['ar_data']['intrinsics']
            if 'image_resolution' in frame['ar_data']:
                image_res = frame['ar_data']['image_resolution']
            if intrinsics and image_res:
                break
        
        # Default values if not found
        if not intrinsics:
            logging.warning("No camera intrinsics found, using defaults")
            intrinsics = {'fx': 800, 'fy': 800, 'cx': 400, 'cy': 300}
        
        if not image_res:
            logging.warning("No image resolution found, using defaults")
            image_res = {'width': 800, 'height': 600}
        
        # Process each frame
        transforms_list = []
        for frame_data in matched_frames:
            ar_data = frame_data['ar_data']
            camera_matrix = ar_data['camera_transform']
            
            # Convert ARKit to NeRF coordinate system
            nerf_transform = self._arkit_to_nerf_transform(camera_matrix)
            
            # Build frame entry
            frame_entry = {
                "file_path": f"./images/{frame_data['frame_path'].name}",
                "transform_matrix": nerf_transform.tolist()
            }
            
            transforms_list.append(frame_entry)
        
        # Calculate camera angle from focal length
        camera_angle_x = 2.0 * np.arctan(image_res['width'] / (2.0 * intrinsics['fx']))
        
        # Build final transforms.json structure
        transforms_json = {
            "camera_angle_x": float(camera_angle_x),
            "fl_x": float(intrinsics['fx']),
            "fl_y": float(intrinsics['fy']),
            "cx": float(intrinsics['cx']),
            "cy": float(intrinsics['cy']),
            "w": int(image_res['width']),
            "h": int(image_res['height']),
            "frames": transforms_list
        }
        
        # Apply scene normalization if requested
        if self.pose_config.get('normalize_scene', True):
            transforms_json = self._normalize_scene(transforms_json)
        
        return transforms_json

    def _arkit_to_nerf_transform(self, arkit_matrix: np.ndarray) -> np.ndarray:
        """Convert ARKit camera-to-world matrix to NeRF format."""
        # ARKit uses right-handed coordinate system: +X right, +Y up, +Z backward
        # NeRF (OpenGL) uses right-handed: +X right, +Y up, +Z forward (toward camera)
        
        # Create conversion matrix from ARKit to NeRF coordinates
        # Flip Z axis to convert from ARKit (+Z backward) to NeRF (+Z forward)
        coord_conversion = np.array([
            [1,  0,  0,  0],
            [0,  1,  0,  0], 
            [0,  0, -1,  0],
            [0,  0,  0,  1]
        ])
        
        # Apply coordinate conversion
        nerf_matrix = arkit_matrix @ coord_conversion
        
        return nerf_matrix

    def _normalize_scene(self, transforms_json: Dict) -> Dict:
        """Normalize scene coordinates to reasonable bounds."""
        frames = transforms_json['frames']
        
        # Extract camera positions
        camera_positions = []
        for frame in frames:
            transform = np.array(frame['transform_matrix'])
            position = transform[:3, 3]
            camera_positions.append(position)
        
        camera_positions = np.array(camera_positions)
        
        # Calculate scene bounds
        scene_center = np.mean(camera_positions, axis=0)
        scene_scale = np.max(np.linalg.norm(camera_positions - scene_center, axis=1))
        
        # Normalize to unit scale
        if scene_scale > 0:
            scale_factor = 1.0 / scene_scale
            
            # Apply normalization to all transforms
            for frame in frames:
                transform = np.array(frame['transform_matrix'])
                # Scale translation
                transform[:3, 3] = (transform[:3, 3] - scene_center) * scale_factor
                frame['transform_matrix'] = transform.tolist()
        
        logging.info(f"Scene normalized: center={scene_center}, scale={scene_scale}")
        return transforms_json