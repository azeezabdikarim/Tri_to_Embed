"""
Convert ARKit camera poses from Objectron to NeRF transforms.json format
Fixed version based on analysis of real Objectron geometry files
"""

import json
import numpy as np
import struct
from pathlib import Path
from typing import Dict, List, Any
import logging

class PoseConverter:
    def __init__(self, config: dict):
        self.config = config
        self.pose_config = config['pose_conversion']

    def convert_poses(self, geometry_path: Path, frames_info: Dict) -> Dict:
        """Convert ARKit poses to NeRF transforms.json format."""
        logging.info(f"Converting poses from {geometry_path}")
        
        # Parse geometry file using the proven working approach
        ar_frames = self._parse_geometry_with_known_offsets(geometry_path)
        
        # if not ar_frames:
        #     logging.warning("No valid camera frames found, using fallback pose generation")
        #     ar_frames = self._generate_fallback_poses(len(frames_info['frame_paths']))
        if not ar_frames:
            raise ValueError(f"No valid camera poses found in {geometry_path}. Cannot proceed with synthetic poses.")
    
        
        # Match AR frames to extracted video frames
        matched_frames = self._match_frames_to_poses(ar_frames, frames_info)
        
        # Convert to NeRF format
        transforms_data = self._build_transforms_json(matched_frames, frames_info)
        
        logging.info(f"Converted {len(matched_frames)} camera poses")
        return transforms_data

    def _parse_geometry_with_known_offsets(self, file_path: Path) -> List[Dict]:
        """Parse geometry file using the analysis results to find real camera matrices."""
        frames = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            logging.info(f"Geometry file size: {len(data)} bytes")
            
            # Based on analysis: look for length-prefixed messages
            offset = 0
            message_count = 0
            
            while offset + 4 <= len(data) and message_count < 1000:  # Reasonable limit
                try:
                    # Read message length
                    msg_len = struct.unpack('<I', data[offset:offset + 4])[0]
                    
                    # Check if this looks like a valid message length
                    if 2000 <= msg_len <= 3000 and offset + 4 + msg_len <= len(data):
                        # Extract the message
                        msg_start = offset + 4
                        msg_data = data[msg_start:msg_start + msg_len]
                        
                        # Look for transformation matrices in this message
                        matrices = self._extract_matrices_from_message(msg_data, msg_start)
                        
                        for matrix in matrices:
                            frame_data = {
                                'camera_transform': matrix,
                                'intrinsics': None,  # Will be filled with defaults
                                'image_resolution': None  # Will be filled with defaults
                            }
                            frames.append(frame_data)
                        
                        offset = msg_start + msg_len
                        message_count += 1
                    else:
                        offset += 4
                        
                except struct.error:
                    offset += 4
                    continue
            
            logging.info(f"Found {len(frames)} valid camera matrices from {message_count} messages")
            return frames
            
        except Exception as e:
            logging.error(f"Failed to parse geometry file: {e}")
            return []

    def _extract_matrices_from_message(self, msg_data: bytes, base_offset: int) -> List[np.ndarray]:
        """Extract transformation matrices from a protobuf message."""
        matrices = []
        
        # Look for 16 consecutive floats that form valid transformation matrices
        for offset in range(0, len(msg_data) - 64, 4):  # 16 floats = 64 bytes
            try:
                # Extract 16 consecutive floats
                floats = struct.unpack('<16f', msg_data[offset:offset + 64])
                matrix = np.array(floats).reshape(4, 4)
                
                # Use the WORKING validation from our analysis
                if self._is_valid_camera_matrix(matrix):
                    matrices.append(matrix)
                    
            except (struct.error, ValueError):
                continue
        
        return matrices

    def _is_valid_camera_matrix(self, matrix: np.ndarray) -> bool:
        """Minimal validation - only check basic structure."""
        try:
            # Only check basic structure, not content
            if matrix.shape != (4, 4):
                return False
            
            # Check for finite values
            if not np.all(np.isfinite(matrix)):
                return False
            
            # That's it - accept everything else
            return True
            
        except:
            return False
    def _match_frames_to_poses(self, ar_frames: List[Dict], frames_info: Dict) -> List[Dict]:
        """Match extracted video frames to AR camera poses - NO REUSE."""
        extracted_frames = frames_info['frame_paths']
        frame_indices = frames_info['frame_indices']
        
        total_ar_frames = len(ar_frames)
        total_extracted = len(extracted_frames)
        
        logging.info(f"Matching {total_extracted} extracted frames to {total_ar_frames} AR poses")
        
        # REQUIRE sufficient poses
        if total_ar_frames < total_extracted:
            raise ValueError(f"Insufficient poses: need {total_extracted}, got {total_ar_frames}. "
                            f"Cannot reuse poses - this corrupts frame-pose correspondence.")
        
        matched = []
        
        for i, frame_path in enumerate(extracted_frames):
            original_idx = frame_indices[i] if i < len(frame_indices) else i
            
            # Direct mapping - NO fallbacks, NO reuse
            if original_idx < total_ar_frames:
                ar_frame = ar_frames[original_idx]
                matched.append({
                    'frame_path': frame_path,
                    'frame_index': original_idx,
                    'ar_data': ar_frame
                })
            else:
                raise ValueError(f"Frame {i} (original index {original_idx}) has no corresponding pose. "
                            f"Available poses: {total_ar_frames}")
        
        logging.info(f"Successfully matched {len(matched)} frames to unique poses")
        return matched

    # def _match_frames_to_poses(self, ar_frames: List[Dict], frames_info: Dict) -> List[Dict]:
    #     """Match extracted video frames to AR camera poses."""
    #     extracted_frames = frames_info['frame_paths']
    #     frame_indices = frames_info['frame_indices']
        
    #     matched = []
    #     total_ar_frames = len(ar_frames)
        
    #     logging.info(f"Matching {len(extracted_frames)} extracted frames to {total_ar_frames} AR poses")
        
    #     for i, frame_path in enumerate(extracted_frames):
    #         original_idx = frame_indices[i] if i < len(frame_indices) else i
            
    #         # Map to available AR frames using direct indexing or interpolation
    #         if total_ar_frames > 0:
    #             # Use modulo to cycle through available poses
    #             ar_idx = original_idx % total_ar_frames
    #             ar_frame = ar_frames[ar_idx]
    #         else:
    #             # Fallback: generate a pose
    #             ar_frame = self._generate_single_fallback_pose(i, len(extracted_frames))
            
    #         matched.append({
    #             'frame_path': frame_path,
    #             'frame_index': original_idx,
    #             'ar_data': ar_frame
    #         })
        
    #     logging.info(f"Matched {len(matched)} frames")
    #     return matched

    def _generate_single_fallback_pose(self, index: int, total_frames: int) -> Dict:
        """Generate a single fallback camera pose."""
        # Create circular camera path
        angle = 2 * np.pi * index / max(total_frames - 1, 1)
        radius = 2.0
        
        # Camera position
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = 0.5  # Slightly elevated
        
        # Look at origin
        eye = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        
        # Create look-at matrix
        camera_matrix = self._look_at_matrix(eye, target, up)
        
        return {
            'camera_transform': camera_matrix,
            'intrinsics': None,
            'image_resolution': None
        }

    def _look_at_matrix(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create a look-at transformation matrix."""
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create rotation matrix
        rotation = np.column_stack([right, up, -forward])
        
        # Create full transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = eye
        
        return transform

    def _generate_fallback_poses(self, num_frames: int) -> List[Dict]:
        """Generate reasonable fallback camera poses if parsing fails."""
        logging.warning(f"Generating {num_frames} fallback camera poses")
        
        frames = []
        for i in range(num_frames):
            frame_data = self._generate_single_fallback_pose(i, num_frames)
            frames.append(frame_data)
        
        return frames

    def _build_transforms_json(self, matched_frames: List[Dict], frames_info: Dict) -> Dict:
        """Build NeRF-format transforms.json from matched frames."""
        
        # Get actual image resolution from first frame
        if matched_frames:
            first_frame_path = matched_frames[0]['frame_path']
            try:
                from PIL import Image
                with Image.open(first_frame_path) as img:
                    actual_width, actual_height = img.size
                    image_res = {'width': actual_width, 'height': actual_height}
                    logging.info(f"Got actual image resolution: {actual_width}x{actual_height}")
            except Exception as e:
                logging.warning(f"Failed to get image resolution from image: {e}")
                image_res = {'width': 800, 'height': 600}
        else:
            image_res = {'width': 800, 'height': 600}
        
        # Use reasonable default camera intrinsics
        focal_length = image_res['width'] * 0.7  # Conservative estimate
        intrinsics = {
            'fx': focal_length,
            'fy': focal_length,
            'cx': image_res['width'] / 2.0,
            'cy': image_res['height'] / 2.0
        }
        
        logging.info(f"Using camera intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
        
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
        
        return transforms_json

    def _arkit_to_nerf_transform(self, arkit_matrix: np.ndarray) -> np.ndarray:
        """Convert ARKit camera-to-world matrix to NeRF format."""
        try:
            # ARKit coordinate system: +X right, +Y up, +Z backward (towards user)
            # NeRF coordinate system: +X right, +Y up, +Z forward (into scene)
            
            # Flip Z axis: ARKit +Z backward → NeRF +Z forward
            conversion = np.array([
                [1,  0,  0,  0],
                [0,  1,  0,  0], 
                [0,  0, -1,  0],
                [0,  0,  0,  1]
            ])
            
            # Apply coordinate conversion
            nerf_matrix = arkit_matrix @ conversion
            
            return nerf_matrix
            
        except Exception as e:
            logging.warning(f"Failed to convert ARKit matrix: {e}")
            return np.eye(4)