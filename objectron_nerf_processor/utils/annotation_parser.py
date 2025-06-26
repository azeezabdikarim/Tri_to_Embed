"""
Parse Objectron annotation protobuf files to extract 3D bounding boxes and object metadata
"""

import struct
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

class AnnotationParser:
   def __init__(self, config: dict):
       self.config = config
       
       # Objectron 3D bounding box keypoint definitions
       # 9 keypoints: center (0) + 8 vertices (1-8)
       self.keypoint_names = {
           0: "center",
           1: "front_bottom_left",   # vertex 1
           2: "front_bottom_right",  # vertex 2  
           3: "front_top_left",      # vertex 3
           4: "front_top_right",     # vertex 4
           5: "back_bottom_left",    # vertex 5
           6: "back_bottom_right",   # vertex 6
           7: "back_top_left",       # vertex 7
           8: "back_top_right"       # vertex 8
       }
       
       # Standard bounding box edges for validation
       self.box_edges = [
           [1, 2], [2, 4], [4, 3], [3, 1],  # front face
           [5, 6], [6, 8], [8, 7], [7, 5],  # back face  
           [1, 5], [2, 6], [3, 7], [4, 8]   # connecting edges
       ]

   def parse_annotations(self, annotation_path: Path, transforms_data: Dict) -> Dict:
       """Parse annotation protobuf file and extract object metadata."""
       logging.info(f"Parsing annotations from {annotation_path}")
       
       try:
           # Parse the annotation protobuf
           annotation_data = self._parse_annotation_protobuf(annotation_path)
           
           if not annotation_data:
               raise ValueError("No valid annotation data found")
           
           # Process objects and bounding boxes
           scene_metadata = self._process_scene_annotations(annotation_data, transforms_data)
           
           logging.info(f"Parsed {len(scene_metadata.get('objects', []))} objects with bounding boxes")
           return scene_metadata
           
       except Exception as e:
           logging.error(f"Failed to parse annotations from {annotation_path}: {e}")
           raise

   def _parse_annotation_protobuf(self, file_path: Path) -> Optional[Dict]:
       """Parse annotation protobuf file - corrected for single message format."""
       try:
           with open(file_path, 'rb') as f:
               data = f.read()
           
           # This is a single protobuf message, not length-prefixed messages
           parsed_data = {
               'objects': [],
               'frame_annotations': []
           }
           
           # Parse the single protobuf message
           parsed_message = self._parse_single_protobuf_message(data)
           if parsed_message:
               parsed_data.update(parsed_message)
           
           return parsed_data if parsed_data['objects'] else None
           
       except Exception as e:
           logging.error(f"Failed to read annotation file {file_path}: {e}")
           return None

   def _parse_single_protobuf_message(self, data: bytes) -> Optional[Dict]:
       """Parse single protobuf message using wire format."""
       try:
           parsed = {'objects': [], 'frame_annotations': []}
           
           # Look for text strings to identify categories
           categories = self._extract_text_strings(data)
           
           # Look for coordinate patterns
           coordinates_3d = self._extract_coordinate_sequences(data)
           
           # Look for rotation matrices and other object data
           rotation_matrices = self._extract_rotation_matrices_improved(data)
           translations = self._extract_translation_vectors_improved(data)
           
           # Combine into objects
           if categories:
               for i, category in enumerate(categories[:1]):  # Usually one object per file
                   obj = {
                       'object_id': i,
                       'category': category,
                       'keypoints_3d': coordinates_3d[i] if i < len(coordinates_3d) else None,
                       'rotation': rotation_matrices[i] if i < len(rotation_matrices) else None,
                       'translation': translations[i] if i < len(translations) else None,
                       'scale': None  # Will extract separately if needed
                   }
                   parsed['objects'].append(obj)
           
           return parsed if parsed['objects'] else None
           
       except Exception as e:
           logging.debug(f"Failed to parse protobuf message: {e}")
           return None

   def _extract_text_strings(self, data: bytes) -> List[str]:
       """Extract text strings from protobuf data."""
       categories = []
       
       # Known categories
       known_categories = ['bike', 'book', 'bottle', 'camera', 'cereal_box', 'chair', 'cup', 'laptop', 'shoe']
       
       # Also look for variations
       category_variations = {
           'motobike': 'bike',  # As seen in the debug output
           'motorcycle': 'bike'
       }
       
       # Search for categories in the data
       for category in known_categories:
           if category.encode('utf-8') in data:
               categories.append(category)
       
       # Search for variations
       for variation, canonical in category_variations.items():
           if variation.encode('utf-8') in data and canonical not in categories:
               categories.append(canonical)
       
       return categories

   def _extract_coordinate_sequences(self, data: bytes) -> List[np.ndarray]:
       """Extract sequences of 3D coordinates that could be keypoints."""
       coordinate_sets = []
       
       # Look for the specific pattern we see in the debug output
       # The hex shows repeated patterns like: 0d000000bf15000000bf1d000000bf
       # This suggests coordinates stored as little-endian floats
       
       i = 0
       while i < len(data) - 36:  # Need at least 9 floats (9*4 = 36 bytes)
           try:
               # Look for a sequence of 9 3D points (27 floats total)
               potential_coords = []
               
               for point_idx in range(9):
                   offset = i + (point_idx * 12)  # Each point is 3 floats = 12 bytes
                   if offset + 12 > len(data):
                       break
                   
                   try:
                       x, y, z = struct.unpack('<3f', data[offset:offset + 12])
                       
                       # Validate coordinates are reasonable
                       if all(-10 < coord < 10 for coord in [x, y, z]):
                           potential_coords.append([x, y, z])
                       else:
                           break
                   except:
                       break
               
               # If we found 9 valid points, this might be a keypoint set
               if len(potential_coords) == 9:
                   coordinate_sets.append(np.array(potential_coords))
                   i += 108  # Skip past this coordinate set
               else:
                   i += 4  # Move to next potential starting point
                   
           except:
               i += 4
       
       return coordinate_sets

   def _extract_rotation_matrices_improved(self, data: bytes) -> List[np.ndarray]:
       """Extract rotation matrices with better validation."""
       rotations = []
       
       for i in range(0, len(data) - 36, 4):
           try:
               # Extract 9 floats for a 3x3 matrix
               matrix_data = struct.unpack('<9f', data[i:i + 36])
               matrix = np.array(matrix_data).reshape(3, 3)
               
               # Better validation for rotation matrices
               if self._is_valid_rotation_matrix(matrix):
                   rotations.append(matrix)
                   
           except:
               continue
       
       return rotations

   def _is_valid_rotation_matrix(self, R: np.ndarray) -> bool:
       """Improved rotation matrix validation."""
       if R.shape != (3, 3):
           return False
       
       # Check if all values are reasonable
       if np.any(np.abs(R) > 2):  # Rotation matrix elements should be [-1, 1]
           return False
       
       # Check orthogonality (R @ R.T ≈ I)
       should_be_identity = R @ R.T
       identity = np.eye(3)
       if not np.allclose(should_be_identity, identity, atol=0.1):
           return False
       
       # Check determinant ≈ 1
       det = np.linalg.det(R)
       if not np.isclose(det, 1.0, atol=0.1):
           return False
       
       return True

   def _extract_translation_vectors_improved(self, data: bytes) -> List[np.ndarray]:
       """Extract translation vectors with better validation."""
       translations = []
       
       for i in range(0, len(data) - 12, 4):
           try:
               x, y, z = struct.unpack('<3f', data[i:i + 12])
               
               # Validate translation vectors are reasonable world coordinates
               if all(-100 < coord < 100 for coord in [x, y, z]):
                   # Additional check: not all zeros
                   if not all(abs(coord) < 1e-6 for coord in [x, y, z]):
                       translations.append(np.array([x, y, z]))
                       
           except:
               continue
       
       return translations

   def _process_scene_annotations(self, annotation_data: Dict, transforms_data: Dict) -> Dict:
       """Process raw annotation data into structured scene metadata."""
       objects = annotation_data.get('objects', [])
       
       processed_objects = []
       for obj in objects:
           # Convert 3D keypoints to bounding box representation
           bbox_3d = self._keypoints_to_bbox(obj.get('keypoints_3d'))
           
           # Create object metadata
           object_metadata = {
               'object_id': obj['object_id'],
               'category': obj['category'],
               'confidence': 1.0,  # Objectron annotations are ground truth
               'keypoints_3d': obj['keypoints_3d'].tolist() if obj.get('keypoints_3d') is not None else None,
               'keypoints_2d': obj['keypoints_2d'].tolist() if obj.get('keypoints_2d') is not None else None,
               'bbox_3d': bbox_3d,
               'rotation_matrix': obj['rotation'].tolist() if obj.get('rotation') is not None else None,
               'translation': obj['translation'].tolist() if obj.get('translation') is not None else None,
               'scale': obj['scale'].tolist() if obj.get('scale') is not None else None
           }
           
           processed_objects.append(object_metadata)
       
       # Create scene metadata
       scene_metadata = {
           'scene_type': 'objectron',
           'total_objects': len(processed_objects),
           'object_categories': list(set(obj['category'] for obj in processed_objects)),
           'objects': processed_objects,
           'annotation_source': 'objectron_ground_truth',
           'coordinate_system': 'camera_centered'
       }
       
       return scene_metadata

   def _keypoints_to_bbox(self, keypoints_3d: Optional[np.ndarray]) -> Optional[Dict]:
       """Convert 9 keypoints to axis-aligned bounding box representation."""
       if keypoints_3d is None:
           return None
       
       center = keypoints_3d[0]  # Center point
       vertices = keypoints_3d[1:]  # 8 corner vertices
       
       # Calculate axis-aligned bounding box
       min_coords = np.min(vertices, axis=0)
       max_coords = np.max(vertices, axis=0)
       size = max_coords - min_coords
       
       # Calculate oriented bounding box from vertices
       # This is more complex but preserves rotation information
       bbox_data = {
           'center': center.tolist(),
           'size': size.tolist(),
           'min_coords': min_coords.tolist(),
           'max_coords': max_coords.tolist(),
           'vertices': vertices.tolist(),
           'volume': float(np.prod(size)),
           'type': 'oriented_3d_bbox'
       }
       
       return bbox_data