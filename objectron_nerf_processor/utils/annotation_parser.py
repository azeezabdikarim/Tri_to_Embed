# objectron_nerf_processor/utils/annotation_parser.py

import struct
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Import the compiled Protobuf schema for annotation data
from utils.schema import annotation_data_pb2 as annotation_protocol

class AnnotationParser:
    """
    Parses Objectron annotation.pbdata files to extract 3D bounding box
    information using the official Protobuf schema.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the AnnotationParser."""
        self.config = config

    def _parse_proto_messages(self, data: bytes) -> List[bytes]:
        """
        Splits the raw .pbdata content into individual Protobuf messages.
        """
        messages = []
        offset = 0
        while offset + 4 <= len(data):
            msg_len = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            if offset + msg_len > len(data) or msg_len == 0:
                break
            messages.append(data[offset:offset + msg_len])
            offset += msg_len
        return messages

    def parse_annotations(self, annotation_path: Path, transforms_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses 3D bounding box annotations and creates scene metadata.

        Args:
            annotation_path: Path to the annotation.pbdata file.
            transforms_data: The transforms dictionary, used for frame count.

        Returns:
            A dictionary containing structured metadata about the scene's objects.
        """
        try:
            with open(annotation_path, 'rb') as f:
                raw_data = f.read()
        except FileNotFoundError:
            # It's okay if annotations don't exist; return minimal metadata
            return {
                'scene_type': 'objectron',
                'parsing_error': 'Annotation file not found.',
                'total_objects': 0,
                'object_categories': [],
                'objects': [],
            }
            
        annotation_frame_messages = self._parse_proto_messages(raw_data)
        
        # This will hold the final list of unique objects found in the sequence
        scene_objects = {}
        object_categories = set()

        # The annotations are per-frame, so we iterate through them
        for frame_index, message_buf in enumerate(annotation_frame_messages):
            frame_annotations = annotation_protocol.FrameAnnotations()
            frame_annotations.ParseFromString(message_buf)

            for annotation in frame_annotations.annotations:
                object_id = annotation.object_id
                category = annotation.category_name
                object_categories.add(category)

                # Get the 9 keypoints (center + 8 vertices) of the 3D box
                keypoints_3d = []
                for kp in annotation.keypoints:
                    keypoints_3d.append([kp.x, kp.y, kp.z])
                
                # We only need to store each unique object once with its 3D box
                if object_id not in scene_objects:
                    scene_objects[object_id] = {
                        "id": object_id,
                        "category": category,
                        "vertices_3d": keypoints_3d
                    }
        
        final_object_list = list(scene_objects.values())

        scene_metadata = {
            'scene_type': 'objectron',
            'annotation_source': 'objectron_schema_parser',
            'coordinate_system': 'camera_centered', # As defined by Objectron
            'total_objects': len(final_object_list),
            'object_categories': list(object_categories),
            'objects': final_object_list,
        }

        return scene_metadata