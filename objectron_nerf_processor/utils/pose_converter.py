# objectron_nerf_processor/utils/pose_converter.py

import struct
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Import the compiled Protobuf schema for ARKit session metadata
from utils.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol

class PoseConverter:
    """
    Converts Objectron geometry.pbdata into NeRF-compatible camera poses
    and intrinsics using the official Protobuf schema.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the PoseConverter."""
        self.config = config

    def _parse_proto_messages(self, data: bytes) -> List[bytes]:
        """
        Splits the raw .pbdata content into individual Protobuf messages.
        Each message is prefixed with its length as a 4-byte little-endian integer.
        """
        messages = []
        offset = 0
        while offset + 4 <= len(data):
            # Read the length of the next message
            msg_len = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            
            # Check for invalid message length
            if offset + msg_len > len(data) or msg_len == 0:
                break
                
            messages.append(data[offset:offset + msg_len])
            offset += msg_len
        return messages

    def _arkit_to_nerf_transform(self, arkit_c2w: np.ndarray) -> np.ndarray:
        """
        Converts a camera-to-world matrix from ARKit's coordinate system
        to the standard NeRF coordinate system.

        ARKit/OpenGL: +X right, +Y up, +Z backward (looks along -Z)
        NeRF/OpenCV:  +X right, +Y down, +Z forward (looks along +Z)

        This conversion corresponds to a 180-degree rotation around the X-axis.
        """
        conversion_matrix = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        return arkit_c2w @ conversion_matrix

    def convert_poses(self, geometry_path: Path, frames_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts camera poses and intrinsics using the official Protobuf schema
        and formats the data for a NeRF `transforms.json` file.

        Args:
            geometry_path: Path to the geometry.pbdata file.
            frames_info: Dictionary containing video metadata like width and height.

        Returns:
            A dictionary structured like a NeRF `transforms.json` file.
        """
        try:
            with open(geometry_path, 'rb') as f:
                raw_data = f.read()
        except FileNotFoundError:
            raise ValueError(f"Geometry file not found at: {geometry_path}")

        frame_messages = self._parse_proto_messages(raw_data)
        
        if not frame_messages:
            raise ValueError(f"Could not parse any messages from {geometry_path}. File might be empty or corrupt.")

        transforms = {
            "w": frames_info.get('width'),
            "h": frames_info.get('height'),
            "frames": []
        }
        
        has_intrinsics = False
        
        for i, message_buf in enumerate(frame_messages):
            # Create an ARFrame object from the schema
            frame_data = ar_metadata_protocol.ARFrame()
            # Parse the binary data into the structured object
            frame_data.ParseFromString(message_buf)
            
            # --- Extract Camera Pose ---
            # The transform is a 16-element list, reshape to a 4x4 matrix
            arkit_pose = np.array(frame_data.camera.transform).reshape(4, 4)
            nerf_pose = self._arkit_to_nerf_transform(arkit_pose)

            # --- Extract Camera Intrinsics (once per sequence) ---
            if not has_intrinsics and len(frame_data.camera.intrinsics) == 4:
                intrinsics = frame_data.camera.intrinsics
                transforms["fl_x"] = float(intrinsics[0])
                transforms["fl_y"] = float(intrinsics[1])
                transforms["cx"] = float(intrinsics[2])
                transforms["cy"] = float(intrinsics[3])
                transforms["camera_angle_x"] = 2 * np.arctan(0.5 * transforms["w"] / transforms["fl_x"])
                has_intrinsics = True
            
            # Match frame file names from the frame_extractor
            frame_filename = frames_info['frame_paths'][i].name
            frame_entry = {
                "file_path": f"images/{frame_filename}",
                "transform_matrix": nerf_pose.tolist()
            }
            transforms["frames"].append(frame_entry)

        if not transforms["frames"]:
            raise ValueError("Failed to extract any valid poses using the schema.")
            
        return transforms