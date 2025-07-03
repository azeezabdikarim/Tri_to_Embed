"""
Extract frames from Objectron videos using FFmpeg
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import logging
import cv2
import re
import numpy as np

class FrameExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.frame_config = config['frame_extraction']
        self.temp_dir = Path(config['paths']['temp_dir'])
        
        # Verify FFmpeg is available
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found. Please install FFmpeg to extract video frames.")

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available in system PATH."""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def extract_frames(self, video_path: Path, sample_id: str) -> Optional[Dict]:
        """Extract frames from video at target FPS."""
        logging.info(f"Extracting frames from {video_path}")
        
        try:
            # Get video information
            video_info = self._get_video_info(video_path)
            if not video_info:
                raise ValueError("Could not get video information")
            
            # Calculate frame extraction parameters
            extraction_params = self._calculate_extraction_params(video_info)
            
            # Create temporary directory for frames
            temp_frames_dir = self.temp_dir / 'frames' / sample_id.replace('/', '_')
            temp_frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames using FFmpeg
            frame_paths, frame_indices = self._extract_frames_ffmpeg(
                video_path, temp_frames_dir, extraction_params
            )
            
            # Validate extracted frames
            if len(frame_paths) < self.frame_config['min_frames']:
                raise ValueError(f"Too few frames extracted: {len(frame_paths)} < {self.frame_config['min_frames']}")
            
            if len(frame_paths) > self.frame_config['max_frames']:
                # Subsample to max frames
                logging.info(f"Subsampling {len(frame_paths)} frames to {self.frame_config['max_frames']}")
                frame_paths, frame_indices = self._subsample_frames(
                    frame_paths, frame_indices, self.frame_config['max_frames']
                )
            
            # Resize frames if necessary
            frame_paths = self._resize_frames_if_needed(frame_paths)
            
            logging.info(f"Successfully extracted {len(frame_paths)} frames")
            
            return {
                'frame_paths': frame_paths,
                'frame_indices': frame_indices,
                'video_info': video_info,
                'total_frames_extracted': len(frame_paths),
                'frame_interval':extraction_params['frame_interval']
            }
            
        except Exception as e:
            logging.error(f"Frame extraction failed for {video_path}: {e}")
            # Cleanup on failure
            temp_frames_dir = self.temp_dir / 'frames' / sample_id.replace('/', '_')
            if temp_frames_dir.exists():
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
            return None

    def _get_video_info(self, video_path: Path) -> Optional[Dict]:
        """Get video metadata using FFprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'v:0', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logging.error(f"FFprobe failed: {result.stderr}")
                return None
            
            import json
            probe_data = json.loads(result.stdout)
            
            if not probe_data.get('streams'):
                return None
            
            video_stream = probe_data['streams'][0]
            
            # Parse frame rate
            fps_str = video_stream.get('r_frame_rate', '30/1')
            fps_parts = fps_str.split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
            
            return {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': fps,
                'duration': float(video_stream.get('duration', 0)),
                'total_frames': int(video_stream.get('nb_frames', 0)) if video_stream.get('nb_frames') else None
            }
            
        except Exception as e:
            logging.error(f"Failed to get video info: {e}")
            return None

    def _calculate_extraction_params(self, video_info: Dict) -> Dict:
        """Calculate frame extraction parameters."""
        source_fps = video_info['fps']
        target_fps = self.frame_config['target_fps']
        
        # Calculate frame skip interval
        if source_fps <= target_fps:
            frame_interval = 1  # Extract all frames
        else:
            frame_interval = int(round(source_fps / target_fps))
        
        # Estimate total frames to extract
        if video_info.get('total_frames'):
            estimated_output_frames = video_info['total_frames'] // frame_interval
        else:
            estimated_output_frames = int(video_info['duration'] * target_fps)
        
        return {
            'source_fps': source_fps,
            'target_fps': target_fps,
            'frame_interval': frame_interval,
            'estimated_frames': estimated_output_frames
        }

    def _extract_frames_ffmpeg(self, video_path: Path, output_dir: Path, 
                             params: Dict) -> tuple[List[Path], List[int]]:
        """Extract frames using FFmpeg."""
        
        # FFmpeg command to extract frames at specified interval
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'select=not(mod(n\\,{params["frame_interval"]}))',
            '-vsync', 'vfr',  # Variable frame rate
            '-frame_pts', 'true',  # Use original frame timestamps
            '-q:v', str(self.frame_config.get('ffmpeg_quality', 2)),  # Quality setting
            str(output_dir / '%06d.png')
        ]
        
        # Add resolution scaling if specified
        max_res = self.frame_config.get('max_resolution')
        if max_res:
            # Add scaling filter
            scale_filter = f'scale=min({max_res}\\,iw):min({max_res}\\,ih):force_original_aspect_ratio=decrease'
            cmd[4] = f'select=not(mod(n\\,{params["frame_interval"]})),{scale_filter}'
        
        logging.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                timeout=self.config['validation']['max_processing_time']
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            # Get list of extracted frames
            frame_files = sorted(output_dir.glob('*.png'))
            
            # Calculate original frame indices
            frame_indices = []
            for i, frame_file in enumerate(frame_files):
                # Extract frame number from filename
                frame_num = int(frame_file.stem)
                original_index = (frame_num - 1) * params['frame_interval']
                frame_indices.append(original_index)
            
            return frame_files, frame_indices
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg extraction timed out")
        except Exception as e:
            raise RuntimeError(f"FFmpeg extraction failed: {e}")

    def _subsample_frames(self, frame_paths: List[Path], frame_indices: List[int], 
                         max_frames: int) -> tuple[List[Path], List[int]]:
        """Subsample frames to maximum count."""
        if len(frame_paths) <= max_frames:
            return frame_paths, frame_indices
        
        # Evenly distribute frames across the sequence
        indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
        
        subsampled_paths = [frame_paths[i] for i in indices]
        subsampled_indices = [frame_indices[i] for i in indices]
        
        # Delete unused frames to save space
        for i, path in enumerate(frame_paths):
            if i not in indices:
                try:
                    path.unlink()
                except:
                    pass
        
        return subsampled_paths, subsampled_indices

    def _resize_frames_if_needed(self, frame_paths: List[Path]) -> List[Path]:
        """Resize frames if they exceed maximum resolution."""
        max_res = self.frame_config.get('max_resolution')
        if not max_res:
            return frame_paths
        
        # Check first frame to see if resizing is needed
        if not frame_paths:
            return frame_paths
        
        try:
            import cv2
            first_frame = cv2.imread(str(frame_paths[0]))
            if first_frame is None:
                return frame_paths
            
            height, width = first_frame.shape[:2]
            max_dim = max(height, width)
            
            if max_dim <= max_res:
                return frame_paths  # No resizing needed
            
            # Calculate new dimensions
            scale = max_res / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            logging.info(f"Resizing frames from {width}x{height} to {new_width}x{new_height}")
            
            # Resize all frames
            for frame_path in frame_paths:
                try:
                    img = cv2.imread(str(frame_path))
                    if img is not None:
                        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                        cv2.imwrite(str(frame_path), resized)
                except Exception as e:
                    logging.warning(f"Failed to resize {frame_path}: {e}")
            
        except ImportError:
            logging.warning("OpenCV not available for frame resizing")
        except Exception as e:
            logging.warning(f"Frame resizing failed: {e}")
        
        return frame_paths