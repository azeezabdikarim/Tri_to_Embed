"""
Objectron dataset downloader - handles fetching samples from Google Cloud Storage
"""

import os
import requests
import time
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from urllib.parse import urljoin
import random

class ObjectronDownloader:
    def __init__(self, config: dict):
        self.config = config
        self.bucket_url = config['objectron']['base_url']
        self.index_url = config['objectron']['index_base_url']
        self.temp_dir = Path(config['paths']['temp_dir'])
        self.timeout = config['objectron']['download_timeout']
        self.retry_attempts = config['objectron']['retry_attempts']
        
        # Create temp subdirectories
        self.video_dir = self.temp_dir / 'videos'
        self.geometry_dir = self.temp_dir / 'geometry'
        self.annotation_dir = self.temp_dir / 'annotations'
        
        for dir_path in [self.video_dir, self.geometry_dir, self.annotation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_category_samples(self, category: str) -> List[str]:
        """Get list of available samples for a category from Objectron index."""
        index_url = f"{self.index_url}/{category}_annotations"
        
        try:
            logging.info(f"Fetching sample index for {category}")
            response = requests.get(index_url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse sample IDs from index file
            samples = []
            for line in response.text.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    samples.append(line)
            
            logging.info(f"Found {len(samples)} samples for {category}")
            return samples
            
        except Exception as e:
            logging.error(f"Failed to fetch index for {category}: {e}")
            return []

    def download_sample(self, sample_id: str) -> Tuple[Path, Path, Path]:
        """Download video, geometry, and annotation files for a sample."""
        logging.debug(f"Downloading sample: {sample_id}")
        
        # Generate file URLs
        video_url = f"{self.bucket_url}/videos/{sample_id}/video.MOV"
        geometry_url = f"{self.bucket_url}/videos/{sample_id}/geometry.pbdata"
        annotation_url = f"{self.bucket_url}/annotations/{sample_id}.pbdata"
        
        # Generate local file paths
        safe_id = sample_id.replace('/', '_').replace('-', '_')
        video_path = self.video_dir / f"{safe_id}.MOV"
        geometry_path = self.geometry_dir / f"{safe_id}.pbdata"
        annotation_path = self.annotation_dir / f"{safe_id}.pbdata"
        
        try:
            # Download each file
            video_path = self._download_file(video_url, video_path, "video")
            geometry_path = self._download_file(geometry_url, geometry_path, "geometry")
            annotation_path = self._download_file(annotation_url, annotation_path, "annotation")
            
            return video_path, geometry_path, annotation_path
            
        except Exception as e:
            # Cleanup partial downloads on failure
            for path in [video_path, geometry_path, annotation_path]:
                if path.exists():
                    try:
                        path.unlink()
                    except:
                        pass
            raise e

    def _download_file(self, url: str, local_path: Path, file_type: str) -> Path:
        """Download a single file with retry logic."""
        if local_path.exists():
            logging.debug(f"File already exists: {local_path}")
            return local_path
        
        for attempt in range(self.retry_attempts):
            try:
                logging.debug(f"Downloading {file_type}: {url} (attempt {attempt + 1})")
                
                # Stream download for large files
                response = requests.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()
                
                # Write file in chunks
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file was downloaded
                if local_path.exists() and local_path.stat().st_size > 0:
                    logging.debug(f"Successfully downloaded {file_type}: {local_path.stat().st_size} bytes")
                    return local_path
                else:
                    raise ValueError(f"Downloaded file is empty: {local_path}")
                    
            except Exception as e:
                logging.warning(f"Download attempt {attempt + 1} failed for {file_type}: {e}")
                if local_path.exists():
                    local_path.unlink()
                
                if attempt < self.retry_attempts - 1:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    logging.debug(f"Waiting {wait_time:.1f}s before retry")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to download {file_type} after {self.retry_attempts} attempts: {e}")

    def cleanup_sample_files(self, sample_id: str):
        """Clean up downloaded files for a specific sample."""
        safe_id = sample_id.replace('/', '_').replace('-', '_')
        
        files_to_delete = [
            self.video_dir / f"{safe_id}.MOV",
            self.geometry_dir / f"{safe_id}.pbdata", 
            self.annotation_dir / f"{safe_id}.pbdata"
        ]
        
        for file_path in files_to_delete:
            if file_path.exists():
                try:
                    file_path.unlink()
                    logging.debug(f"Deleted: {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to delete {file_path}: {e}")