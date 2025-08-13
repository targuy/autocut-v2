"""
File handling utilities for video processing.
"""

from typing import List, Optional
import logging
from pathlib import Path
import mimetypes

logger = logging.getLogger(__name__)


class FileHandler:
    """
    Handles file validation and management for video processing.
    """
    
    SUPPORTED_VIDEO_FORMATS = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'
    }
    
    SUPPORTED_MIME_TYPES = {
        'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo',
        'video/x-matroska', 'video/x-ms-wmv', 'video/x-flv', 'video/webm'
    }
    
    def __init__(self):
        """Initialize file handler."""
        pass
    
    def validate_video_file(self, file_path: str) -> bool:
        """
        Validate if file is a supported video format.
        
        Args:
            file_path: Path to video file
            
        Returns:
            True if file is valid video format
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Check if it's a file (not directory)
            if not path.is_file():
                logger.error(f"Path is not a file: {file_path}")
                return False
            
            # Check file extension
            if path.suffix.lower() not in self.SUPPORTED_VIDEO_FORMATS:
                logger.error(f"Unsupported video format: {path.suffix}")
                return False
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type not in self.SUPPORTED_MIME_TYPES:
                logger.warning(f"Questionable MIME type: {mime_type}")
            
            # Check file size (not empty)
            if path.stat().st_size == 0:
                logger.error(f"File is empty: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating video file {file_path}: {str(e)}")
            return False
    
    def find_video_files(
        self, 
        directory: str, 
        recursive: bool = True
    ) -> List[str]:
        """
        Find all video files in directory.
        
        Args:
            directory: Directory to search
            recursive: Search subdirectories
            
        Returns:
            List of video file paths
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists() or not dir_path.is_dir():
                logger.error(f"Directory not found: {directory}")
                return []
            
            video_files = []
            pattern = "**/*" if recursive else "*"
            
            for file_path in dir_path.glob(pattern):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.SUPPORTED_VIDEO_FORMATS):
                    if self.validate_video_file(str(file_path)):
                        video_files.append(str(file_path))
            
            logger.info(f"Found {len(video_files)} video files in {directory}")
            return sorted(video_files)
            
        except Exception as e:
            logger.error(f"Error finding video files in {directory}: {str(e)}")
            return []
    
    def create_output_directory(self, output_path: str) -> bool:
        """
        Create output directory if it doesn't exist.
        
        Args:
            output_path: Output directory path
            
        Returns:
            True if directory created or exists
        """
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating output directory {output_path}: {str(e)}")
            return False
