"""
Scene detection module for identifying video segments.
"""

from typing import List, Tuple, Optional
import logging
import subprocess
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class SceneDetector:
    """
    Detects scene changes in video content using various algorithms.
    """
    
    def __init__(self, config: dict):
        """
        Initialize scene detector.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.threshold = config.get('scenes', {}).get('threshold', 0.3)
        self.min_scene_length = config.get('min_scene_length', 1.0)  # seconds
        self.detection_method = config.get('scenes', {}).get('method', 'ffmpeg')
        self.enabled = config.get('scenes', {}).get('enabled', True)
        
    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries in video.
        
        Args:
            video_path: Path to video file to analyze
            
        Returns:
            List of tuples containing (start_time, end_time) for each scene
        """
        try:
            if not self.enabled:
                logger.info("Scene detection disabled")
                return [(0, 60.0)]  # Return single scene for mock
            
            logger.info(f"Detecting scenes using {self.detection_method} method")
            
            if self.detection_method == 'ffmpeg':
                return self._detect_scenes_ffmpeg(video_path)
            elif self.detection_method == 'pyscenedetect':
                return self._detect_scenes_pyscenedetect(video_path)
            else:
                logger.warning(f"Unknown detection method: {self.detection_method}")
                return self._detect_scenes_ffmpeg(video_path)
                
        except Exception as e:
            logger.error(f"Error in scene detection: {str(e)}")
            return [(0, 60.0)]  # Return single scene on error
    
    def _detect_scenes_ffmpeg(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Detect scenes using FFmpeg scene filter.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of scene boundaries
        """
        try:
            logger.info("Using FFmpeg scene detection")
            
            # First, get video duration
            duration_cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', video_path
            ]
            
            try:
                duration_result = subprocess.run(
                    duration_cmd, capture_output=True, text=True, check=True
                )
                duration = float(duration_result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError) as e:
                logger.warning(f"Could not get video duration: {e}")
                duration = 60.0  # Default fallback
            
            # Run FFmpeg scene detection
            scene_cmd = [
                'ffmpeg', '-i', video_path,
                '-filter:v', f'select=gt(scene\\,{self.threshold}),showinfo',
                '-f', 'null', '-'
            ]
            
            try:
                result = subprocess.run(
                    scene_cmd, capture_output=True, text=True, check=False
                )
                
                # Parse scene detection output from stderr
                scene_times = []
                lines = result.stderr.split('\n')
                
                for line in lines:
                    if 'showinfo' in line and 'pts_time:' in line:
                        try:
                            # Extract timestamp from showinfo output
                            pts_start = line.find('pts_time:') + 9
                            pts_end = line.find(' ', pts_start)
                            if pts_end == -1:
                                pts_end = len(line)
                            timestamp = float(line[pts_start:pts_end])
                            scene_times.append(timestamp)
                        except (ValueError, IndexError):
                            continue
                
                # Convert scene times to scene boundaries
                scenes = []
                if scene_times:
                    # Add start of video
                    if scene_times[0] > 0:
                        scenes.append((0.0, scene_times[0]))
                    
                    # Add scenes between detected boundaries
                    for i in range(len(scene_times) - 1):
                        scenes.append((scene_times[i], scene_times[i + 1]))
                    
                    # Add final scene to end of video
                    if scene_times[-1] < duration:
                        scenes.append((scene_times[-1], duration))
                else:
                    # No scene changes detected, return single scene
                    scenes = [(0.0, duration)]
                
                # Filter out scenes shorter than minimum length
                filtered_scenes = []
                for start, end in scenes:
                    if end - start >= self.min_scene_length:
                        filtered_scenes.append((start, end))
                
                if not filtered_scenes:
                    # If all scenes were too short, return single scene
                    filtered_scenes = [(0.0, duration)]
                
                logger.info(f"Detected {len(filtered_scenes)} scenes using FFmpeg method")
                return filtered_scenes
                
            except Exception as e:
                logger.error(f"FFmpeg scene detection failed: {e}")
                # Return single scene as fallback
                return [(0.0, duration)]
                
        except Exception as e:
            logger.error(f"Error in FFmpeg scene detection: {e}")
            return [(0.0, 60.0)]  # Return fallback scene
    
    def _detect_scenes_pyscenedetect(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Detect scenes using PySceneDetect.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of scene boundaries
        """
        try:
            logger.info("Using PySceneDetect")
            
            # Try to import PySceneDetect
            try:
                from scenedetect import detect, ContentDetector
            except ImportError:
                logger.warning("PySceneDetect not available, falling back to FFmpeg")
                return self._detect_scenes_ffmpeg(video_path)
            
            # Detect scenes using ContentDetector
            scene_list = detect(video_path, ContentDetector(threshold=self.threshold))
            
            if not scene_list:
                logger.warning("No scenes detected, returning single scene")
                # Get video duration as fallback
                try:
                    duration_cmd = [
                        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', video_path
                    ]
                    duration_result = subprocess.run(
                        duration_cmd, capture_output=True, text=True, check=True
                    )
                    duration = float(duration_result.stdout.strip())
                except:
                    duration = 60.0
                return [(0.0, duration)]
            
            # Convert scene list to our format
            scenes = []
            for i, scene in enumerate(scene_list):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                
                # Filter by minimum scene length
                if end_time - start_time >= self.min_scene_length:
                    scenes.append((start_time, end_time))
            
            if not scenes:
                # All scenes were too short, return single scene
                total_duration = scene_list[-1][1].get_seconds() if scene_list else 60.0
                scenes = [(0.0, total_duration)]
            
            logger.info(f"Detected {len(scenes)} scenes using PySceneDetect method")
            return scenes
            
        except Exception as e:
            logger.error(f"PySceneDetect failed: {e}, falling back to FFmpeg")
            return self._detect_scenes_ffmpeg(video_path)
