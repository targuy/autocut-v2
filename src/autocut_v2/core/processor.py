"""
Core video processing module for AutoCut v2.
"""

from typing import Optional, Dict, Any, List
import logging
import json
import time
import subprocess
import re
from datetime import datetime
from pathlib import Path
# import cv2
# from moviepy.editor import VideoFileClip

from ..utils.file_handler import FileHandler
from ..utils.config import Config
from .scene_detector import SceneDetector
from .workflow import WorkflowOptimizer
from .criteria import CriteriaManager, FrameData
from ..processors.video_cutter import VideoCutter

logger = logging.getLogger(__name__)


class AutoCut:
    """
    Main video processing class for AutoCut v2.
    
    Handles video loading, processing, and saving with various automation features.
    Implements the intelligent adaptive workflow from specifications.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AutoCut processor.
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = Config(config or {})
        self.file_handler = FileHandler()
        self.scene_detector = SceneDetector(self.config.to_dict())
        self.video_cutter = VideoCutter(self.config.to_dict())
        self.workflow_optimizer = WorkflowOptimizer(self.config.to_dict())
        self.criteria_manager = CriteriaManager(self.config.to_dict())
        
    def process_video(
        self,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a video file with automated cutting and editing.
        
        Uses intelligent adaptive workflow to optimize processing order.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary containing processing results and metadata
        """
        try:
            # Validate input file
            if not self.file_handler.validate_video_file(input_path):
                raise ValueError(f"Invalid or unsupported video file: {input_path}")
            
            logger.info(f"Processing video: {input_path}")
            
            # Create execution plan using workflow optimizer
            execution_plan = self.workflow_optimizer.create_execution_plan(input_path)
            logger.info(f"Execution plan created with {len(execution_plan['steps'])} steps")
            logger.info(f"Estimated total time: {execution_plan['estimated_total_time']:.1f}s")
            
            # Create output directory
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create temp directory for intermediate files
            temp_dir = output_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            # STEP 1: Video Analysis (IMPLEMENTED)
            logger.info("Starting Step 1: Video Analysis")
            analysis_result = self._execute_step_1_analysis(input_path, temp_dir)
            logger.info(f"Step 1 completed: {analysis_result['summary']}")
            
            # Save intermediate results for verification
            analysis_file = temp_dir / "step1_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_result, f, indent=2, default=str)
            logger.info(f"Step 1 results saved to: {analysis_file}")
            
            # STEP 2: Scene Detection (IMPLEMENTED)
            logger.info("Starting Step 2: Scene Detection")
            scene_result = self._execute_step_2_scene_detection(input_path, temp_dir, analysis_result)
            logger.info(f"Step 2 completed: {scene_result['summary']}")
            
            # Save scene detection results
            scene_file = temp_dir / "step2_scenes.json"
            with open(scene_file, 'w') as f:
                json.dump(scene_result, f, indent=2, default=str)
            logger.info(f"Step 2 results saved to: {scene_file}")
            
            # STEP 3: Content Analysis (IMPLEMENTED)
            logger.info("Starting Step 3: Content Analysis")
            content_result = self._execute_step_3_content_analysis(input_path, temp_dir, analysis_result, scene_result)
            logger.info(f"Step 3 completed: {content_result['summary']}")
            
            # Save content analysis results
            content_file = temp_dir / "step3_content.json"
            with open(content_file, 'w') as f:
                json.dump(content_result, f, indent=2, default=str)
            logger.info(f"Step 3 results saved to: {content_file}")
            
            # STEP 4: Video Cutting & Filtering (NEW IMPLEMENTATION)
            logger.info("Starting Step 4: Video Cutting & Filtering")
            cutting_result = self._execute_step_4_cutting_filtering(input_path, temp_dir, analysis_result, scene_result, content_result)
            logger.info(f"Step 4 completed: {cutting_result['summary']}")
            
            # Save cutting results
            cutting_file = temp_dir / "step4_cutting.json"
            with open(cutting_file, 'w') as f:
                json.dump(cutting_result, f, indent=2, default=str)
            logger.info(f"Step 4 results saved to: {cutting_file}")
            
            # STEP 5: Video Assembly & Post-Production (NEW IMPLEMENTATION)
            logger.info("Starting Step 5: Video Assembly & Post-Production")
            assembly_result = self._execute_step_5_assembly_postprod(input_path, output_path, temp_dir, analysis_result, scene_result, content_result, cutting_result)
            logger.info(f"Step 5 completed: {assembly_result['summary']}")
            
            # Save assembly results
            assembly_file = temp_dir / "step5_assembly.json"
            with open(assembly_file, 'w') as f:
                json.dump(assembly_result, f, indent=2, default=str)
            logger.info(f"Step 5 results saved to: {assembly_file}")
            
            # Complete processing result
            result = {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'step1_analysis': analysis_result,
                'step2_scenes': scene_result,
                'step3_content': content_result,
                'step4_cutting': cutting_result,
                'step5_assembly': assembly_result,
                'temp_dir': str(temp_dir),
                'execution_plan': execution_plan,
                'workflow_optimizations': execution_plan.get('optimization_decisions', {}),
                'current_step': 5,
                'total_steps': len(execution_plan['steps']),
                'final_output': assembly_result.get('output_file')
            }
            
            logger.info(f"Complete video processing finished successfully")
            logger.info(f"Final output: {assembly_result.get('output_file', 'Not generated')}")
            logger.info(f"Temporary files available in: {temp_dir}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video {input_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'input_path': input_path
            }
    
    def _execute_step_1_analysis(self, input_path: str, temp_dir: Path) -> Dict[str, Any]:
        """
        Execute Step 1: Video Analysis
        
        This step analyzes the video file to extract metadata, basic statistics,
        and prepare for subsequent processing steps.
        
        Args:
            input_path: Path to input video
            temp_dir: Directory for temporary files
            
        Returns:
            Dictionary with analysis results
        """
        step_start = time.time()
        
        try:
            # Get basic file information
            file_path = Path(input_path)
            file_stats = file_path.stat()
            
            # Use ffprobe to get video metadata
            ffprobe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
                str(file_path)
            ]
            
            logger.debug(f"Running ffprobe: {' '.join(ffprobe_cmd)}")
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                
                # Extract video stream info
                video_stream = None
                audio_stream = None
                
                for stream in probe_data.get('streams', []):
                    if stream.get('codec_type') == 'video' and not video_stream:
                        video_stream = stream
                    elif stream.get('codec_type') == 'audio' and not audio_stream:
                        audio_stream = stream
                
                # Build analysis result
                analysis = {
                    'timestamp': datetime.now().isoformat(),
                    'file_info': {
                        'path': str(file_path),
                        'size_bytes': file_stats.st_size,
                        'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                        'format': probe_data.get('format', {}).get('format_name', 'unknown')
                    },
                    'video_info': {
                        'codec': video_stream.get('codec_name', 'unknown') if video_stream else None,
                        'width': int(video_stream.get('width', 0)) if video_stream else 0,
                        'height': int(video_stream.get('height', 0)) if video_stream else 0,
                        'fps': eval(video_stream.get('r_frame_rate', '0/1')) if video_stream else 0,
                        'duration': float(video_stream.get('duration', 0)) if video_stream else 0,
                        'bitrate': int(video_stream.get('bit_rate', 0)) if video_stream and video_stream.get('bit_rate') else 0
                    },
                    'audio_info': {
                        'codec': audio_stream.get('codec_name', 'unknown') if audio_stream else None,
                        'channels': int(audio_stream.get('channels', 0)) if audio_stream else 0,
                        'sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream else 0,
                        'bitrate': int(audio_stream.get('bit_rate', 0)) if audio_stream and audio_stream.get('bit_rate') else 0
                    },
                    'processing_info': {
                        'analysis_duration': time.time() - step_start,
                        'ffprobe_success': True
                    },
                    'summary': f"Video analyzed: {video_stream.get('width', '?') if video_stream else '?'}x{video_stream.get('height', '?') if video_stream else '?'} @ {eval(video_stream.get('r_frame_rate', '0/1')) if video_stream else 0:.2f}fps, {float(video_stream.get('duration', 0)) if video_stream else 0:.1f}s duration"
                }
                
            else:
                # Fallback if ffprobe fails
                logger.warning(f"ffprobe failed: {result.stderr}")
                analysis = {
                    'timestamp': datetime.now().isoformat(),
                    'file_info': {
                        'path': str(file_path),
                        'size_bytes': file_stats.st_size,
                        'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                        'format': 'unknown'
                    },
                    'video_info': {'error': 'ffprobe_failed'},
                    'audio_info': {'error': 'ffprobe_failed'},
                    'processing_info': {
                        'analysis_duration': time.time() - step_start,
                        'ffprobe_success': False,
                        'ffprobe_error': result.stderr
                    },
                    'summary': f"Basic file analysis: {file_stats.st_size / (1024*1024):.1f}MB file"
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in step 1 analysis: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'processing_info': {
                    'analysis_duration': time.time() - step_start,
                    'success': False
                },
                'summary': f"Analysis failed: {str(e)}"
            }
    
    def _execute_step_2_scene_detection(self, input_path: str, temp_dir: Path, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Step 2: Scene Detection
        
        This step detects scene changes in the video using ffmpeg or other methods.
        
        Args:
            input_path: Path to input video
            temp_dir: Directory for temporary files
            analysis_result: Results from step 1 analysis
            
        Returns:
            Dictionary with scene detection results
        """
        step_start = time.time()
        
        try:
            # Get scene detection settings from config
            scene_config = self.config.get('scenes', {})
            enabled = scene_config.get('enabled', True)
            method = scene_config.get('method', 'ffmpeg')
            threshold = scene_config.get('threshold', 0.3)
            
            if not enabled:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'scenes': [],
                    'method': 'disabled',
                    'processing_info': {
                        'duration': time.time() - step_start,
                        'scene_count': 0
                    },
                    'summary': 'Scene detection disabled in config'
                }
            
            logger.info(f"Using scene detection method: {method}, threshold: {threshold}")
            
            if method == 'ffmpeg':
                return self._detect_scenes_ffmpeg(input_path, temp_dir, threshold, step_start, analysis_result)
            else:
                # Fallback to simple time-based scenes
                return self._detect_scenes_simple(input_path, analysis_result, step_start)
                
        except Exception as e:
            logger.error(f"Error in step 2 scene detection: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'processing_info': {
                    'duration': time.time() - step_start,
                    'success': False
                },
                'summary': f"Scene detection failed: {str(e)}"
            }
    
    def _detect_scenes_ffmpeg(self, input_path: str, temp_dir: Path, threshold: float, step_start: float, analysis_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use ffmpeg to detect scene changes."""
        # Create output file for scene detection
        scenes_file = temp_dir / "scenes_ffmpeg.txt"
        
        # ffmpeg command for scene detection
        ffmpeg_cmd = [
            'ffmpeg', '-i', str(input_path),
            '-vf', f'select=gt(scene\\,{threshold}),showinfo',
            '-f', 'null', '-'
        ]
        
        logger.debug(f"Running ffmpeg scene detection: {' '.join(ffmpeg_cmd)}")
        
        try:
            result = subprocess.run(
                ffmpeg_cmd, 
                capture_output=True, 
                text=True, 
                timeout=120  # 2 minute timeout
            )
            
            # Parse scene changes from stderr (ffmpeg outputs info to stderr)
            scene_times = []
            if result.stderr:
                # Look for timestamp patterns in ffmpeg output
                time_pattern = r'pts_time:(\d+\.?\d*)'
                matches = re.findall(time_pattern, result.stderr)
                scene_times = [float(t) for t in matches[:20]]  # Limit to first 20 scenes
            
            # Add beginning and end scenes
            if 0.0 not in scene_times:
                scene_times.insert(0, 0.0)
            
            # Get duration from analysis
            duration = 300.0  # Default fallback
            if analysis_result and 'video_info' in analysis_result and 'duration' in analysis_result['video_info']:
                duration = analysis_result['video_info']['duration']
            
            if scene_times and scene_times[-1] < duration - 5:
                scene_times.append(duration)
            
            # Create scene objects
            scenes = []
            for i, start_time in enumerate(scene_times[:-1]):
                end_time = scene_times[i + 1]
                scenes.append({
                    'id': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'description': f"Scene {i + 1}"
                })
            
            # Handle single scene case
            if not scenes and duration > 0:
                scenes.append({
                    'id': 1,
                    'start_time': 0.0,
                    'end_time': duration,
                    'duration': duration,
                    'description': "Single scene (no cuts detected)"
                })
            
            return {
                'timestamp': datetime.now().isoformat(),
                'method': 'ffmpeg',
                'threshold': threshold,
                'scenes': scenes,
                'processing_info': {
                    'duration': time.time() - step_start,
                    'scene_count': len(scenes),
                    'ffmpeg_success': result.returncode == 0
                },
                'summary': f"Detected {len(scenes)} scenes using ffmpeg (threshold: {threshold})"
            }
            
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg scene detection timed out, falling back to simple method")
            return self._detect_scenes_simple(input_path, {'video_info': {'duration': 300}}, step_start)
        except Exception as e:
            logger.warning(f"ffmpeg scene detection failed: {e}, falling back to simple method")
            return self._detect_scenes_simple(input_path, {'video_info': {'duration': 300}}, step_start)
    
    def _detect_scenes_simple(self, input_path: str, analysis_result: Dict[str, Any], step_start: float) -> Dict[str, Any]:
        """Simple time-based scene detection as fallback."""
        # Get duration from analysis
        duration = 300.0  # Default
        if 'video_info' in analysis_result and 'duration' in analysis_result['video_info']:
            duration = float(analysis_result['video_info']['duration'])
        
        # Create scenes every 30 seconds
        scene_length = 30.0
        scenes = []
        current_time = 0.0
        scene_id = 1
        
        while current_time < duration:
            end_time = min(current_time + scene_length, duration)
            scenes.append({
                'id': scene_id,
                'start_time': current_time,
                'end_time': end_time,
                'duration': end_time - current_time,
                'description': f"Time-based scene {scene_id}"
            })
            current_time = end_time
            scene_id += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'method': 'simple_time_based',
            'scenes': scenes,
            'processing_info': {
                'duration': time.time() - step_start,
                'scene_count': len(scenes)
            },
            'summary': f"Created {len(scenes)} time-based scenes (30s each)"
        }
    
    def _execute_step_3_content_analysis(self, input_path: str, temp_dir: Path, analysis_result: Dict[str, Any], scene_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Step 3: Content Analysis
        
        This step analyzes each scene for content criteria (faces, NSFW, etc.)
        and identifies valid clips within scenes instead of binary scene decisions.
        
        Args:
            input_path: Path to input video
            temp_dir: Directory for temporary files
            analysis_result: Results from step 1 analysis
            scene_result: Results from step 2 scene detection
            
        Returns:
            Dictionary with content analysis results including clips
        """
        step_start = time.time()
        
        try:
            # Get criteria configuration
            criteria_config = self.config.get('criteria', {})
            
            # Get scenes from step 2
            scenes = scene_result.get('scenes', [])
            if not scenes:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'error': 'No scenes available for analysis',
                    'processing_info': {
                        'duration': time.time() - step_start,
                        'success': False
                    },
                    'summary': 'Content analysis skipped - no scenes found'
                }
            
            logger.info(f"Analyzing {len(scenes)} scenes for content criteria")
            
            # Analyze each scene to identify clips
            scene_analyses = []
            frames_dir = temp_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            for scene in scenes:
                scene_analysis = self._analyze_scene_content(
                    input_path, scene, frames_dir, criteria_config, analysis_result
                )
                scene_analyses.append(scene_analysis)
                
                # Log progress
                clips_found = len(scene_analysis.get('clips', []))
                logger.debug(f"Scene {scene['id']} analysis: {clips_found} clips found")
            
            # Calculate overall statistics
            total_clips = sum(len(s.get('clips', [])) for s in scene_analyses)
            kept_scenes = [s for s in scene_analyses if s.get('clips', [])]
            rejected_scenes = [s for s in scene_analyses if not s.get('clips', [])]
            
            # Calculate total durations
            total_original_duration = sum(s.get('duration', 0) for s in scene_analyses)
            total_clip_duration = sum(
                sum(clip['duration'] for clip in s.get('clips', []))
                for s in scene_analyses
            )
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'criteria_applied': list(criteria_config.keys()),
                'scenes_analyzed': len(scene_analyses),
                'scenes_with_clips': len(kept_scenes),
                'scenes_without_clips': len(rejected_scenes),
                'total_clips_found': total_clips,
                'duration_original': total_original_duration,
                'duration_clips': total_clip_duration,
                'duration_reduction': total_original_duration - total_clip_duration,
                'scene_analyses': scene_analyses,
                'frames_extracted': len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0,
                'processing_info': {
                    'duration': time.time() - step_start,
                    'sample_rate_used': self.config.get('sample_rate', 0.1),
                    'analysis_method': 'frame_by_frame_clip_identification'
                },
                'summary': f"Analyzed {len(scenes)} scenes: found {total_clips} valid clips in {len(kept_scenes)} scenes ({total_clip_duration:.1f}s total)"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in step 3 content analysis: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'processing_info': {
                    'duration': time.time() - step_start,
                    'success': False
                },
                'summary': f"Content analysis failed: {str(e)}"
            }
    
    def _analyze_scene_content(self, input_path: str, scene: Dict[str, Any], frames_dir: Path, criteria_config: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze content of a single scene frame by frame to identify valid clips.
        
        This is the core implementation that addresses the user's requirement:
        "chaque scene doit etre découpée en clips là où les criteres sont OK"
        """
        try:
            scene_id = scene.get('id', 0)
            start_time = scene['start_time']
            end_time = scene['end_time']
            duration = end_time - start_time
            
            logger.debug(f"Analyzing scene {scene_id}: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s)")
            
            # Get sample rate for frame extraction
            sample_rate = self.config.get('sample_rate', 0.25)  # Default 0.25 fps
            frame_interval = 1.0 / sample_rate  # Time between frames
            
            # Calculate frame extraction times
            frame_times = []
            current_time = start_time + 0.5  # Start slightly after scene start
            while current_time < end_time - 0.5:  # End slightly before scene end
                frame_times.append(current_time)
                current_time += frame_interval
            
            if not frame_times:  # Very short scene
                frame_times = [start_time + duration / 2]
            
            logger.debug(f"Scene {scene_id}: extracting {len(frame_times)} frames at {sample_rate} fps")
            
            # Extract and analyze frames
            frame_analyses = []
            for i, frame_time in enumerate(frame_times):
                frame_file = frames_dir / f"scene_{scene_id:03d}_frame_{i+1:03d}.jpg"
                
                # Extract frame using ffmpeg
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-ss', str(frame_time),
                    '-i', str(input_path),
                    '-vframes', '1',
                    '-q:v', '2',  # High quality
                    str(frame_file)
                ]
                
                try:
                    result = subprocess.run(
                        ffmpeg_cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    
                    if result.returncode == 0 and frame_file.exists():
                        # Analyze this frame
                        frame_analysis = self._analyze_single_frame(str(frame_file), frame_time, criteria_config, analysis_result)
                        frame_analyses.append(frame_analysis)
                        logger.debug(f"Frame {i+1}/{len(frame_times)} at {frame_time:.1f}s: {frame_analysis['decision']}")
                    else:
                        logger.warning(f"Failed to extract frame {i+1} for scene {scene_id}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"Frame extraction timed out for scene {scene_id}, frame {i+1}")
                except Exception as e:
                    logger.warning(f"Frame extraction error for scene {scene_id}, frame {i+1}: {e}")
            
            # Identify valid clips from frame-by-frame analysis
            clips = self._identify_clips_from_frames(frame_analyses, start_time, end_time, scene_id)
            
            # Calculate scene-level statistics
            total_frames = len(frame_analyses)
            good_frames = sum(1 for f in frame_analyses if f['decision'] == 'keep')
            scene_quality_score = good_frames / total_frames if total_frames > 0 else 0
            
            return {
                'scene_id': scene_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'frames_analyzed': total_frames,
                'frames_good': good_frames,
                'quality_score': scene_quality_score,
                'clips': clips,
                'frame_analyses': frame_analyses,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing scene {scene.get('id', 0)}: {str(e)}")
            return {
                'scene_id': scene.get('id', 0),
                'error': str(e),
                'clips': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_single_frame(self, frame_path: str, timestamp: float, criteria_config: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single frame using CriteriaManager."""
        try:
            # Load frame image
            from PIL import Image
            frame_image = Image.open(frame_path)
            
            # Get video info
            video_info = analysis_result.get('video_info', {})
            
            # Create FrameData object
            frame_data = FrameData(
                frame=frame_image,
                timestamp=timestamp,
                frame_index=0,  # Not used for single frame analysis
                video_info={
                    'width': video_info.get('width', 1280),
                    'height': video_info.get('height', 720),
                    'fps': video_info.get('fps', 24),
                    'duration': video_info.get('duration', 0)
                },
                metadata={'file_path': frame_path}
            )
            
            # Analyze with CriteriaManager
            criteria_results = self.criteria_manager.analyze_frames([frame_data])
            
            # Calculate overall decision based on filtering config
            filtering_config = self.config.get('filtering', {})
            criteria_threshold = filtering_config.get('criteria_threshold', 0.2)
            rejection_strategy = filtering_config.get('rejection_strategy', 'any')
            min_criteria_count = filtering_config.get('min_criteria_count', 1)
            
            positive_criteria = sum(1 for result in criteria_results.values() if result['meets_criteria'])
            total_criteria = len(criteria_results)
            
            # Decision logic
            if total_criteria == 0:
                decision = 'keep'
            elif rejection_strategy == 'percentage':
                decision = 'keep' if positive_criteria >= total_criteria * criteria_threshold else 'reject'
            elif rejection_strategy == 'count':
                decision = 'keep' if positive_criteria >= min_criteria_count else 'reject'
            elif rejection_strategy == 'any':
                decision = 'keep' if positive_criteria > 0 else 'reject'
            elif rejection_strategy == 'all':
                decision = 'keep' if positive_criteria == total_criteria else 'reject'
            else:
                decision = 'keep' if positive_criteria >= total_criteria * criteria_threshold else 'reject'
            
            return {
                'timestamp': timestamp,
                'decision': decision,
                'criteria_results': criteria_results,
                'positive_criteria': positive_criteria,
                'total_criteria': total_criteria
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing frame {frame_path}: {e}")
            return {
                'timestamp': timestamp,
                'decision': 'reject',
                'error': str(e),
                'criteria_results': {}
            }
    
    def _identify_clips_from_frames(self, frame_analyses: List[Dict[str, Any]], scene_start: float, scene_end: float, scene_id: int) -> List[Dict[str, Any]]:
        """
        Identify continuous clips from frame-by-frame analysis.
        
        This implements the core logic for "découper en clips là où les critères sont OK"
        """
        if not frame_analyses:
            return []
        
        # Get filtering parameters
        filtering_config = self.config.get('filtering', {})
        min_clip_duration = filtering_config.get('min_clip_duration', 2.0)  # Minimum 2 seconds
        max_gap = filtering_config.get('max_gap', 1.0)  # Maximum 1 second gap to bridge
        
        clips = []
        current_clip_start = None
        last_good_frame_time = None
        
        for frame_analysis in frame_analyses:
            frame_time = frame_analysis['timestamp']
            is_good = frame_analysis['decision'] == 'keep'
            
            if is_good:
                if current_clip_start is None:
                    # Start new clip
                    current_clip_start = frame_time
                last_good_frame_time = frame_time
            else:
                # Bad frame - check if we should end current clip
                if current_clip_start is not None and last_good_frame_time is not None:
                    # Check gap size
                    gap_size = frame_time - last_good_frame_time
                    if gap_size > max_gap:
                        # Gap too large, end current clip
                        clip_duration = last_good_frame_time - current_clip_start
                        if clip_duration >= min_clip_duration:
                            clips.append({
                                'start_time': current_clip_start,
                                'end_time': last_good_frame_time,
                                'duration': clip_duration,
                                'scene_id': scene_id,
                                'clip_id': len(clips) + 1,
                                'quality': 'good'
                            })
                        current_clip_start = None
                        last_good_frame_time = None
        
        # Handle clip that extends to end of scene
        if current_clip_start is not None and last_good_frame_time is not None:
            clip_duration = last_good_frame_time - current_clip_start
            if clip_duration >= min_clip_duration:
                clips.append({
                    'start_time': current_clip_start,
                    'end_time': last_good_frame_time,
                    'duration': clip_duration,
                    'scene_id': scene_id,
                    'clip_id': len(clips) + 1,
                    'quality': 'good'
                })
        
        logger.debug(f"Scene {scene_id}: identified {len(clips)} clips from {len(frame_analyses)} frame analyses")
        return clips
    
    def _execute_step_4_cutting_filtering(self, input_path: str, temp_dir: Path, analysis_result: Dict[str, Any], scene_result: Dict[str, Any], content_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Step 4: Video Cutting & Filtering
        
        This step cuts the video into clips based on the frame-by-frame analysis results.
        """
        step_start = time.time()
        
        try:
            logger.info("Starting video cutting and filtering...")
            
            # Collect all clips from scene analyses
            scene_analyses = content_result.get('scene_analyses', [])
            all_clips = []
            
            for scene_analysis in scene_analyses:
                clips = scene_analysis.get('clips', [])
                all_clips.extend(clips)
            
            if not all_clips:
                logger.warning("No clips found to cut")
                return {
                    'timestamp': datetime.now().isoformat(),
                    'clips_cut': 0,
                    'processing_info': {
                        'duration': time.time() - step_start,
                        'success': True
                    },
                    'summary': 'No clips found to cut'
                }
            
            # Cut each clip
            cut_clips = []
            for i, clip in enumerate(all_clips):
                clip_info = self._cut_video_clip(input_path, clip, i, temp_dir)
                if clip_info:
                    cut_clips.append(clip_info)
                    logger.info(f"✅ Cut clip {i+1}/{len(all_clips)}: {clip['start_time']:.1f}s-{clip['end_time']:.1f}s ({clip['duration']:.1f}s)")
                else:
                    logger.warning(f"❌ Failed to cut clip {i+1}/{len(all_clips)}")
            
            # Calculate statistics
            total_clip_duration = sum(clip['duration'] for clip in cut_clips)
            original_duration = analysis_result.get('video_info', {}).get('duration', 0)
            reduction_percentage = ((original_duration - total_clip_duration) / original_duration * 100) if original_duration > 0 else 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'clips_cut': len(cut_clips),
                'clips_requested': len(all_clips),
                'total_clip_duration': total_clip_duration,
                'original_duration': original_duration,
                'reduction_percentage': reduction_percentage,
                'cut_clips': cut_clips,
                'processing_info': {
                    'duration': time.time() - step_start,
                    'success': True
                },
                'summary': f"Cut {len(cut_clips)}/{len(all_clips)} clips ({total_clip_duration:.1f}s total, {reduction_percentage:.1f}% reduction)"
            }
            
        except Exception as e:
            logger.error(f"Error in step 4 cutting: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'processing_info': {
                    'duration': time.time() - step_start,
                    'success': False
                },
                'summary': f"Video cutting failed: {str(e)}"
            }
    
    def _cut_video_clip(self, input_path: str, clip: Dict[str, Any], clip_index: int, temp_dir: Path) -> Optional[Dict[str, Any]]:
        """Cut a single video clip."""
        try:
            start_time = clip['start_time']
            end_time = clip['end_time']
            duration = end_time - start_time
            
            # Create output filename
            clip_filename = f"clip_{clip_index+1:03d}_{start_time:.1f}s-{end_time:.1f}s.mp4"
            clip_path = temp_dir / clip_filename
            
            # Use ffmpeg to extract clip
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # -y to overwrite
                '-i', str(input_path),
                '-ss', str(start_time),  # Start time
                '-t', str(duration),     # Duration
                '-c', 'copy',            # Copy without re-encoding for speed
                '-avoid_negative_ts', 'make_zero',
                str(clip_path)
            ]
            
            logger.debug(f"Cutting clip {clip_index+1}: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and clip_path.exists():
                file_size = clip_path.stat().st_size
                return {
                    'clip_index': clip_index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'file_path': str(clip_path),
                    'file_size': file_size,
                    'scene_id': clip.get('scene_id', 0),
                    'clip_id': clip.get('clip_id', 0)
                }
            else:
                logger.error(f"Failed to cut clip {clip_index+1}: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error cutting clip {clip_index+1}: {str(e)}")
            return None
    
    def _execute_step_5_assembly_postprod(self, input_path: str, output_path: str, temp_dir: Path, analysis_result: Dict[str, Any], scene_result: Dict[str, Any], content_result: Dict[str, Any], cutting_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Step 5: Video Assembly & Post-Production
        
        This step assembles the cut clips into a final output video.
        """
        step_start = time.time()
        
        try:
            logger.info("Starting video assembly and post-production...")
            
            cut_clips = cutting_result.get('cut_clips', [])
            
            if not cut_clips:
                logger.warning("No clips to assemble")
                return {
                    'timestamp': datetime.now().isoformat(),
                    'output_file': None,
                    'processing_info': {
                        'duration': time.time() - step_start,
                        'success': False
                    },
                    'summary': 'No clips to assemble'
                }
            
            # Create file list for ffmpeg concat
            concat_file = temp_dir / "concat_list.txt"
            with open(concat_file, 'w') as f:
                for clip in cut_clips:
                    f.write(f"file '{clip['file_path']}'\n")
            
            # Use ffmpeg to concatenate clips
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path)
            ]
            
            logger.info(f"Assembling {len(cut_clips)} clips into: {output_path}")
            logger.debug(f"Assembly command: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0 and Path(output_path).exists():
                output_size = Path(output_path).stat().st_size
                total_duration = sum(clip['duration'] for clip in cut_clips)
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'output_file': output_path,
                    'output_size': output_size,
                    'clips_assembled': len(cut_clips),
                    'total_duration': total_duration,
                    'processing_info': {
                        'duration': time.time() - step_start,
                        'success': True
                    },
                    'summary': f"Assembled {len(cut_clips)} clips into {output_path} ({total_duration:.1f}s, {output_size/(1024*1024):.1f}MB)"
                }
            else:
                logger.error(f"Assembly failed: {result.stderr}")
                return {
                    'timestamp': datetime.now().isoformat(),
                    'error': result.stderr,
                    'processing_info': {
                        'duration': time.time() - step_start,
                        'success': False
                    },
                    'summary': f"Assembly failed: {result.stderr}"
                }
                
        except Exception as e:
            logger.error(f"Error in step 5 assembly: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'processing_info': {
                    'duration': time.time() - step_start,
                    'success': False
                },
                'summary': f"Assembly failed: {str(e)}"
            }
    
    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.mp4",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple video files in batch.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for output videos
            file_pattern: Glob pattern for matching video files
            **kwargs: Additional processing parameters
            
        Returns:
            List of processing results for each file
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find video files
        video_files = list(input_path.glob(file_pattern))
        logger.info(f"Found {len(video_files)} video files to process")
        
        results = []
        for video_file in video_files:
            output_file = output_path / f"processed_{video_file.name}"
            result = self.process_video(str(video_file), str(output_file), **kwargs)
            results.append(result)
        
        return results
