"""
Intelligent adaptive workflow for video processing.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStep(Enum):
    """Available workflow steps."""
    PRE_ANALYSIS = "pre_analysis"
    NORMALIZATION = "normalization"
    SCENE_DETECTION = "scene_detection"
    CONTENT_ANALYSIS = "content_analysis"
    CUTTING_ENRICHMENT = "cutting_enrichment"
    POST_PRODUCTION = "post_production"


@dataclass
class StepMetrics:
    """Metrics for a workflow step."""
    step: WorkflowStep
    duration: float
    cost_estimate: float
    success: bool
    details: Dict[str, Any]


@dataclass
class VideoProfile:
    """Video characteristics for optimization decisions."""
    resolution: Tuple[int, int]
    fps: float
    duration: float
    format: str
    codec: str
    file_size: int
    estimated_complexity: float


class WorkflowOptimizer:
    """
    Optimizes workflow execution order based on content and resources.
    
    Implements the adaptive workflow logic from specifications:
    - Determines optimal order of steps
    - Skips unnecessary normalization
    - Estimates processing costs
    - Makes intelligent decisions based on heuristics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize workflow optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.auto_optimize = config.get('workflow', {}).get('auto_optimize', True)
        self.skip_normalize_if_conform = config.get('workflow', {}).get('skip_normalize_if_conform', True)
        self.force_normalize = config.get('workflow', {}).get('force_normalize', False)
        
        # Target format settings
        self.target_width = config.get('normalize', {}).get('target_width', 1280)
        self.target_height = config.get('normalize', {}).get('target_height', 720)
        self.target_fps = config.get('normalize', {}).get('target_fps', 24)
    
    def analyze_video_profile(self, video_path: str) -> VideoProfile:
        """
        Analyze video characteristics for optimization decisions.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoProfile with video characteristics
        """
        try:
            # In real implementation, use ffprobe or moviepy to get video info
            # For now, return mock profile
            file_size = Path(video_path).stat().st_size
            
            return VideoProfile(
                resolution=(1920, 1080),  # Mock values
                fps=30.0,
                duration=300.0,  # 5 minutes
                format="mp4",
                codec="h264",
                file_size=file_size,
                estimated_complexity=0.7  # 0.0 = simple, 1.0 = complex
            )
            
        except Exception as e:
            logger.error(f"Error analyzing video profile: {str(e)}")
            # Return safe defaults
            return VideoProfile(
                resolution=(1280, 720),
                fps=24.0,
                duration=60.0,
                format="unknown",
                codec="unknown",
                file_size=0,
                estimated_complexity=0.5
            )
    
    def should_normalize(self, profile: VideoProfile) -> Tuple[bool, str]:
        """
        Determine if video should be normalized.
        
        Args:
            profile: Video profile
            
        Returns:
            Tuple of (should_normalize, reason)
        """
        if self.force_normalize:
            return True, "Force normalize enabled"
        
        if not self.skip_normalize_if_conform:
            return True, "Skip normalize disabled"
        
        # Check if video already conforms to target format
        width, height = profile.resolution
        
        format_conforms = (
            width <= self.target_width and
            height <= self.target_height and
            abs(profile.fps - self.target_fps) <= 1.0
        )
        
        if format_conforms:
            return False, "Video already conforms to target format"
        
        # Estimate normalization cost vs analysis cost
        normalization_cost = self._estimate_normalization_cost(profile)
        analysis_cost = self._estimate_analysis_cost(profile)
        
        # If normalization takes more than 2x analysis time, skip it
        if normalization_cost > 2 * analysis_cost:
            return False, f"Normalization too expensive ({normalization_cost:.1f}s vs {analysis_cost:.1f}s)"
        
        return True, f"Normalization beneficial ({normalization_cost:.1f}s vs {analysis_cost:.1f}s)"
    
    def optimize_workflow_order(self, profile: VideoProfile) -> List[WorkflowStep]:
        """
        Determine optimal order of workflow steps.
        
        Args:
            profile: Video profile
            
        Returns:
            Ordered list of workflow steps
        """
        if not self.auto_optimize:
            # Default order
            return [
                WorkflowStep.PRE_ANALYSIS,
                WorkflowStep.NORMALIZATION,
                WorkflowStep.SCENE_DETECTION,
                WorkflowStep.CONTENT_ANALYSIS,
                WorkflowStep.CUTTING_ENRICHMENT,
                WorkflowStep.POST_PRODUCTION
            ]
        
        steps = []
        
        # Always start with pre-analysis
        steps.append(WorkflowStep.PRE_ANALYSIS)
        
        # Determine normalization necessity
        should_norm, reason = self.should_normalize(profile)
        logger.info(f"Normalization decision: {should_norm} - {reason}")
        
        # Estimate rejection probability
        rejection_prob = self._estimate_rejection_probability(profile)
        
        # If high rejection probability, analyze first to fail fast
        if rejection_prob > 0.7 and should_norm:
            logger.info(f"High rejection probability ({rejection_prob:.2f}), analyzing before normalization")
            steps.extend([
                WorkflowStep.SCENE_DETECTION,
                WorkflowStep.CONTENT_ANALYSIS,
                WorkflowStep.NORMALIZATION,
                WorkflowStep.CUTTING_ENRICHMENT,
                WorkflowStep.POST_PRODUCTION
            ])
        else:
            # Standard order
            if should_norm:
                steps.append(WorkflowStep.NORMALIZATION)
            
            steps.extend([
                WorkflowStep.SCENE_DETECTION,
                WorkflowStep.CONTENT_ANALYSIS,
                WorkflowStep.CUTTING_ENRICHMENT,
                WorkflowStep.POST_PRODUCTION
            ])
        
        # Remove disabled steps
        if not self.config.get('scenes', {}).get('enabled', True):
            steps = [s for s in steps if s != WorkflowStep.SCENE_DETECTION]
        
        return steps
    
    def _estimate_normalization_cost(self, profile: VideoProfile) -> float:
        """
        Estimate time cost of video normalization.
        
        Args:
            profile: Video profile
            
        Returns:
            Estimated time in seconds
        """
        # Simple heuristic based on resolution, duration, and complexity
        width, height = profile.resolution
        pixel_count = width * height
        
        # Base cost: ~1x realtime for 1080p video
        base_cost = profile.duration * (pixel_count / (1920 * 1080))
        
        # Adjust for complexity and codec
        complexity_factor = 1.0 + profile.estimated_complexity
        
        return base_cost * complexity_factor
    
    def _estimate_analysis_cost(self, profile: VideoProfile) -> float:
        """
        Estimate time cost of content analysis.
        
        Args:
            profile: Video profile
            
        Returns:
            Estimated time in seconds
        """
        # Analysis cost depends on sampling rate and enabled criteria
        sample_rate = self.config.get('sample_rate', 0.1)  # frames per second
        total_frames = profile.duration * sample_rate
        
        # Base analysis time per frame (depends on criteria enabled)
        time_per_frame = 0.1  # seconds
        
        # Adjust for enabled criteria
        criteria_count = 0
        if self.config.get('criteria', {}).get('nsfw', {}).get('enabled', True):
            criteria_count += 1
        if self.config.get('criteria', {}).get('face', {}).get('enabled', True):
            criteria_count += 1
        if self.config.get('criteria', {}).get('gender', {}).get('enabled', True):
            criteria_count += 1
        if self.config.get('criteria', {}).get('pose', {}).get('enabled', True):
            criteria_count += 1
        
        return total_frames * time_per_frame * max(criteria_count, 1)
    
    def _estimate_rejection_probability(self, profile: VideoProfile) -> float:
        """
        Estimate probability of content rejection based on heuristics.
        
        Args:
            profile: Video profile
            
        Returns:
            Probability of rejection (0.0 to 1.0)
        """
        # This would use historical data in a real implementation
        # For now, use simple heuristics
        
        base_prob = 0.2  # 20% base rejection rate
        
        # Adjust based on video characteristics
        if profile.estimated_complexity > 0.8:
            base_prob += 0.2  # Complex videos more likely to have issues
        
        if profile.duration > 600:  # 10+ minutes
            base_prob += 0.1  # Longer videos more likely to have issues
        
        return min(base_prob, 0.9)  # Cap at 90%
    
    def create_execution_plan(self, video_path: str) -> Dict[str, Any]:
        """
        Create complete execution plan for video processing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Execution plan dictionary
        """
        profile = self.analyze_video_profile(video_path)
        steps = self.optimize_workflow_order(profile)
        
        plan = {
            'video_path': video_path,
            'profile': profile,
            'steps': steps,
            'estimated_total_time': sum([
                self._estimate_step_cost(step, profile) for step in steps
            ]),
            'optimization_decisions': {
                'auto_optimize': self.auto_optimize,
                'normalize_decision': self.should_normalize(profile),
                'rejection_probability': self._estimate_rejection_probability(profile)
            }
        }
        
        return plan
    
    def _estimate_step_cost(self, step: WorkflowStep, profile: VideoProfile) -> float:
        """Estimate cost for individual workflow step."""
        if step == WorkflowStep.NORMALIZATION:
            return self._estimate_normalization_cost(profile)
        elif step == WorkflowStep.CONTENT_ANALYSIS:
            return self._estimate_analysis_cost(profile)
        elif step == WorkflowStep.SCENE_DETECTION:
            return profile.duration * 0.1  # Fast scene detection
        else:
            return profile.duration * 0.05  # Other steps are relatively fast
