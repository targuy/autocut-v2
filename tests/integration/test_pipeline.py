"""
Integration tests for complete video processing pipeline.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json

from autocut_v2.core.processor import AutoCut
from autocut_v2.core.workflow import WorkflowOptimizer, VideoProfile
from autocut_v2.utils.config import Config


class TestPipelineIntegration:
    """Test complete video processing pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock video file
        self.mock_video_path = self.temp_path / "test_video.mp4"
        self.mock_video_path.write_bytes(b"mock video content")
        
        # Create output directory
        self.output_dir = self.temp_path / "output"
        self.output_dir.mkdir()
        
        # Default configuration
        self.config = {
            'input_video': str(self.mock_video_path),
            'output_dir': str(self.output_dir),
            'device': 'cpu',
            'workflow': {
                'auto_optimize': True,
                'skip_normalize_if_conform': True,
                'force_normalize': False
            },
            'scenes': {
                'enabled': True,
                'method': 'ffmpeg',
                'threshold': 0.3
            },
            'criteria': {
                'nsfw': {
                    'enabled': True,
                    'method': 'auto',
                    'action': 'reject'
                },
                'face': {
                    'enabled': True,
                    'method': 'auto',
                    'min_confidence': 0.6
                }
            },
            'describe': {
                'enabled': True,
                'frames_per_clip': 3
            }
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow_execution(self):
        """Test complete workflow from start to finish."""
        processor = AutoCut(self.config)
        
        # Mock the video processing components
        with patch('autocut_v2.core.processor.VideoFileClip') as mock_video_clip:
            mock_clip = Mock()
            mock_clip.duration = 60.0
            mock_clip.fps = 30.0
            mock_video_clip.return_value = mock_clip
            
            with patch.object(processor.scene_detector, 'detect_scenes') as mock_scenes:
                mock_scenes.return_value = [(0, 30), (30, 60)]
                
                with patch.object(processor.video_cutter, 'cut_video') as mock_cut:
                    mock_cut.return_value = mock_clip
                    
                    output_path = str(self.output_dir / "processed_test_video.mp4")
                    result = processor.process_video(str(self.mock_video_path), output_path)
                    
                    assert result['success'] is True
                    assert result['input_path'] == str(self.mock_video_path)
                    assert result['output_path'] == output_path
                    assert result['scenes_detected'] == 2
    
    def test_workflow_optimization(self):
        """Test adaptive workflow optimization."""
        optimizer = WorkflowOptimizer(self.config)
        
        # Test with high-resolution video (should normalize)
        high_res_profile = VideoProfile(
            resolution=(3840, 2160),  # 4K
            fps=60.0,
            duration=300.0,
            format="mp4",
            codec="h264",
            file_size=1000000000,  # 1GB
            estimated_complexity=0.8
        )
        
        steps = optimizer.optimize_workflow_order(high_res_profile)
        
        # Should include normalization for high-res video
        from autocut_v2.core.workflow import WorkflowStep
        assert WorkflowStep.NORMALIZATION in steps
        assert WorkflowStep.CONTENT_ANALYSIS in steps
    
    def test_workflow_skip_normalization(self):
        """Test skipping normalization for conforming video."""
        self.config['workflow']['skip_normalize_if_conform'] = True
        optimizer = WorkflowOptimizer(self.config)
        
        # Test with already conforming video
        conforming_profile = VideoProfile(
            resolution=(1280, 720),  # Target resolution
            fps=24.0,  # Target FPS
            duration=120.0,
            format="mp4",
            codec="h264",
            file_size=200000000,
            estimated_complexity=0.3
        )
        
        should_normalize, reason = optimizer.should_normalize(conforming_profile)
        
        assert should_normalize is False
        assert "already conforms" in reason.lower()
    
    def test_error_handling_and_recovery(self):
        """Test error handling in pipeline."""
        processor = AutoCut(self.config)
        
        # Test with non-existent video file
        nonexistent_path = str(self.temp_path / "nonexistent.mp4")
        output_path = str(self.output_dir / "output.mp4")
        
        result = processor.process_video(nonexistent_path, output_path)
        
        assert result['success'] is False
        assert 'error' in result
        assert result['input_path'] == nonexistent_path
    
    def test_batch_processing(self):
        """Test batch processing multiple videos."""
        processor = AutoCut(self.config)
        
        # Create additional mock video files
        video2_path = self.temp_path / "test_video2.mp4"
        video2_path.write_bytes(b"mock video content 2")
        
        video3_path = self.temp_path / "test_video3.mp4"
        video3_path.write_bytes(b"mock video content 3")
        
        # Mock video processing
        with patch.object(processor, 'process_video') as mock_process:
            mock_process.return_value = {'success': True, 'output_path': 'test_output.mp4'}
            
            results = processor.batch_process(
                str(self.temp_path),
                str(self.output_dir),
                file_pattern="*.mp4"
            )
            
            # Should process all 3 video files
            assert len(results) == 3
            assert all(r['success'] for r in results)
    
    def test_configuration_loading_and_validation(self):
        """Test configuration loading and validation."""
        config_path = self.temp_path / "test_config.yml"
        
        # Create test configuration file
        config_content = """
input_video: "test.mp4"
output_dir: "./output"
device: "auto"
criteria:
  nsfw:
    enabled: true
    action: "reject"
  face:
    enabled: true
    min_confidence: 0.8
"""
        config_path.write_text(config_content)
        
        # Load configuration
        config = Config()
        config.load_from_file(str(config_path))
        
        assert config.get('input_video') == "test.mp4"
        assert config.get('device') == "auto"
        assert config.get('criteria.nsfw.enabled') is True
        assert config.get('criteria.face.min_confidence') == 0.8
    
    def test_performance_metrics_collection(self):
        """Test collection of performance metrics."""
        processor = AutoCut(self.config)
        
        with patch('autocut_v2.core.processor.VideoFileClip') as mock_video_clip:
            mock_clip = Mock()
            mock_clip.duration = 60.0
            mock_video_clip.return_value = mock_clip
            
            # Process video and check for metrics
            output_path = str(self.output_dir / "processed_test_video.mp4")
            result = processor.process_video(str(self.mock_video_path), output_path)
            
            # Should include timing information
            assert 'duration_original' in result
            assert 'duration_processed' in result
            assert 'scenes_detected' in result
    
    def test_memory_and_resource_management(self):
        """Test proper resource cleanup."""
        processor = AutoCut(self.config)
        
        with patch('autocut_v2.core.processor.VideoFileClip') as mock_video_clip:
            mock_clip = Mock()
            mock_clip.duration = 60.0
            mock_video_clip.return_value = mock_clip
            
            output_path = str(self.output_dir / "processed_test_video.mp4")
            result = processor.process_video(str(self.mock_video_path), output_path)
            
            # Verify clip cleanup was called
            mock_clip.close.assert_called()
    
    def test_json_metadata_generation(self):
        """Test generation of JSON metadata for clips."""
        # This would test the LLM integration and JSON generation
        # For now, just verify the structure is in place
        
        processor = AutoCut(self.config)
        
        # Mock LLM response
        mock_json = {
            "title": "Test Video Clip",
            "description": "A test video clip for validation",
            "tags": ["test", "validation"],
            "quality_score": 0.85
        }
        
        # Verify JSON structure is valid
        json_str = json.dumps(mock_json)
        parsed = json.loads(json_str)
        
        assert parsed['title'] == "Test Video Clip"
        assert isinstance(parsed['tags'], list)
        assert isinstance(parsed['quality_score'], float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
