"""
Unit tests for criteria plugins.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from autocut_v2.core.criteria import (
    CriterionPlugin, NSFWCriterion, FaceCriterion,
    FrameData, CriterionResult, CriterionStatus
)


class TestNSFWCriterion:
    """Test NSFW detection criterion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'method': 'auto',
            'action': 'reject',
            'mode': 'high'
        }
        self.criterion = NSFWCriterion(self.config)
        
        # Mock frame data
        self.frame_data = FrameData(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=1.0,
            frame_index=30,
            video_info={'fps': 30, 'duration': 10.0}
        )
    
    def test_get_available_methods(self):
        """Test getting available NSFW detection methods."""
        methods = self.criterion.get_available_methods()
        assert isinstance(methods, list)
        # Should return empty list if no libraries installed
    
    def test_get_fallback_methods(self):
        """Test fallback method chain."""
        fallbacks = self.criterion.get_fallback_methods()
        expected = ['nsfw_image_detector', 'transformers', 'opennsfw2']
        assert fallbacks == expected
    
    def test_check_with_mock_detection(self):
        """Test NSFW check with mocked detection."""
        with patch.object(self.criterion, '_nsfw_image_detector') as mock_method:
            mock_method.return_value = CriterionResult(
                status=CriterionStatus.PASS,
                confidence=0.9,
                details={'nsfw_score': 0.1},
                method_used='nsfw_image_detector'
            )
            
            result = self.criterion.check(self.frame_data)
            
            assert result.status == CriterionStatus.PASS
            assert result.confidence == 0.9
            assert result.method_used == 'nsfw_image_detector'
    
    def test_method_availability_check(self):
        """Test method availability checking."""
        # Should return False for unavailable methods
        assert not self.criterion._is_method_available('nsfw_image_detector')
        assert not self.criterion._is_method_available('transformers')
        assert not self.criterion._is_method_available('opennsfw2')
    
    def test_nsfw_image_detector_implementation(self):
        """Test NSFW image detector implementation."""
        result = self.criterion._nsfw_image_detector(self.frame_data, 0.0)
        
        assert isinstance(result, CriterionResult)
        assert result.method_used == 'nsfw_image_detector'
        assert 'nsfw_score' in result.details
        assert result.processing_time >= 0
    
    def test_rejection_action(self):
        """Test rejection action for high NSFW score."""
        # Mock high NSFW score
        with patch.object(self.criterion, '_nsfw_image_detector') as mock_method:
            mock_method.return_value = CriterionResult(
                status=CriterionStatus.FAIL,
                confidence=0.8,
                details={'nsfw_score': 0.8},
                method_used='nsfw_image_detector'
            )
            
            result = self.criterion.check(self.frame_data)
            assert result.status == CriterionStatus.FAIL


class TestFaceCriterion:
    """Test face detection criterion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'method': 'auto',
            'min_confidence': 0.6,
            'min_area_pct': 1.0
        }
        self.criterion = FaceCriterion(self.config)
        
        self.frame_data = FrameData(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=1.0,
            frame_index=30,
            video_info={'fps': 30, 'duration': 10.0}
        )
    
    def test_get_available_methods(self):
        """Test getting available face detection methods."""
        methods = self.criterion.get_available_methods()
        assert isinstance(methods, list)
    
    def test_get_fallback_methods(self):
        """Test fallback method chain."""
        fallbacks = self.criterion.get_fallback_methods()
        expected = ['ultralytics', 'huggingface', 'mediapipe', 'opencv']
        assert fallbacks == expected
    
    def test_check_with_valid_face(self):
        """Test face check with valid face detection."""
        result = self.criterion.check(self.frame_data)
        
        assert isinstance(result, CriterionResult)
        assert result.method_used in ['ultralytics', 'huggingface', 'mediapipe', 'opencv']
        assert 'faces_detected' in result.details
        assert 'valid_faces' in result.details
    
    def test_confidence_filtering(self):
        """Test face confidence filtering."""
        # High confidence requirement
        high_conf_config = self.config.copy()
        high_conf_config['min_confidence'] = 0.95
        
        criterion = FaceCriterion(high_conf_config)
        result = criterion.check(self.frame_data)
        
        # With mocked 0.95 confidence, should pass
        # (implementation currently mocks high confidence)
        assert isinstance(result, CriterionResult)
    
    def test_area_percentage_filtering(self):
        """Test face area percentage filtering."""
        # High area requirement
        high_area_config = self.config.copy()
        high_area_config['min_area_pct'] = 10.0  # Very high requirement
        
        criterion = FaceCriterion(high_area_config)
        result = criterion.check(self.frame_data)
        
        # With high area requirement, might fail
        assert isinstance(result, CriterionResult)


class TestCriterionPlugin:
    """Test base criterion plugin functionality."""
    
    def test_abstract_methods(self):
        """Test that CriterionPlugin is abstract."""
        with pytest.raises(TypeError):
            # Should not be able to instantiate abstract class
            CriterionPlugin({})
    
    def test_try_methods_fallback(self):
        """Test fallback method chain execution."""
        # Create concrete implementation for testing
        class TestCriterion(CriterionPlugin):
            def check(self, frame_data):
                return self._try_methods(frame_data, ['method1', 'method2'])
            
            def get_available_methods(self):
                return ['method1', 'method2']
            
            def get_fallback_methods(self):
                return ['method1', 'method2']
            
            def _is_method_available(self, method):
                return True
            
            def _execute_method(self, method, frame_data):
                if method == 'method1':
                    # First method fails
                    return CriterionResult(
                        status=CriterionStatus.ERROR,
                        confidence=0.0,
                        details={'error': 'Method 1 failed'},
                        method_used=method,
                        error_message='Method 1 failed'
                    )
                else:
                    # Second method succeeds
                    return CriterionResult(
                        status=CriterionStatus.PASS,
                        confidence=0.8,
                        details={'success': True},
                        method_used=method
                    )
        
        criterion = TestCriterion({})
        frame_data = FrameData(
            frame=np.zeros((100, 100, 3)),
            timestamp=1.0,
            frame_index=1,
            video_info={}
        )
        
        result = criterion.check(frame_data)
        
        # Should succeed with method2 after method1 fails
        assert result.status == CriterionStatus.PASS
        assert result.method_used == 'method2'
        assert result.confidence == 0.8


if __name__ == '__main__':
    pytest.main([__file__])
