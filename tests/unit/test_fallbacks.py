"""
Test fallback mechanisms for analysis criteria.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from autocut_v2.core.criteria import (
    NSFWCriterion, FaceCriterion, CriterionResult, CriterionStatus, FrameData
)


class TestFallbackChains:
    """Test fallback chain behavior for different criteria."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.frame_data = FrameData(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=1.0,
            frame_index=30,
            video_info={'fps': 30, 'duration': 10.0}
        )
    
    def test_nsfw_fallback_chain_execution(self):
        """Test NSFW criterion fallback chain."""
        config = {
            'method': 'auto',
            'fallback_chain': ['nsfw_image_detector', 'transformers', 'opennsfw2']
        }
        criterion = NSFWCriterion(config)
        
        # Mock all methods as unavailable except the last one
        with patch.object(criterion, '_is_method_available') as mock_available:
            mock_available.side_effect = lambda method: method == 'opennsfw2'
            
            with patch.object(criterion, '_opennsfw2') as mock_opennsfw2:
                mock_opennsfw2.return_value = CriterionResult(
                    status=CriterionStatus.PASS,
                    confidence=0.9,
                    details={'nsfw_score': 0.1},
                    method_used='opennsfw2'
                )
                
                result = criterion.check(self.frame_data)
                
                assert result.method_used == 'opennsfw2'
                assert result.status == CriterionStatus.PASS
                mock_opennsfw2.assert_called_once()
    
    def test_face_fallback_chain_execution(self):
        """Test face criterion fallback chain."""
        config = {
            'method': 'auto',
            'fallback_chain': ['ultralytics', 'huggingface', 'mediapipe', 'opencv'],
            'min_confidence': 0.6,
            'min_area_pct': 1.0
        }
        criterion = FaceCriterion(config)
        
        # Mock methods to simulate ultralytics failing, mediapipe succeeding
        with patch.object(criterion, '_is_method_available') as mock_available:
            mock_available.side_effect = lambda method: method in ['mediapipe', 'opencv']
            
            # Mock execution to simulate first available method working
            with patch.object(criterion, '_execute_method') as mock_execute:
                mock_execute.return_value = CriterionResult(
                    status=CriterionStatus.PASS,
                    confidence=0.8,
                    details={
                        'faces_detected': 1,
                        'valid_faces': 1,
                        'faces': [{'bbox': (100, 100, 200, 200), 'confidence': 0.8}]
                    },
                    method_used='mediapipe'
                )
                
                result = criterion.check(self.frame_data)
                
                assert result.method_used == 'mediapipe'
                assert result.status == CriterionStatus.PASS
    
    def test_all_methods_fail(self):
        """Test behavior when all fallback methods fail."""
        config = {'method': 'auto'}
        criterion = NSFWCriterion(config)
        
        # Mock all methods as unavailable
        with patch.object(criterion, '_is_method_available') as mock_available:
            mock_available.return_value = False
            
            result = criterion.check(self.frame_data)
            
            assert result.status == CriterionStatus.ERROR
            assert result.method_used == 'none'
            assert 'All methods failed' in result.details.get('error', '')
    
    def test_method_exception_handling(self):
        """Test handling of exceptions in method execution."""
        config = {'method': 'auto'}
        criterion = NSFWCriterion(config)
        
        with patch.object(criterion, '_is_method_available') as mock_available:
            mock_available.return_value = True
            
            with patch.object(criterion, '_execute_method') as mock_execute:
                mock_execute.side_effect = Exception("Method execution failed")
                
                result = criterion.check(self.frame_data)
                
                assert result.status == CriterionStatus.ERROR
                assert "Method execution failed" in str(result.error_message)
    
    def test_specific_method_selection(self):
        """Test using specific method instead of auto."""
        config = {
            'method': 'transformers',  # Specific method
            'action': 'reject'
        }
        criterion = NSFWCriterion(config)
        
        with patch.object(criterion, '_is_method_available') as mock_available:
            mock_available.return_value = True
            
            with patch.object(criterion, '_transformers_nsfw') as mock_method:
                mock_method.return_value = CriterionResult(
                    status=CriterionStatus.PASS,
                    confidence=0.9,
                    details={'nsfw_score': 0.1},
                    method_used='transformers'
                )
                
                result = criterion.check(self.frame_data)
                
                assert result.method_used == 'transformers'
                mock_method.assert_called_once()
    
    def test_precision_degradation_logging(self):
        """Test logging when falling back to lower precision methods."""
        config = {'method': 'auto'}
        criterion = FaceCriterion(config)
        
        # Simulate falling back from ultralytics to opencv (lower precision)
        with patch.object(criterion, '_is_method_available') as mock_available:
            mock_available.side_effect = lambda method: method == 'opencv'
            
            with patch.object(criterion, '_execute_method') as mock_execute:
                mock_execute.return_value = CriterionResult(
                    status=CriterionStatus.PASS,
                    confidence=0.7,  # Lower confidence
                    details={'faces_detected': 1},
                    method_used='opencv'
                )
                
                with patch.object(criterion.logger, 'warning') as mock_log:
                    result = criterion.check(self.frame_data)
                    
                    assert result.method_used == 'opencv'
                    # Should log about unavailable higher-precision methods
                    assert mock_log.call_count >= 1


if __name__ == '__main__':
    pytest.main([__file__])
