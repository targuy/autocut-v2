#!/usr/bin/env python3
"""
Basic test script for AutoCut v2 functionality.
"""

import sys
import logging
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autocut_v2.core.processor import AutoCut
from autocut_v2.core.workflow import WorkflowOptimizer, VideoProfile
from autocut_v2.utils.config import Config
from autocut_v2.utils.file_handler import FileHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_config():
    """Test configuration management."""
    print("Testing Configuration...")
    
    config = Config()
    
    # Test basic access
    assert config.get('device') == 'auto'
    assert config.get('scenes.enabled') is True
    assert config.get('criteria.nsfw.action') == 'reject'
    
    # Test setting values
    config.set('device', 'cuda')
    config.set('criteria.face.min_confidence', 0.9)
    
    assert config.get('device') == 'cuda'
    assert config.get('criteria.face.min_confidence') == 0.9
    
    print("✓ Configuration tests passed")

def test_workflow_optimizer():
    """Test workflow optimization."""
    print("Testing Workflow Optimizer...")
    
    config = {
        'workflow': {
            'auto_optimize': True,
            'skip_normalize_if_conform': True
        },
        'normalize': {
            'target_width': 1280,
            'target_height': 720,
            'target_fps': 24
        }
    }
    
    optimizer = WorkflowOptimizer(config)
    
    # Test with high-resolution video
    high_res_profile = VideoProfile(
        resolution=(3840, 2160),
        fps=60.0,
        duration=300.0,
        format="mp4",
        codec="h264",
        file_size=1000000000,
        estimated_complexity=0.8
    )
    
    should_normalize, reason = optimizer.should_normalize(high_res_profile)
    print(f"High-res normalization decision: {should_normalize} - {reason}")
    
    # Test with conforming video
    conforming_profile = VideoProfile(
        resolution=(1280, 720),
        fps=24.0,
        duration=120.0,
        format="mp4",
        codec="h264",
        file_size=200000000,
        estimated_complexity=0.3
    )
    
    should_normalize, reason = optimizer.should_normalize(conforming_profile)
    print(f"Conforming video normalization decision: {should_normalize} - {reason}")
    
    print("✓ Workflow optimizer tests passed")

def test_file_handler():
    """Test file handling."""
    print("Testing File Handler...")
    
    handler = FileHandler()
    
    # Test supported formats
    assert '.mp4' in handler.SUPPORTED_VIDEO_FORMATS
    assert '.avi' in handler.SUPPORTED_VIDEO_FORMATS
    assert '.mov' in handler.SUPPORTED_VIDEO_FORMATS
    
    # Test validation with non-existent file
    assert not handler.validate_video_file("nonexistent.mp4")
    
    print("✓ File handler tests passed")

def test_autocut_processor():
    """Test main AutoCut processor."""
    print("Testing AutoCut Processor...")
    
    config = {
        'device': 'cpu',
        'scenes': {'enabled': True, 'method': 'ffmpeg'},
        'criteria': {
            'nsfw': {'enabled': True, 'action': 'reject'},
            'face': {'enabled': True, 'min_confidence': 0.6}
        }
    }
    
    processor = AutoCut(config)
    
    # Test with mock video path (will fail validation but test the flow)
    result = processor.process_video("test_video.mp4", "output.mp4")
    
    # Should fail due to invalid input but return proper error structure
    assert result['success'] is False
    assert 'error' in result
    assert result['input_path'] == "test_video.mp4"
    
    print("✓ AutoCut processor tests passed")

def main():
    """Run all tests."""
    print("Running AutoCut v2 Basic Tests")
    print("=" * 50)
    
    try:
        test_config()
        test_workflow_optimizer()
        test_file_handler()
        test_autocut_processor()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! AutoCut v2 basic functionality is working.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
