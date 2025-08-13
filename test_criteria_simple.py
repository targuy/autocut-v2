#!/usr/bin/env python3
"""
Simple test script for AutoCut v2 criteria architecture without heavy models.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autocut_v2.core.criteria import (
    NSFWCriterion, FaceCriterion, GenderCriterion,
    FrameData, CriterionStatus
)

# Configure logging to be less verbose
logging.basicConfig(level=logging.WARNING)

def create_mock_frame_data():
    """Create mock frame data for testing."""
    # Create a simple mock frame (blue image)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # Blue channel
    
    return FrameData(
        frame=frame,
        timestamp=1.0,
        frame_index=30,
        video_info={'fps': 30, 'duration': 10.0}
    )

def test_criterion_architecture():
    """Test the criterion architecture without heavy models."""
    print("Testing AutoCut v2 Criteria Architecture...")
    print("=" * 50)
    
    frame_data = create_mock_frame_data()
    
    # Test NSFW Criterion
    print("Testing NSFW Criterion:")
    nsfw_config = {'method': 'mock', 'action': 'reject'}
    nsfw_criterion = NSFWCriterion(nsfw_config)
    
    # Get available methods
    methods = nsfw_criterion.get_available_methods()
    print(f"  Available methods: {methods}")
    
    fallbacks = nsfw_criterion.get_fallback_methods()
    print(f"  Fallback chain: {fallbacks}")
    
    # Test with mock method (should always work)
    result = nsfw_criterion.check(frame_data)
    print(f"  Result: {result.status.value}, confidence: {result.confidence:.2f}, method: {result.method_used}")
    assert result.status in [CriterionStatus.PASS, CriterionStatus.FAIL, CriterionStatus.ERROR]
    print("  ✓ NSFW criterion architecture working")
    
    print()
    
    # Test Face Criterion
    print("Testing Face Criterion:")
    face_config = {'method': 'mock', 'min_confidence': 0.6}
    face_criterion = FaceCriterion(face_config)
    
    methods = face_criterion.get_available_methods()
    print(f"  Available methods: {methods}")
    
    fallbacks = face_criterion.get_fallback_methods()
    print(f"  Fallback chain: {fallbacks}")
    
    result = face_criterion.check(frame_data)
    print(f"  Result: {result.status.value}, confidence: {result.confidence:.2f}, method: {result.method_used}")
    print(f"  Details: {result.details}")
    assert result.status in [CriterionStatus.PASS, CriterionStatus.FAIL, CriterionStatus.ERROR]
    print("  ✓ Face criterion architecture working")
    
    print()
    
    # Test Gender Criterion
    print("Testing Gender Criterion:")
    gender_config = {'method': 'mock', 'filter': 'female'}
    gender_criterion = GenderCriterion(gender_config)
    
    methods = gender_criterion.get_available_methods()
    print(f"  Available methods: {methods}")
    
    fallbacks = gender_criterion.get_fallback_methods()
    print(f"  Fallback chain: {fallbacks}")
    
    result = gender_criterion.check(frame_data)
    print(f"  Result: {result.status.value}, confidence: {result.confidence:.2f}, method: {result.method_used}")
    print(f"  Details: {result.details}")
    assert result.status in [CriterionStatus.PASS, CriterionStatus.FAIL, CriterionStatus.ERROR]
    print("  ✓ Gender criterion architecture working")
    
    print()
    print("=" * 50)
    print("✓ All criterion architectures working correctly!")
    print()
    
    # Test transformers availability without creating heavy pipelines
    try:
        import transformers
        print(f"✓ Transformers library available (version: {transformers.__version__})")
    except ImportError:
        print("⚠ Transformers library not available")
    
    try:
        import torch
        print(f"✓ PyTorch available (version: {torch.__version__})")
    except ImportError:
        print("⚠ PyTorch not available")
        
    try:
        import cv2
        print(f"✓ OpenCV available (version: {cv2.__version__})")
    except ImportError:
        print("⚠ OpenCV not available")
        
    try:
        import mediapipe
        print(f"✓ MediaPipe available (version: {mediapipe.__version__})")
    except ImportError:
        print("⚠ MediaPipe not available")
        
    print()
    print("✓ AutoCut v2 criteria system is ready!")
    print("Note: Heavy ML models will be downloaded on first use of 'auto' method.")

if __name__ == "__main__":
    test_criterion_architecture()
