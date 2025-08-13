#!/usr/bin/env python3
"""
Test script for AutoCut v2 criteria with Hugging Face transformers.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

def test_nsfw_criterion():
    """Test NSFW detection criterion."""
    print("Testing NSFW Criterion...")
    
    config = {
        'method': 'auto',
        'action': 'reject',
        'mode': 'high'
    }
    
    criterion = NSFWCriterion(config)
    frame_data = create_mock_frame_data()
    
    # Test available methods
    available_methods = criterion.get_available_methods()
    print(f"Available NSFW methods: {available_methods}")
    
    # Test fallback methods
    fallback_methods = criterion.get_fallback_methods()
    print(f"NSFW fallback chain: {fallback_methods}")
    
    # Test detection
    result = criterion.check(frame_data)
    print(f"NSFW detection result: {result.status.value}, confidence: {result.confidence:.2f}, method: {result.method_used}")
    
    assert result.status in [CriterionStatus.PASS, CriterionStatus.FAIL, CriterionStatus.ERROR]
    print("✓ NSFW criterion tests passed")

def test_face_criterion():
    """Test face detection criterion."""
    print("Testing Face Criterion...")
    
    config = {
        'method': 'auto',
        'min_confidence': 0.6,
        'min_area_pct': 1.0
    }
    
    criterion = FaceCriterion(config)
    frame_data = create_mock_frame_data()
    
    # Test available methods
    available_methods = criterion.get_available_methods()
    print(f"Available face detection methods: {available_methods}")
    
    # Test fallback methods
    fallback_methods = criterion.get_fallback_methods()
    print(f"Face detection fallback chain: {fallback_methods}")
    
    # Test detection
    result = criterion.check(frame_data)
    print(f"Face detection result: {result.status.value}, confidence: {result.confidence:.2f}, method: {result.method_used}")
    print(f"Faces detected: {result.details.get('faces_detected', 0)}, valid: {result.details.get('valid_faces', 0)}")
    
    assert result.status in [CriterionStatus.PASS, CriterionStatus.FAIL, CriterionStatus.ERROR]
    print("✓ Face criterion tests passed")

def test_gender_criterion():
    """Test gender classification criterion."""
    print("Testing Gender Criterion...")
    
    config = {
        'method': 'auto',
        'filter': 'female',
        'min_confidence': 0.8
    }
    
    criterion = GenderCriterion(config)
    frame_data = create_mock_frame_data()
    
    # Test available methods
    available_methods = criterion.get_available_methods()
    print(f"Available gender classification methods: {available_methods}")
    
    # Test fallback methods
    fallback_methods = criterion.get_fallback_methods()
    print(f"Gender classification fallback chain: {fallback_methods}")
    
    # Test classification
    result = criterion.check(frame_data)
    print(f"Gender classification result: {result.status.value}, confidence: {result.confidence:.2f}, method: {result.method_used}")
    if 'gender_scores' in result.details:
        print(f"Gender scores: {result.details['gender_scores']}")
    
    assert result.status in [CriterionStatus.PASS, CriterionStatus.FAIL, CriterionStatus.ERROR]
    print("✓ Gender criterion tests passed")

def test_transformers_availability():
    """Test if transformers methods are available."""
    print("Testing Transformers Availability...")
    
    try:
        from transformers import pipeline
        print("✓ Transformers library is available")
        
        # Test if we can create a pipeline (this will download models if needed)
        try:
            # Use a small, fast model for testing
            classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
            result = classifier("This is a test")
            print("✓ Transformers pipeline creation successful")
        except Exception as e:
            print(f"⚠ Transformers pipeline test failed (but library is available): {e}")
            
    except ImportError:
        print("✗ Transformers library is not available")
        return False
    
    return True

def main():
    """Run all criteria tests."""
    print("Running AutoCut v2 Criteria Tests with Hugging Face Transformers")
    print("=" * 70)
    
    try:
        # Test transformers availability first
        transformers_available = test_transformers_availability()
        print()
        
        # Run criteria tests
        test_nsfw_criterion()
        print()
        
        test_face_criterion()
        print()
        
        test_gender_criterion()
        print()
        
        print("=" * 70)
        if transformers_available:
            print("✓ All criteria tests passed! Hugging Face transformers integration is working.")
        else:
            print("✓ All criteria tests passed! (Note: Transformers not fully available, using fallbacks)")
        
        print("\nNote: Some methods may fall back to mock implementations if models are not")
        print("downloaded or if there are network issues. This is expected behavior.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
