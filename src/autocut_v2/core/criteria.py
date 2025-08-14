"""
Plugin architecture for video analysis criteria.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
import os
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Import device utilities
from ..utils.config import get_device_for_ml, get_device_id_for_transformers

logger = logging.getLogger(__name__)


class CriterionStatus(Enum):
    """Status of criterion evaluation."""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class FrameData:
    """Data structure for frame analysis."""
    frame: Any  # numpy array or PIL Image
    timestamp: float
    frame_index: int
    video_info: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CriterionResult:
    """Result of criterion evaluation."""
    status: CriterionStatus
    confidence: float
    details: Dict[str, Any]
    method_used: str
    processing_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class FaceDetection:
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    area_percentage: float = 0.0


class CriterionPlugin(ABC):
    """
    Abstract base class for all analysis criteria plugins.
    
    Each criterion implements multiple methods with fallback chains
    for robustness and adaptability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize criterion plugin.
        
        Args:
            config: Configuration dictionary for this criterion
        """
        self.config = config
        self.method = config.get('method', 'auto')
        self.fallback_chain = config.get('fallback_chain', [])
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def check(self, frame_data: FrameData) -> CriterionResult:
        """
        Check if frame meets this criterion.
        
        Args:
            frame_data: Frame data to analyze
            
        Returns:
            CriterionResult with evaluation details
        """
        pass
    
    @abstractmethod
    def get_available_methods(self) -> List[str]:
        """
        Get list of available implementation methods.
        
        Returns:
            List of method names available for this criterion
        """
        pass
    
    @abstractmethod
    def get_fallback_methods(self) -> List[str]:
        """
        Get ordered list of fallback methods.
        
        Returns:
            List of method names in fallback order
        """
        pass
    
    def _try_methods(self, frame_data: FrameData, methods: List[str]) -> CriterionResult:
        """
        Try multiple methods with fallback chain.
        
        Args:
            frame_data: Frame data to analyze
            methods: List of methods to try in order
            
        Returns:
            CriterionResult from first successful method
        """
        last_error = None
        
        for method in methods:
            try:
                if self._is_method_available(method):
                    self.logger.debug(f"Trying method: {method}")
                    result = self._execute_method(method, frame_data)
                    
                    if result.status != CriterionStatus.ERROR:
                        return result
                    else:
                        last_error = result.error_message
                        
                else:
                    self.logger.warning(f"Method {method} not available")
                    
            except Exception as e:
                self.logger.warning(f"Method {method} failed: {str(e)}")
                last_error = str(e)
        
        # All methods failed
        return CriterionResult(
            status=CriterionStatus.ERROR,
            confidence=0.0,
            details={'error': 'All methods failed'},
            method_used='none',
            error_message=last_error
        )
    
    @abstractmethod
    def _is_method_available(self, method: str) -> bool:
        """Check if method is available/installed."""
        pass
    
    @abstractmethod
    def _execute_method(self, method: str, frame_data: FrameData) -> CriterionResult:
        """Execute specific method implementation."""
        pass


class NSFWCriterion(CriterionPlugin):
    """NSFW content detection criterion."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._transformers_pipeline = None
        self._nsfw_detector_model = None
        self._opennsfw2_model = None
    
    def get_available_methods(self) -> List[str]:
        """Get list of available NSFW detection methods."""
        methods = ['mock']  # Mock is always available
        
        # Check if specific libraries are available
        try:
            import nsfw_detector  # noqa: F401
            methods.append('nsfw_image_detector')
        except ImportError:
            pass
        
        try:
            import transformers  # noqa: F401
            methods.append('transformers')
        except ImportError:
            pass
        
        try:
            import opennsfw2  # noqa: F401
            methods.append('opennsfw2')
        except ImportError:
            pass
            
        return methods
    
    def get_fallback_methods(self) -> List[str]:
        """Get NSFW detection fallback chain."""
        # tests expect opennsfw2 to be included
        return ['nsfw_image_detector', 'transformers', 'opennsfw2']
    
    def check(self, frame_data: FrameData) -> CriterionResult:
        """Check NSFW content in frame."""
        methods = self.get_fallback_methods() if self.method == 'auto' else [self.method]
        return self._try_methods(frame_data, methods)
    
    def _is_method_available(self, method: str) -> bool:
        """Check if NSFW method is available."""
        if method == 'mock':
            return True
        elif method == 'nsfw_image_detector':
            try:
                import nsfw_detector
                return True
            except ImportError:
                return False
        elif method == 'transformers':
            try:
                from transformers import pipeline
                return True
            except ImportError:
                return False
        elif method == 'opennsfw2':
            try:
                import opennsfw2  # noqa: F401
                return True
            except ImportError:
                return False
        return False
    
    def _execute_method(self, method: str, frame_data: FrameData) -> CriterionResult:
        """Execute NSFW detection method."""
        start_time = time.time()
        
        try:
            if method == 'mock':
                return self._mock_nsfw(frame_data, start_time)
            elif method == 'nsfw_image_detector':
                return self._nsfw_image_detector(frame_data, start_time)
            elif method == 'transformers':
                return self._transformers_nsfw(frame_data, start_time)
            elif method == 'opennsfw2':
                return self._opennsfw2_nsfw(frame_data, start_time)
            else:
                return CriterionResult(
                    status=CriterionStatus.ERROR,
                    confidence=0.0,
                    details={'error': f'Unknown method: {method}'},
                    method_used=method,
                    processing_time=time.time() - start_time,
                    error_message=f'Unknown method: {method}'
                )
        except Exception as e:
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': str(e)},
                method_used=method,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _mock_nsfw(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Fallback NSFW detection when no models are available."""
        logger.warning("Using fallback NSFW detection - no models available")
        
        # Conservative approach: assume content might be NSFW when we can't analyze
        action = self.config.get('action', 'reject')
        threshold = self.config.get('threshold', 0.5)
        
        if action == 'reject':
            # When rejecting NSFW, be conservative and fail when uncertain
            status = CriterionStatus.FAIL
            confidence = 0.7  # Conservative confidence
            nsfw_score = 0.8  # Assume potentially NSFW
        else:  # 'allow'
            # When allowing NSFW, be permissive 
            status = CriterionStatus.PASS
            confidence = 0.3  # Low confidence but passing
            nsfw_score = 0.2  # Assume probably safe
        
        return CriterionResult(
            status=status,
            confidence=confidence,
            details={
                'nsfw_score': nsfw_score, 
                'threshold': threshold, 
                'fallback': True,
                'reason': 'No NSFW detection models available'
            },
            method_used='fallback',
            processing_time=time.time() - start_time,
            error_message="NSFW detection unavailable - using conservative fallback"
        )
    
    def _nsfw_image_detector(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """NSFW detection using nsfw_image_detector library."""
        try:
            import nsfw_detector
            from PIL import Image
            import numpy as np
            
            # Check if frame data is valid
            if frame_data.frame is None:
                raise ValueError("Frame data is None - no image to analyze")
            
            # Initialize detector once and cache it
            if self._nsfw_detector_model is None:
                # Get device configuration
                config_device = self.config.get('device', 'auto')
                device = get_device_for_ml(config_device)
                
                self.logger.info(f"NSFW Detector Config device: {config_device}, Detected device: {device}")
                
                # Initialize NSFW detector with device preference
                try:
                    if device == "cuda":
                        self._nsfw_detector_model = nsfw_detector.predict.load_model()
                        self.logger.info("NSFW detector loaded successfully")
                    else:
                        self._nsfw_detector_model = nsfw_detector.predict.load_model()
                        self.logger.info("NSFW detector loaded on CPU")
                except Exception as e:
                    self.logger.warning(f"Failed to load NSFW detector: {e}")
                    raise
            
            # Convert frame to the format expected by nsfw_detector
            if isinstance(frame_data.frame, np.ndarray):
                # Convert BGR to RGB if needed
                if len(frame_data.frame.shape) == 3 and frame_data.frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame_data.frame
                image = Image.fromarray(rgb_frame)
            elif isinstance(frame_data.frame, Image.Image):
                image = frame_data.frame
            else:
                raise ValueError(f"Unsupported frame type: {type(frame_data.frame)}")
            
            # Save image temporarily for nsfw_detector (it expects file path)
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                temp_path = tmp_file.name
                image.save(temp_path, 'JPEG')
            
            try:
                # Run NSFW detection
                results = nsfw_detector.predict.classify(self._nsfw_detector_model, temp_path)
                
                # Extract NSFW probability
                # Results format: {image_path: {'safe': prob, 'unsafe': prob}}
                if temp_path in results:
                    prediction = results[temp_path]
                    nsfw_score = prediction.get('unsafe', 0.0)
                    safe_score = prediction.get('safe', 1.0)
                else:
                    raise ValueError("No prediction results returned")
                
                # Clean up temporary file
                os.unlink(temp_path)
                
            except Exception as e:
                # Clean up temporary file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
            
            # Determine if criteria is met
            action = self.config.get('action', 'reject')
            threshold = self.config.get('threshold', 0.5)
            
            if action == 'reject':
                # Reject NSFW content: PASS if safe, FAIL if NSFW
                status = CriterionStatus.FAIL if nsfw_score > threshold else CriterionStatus.PASS
                confidence = safe_score if status == CriterionStatus.PASS else nsfw_score
            else:  # 'allow'
                # Allow NSFW content: PASS if NSFW, FAIL if safe
                status = CriterionStatus.PASS if nsfw_score > threshold else CriterionStatus.FAIL
                confidence = nsfw_score if status == CriterionStatus.PASS else safe_score
            
            return CriterionResult(
                status=status,
                confidence=confidence,
                details={
                    'nsfw_score': nsfw_score,
                    'safe_score': safe_score,
                    'threshold': threshold,
                    'action': action,
                    'raw_results': prediction
                },
                method_used='nsfw_image_detector',
                processing_time=time.time() - start_time
            )
            
        except ImportError:
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': 'nsfw_detector library not installed'},
                method_used='nsfw_image_detector',
                processing_time=time.time() - start_time,
                error_message='nsfw_detector library not installed. Install with: pip install nsfw-detector'
            )
        except Exception as e:
            self.logger.error(f"NSFW image detector failed: {str(e)}")
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': str(e)},
                method_used='nsfw_image_detector',
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _transformers_nsfw(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """NSFW detection using Hugging Face transformers."""
        try:
            from transformers import pipeline
            from PIL import Image
            import numpy as np
            
            # Check if frame data is valid
            if frame_data.frame is None:
                raise ValueError("Frame data is None - no image to analyze")
            
            # Convert frame to PIL Image if it's numpy array
            if isinstance(frame_data.frame, np.ndarray):
                image = Image.fromarray(frame_data.frame)
            else:
                image = frame_data.frame
            
            # Initialize pipeline once and cache it
            if self._transformers_pipeline is None:
                # Get device configuration
                config_device = self.config.get('device', 'auto')
                device = get_device_for_ml(config_device)
                device_id = get_device_id_for_transformers(device)
                
                self.logger.info(f"NSFW Config device: {config_device}, Detected device: {device} (device_id: {device_id})")
                
                # Use a popular NSFW detection model from Hugging Face
                # Examples: "Falconsai/nsfw_image_detection", "michelecafagna26/t5-base-tag-generation"
                try:
                    # Force PyTorch backend and explicit device
                    import torch
                    
                    # Check CUDA availability at pipeline creation time
                    if device == "cuda" and torch.cuda.is_available():
                        self.logger.info(f"Creating NSFW pipeline on CUDA device {device_id}")
                        self._transformers_pipeline = pipeline(
                            "image-classification",
                            model="Falconsai/nsfw_image_detection",
                            device=device_id,  # Use proper device detection
                            torch_dtype=torch.float16,  # Use FP16 for GPU
                            framework="pt"  # Force PyTorch
                        )
                        # Verify model is on CUDA
                        if hasattr(self._transformers_pipeline.model, 'device'):
                            self.logger.info(f"NSFW model device: {self._transformers_pipeline.model.device}")
                        else:
                            self.logger.info("NSFW model device info not available")
                    else:
                        self.logger.info(f"Creating NSFW pipeline on CPU (CUDA available: {torch.cuda.is_available()})")
                        self._transformers_pipeline = pipeline(
                            "image-classification",
                            model="Falconsai/nsfw_image_detection",
                            device=-1,  # Force CPU
                            framework="pt"
                        )
                    
                    self.logger.info(f"NSFW model loaded successfully on device: {device}")
                except Exception as e:
                    self.logger.warning(f"Failed to load NSFW model on {device}, falling back to CPU: {e}")
                    self._transformers_pipeline = pipeline(
                        "image-classification",
                        model="Falconsai/nsfw_image_detection",
                        device=-1,  # Force CPU fallback
                        framework="pt"
                    )
            
            results = self._transformers_pipeline(image)
            
            # Extract NSFW probability
            nsfw_score = 0.0
            for result in results:
                if 'nsfw' in result['label'].lower():
                    nsfw_score = result['score']
                    break
            
            action = self.config.get('action', 'reject')
            threshold = 0.5
            
            if action == 'reject':
                status = CriterionStatus.FAIL if nsfw_score > threshold else CriterionStatus.PASS
            else:
                status = CriterionStatus.PASS if nsfw_score > threshold else CriterionStatus.FAIL
            
            return CriterionResult(
                status=status,
                confidence=1.0 - nsfw_score if status == CriterionStatus.PASS else nsfw_score,
                details={
                    'nsfw_score': nsfw_score,
                    'threshold': threshold,
                    'raw_results': results
                },
                method_used='transformers',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Transformers NSFW detection failed: {str(e)}")
            # Fallback to mock implementation
            nsfw_score = 0.2
            return CriterionResult(
                status=CriterionStatus.PASS,
                confidence=0.8,
                details={'nsfw_score': nsfw_score, 'error': str(e)},
                method_used='transformers',
                processing_time=time.time() - start_time
            )
    
    def _opennsfw2_nsfw(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """NSFW detection using OpenNSFW2."""
        try:
            # Try to import OpenNSFW2
            try:
                import opennsfw2
            except ImportError:
                self.logger.warning("OpenNSFW2 not available, using heuristic fallback")
                return self._heuristic_nsfw_fallback(frame_data, start_time)
            
            # Check if frame data is valid
            if frame_data.frame is None:
                raise ValueError("Frame data is None - no image to analyze")
            
            # Initialize model once and cache it
            if self._opennsfw2_model is None:
                self._opennsfw2_model = opennsfw2.make_open_nsfw_model()
                self.logger.info("OpenNSFW2 model loaded successfully")
            
            # Convert frame to the format expected by OpenNSFW2
            if isinstance(frame_data.frame, np.ndarray):
                # OpenNSFW2 expects PIL Image or path
                from PIL import Image
                if len(frame_data.frame.shape) == 3 and frame_data.frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame_data.frame
                image = Image.fromarray(rgb_frame)
            else:
                image = frame_data.frame
            
            # Run NSFW detection
            nsfw_score = opennsfw2.predict_image(image, self._opennsfw2_model)
            
            # Determine if criteria is met
            action = self.config.get('action', 'reject')
            threshold = self.config.get('threshold', 0.5)
            
            if action == 'reject':
                # Reject NSFW content: PASS if safe, FAIL if NSFW
                status = CriterionStatus.FAIL if nsfw_score > threshold else CriterionStatus.PASS
                confidence = 1.0 - nsfw_score if status == CriterionStatus.PASS else nsfw_score
            else:  # 'allow'
                # Allow NSFW content: PASS if NSFW, FAIL if safe
                status = CriterionStatus.PASS if nsfw_score > threshold else CriterionStatus.FAIL
                confidence = nsfw_score if status == CriterionStatus.PASS else 1.0 - nsfw_score
            
            return CriterionResult(
                status=status,
                confidence=confidence,
                details={
                    'nsfw_score': nsfw_score,
                    'threshold': threshold,
                    'action': action
                },
                method_used='opennsfw2',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"OpenNSFW2 NSFW detection failed: {str(e)}")
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': str(e)},
                method_used='opennsfw2',
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _heuristic_nsfw_fallback(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Simple heuristic-based NSFW detection fallback."""
        try:
            if frame_data.frame is None:
                raise ValueError("Frame data is None")
            
            # Simple heuristic: analyze skin tone percentage and image brightness
            frame = frame_data.frame
            
            # Convert to HSV for better skin detection
            if len(frame.shape) == 3:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Define skin color range in HSV
                lower_skin = np.array([0, 20, 70])
                upper_skin = np.array([20, 255, 255])
                
                # Create mask for skin tones
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                skin_percentage = np.sum(skin_mask > 0) / (frame.shape[0] * frame.shape[1])
                
                # Simple heuristic: high skin percentage might indicate NSFW
                # This is a very basic approximation
                nsfw_score = min(0.8, skin_percentage * 2)  # Cap at 0.8 for uncertainty
            else:
                # Grayscale image, use conservative estimate
                nsfw_score = 0.3
            
            action = self.config.get('action', 'reject')
            threshold = self.config.get('threshold', 0.5)
            
            if action == 'reject':
                status = CriterionStatus.FAIL if nsfw_score > threshold else CriterionStatus.PASS
                confidence = 1.0 - nsfw_score if status == CriterionStatus.PASS else nsfw_score
            else:
                status = CriterionStatus.PASS if nsfw_score > threshold else CriterionStatus.FAIL
                confidence = nsfw_score if status == CriterionStatus.PASS else 1.0 - nsfw_score
            
            return CriterionResult(
                status=status,
                confidence=confidence * 0.5,  # Lower confidence for heuristic method
                details={
                    'nsfw_score': nsfw_score,
                    'threshold': threshold,
                    'action': action,
                    'skin_percentage': skin_percentage if len(frame.shape) == 3 else 'N/A',
                    'heuristic': True
                },
                method_used='opennsfw2_heuristic',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Heuristic NSFW fallback failed: {str(e)}")
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': str(e)},
                method_used='opennsfw2_heuristic',
                processing_time=time.time() - start_time,
                error_message=str(e)
            )


class FaceCriterion(CriterionPlugin):
    """Face detection and validation criterion."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._ultralytics_model = None
        self._huggingface_pipeline = None
        self._mediapipe_detector = None
        self._opencv_cascade = None
    
    def get_available_methods(self) -> List[str]:
        """Get available face detection methods."""
        methods = []
        
        try:
            from ultralytics import YOLO  # noqa: F401
            methods.append('ultralytics')
        except ImportError:
            pass
        
        try:
            from transformers import pipeline  # noqa: F401
            # expose as 'huggingface' in public API/tests
            methods.append('huggingface')
        except ImportError:
            pass
        
        try:
            import mediapipe
            methods.append('mediapipe')
        except ImportError:
            pass
        
        try:
            import cv2
            methods.append('opencv')
        except ImportError:
            pass
        
        # Suggest installation commands for missing dependencies
        if 'ultralytics' not in methods:
            self.logger.info("To use 'ultralytics', install with: pip install ultralytics")
        if 'transformers' not in methods:
            self.logger.info("To use 'transformers', install with: pip install transformers")
        if 'mediapipe' not in methods:
            self.logger.info("To use 'mediapipe', install with: pip install mediapipe")
        if 'opencv' not in methods:
            self.logger.info("To use 'opencv', install with: pip install opencv-python")
            
        return methods
    
    def get_fallback_methods(self) -> List[str]:
        """Get face detection fallback chain."""
        # tests expect this order and labels
        return ['ultralytics', 'huggingface', 'mediapipe', 'opencv']
    
    def check(self, frame_data: FrameData) -> CriterionResult:
        """Check face detection in frame."""
        methods = self.get_fallback_methods() if self.method == 'auto' else [self.method]
        return self._try_methods(frame_data, methods)
    
    def _is_method_available(self, method: str) -> bool:
        """Check if face detection method is available."""
        if method == 'huggingface':
            try:
                from transformers import pipeline  # noqa: F401
                return True
            except ImportError:
                return False
        elif method == 'ultralytics':
            try:
                import ultralytics
                return True
            except ImportError:
                self.logger.info("To use 'ultralytics', install with: pip install ultralytics")
                return False
        elif method == 'mediapipe':
            try:
                import mediapipe
                return True
            except ImportError:
                self.logger.info("To use 'mediapipe', install with: pip install mediapipe")
                return False
        elif method == 'opencv':
            try:
                import cv2
                return True
            except ImportError:
                return False
        return False
    
    def _execute_method(self, method: str, frame_data: FrameData) -> CriterionResult:
        """Execute face detection method."""
        start_time = time.time()
        
        try:
            if method == 'huggingface':
                return self._transformers_face_detection(frame_data, start_time)
            elif method == 'ultralytics':
                return self._ultralytics_face_detection(frame_data, start_time)
            elif method == 'mediapipe':
                return self._mediapipe_face_detection(frame_data, start_time)
            elif method == 'opencv':
                return self._opencv_face_detection(frame_data, start_time)
            else:
                return CriterionResult(
                    status=CriterionStatus.ERROR,
                    confidence=0.0,
                    details={'error': f'Unknown method: {method}'},
                    method_used=method,
                    processing_time=time.time() - start_time,
                    error_message=f'Unknown method: {method}'
                )
        except Exception as e:
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': str(e)},
                method_used=method,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _transformers_face_detection(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Face detection using Hugging Face transformers (exposed as 'huggingface')."""
        try:
            from transformers import pipeline
            from PIL import Image
            import numpy as np
            
            # Check if frame data is valid
            if frame_data.frame is None:
                raise ValueError("Frame data is None - no image to analyze")
            
            # Convert frame to PIL Image if it's numpy array
            if isinstance(frame_data.frame, np.ndarray):
                image = Image.fromarray(frame_data.frame)
            else:
                image = frame_data.frame
            
            # Initialize pipeline once and cache it
            if self._huggingface_pipeline is None:
                # Get device configuration
                config_device = self.config.get('device', 'auto')
                device = get_device_for_ml(config_device)
                device_id = get_device_id_for_transformers(device)
                
                self.logger.info(f"Using device: {device} (device_id: {device_id}) for face detection")
                
                # Use face detection pipeline
                # Popular models: "facebook/detr-resnet-50", "microsoft/DialoGPT-medium"
                self._huggingface_pipeline = pipeline(
                    "object-detection",
                    model="facebook/detr-resnet-50",
                    device=device_id  # Use proper device detection
                )
            
            results = self._huggingface_pipeline(image)
            
            # Filter for faces (person detection)
            faces = []
            image_width, image_height = image.size
            
            for result in results:
                if result['label'].lower() in ['person', 'face']:
                    bbox = result['box']
                    confidence = result['score']
                    
                    # Calculate area percentage
                    area = (bbox['xmax'] - bbox['xmin']) * (bbox['ymax'] - bbox['ymin'])
                    area_percentage = (area / (image_width * image_height)) * 100
                    
                    faces.append(FaceDetection(
                        bbox=(bbox['xmin'], bbox['ymin'], 
                              bbox['xmax'] - bbox['xmin'], 
                              bbox['ymax'] - bbox['ymin']),
                        confidence=confidence,
                        area_percentage=area_percentage
                    ))
            
            min_confidence = self.config.get('min_confidence', 0.6)
            min_area_pct = self.config.get('min_area_pct', 1.0)
            
            valid_faces = [
                face for face in faces 
                if face.confidence >= min_confidence and face.area_percentage >= min_area_pct
            ]
            
            status = CriterionStatus.PASS if valid_faces else CriterionStatus.FAIL
            
            return CriterionResult(
                status=status,
                confidence=max([f.confidence for f in valid_faces], default=0.0),
                details={
                    'faces_detected': len(faces),
                    'valid_faces': len(valid_faces),
                    'faces': [
                        {
                            'bbox': face.bbox,
                            'confidence': face.confidence,
                            'area_pct': face.area_percentage
                        } for face in faces
                    ]
                },
                method_used='huggingface',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            # return ERROR to allow _try_methods to fall through to next backend
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': str(e)},
                method_used='huggingface',
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _ultralytics_face_detection(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Face detection using Ultralytics YOLO."""
        try:
            from ultralytics import YOLO
            import numpy as np
            from PIL import Image
            
            # Check if frame data is valid
            if frame_data.frame is None:
                raise ValueError("Frame data is None - no image to analyze")
            
            # Initialize model once and cache it
            if self._ultralytics_model is None:
                # Get device configuration
                config_device = self.config.get('device', 'auto')
                device = get_device_for_ml(config_device)
                
                self.logger.info(f"Ultralytics Config device: {config_device}, Detected device: {device}")
                
                # Load YOLOv8 model
                self._ultralytics_model = YOLO('yolov8n.pt')  # Nano model for speed
                
                # Explicitly move model to device and log the result
                try:
                    if device == "cuda":
                        import torch
                        if torch.cuda.is_available():
                            self._ultralytics_model.to(device)
                            self.logger.info(f"YOLO model moved to CUDA successfully")
                        else:
                            self.logger.warning(f"CUDA requested but not available, using CPU")
                            device = "cpu"
                    else:
                        self._ultralytics_model.to(device)
                        self.logger.info(f"YOLO model using device: {device}")
                except Exception as e:
                    self.logger.warning(f"Failed to move YOLO model to {device}, using default: {e}")
                    device = "cpu"
            
            # Ensure frame is in the right format for YOLO
            if isinstance(frame_data.frame, Image.Image):
                # YOLO can handle PIL Images directly
                image_input = frame_data.frame
            elif isinstance(frame_data.frame, np.ndarray):
                image_input = frame_data.frame
            else:
                raise ValueError(f"Unsupported frame type: {type(frame_data.frame)}")
            
            # Run inference
            results = self._ultralytics_model(image_input)  # YOLO will use the device the model is on
            
            faces = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is a person (class 0 in COCO)
                        if int(box.cls) == 0:  # Person class
                            confidence = float(box.conf)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Calculate area percentage
                            if isinstance(frame_data.frame, Image.Image):
                                frame_width, frame_height = frame_data.frame.size
                            else:
                                frame_height, frame_width = frame_data.frame.shape[:2]
                            area = (x2 - x1) * (y2 - y1)
                            area_percentage = (area / (frame_width * frame_height)) * 100
                            
                            faces.append(FaceDetection(
                                bbox=(int(x1), int(y1), int(x2-x1), int(y2-y1)),
                                confidence=confidence,
                                area_percentage=area_percentage
                            ))
            
            min_confidence = self.config.get('min_confidence', 0.6)
            min_area_pct = self.config.get('min_area_pct', 1.0)
            
            valid_faces = [
                face for face in faces 
                if face.confidence >= min_confidence and face.area_percentage >= min_area_pct
            ]
            
            status = CriterionStatus.PASS if valid_faces else CriterionStatus.FAIL
            
            return CriterionResult(
                status=status,
                confidence=max([f.confidence for f in valid_faces], default=0.0),
                details={
                    'faces_detected': len(faces),
                    'valid_faces': len(valid_faces),
                    'faces': [
                        {
                            'bbox': face.bbox,
                            'confidence': face.confidence,
                            'area_pct': face.area_percentage
                        } for face in faces
                    ]
                },
                method_used='ultralytics',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Ultralytics face detection failed: {str(e)}")
            return self._mock_face_detection(frame_data, start_time)
    
    def _mediapipe_face_detection(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Face detection using MediaPipe."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import numpy as np
            import cv2

            # Initialize detector once and cache it
            if self._mediapipe_detector is None:
                # This implementation uses the new MediaPipe Tasks API.
                # It requires a face detector model file (e.g., 'blaze_face_short_range.tflite').
                # For simplicity, we'll assume a model is available or handle its absence.
                
                model_path = self.config.get('mediapipe_model_path', 'blaze_face_short_range.tflite')
                try:
                    base_options = python.BaseOptions(model_asset_path=model_path)
                except Exception:
                    self.logger.warning(f"Could not load MediaPipe model from {model_path}. "
                                        "Download it or provide a valid path. "
                                        "See: https://developers.google.com/mediapipe/solutions/vision/face_detector/python")
                    # Fallback to legacy API if model file is the issue
                    return self._mediapipe_legacy_face_detection(frame_data, start_time)

                options = vision.FaceDetectorOptions(base_options=base_options,
                                                      min_detection_confidence=0.5)
                self._mediapipe_detector = vision.FaceDetector.create_from_options(options)

            # Convert frame to MediaPipe Image
            if len(frame_data.frame.shape) == 3 and frame_data.frame.shape[2] == 3:
                 rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
            else:
                 rgb_frame = frame_data.frame # Assume it's already RGB or grayscale
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            detection_result = self._mediapipe_detector.detect(mp_image)
            
            faces = []
            if detection_result.detections:
                frame_height, frame_width = rgb_frame.shape[:2]
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    confidence = detection.categories[0].score
                    
                    # Bbox is (origin_x, origin_y, width, height)
                    area_percentage = (bbox.width * bbox.height) / (frame_width * frame_height) * 100
                    
                    faces.append(FaceDetection(
                        bbox=(bbox.origin_x, bbox.origin_y, bbox.width, bbox.height),
                        confidence=confidence,
                        area_percentage=area_percentage
                    ))

            min_confidence = self.config.get('min_confidence', 0.6)
            min_area_pct = self.config.get('min_area_pct', 1.0)
            
            valid_faces = [
                face for face in faces 
                if face.confidence >= min_confidence and face.area_percentage >= min_area_pct
            ]
            
            status = CriterionStatus.PASS if valid_faces else CriterionStatus.FAIL
            
            return CriterionResult(
                status=status,
                confidence=max([f.confidence for f in valid_faces], default=0.0),
                details={
                    'faces_detected': len(faces),
                    'valid_faces': len(valid_faces),
                    'faces': [
                        {
                            'bbox': face.bbox,
                            'confidence': face.confidence,
                            'area_pct': face.area_percentage
                        } for face in faces
                    ]
                },
                method_used='mediapipe',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"MediaPipe face detection failed: {str(e)}")
            return self._mock_face_detection(frame_data, start_time)

    def _mediapipe_legacy_face_detection(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Face detection using legacy MediaPipe Solutions API."""
        try:
            import mediapipe as mp
            import cv2
            
            # Check if face_detection is available in solutions
            if not hasattr(mp.solutions, 'face_detection'):
                self.logger.error("MediaPipe face_detection solution not available")
                return self._mock_face_detection(frame_data, start_time)
            
            mp_face_detection = mp.solutions.face_detection
            
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                # Convert BGR to RGB if needed
                if len(frame_data.frame.shape) == 3:
                    rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame_data.frame
                
                results = face_detection.process(rgb_frame)
                
                faces = []
                if results.detections:
                    frame_height, frame_width = frame_data.frame.shape[:2]
                    
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        confidence = detection.score[0]
                        
                        # Convert relative coordinates to absolute
                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        w = int(bbox.width * frame_width)
                        h = int(bbox.height * frame_height)
                        
                        # Calculate area percentage
                        area_percentage = (bbox.width * bbox.height) * 100
                        
                        faces.append(FaceDetection(
                            bbox=(x, y, w, h),
                            confidence=confidence,
                            area_percentage=area_percentage
                        ))
                
                min_confidence = self.config.get('min_confidence', 0.6)
                min_area_pct = self.config.get('min_area_pct', 1.0)
                
                valid_faces = [
                    face for face in faces 
                    if face.confidence >= min_confidence and face.area_percentage >= min_area_pct
                ]
                
                status = CriterionStatus.PASS if valid_faces else CriterionStatus.FAIL
                
                return CriterionResult(
                    status=status,
                    confidence=max([f.confidence for f in valid_faces], default=0.0),
                    details={
                        'faces_detected': len(faces),
                        'valid_faces': len(valid_faces),
                        'faces': [
                            {
                                'bbox': face.bbox,
                                'confidence': face.confidence,
                                'area_pct': face.area_percentage
                            } for face in faces
                        ]
                    },
                    method_used='mediapipe_legacy',
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            self.logger.error(f"MediaPipe legacy face detection failed: {str(e)}")
            return self._mock_face_detection(frame_data, start_time)
    
    def _opencv_face_detection(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Face detection using OpenCV Haar Cascades."""
        try:
            import cv2
            
            # Initialize cascade once and cache it
            if self._opencv_cascade is None:
                # Load the cascade with fallback for different OpenCV versions
                import os
                cascade_path = None
                
                # Try modern OpenCV with cv2.data if available
                if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                    try:
                        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        if not os.path.exists(cascade_path):
                            cascade_path = None
                    except (AttributeError, TypeError):
                        cascade_path = None
                
                # Fallback to common installation paths
                if cascade_path is None:
                    possible_paths = [
                        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                        '/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                        os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml'),
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            cascade_path = path
                            break
                    
                    if cascade_path is None:
                        raise FileNotFoundError("Could not find OpenCV Haar cascade file")
                
                self._opencv_cascade = cv2.CascadeClassifier(cascade_path)
                
                # Check if cascade loaded successfully
                if self._opencv_cascade.empty():
                    raise ValueError("Failed to load Haar cascade classifier")
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_rects = self._opencv_cascade.detectMultiScale(gray, 1.3, 5)
            
            faces = []
            frame_height, frame_width = frame_data.frame.shape[:2]
            
            for (x, y, w, h) in face_rects:
                # Calculate area percentage
                area_percentage = ((w * h) / (frame_width * frame_height)) * 100
                
                faces.append(FaceDetection(
                    bbox=(x, y, w, h),
                    confidence=0.8,  # OpenCV doesn't provide confidence scores
                    area_percentage=area_percentage
                ))
            
            min_confidence = self.config.get('min_confidence', 0.6)
            min_area_pct = self.config.get('min_area_pct', 1.0)
            
            valid_faces = [
                face for face in faces 
                if face.confidence >= min_confidence and face.area_percentage >= min_area_pct
            ]
            
            status = CriterionStatus.PASS if valid_faces else CriterionStatus.FAIL
            
            return CriterionResult(
                status=status,
                confidence=0.8 if valid_faces else 0.0,
                details={
                    'faces_detected': len(faces),
                    'valid_faces': len(valid_faces),
                    'faces': [
                        {
                            'bbox': face.bbox,
                            'confidence': face.confidence,
                            'area_pct': face.area_percentage
                        } for face in faces
                    ]
                },
                method_used='opencv',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"OpenCV face detection failed: {str(e)}")
            return self._mock_face_detection(frame_data, start_time)
    
    def _mock_face_detection(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Fallback face detection when no models are available."""
        logger.warning("Using fallback face detection - no models available")
        
        min_confidence = self.config.get('min_confidence', 0.6)
        min_area_pct = self.config.get('min_area_pct', 1.0)
        
        # Conservative approach: assume no faces when we can't detect
        # This is safer than false positives in most video processing scenarios
        status = CriterionStatus.FAIL
        confidence = 0.0
        
        return CriterionResult(
            status=status,
            confidence=confidence,
            details={
                'faces_detected': 0,
                'valid_faces': 0,
                'faces': [],
                'fallback': True,
                'reason': 'No face detection models available',
                'min_confidence': min_confidence,
                'min_area_pct': min_area_pct
            },
            method_used='fallback',
            processing_time=time.time() - start_time,
            error_message="Face detection unavailable - using conservative fallback"
        )
    

class GenderCriterion(CriterionPlugin):
    """Gender classification criterion using Hugging Face transformers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GenderCriterion with cached models."""
        super().__init__(config)
        # Cache for models to avoid reloading
        self._transformers_pipeline = None
        self._deepface_model = None
    
    def get_available_methods(self) -> List[str]:
        """Get available gender classification methods."""
        methods = ['mock']  # Mock is always available
        
        try:
            from transformers import pipeline
            methods.append('transformers')
        except ImportError:
            pass
        
        try:
            import deepface
            methods.append('deepface')
        except ImportError:
            pass
        
        if 'transformers' not in methods:
            self.logger.info("To use 'transformers', install with: pip install transformers")
        if 'deepface' not in methods:
            self.logger.info("To use 'deepface', install with: pip install deepface")
            
        return methods
    
    def get_fallback_methods(self) -> List[str]:
        """Get gender classification fallback chain."""
        return ['transformers', 'deepface']
    
    def check(self, frame_data: FrameData) -> CriterionResult:
        """Check gender classification in frame."""
        methods = self.get_fallback_methods() if self.method == 'auto' else [self.method]
        return self._try_methods(frame_data, methods)
    
    def _is_method_available(self, method: str) -> bool:
        """Check if gender classification method is available."""
        if method == 'mock':
            return True
        elif method == 'transformers':
            try:
                from transformers import pipeline
                return True
            except ImportError:
                return False
        elif method == 'deepface':
            try:
                import deepface
                return True
            except ImportError:
                return False
        return False
    
    def _execute_method(self, method: str, frame_data: FrameData) -> CriterionResult:
        """Execute gender classification method."""
        start_time = time.time()
        
        try:
            if method == 'mock':
                return self._mock_gender_classification(frame_data, start_time)
            elif method == 'transformers':
                return self._transformers_gender_classification(frame_data, start_time)
            elif method == 'deepface':
                return self._deepface_gender_classification(frame_data, start_time)
            else:
                return CriterionResult(
                    status=CriterionStatus.ERROR,
                    confidence=0.0,
                    details={'error': f'Unknown method: {method}'},
                    method_used=method,
                    processing_time=time.time() - start_time,
                    error_message=f'Unknown method: {method}'
                )
        except Exception as e:
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': str(e)},
                method_used=method,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _transformers_gender_classification(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Gender classification using Hugging Face transformers."""
        try:
            from transformers import pipeline
            from PIL import Image
            import numpy as np
            
            # Check if frame data is valid
            if frame_data.frame is None:
                raise ValueError("Frame data is None - no image to analyze")
            
            # Convert frame to PIL Image if it's numpy array
            if isinstance(frame_data.frame, np.ndarray):
                image = Image.fromarray(frame_data.frame)
            else:
                image = frame_data.frame
            
            # Initialize pipeline once and cache it
            if self._transformers_pipeline is None:
                # Get device configuration
                config_device = self.config.get('device', 'auto')
                device = get_device_for_ml(config_device)
                device_id = get_device_id_for_transformers(device)
                
                self.logger.info(f"Gender Config device: {config_device}, Detected device: {device} (device_id: {device_id})")
                
                # Use gender classification model
                # Example models: "rizvandwiki/gender-classification", "salt-ai/age-and-gender"
                try:
                    # Force PyTorch backend and explicit device
                    import torch
                    self._transformers_pipeline = pipeline(
                        "image-classification",
                        model="rizvandwiki/gender-classification",
                        device=device_id,  # Use proper device detection
                        torch_dtype=torch.float16 if device_id >= 0 else torch.float32,  # Use FP16 for GPU
                        framework="pt"  # Force PyTorch
                    )
                    self.logger.info(f"Gender model loaded successfully on device: {device}")
                except Exception as e:
                    self.logger.warning(f"Failed to load Gender model on {device}, falling back to CPU: {e}")
                    self._transformers_pipeline = pipeline(
                        "image-classification",
                        model="rizvandwiki/gender-classification",
                        device=-1,  # Force CPU fallback
                        framework="pt"
                    )
            
            results = self._transformers_pipeline(image)
            
            # Extract gender probabilities
            gender_scores = {}
            for result in results:
                gender_scores[result['label'].lower()] = result['score']
            
            # Determine if criteria is met
            gender_filter = self.config.get('filter', 'any').lower()
            min_confidence = self.config.get('min_confidence', 0.8)
            
            female_score = gender_scores.get('female', 0.0)
            male_score = gender_scores.get('male', 0.0)
            
            if gender_filter == 'female':
                status = CriterionStatus.PASS if female_score >= min_confidence else CriterionStatus.FAIL
                confidence = female_score
            elif gender_filter == 'male':
                status = CriterionStatus.PASS if male_score >= min_confidence else CriterionStatus.FAIL
                confidence = male_score
            else:  # 'any'
                max_score = max(female_score, male_score)
                status = CriterionStatus.PASS if max_score >= min_confidence else CriterionStatus.FAIL
                confidence = max_score
            
            return CriterionResult(
                status=status,
                confidence=confidence,
                details={
                    'gender_scores': gender_scores,
                    'filter': gender_filter,
                    'min_confidence': min_confidence,
                    'raw_results': results
                },
                method_used='transformers',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Transformers gender classification failed: {str(e)}")
            # Fallback to mock implementation
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': str(e)},
                method_used='transformers',
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _deepface_gender_classification(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Gender classification using DeepFace."""
        try:
            from deepface import DeepFace
            import numpy as np
            
            # Analyze the frame
            results = DeepFace.analyze(
                img_path=frame_data.frame,
                actions=['gender'],
                enforce_detection=False
            )
            
            if isinstance(results, list):
                result = results[0]
            else:
                result = results
            
            # Safely access gender scores with type checking
            if isinstance(result, dict) and 'gender' in result:
                gender_scores = result['gender']
            else:
                raise ValueError(f"Unexpected result format from DeepFace: {type(result)}")
            
            # Determine if criteria is met
            gender_filter = self.config.get('filter', 'any').lower()
            min_confidence = self.config.get('min_confidence', 0.8)
            
            female_score = gender_scores.get('Woman', 0.0) / 100.0
            male_score = gender_scores.get('Man', 0.0) / 100.0
            
            if gender_filter == 'female':
                status = CriterionStatus.PASS if female_score >= min_confidence else CriterionStatus.FAIL
                confidence = female_score
            elif gender_filter == 'male':
                status = CriterionStatus.PASS if male_score >= min_confidence else CriterionStatus.FAIL
                confidence = male_score
            else:  # 'any'
                max_score = max(female_score, male_score)
                status = CriterionStatus.PASS if max_score >= min_confidence else CriterionStatus.FAIL
                confidence = max_score
            
            return CriterionResult(
                status=status,
                confidence=confidence,
                details={
                    'gender_scores': {'female': female_score, 'male': male_score},
                    'filter': gender_filter,
                    'min_confidence': min_confidence,
                    'raw_results': result
                },
                method_used='deepface',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"DeepFace gender classification failed: {str(e)}")
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': str(e)},
                method_used='deepface',
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _mock_gender_classification(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Fallback gender classification when no models are available."""
        logger.warning("Using fallback gender classification - no models available")
        
        gender_filter = self.config.get('filter', 'any').lower()
        min_confidence = self.config.get('min_confidence', 0.8)
        
        # Conservative approach: when we can't classify, fail specific filters
        # but pass 'any' filter to avoid blocking all content
        if gender_filter == 'any':
            status = CriterionStatus.PASS
            confidence = 0.5  # Neutral confidence
            gender_scores = {'female': 0.5, 'male': 0.5}
        else:
            # For specific gender filters, fail when we can't detect
            status = CriterionStatus.FAIL
            confidence = 0.0
            gender_scores = {'female': 0.0, 'male': 0.0}
        
        return CriterionResult(
            status=status,
            confidence=confidence,
            details={
                'gender_scores': gender_scores,
                'filter': gender_filter,
                'min_confidence': min_confidence,
                'fallback': True,
                'reason': 'No gender classification models available'
            },
            method_used='fallback',
            processing_time=time.time() - start_time,
            error_message="Gender classification unavailable - using conservative fallback"
        )


class PoseCriterion(CriterionPlugin):
    """Pose detection criterion using MediaPipe."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._mediapipe_pose = None
        self._mediapipe_holistic = None
    
    def get_available_methods(self) -> List[str]:
        methods = []
        if self._is_method_available('mediapipe'):
            methods.append('mediapipe')
        methods.append('fallback')  # Always available as fallback
        return methods
    
    def get_fallback_methods(self) -> List[str]:
        return ['fallback']
    
    def check(self, frame_data: FrameData) -> CriterionResult:
        """Check pose detection in frame."""
        methods = self.get_fallback_methods() if self.method == 'auto' else [self.method]
        return self._try_methods(frame_data, methods)
    
    def _is_method_available(self, method: str) -> bool:
        if method == 'mediapipe':
            try:
                import mediapipe as mp
                return True
            except ImportError:
                return False
        elif method == 'fallback':
            return True
        return False
    
    def _execute_method(self, method: str, frame_data: FrameData) -> CriterionResult:
        start_time = time.time()
        
        if method == 'mediapipe':
            return self._mediapipe_pose_detection(frame_data, start_time)
        elif method == 'fallback':
            return self._fallback_pose_detection(frame_data, start_time)
        else:
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': f'Unknown method: {method}'},
                method_used=method,
                processing_time=time.time() - start_time,
                error_message=f"Unknown pose detection method: {method}"
            )
    
    def _mediapipe_pose_detection(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Pose detection using MediaPipe."""
        try:
            import mediapipe as mp
            
            # Initialize MediaPipe pose if needed
            if self._mediapipe_pose is None:
                self._mediapipe_pose = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=self.config.get('model_complexity', 1),
                    enable_segmentation=False,
                    min_detection_confidence=self.config.get('min_detection_confidence', 0.5)
                )
            
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self._mediapipe_pose.process(rgb_frame)
            
            pose_filter = self.config.get('filter', 'any')  # 'frontal', 'profile', 'any'
            min_confidence = self.config.get('min_confidence', 0.7)
            
            if results.pose_landmarks:
                # Analyze pose orientation and visibility
                pose_analysis = self._analyze_pose(results.pose_landmarks, rgb_frame.shape)
                
                # Check if pose matches filter criteria
                if pose_filter == 'any':
                    status = CriterionStatus.PASS
                    confidence = pose_analysis['confidence']
                elif pose_filter == 'frontal' and pose_analysis['orientation'] == 'frontal':
                    status = CriterionStatus.PASS if pose_analysis['confidence'] >= min_confidence else CriterionStatus.FAIL
                    confidence = pose_analysis['confidence']
                elif pose_filter == 'profile' and pose_analysis['orientation'] == 'profile':
                    status = CriterionStatus.PASS if pose_analysis['confidence'] >= min_confidence else CriterionStatus.FAIL
                    confidence = pose_analysis['confidence']
                else:
                    status = CriterionStatus.FAIL
                    confidence = pose_analysis['confidence']
            else:
                status = CriterionStatus.FAIL
                confidence = 0.0
                pose_analysis = {'orientation': 'none', 'confidence': 0.0}
            
            return CriterionResult(
                status=status,
                confidence=confidence,
                details={
                    'pose_detected': results.pose_landmarks is not None,
                    'orientation': pose_analysis['orientation'],
                    'pose_confidence': pose_analysis['confidence'],
                    'filter': pose_filter,
                    'min_confidence': min_confidence
                },
                method_used='mediapipe',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"MediaPipe pose detection failed: {str(e)}")
            return self._fallback_pose_detection(frame_data, start_time)
    
    def _analyze_pose(self, landmarks, frame_shape) -> dict:
        """Analyze pose landmarks to determine orientation and confidence."""
        try:
            import mediapipe as mp
            
            # Get key landmarks
            nose = landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            left_shoulder = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_ear = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EAR]
            
            # Calculate shoulder width ratio (helps determine if facing forward)
            shoulder_diff = abs(left_shoulder.x - right_shoulder.x)
            
            # Calculate ear visibility (helps determine profile vs frontal)
            left_ear_vis = left_ear.visibility if hasattr(left_ear, 'visibility') else 0.5
            right_ear_vis = right_ear.visibility if hasattr(right_ear, 'visibility') else 0.5
            
            # Determine orientation
            if shoulder_diff > 0.15 and min(left_ear_vis, right_ear_vis) > 0.3:
                orientation = 'frontal'
                confidence = min(0.9, shoulder_diff * 2 + (left_ear_vis + right_ear_vis) / 2)
            elif max(left_ear_vis, right_ear_vis) > 0.7 and min(left_ear_vis, right_ear_vis) < 0.3:
                orientation = 'profile'
                confidence = max(left_ear_vis, right_ear_vis) * 0.8
            else:
                orientation = 'unclear'
                confidence = 0.5
            
            return {
                'orientation': orientation,
                'confidence': confidence
            }
            
        except Exception:
            return {'orientation': 'unclear', 'confidence': 0.5}
    
    def _fallback_pose_detection(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Fallback pose detection when MediaPipe is not available."""
        logger.warning("Using fallback pose detection - MediaPipe not available")
        
        pose_filter = self.config.get('filter', 'any')
        
        # Conservative approach: pass 'any' filter, fail specific filters
        if pose_filter == 'any':
            status = CriterionStatus.PASS
            confidence = 0.5
        else:
            status = CriterionStatus.FAIL
            confidence = 0.0
        
        return CriterionResult(
            status=status,
            confidence=confidence,
            details={
                'pose_detected': False,
                'orientation': 'unknown',
                'pose_confidence': 0.0,
                'filter': pose_filter,
                'fallback': True,
                'reason': 'MediaPipe pose detection not available'
            },
            method_used='fallback',
            processing_time=time.time() - start_time,
            error_message="Pose detection unavailable - using conservative fallback"
        )


class VisibilityCriterion(CriterionPlugin):
    """Visibility and image quality analysis criterion using OpenCV."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # No model caching needed for basic OpenCV operations
    
    def get_available_methods(self) -> List[str]:
        """Get available visibility analysis methods."""
        methods = []
        
        try:
            import cv2  # noqa: F401
            methods.append('opencv')
        except ImportError:
            pass
        
        methods.append('fallback')  # Always available as fallback
        return methods
    
    def get_fallback_methods(self) -> List[str]:
        """Get visibility analysis fallback chain."""
        return ['opencv', 'fallback']
    
    def check(self, frame_data: FrameData) -> CriterionResult:
        """Check visibility quality in frame."""
        methods = self.get_fallback_methods() if self.method == 'auto' else [self.method]
        return self._try_methods(frame_data, methods)
    
    def _is_method_available(self, method: str) -> bool:
        """Check if visibility method is available."""
        if method == 'opencv':
            try:
                import cv2  # noqa: F401
                return True
            except ImportError:
                return False
        elif method == 'fallback':
            return True
        return False
    
    def _execute_method(self, method: str, frame_data: FrameData) -> CriterionResult:
        """Execute visibility analysis method."""
        start_time = time.time()
        
        try:
            if method == 'opencv':
                return self._opencv_visibility_analysis(frame_data, start_time)
            elif method == 'fallback':
                return self._fallback_visibility_analysis(frame_data, start_time)
            else:
                return CriterionResult(
                    status=CriterionStatus.ERROR,
                    confidence=0.0,
                    details={'error': f'Unknown method: {method}'},
                    method_used=method,
                    processing_time=time.time() - start_time,
                    error_message=f'Unknown method: {method}'
                )
        except Exception as e:
            return CriterionResult(
                status=CriterionStatus.ERROR,
                confidence=0.0,
                details={'error': str(e)},
                method_used=method,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _opencv_visibility_analysis(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Analyze image visibility using OpenCV."""
        try:
            import cv2
            import numpy as np
            
            # Check if frame data is valid
            if frame_data.frame is None:
                raise ValueError("Frame data is None - no image to analyze")
            
            frame = frame_data.frame
            
            # Convert to grayscale for analysis
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # 1. Brightness analysis
            brightness = np.mean(gray)
            brightness_score = self._evaluate_brightness(brightness)
            
            # 2. Contrast analysis
            contrast = np.std(gray)
            contrast_score = self._evaluate_contrast(contrast)
            
            # 3. Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = self._evaluate_sharpness(laplacian_var)
            
            # 4. Noise analysis using standard deviation in local regions
            noise_score = self._evaluate_noise(gray)
            
            # Combine scores with weights
            weights = {
                'brightness': 0.25,
                'contrast': 0.25,
                'sharpness': 0.35,
                'noise': 0.15
            }
            
            overall_score = (
                brightness_score * weights['brightness'] +
                contrast_score * weights['contrast'] +
                blur_score * weights['sharpness'] +
                noise_score * weights['noise']
            )
            
            # Determine status based on threshold
            min_quality = self.config.get('min_quality', 0.6)
            status = CriterionStatus.PASS if overall_score >= min_quality else CriterionStatus.FAIL
            
            return CriterionResult(
                status=status,
                confidence=overall_score,
                details={
                    'overall_quality': overall_score,
                    'brightness': brightness,
                    'brightness_score': brightness_score,
                    'contrast': contrast,
                    'contrast_score': contrast_score,
                    'sharpness_variance': laplacian_var,
                    'sharpness_score': blur_score,
                    'noise_score': noise_score,
                    'min_quality': min_quality,
                    'weights': weights
                },
                method_used='opencv',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"OpenCV visibility analysis failed: {str(e)}")
            return self._fallback_visibility_analysis(frame_data, start_time)
    
    def _evaluate_brightness(self, brightness: float) -> float:
        """Evaluate brightness quality (0.0 to 1.0)."""
        # Optimal brightness range: 80-180 (out of 255)
        optimal_min, optimal_max = 80, 180
        
        if optimal_min <= brightness <= optimal_max:
            return 1.0
        elif brightness < optimal_min:
            # Too dark
            return max(0.0, brightness / optimal_min)
        else:
            # Too bright
            return max(0.0, 1.0 - (brightness - optimal_max) / (255 - optimal_max))
    
    def _evaluate_contrast(self, contrast: float) -> float:
        """Evaluate contrast quality (0.0 to 1.0)."""
        # Good contrast typically has std > 40
        min_contrast = 20
        optimal_contrast = 60
        
        if contrast >= optimal_contrast:
            return 1.0
        elif contrast >= min_contrast:
            return contrast / optimal_contrast
        else:
            return max(0.0, contrast / min_contrast * 0.5)
    
    def _evaluate_sharpness(self, laplacian_var: float) -> float:
        """Evaluate sharpness based on Laplacian variance (0.0 to 1.0)."""
        # Higher variance indicates sharper image
        min_sharpness = 100  # Below this is considered blurry
        good_sharpness = 500  # Above this is considered sharp
        
        if laplacian_var >= good_sharpness:
            return 1.0
        elif laplacian_var >= min_sharpness:
            return laplacian_var / good_sharpness
        else:
            return max(0.0, laplacian_var / min_sharpness * 0.3)
    
    def _evaluate_noise(self, gray_image) -> float:
        """Evaluate noise level (0.0 to 1.0, higher is better)."""
        try:
            import cv2
            import numpy as np
            
            # Use bilateral filter to estimate noise
            filtered = cv2.bilateralFilter(gray_image, 9, 75, 75)
            noise_variance = np.var(gray_image.astype(np.float32) - filtered.astype(np.float32))
            
            # Lower noise variance is better
            max_acceptable_noise = 50
            if noise_variance <= max_acceptable_noise:
                return 1.0 - (noise_variance / max_acceptable_noise) * 0.5
            else:
                return max(0.0, 0.5 - (noise_variance - max_acceptable_noise) / max_acceptable_noise * 0.5)
                
        except Exception:
            # Fallback: assume moderate noise
            return 0.7
    
    def _fallback_visibility_analysis(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Fallback visibility analysis when OpenCV is not available."""
        logger.warning("Using fallback visibility analysis - OpenCV not available")
        
        min_quality = self.config.get('min_quality', 0.6)
        
        # Conservative approach: assume moderate quality when we can't analyze
        status = CriterionStatus.PASS
        confidence = 0.5  # Neutral confidence
        
        return CriterionResult(
            status=status,
            confidence=confidence,
            details={
                'overall_quality': confidence,
                'fallback': True,
                'reason': 'OpenCV not available for visibility analysis',
                'min_quality': min_quality
            },
            method_used='fallback',
            processing_time=time.time() - start_time,
            error_message="Visibility analysis unavailable - using moderate quality assumption"
        )


class CriteriaManager:
    """
    Manager for all video analysis criteria.
    
    Coordinates the application of multiple criteria plugins
    to analyze video frames and scenes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the criteria manager with configuration."""
        self.config = config.get('criteria', {})
        self.global_config = config  # Store global config for device settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize available criteria plugins with merged config
        self.criteria_plugins = {}
        for criteria_name in ['nsfw', 'face', 'gender', 'pose', 'visibility']:
            criteria_config = self.config.get(criteria_name, {}).copy()
            # Merge global device setting into each criterion's config
            if 'device' not in criteria_config:
                criteria_config['device'] = self.global_config.get('device', 'auto')
            
            if criteria_name == 'nsfw':
                self.criteria_plugins[criteria_name] = NSFWCriterion(criteria_config)
            elif criteria_name == 'face':
                self.criteria_plugins[criteria_name] = FaceCriterion(criteria_config)
            elif criteria_name == 'gender':
                self.criteria_plugins[criteria_name] = GenderCriterion(criteria_config)
            elif criteria_name == 'pose':
                self.criteria_plugins[criteria_name] = PoseCriterion(criteria_config)
            elif criteria_name == 'visibility':
                self.criteria_plugins[criteria_name] = VisibilityCriterion(criteria_config)
        
        self.logger.info(f"Initialized CriteriaManager with {len(self.criteria_plugins)} criteria plugins, device: {self.global_config.get('device', 'auto')}")
    
    def get_available_criteria(self) -> List[str]:
        """Get list of available criteria."""
        return list(self.criteria_plugins.keys())
    
    def get_enabled_criteria(self) -> List[str]:
        """Get list of enabled criteria based on configuration."""
        enabled = []
        for criteria_name in self.criteria_plugins.keys():
            criteria_config = self.config.get(criteria_name, {})
            if criteria_config.get('enabled', True):
                enabled.append(criteria_name)
        return enabled
    
    def analyze_frames(self, frames: List[FrameData]) -> Dict[str, Any]:
        """
        Analyze a list of frames against all enabled criteria.
        
        Args:
            frames: List of frame data to analyze
            
        Returns:
            Dictionary containing analysis results for each criteria
        """
        enabled_criteria = self.get_enabled_criteria()
        results = {}
        
        for criteria_name in enabled_criteria:
            plugin = self.criteria_plugins[criteria_name]
            criteria_results = []
            
            self.logger.debug(f"Analyzing {len(frames)} frames with {criteria_name} criterion")
            
            for frame_data in frames:
                try:
                    result = plugin.check(frame_data)  # Use check instead of apply
                    criteria_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error applying {criteria_name} criterion: {e}")
                    criteria_results.append(CriterionResult(
                        status=CriterionStatus.ERROR,
                        confidence=0.0,
                        details={'error': str(e)},
                        method_used='error',
                        error_message=str(e)
                    ))
            
            # Aggregate results for this criterion
            results[criteria_name] = self._aggregate_criterion_results(criteria_results)
        
        return results
    
    def _aggregate_criterion_results(self, results: List[CriterionResult]) -> Dict[str, Any]:
        """
        Aggregate multiple criterion results into a single summary.
        
        Args:
            results: List of criterion results for frames
            
        Returns:
            Aggregated result summary
        """
        if not results:
            return {
                'detected': False,
                'confidence': 0.0,
                'meets_criteria': False,
                'status_counts': {},
                'avg_confidence': 0.0
            }
        
        # Count statuses
        status_counts = {}
        confidences = []
        pass_count = 0
        
        for result in results:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            confidences.append(result.confidence)
            if result.status == CriterionStatus.PASS:
                pass_count += 1
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Determine if criterion is met based on majority of frames
        detected = pass_count > len(results) / 2
        meets_criteria = detected and avg_confidence >= 0.6  # Default threshold
        
        return {
            'detected': detected,
            'confidence': avg_confidence,
            'meets_criteria': meets_criteria,
            'status_counts': status_counts,
            'total_frames': len(results),
            'pass_count': pass_count,
            'avg_confidence': avg_confidence
        }
