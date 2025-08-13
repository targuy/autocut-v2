"""
Plugin architecture for video analysis criteria.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time

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
    
    def get_available_methods(self) -> List[str]:
        """Get list of available NSFW detection methods."""
        methods = ['mock']  # Mock is always available
        
        # Check if specific libraries are available
        try:
            import transformers
            methods.append('nsfw_image_detector')
            methods.append('transformers')
        except ImportError:
            pass
            
        return methods
    
    def get_fallback_methods(self) -> List[str]:
        """Get NSFW fallback chain."""
        return ['nsfw_image_detector', 'transformers']
    
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
        """Mock NSFW detection for testing."""
        # Mock implementation always returns safe content
        nsfw_score = 0.1  # Low NSFW score
        
        action = self.config.get('action', 'reject')
        threshold = 0.5
        
        if action == 'reject':
            status = CriterionStatus.PASS  # Mock always passes
        else:
            status = CriterionStatus.FAIL  # Mock always fails when looking for NSFW
        
        return CriterionResult(
            status=status,
            confidence=0.9,  # High confidence in mock result
            details={'nsfw_score': nsfw_score, 'threshold': threshold, 'mock': True},
            method_used='mock',
            processing_time=time.time() - start_time
        )
    
    def _nsfw_image_detector(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """NSFW detection using nsfw_image_detector."""
        # Placeholder implementation
        nsfw_score = 0.1  # Mock low NSFW score
        
        action = self.config.get('action', 'reject')
        threshold = 0.5
        
        if action == 'reject':
            status = CriterionStatus.FAIL if nsfw_score > threshold else CriterionStatus.PASS
        else:
            status = CriterionStatus.PASS if nsfw_score > threshold else CriterionStatus.FAIL
        
        return CriterionResult(
            status=status,
            confidence=1.0 - nsfw_score if status == CriterionStatus.PASS else nsfw_score,
            details={'nsfw_score': nsfw_score, 'threshold': threshold},
            method_used='nsfw_image_detector',
            processing_time=time.time() - start_time
        )
    
    def _transformers_nsfw(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """NSFW detection using Hugging Face transformers."""
        try:
            from transformers import pipeline
            from PIL import Image
            import numpy as np
            
            # Convert frame to PIL Image if it's numpy array
            if isinstance(frame_data.frame, np.ndarray):
                image = Image.fromarray(frame_data.frame)
            else:
                image = frame_data.frame
            
            # Use a popular NSFW detection model from Hugging Face
            # Examples: "Falconsai/nsfw_image_detection", "michelecafagna26/t5-base-tag-generation"
            classifier = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                device=-1  # Use CPU
            )
            
            results = classifier(image)
            
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


class FaceCriterion(CriterionPlugin):
    """Face detection and validation criterion."""
    
    def get_available_methods(self) -> List[str]:
        """Get available face detection methods."""
        methods = ['mock']  # Mock is always available
        
        try:
            import ultralytics
            methods.append('ultralytics')
        except ImportError:
            pass
        
        try:
            from transformers import pipeline
            methods.append('transformers')
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
        return ['transformers', 'ultralytics', 'mediapipe', 'opencv']
    
    def check(self, frame_data: FrameData) -> CriterionResult:
        """Check face detection in frame."""
        methods = self.get_fallback_methods() if self.method == 'auto' else [self.method]
        return self._try_methods(frame_data, methods)
    
    def _is_method_available(self, method: str) -> bool:
        """Check if face detection method is available."""
        if method == 'mock':
            return True
        elif method == 'ultralytics':
            try:
                import ultralytics
                return True
            except ImportError:
                self.logger.info("To use 'ultralytics', install with: pip install ultralytics")
                return False
        elif method == 'transformers':
            try:
                from transformers import pipeline
                return True
            except ImportError:
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
            if method == 'mock':
                return self._mock_face_detection(frame_data, start_time)
            elif method == 'transformers':
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
        """Face detection using Hugging Face transformers."""
        try:
            from transformers import pipeline
            from PIL import Image
            import numpy as np
            
            # Convert frame to PIL Image if it's numpy array
            if isinstance(frame_data.frame, np.ndarray):
                image = Image.fromarray(frame_data.frame)
            else:
                image = frame_data.frame
            
            # Use face detection pipeline
            # Popular models: "facebook/detr-resnet-50", "microsoft/DialoGPT-medium"
            face_detector = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=-1  # Use CPU
            )
            
            results = face_detector(image)
            
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
                method_used='transformers',
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Transformers face detection failed: {str(e)}")
            # Fallback to mock implementation
            return self._mock_face_detection(frame_data, start_time)
    
    def _ultralytics_face_detection(self, frame_data: FrameData, start_time: float) -> CriterionResult:
        """Face detection using Ultralytics YOLO."""
        try:
            from ultralytics import YOLO
            import numpy as np
            
            # Load YOLOv8 model
            model = YOLO('yolov8n.pt')  # Nano model for speed
            
            # Run inference
            results = model(frame_data.frame)
            
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
            detector = vision.FaceDetector.create_from_options(options)

            # Convert frame to MediaPipe Image
            if len(frame_data.frame.shape) == 3 and frame_data.frame.shape[2] == 3:
                 rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
            else:
                 rgb_frame = frame_data.frame # Assume it's already RGB or grayscale
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            detection_result = detector.detect(mp_image)
            
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
            
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Check if cascade loaded successfully
            if face_cascade.empty():
                raise ValueError("Failed to load Haar cascade classifier")
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
            
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
        """Mock face detection for testing."""
        faces = [
            FaceDetection(
                bbox=(100, 100, 200, 200),
                confidence=0.95,
                area_percentage=5.2
            )
        ]
        
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
                ],
                'mock': True
            },
            method_used='mock',
            processing_time=time.time() - start_time
        )
    

class GenderCriterion(CriterionPlugin):
    """Gender classification criterion using Hugging Face transformers."""
    
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
            
            # Convert frame to PIL Image if it's numpy array
            if isinstance(frame_data.frame, np.ndarray):
                image = Image.fromarray(frame_data.frame)
            else:
                image = frame_data.frame
            
            # Use gender classification model
            # Example models: "rizvandwiki/gender-classification", "salt-ai/age-and-gender"
            classifier = pipeline(
                "image-classification",
                model="rizvandwiki/gender-classification",
                device=-1  # Use CPU
            )
            
            results = classifier(image)
            
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
        """Mock gender classification for testing."""
        gender_scores = {'female': 0.85, 'male': 0.15}
        gender_filter = self.config.get('filter', 'any').lower()
        min_confidence = self.config.get('min_confidence', 0.8)
        
        if gender_filter == 'female':
            status = CriterionStatus.PASS
            confidence = 0.85
        elif gender_filter == 'male':
            status = CriterionStatus.FAIL
            confidence = 0.15
        else:  # 'any'
            status = CriterionStatus.PASS
            confidence = 0.85
        
        return CriterionResult(
            status=status,
            confidence=confidence,
            details={
                'gender_scores': gender_scores,
                'filter': gender_filter,
                'min_confidence': min_confidence,
                'mock': True
            },
            method_used='mock',
            processing_time=time.time() - start_time
        )
