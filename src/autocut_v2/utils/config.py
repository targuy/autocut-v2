"""
Configuration management for AutoCut v2.
"""

from typing import Dict, Any, Optional
import json
# import yaml  # Will be installed later
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_device_for_ml(config_device: str = "auto") -> str:
    """
    Determine the best device for ML models based on config and availability.
    
    Args:
        config_device: Device setting from config ("auto", "cuda", "cuda:0", "mps", "cpu")
        
    Returns:
        Device string for transformers/pytorch ("cuda", "cpu", etc.)
    """
    logger.info(f"get_device_for_ml called with config_device: {config_device}")
    
    if config_device == "auto":
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            if cuda_available:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                logger.info(f"CUDA device count: {device_count}")
                logger.info(f"Current CUDA device: {current_device}")
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Using MPS (Apple Silicon)")
                return "mps"
            else:
                logger.info("Falling back to CPU")
                return "cpu"
        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}")
            return "cpu"
    elif config_device.startswith("cuda"):
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA explicitly requested, torch.cuda.is_available(): {cuda_available}")
            
            # WORKAROUND: If CUDA is explicitly requested and we have GPU devices, force CUDA usage
            # This bypasses any context-specific issues with torch.cuda.is_available()
            if cuda_available:
                logger.info(f"CUDA confirmed available, returning: {config_device}")
                return config_device
            else:
                # Check if we can detect CUDA devices through nvidia-smi or device count
                try:
                    device_count = torch.cuda.device_count()
                    logger.warning(f"CUDA device count: {device_count}")
                    if device_count > 0:
                        logger.info(f"CUDA devices detected ({device_count}), forcing CUDA despite torch.cuda.is_available()=False")
                        return config_device
                except Exception as e:
                    logger.warning(f"Error getting CUDA device count: {e}")
                
                logger.warning(f"CUDA requested but torch.cuda.is_available() returned False, falling back to CPU")
                return "cpu"
        except ImportError:
            logger.warning(f"CUDA requested but PyTorch not available, falling back to CPU")
            return "cpu"
    else:
        logger.info(f"Using specified device: {config_device}")
        return config_device


def get_device_id_for_transformers(device: str) -> int:
    """
    Convert device string to device ID for Transformers pipeline.
    
    Args:
        device: Device string ("cuda", "cpu", "mps", "cuda:0", etc.)
        
    Returns:
        Device ID for transformers pipeline (-1 for CPU, 0+ for GPU)
    """
    if device == "cpu":
        return -1
    elif device.startswith("cuda"):
        if ":" in device:
            try:
                return int(device.split(":")[1])
            except (ValueError, IndexError):
                return 0
        else:
            return 0
    elif device == "mps":
        return 0  # MPS uses device 0
    else:
        return -1  # Default to CPU for unknown devices


class Config:
    """
    Manages configuration settings for video processing.
    """
    
    DEFAULT_CONFIG = {
        # Scene detection settings
        'scenes': {
            'enabled': True,
            'threshold': 0.3,
            'method': 'ffmpeg',
            'generate_timeline': True
        },
        
        # Video cutting settings
        'min_clip_duration': 2.0,
        'max_duration': 300.0,
        'fade_duration': 0.5,
        'remove_silence': False,
        'silence_threshold': -30,
        
        # Workflow settings
        'workflow': {
            'auto_optimize': True,
            'skip_normalize_if_conform': True,
            'force_normalize': False
        },
        
        # Normalization settings
        'normalize': {
            'enabled': True,
            'target_width': 1280,
            'target_height': 720,
            'target_fps': 24,
            'codec': 'auto'
        },
        
        # Criteria settings
        'criteria': {
            'nsfw': {
                'enabled': True,
                'method': 'auto',
                'action': 'reject',
                'mode': 'high'
            },
            'face': {
                'enabled': True,
                'method': 'auto',
                'min_confidence': 0.6,
                'min_area_pct': 1.0
            },
            'gender': {
                'enabled': True,
                'method': 'auto',
                'filter': 'female',
                'min_confidence': 0.8
            },
            'pose': {
                'enabled': True,
                'max_pitch': 35,
                'max_yaw': 75,
                'max_roll': 35
            }
        },
        
        # LLM description settings
        'describe': {
            'enabled': True,
            'frames_per_clip': 5,
            'max_retries': 3,
            'fallback_on_error': 'skip'
        },
        
        # Output settings
        'output_codec': 'libx264',
        'audio_codec': 'aac',
        'output_quality': 'medium',
        
        # Processing settings
        'temp_dir': './temp',
        'parallel_processing': False,
        'max_workers': 4,
        'device': 'auto',
        'sample_rate': 0.1,
        'max_gap': 3.0
    }
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_data: Configuration dictionary or None for defaults
        """
        self._config = self.DEFAULT_CONFIG.copy()
        
        if config_data:
            self._update_nested(self._config, config_data)
    
    def _update_nested(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Update nested dictionary recursively.
        
        Args:
            base_dict: Dictionary to update
            update_dict: Updates to apply
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_nested(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'scenes.enabled')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, config_data: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            config_data: Dictionary of configuration updates
        """
        self._update_nested(self._config, config_data)
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML)
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.warning(f"Configuration file not found: {file_path}")
                return
            
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    try:
                        import yaml
                        config_data = yaml.safe_load(f)
                    except ImportError:
                        logger.error("PyYAML not installed, cannot load YAML config")
                        return
                elif path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {path.suffix}")
                    return
            
            self.update(config_data)
            logger.info(f"Loaded configuration from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file {file_path}: {str(e)}")
    
    def save_to_file(self, file_path: str, format_type: str = 'json') -> None:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration
            format_type: File format ('yaml' or 'json')
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                if format_type.lower() == 'yaml':
                    try:
                        import yaml
                        yaml.dump(self._config, f, default_flow_style=False)
                    except ImportError:
                        logger.error("PyYAML not installed, saving as JSON instead")
                        json.dump(self._config, f, indent=2)
                elif format_type.lower() == 'json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Saved configuration to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment."""
        self.set(key, value)
