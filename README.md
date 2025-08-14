# AutoCut v2

**Intelligent Video Processing and Cutting Tool**

AutoCut v2 is a Python-based video editing automation tool that provides automated video cutting, scene detection, content analysis, and batch processing capabilities using advanced AI models.

## 🚀 Quick Installation

Choose your platform and run the appropriate installation script:

### Windows
```powershell
# Download and run the Windows installer
curl -O https://raw.githubusercontent.com/targuy/autocut-v2/main/scripts/install-windows.ps1
powershell -ExecutionPolicy Bypass -File install-windows.ps1
```

### macOS
```bash
# Download and run the macOS installer
curl -O https://raw.githubusercontent.com/targuy/autocut-v2/main/scripts/install-macos.sh
chmod +x install-macos.sh
./install-macos.sh
```

### Linux
```bash
# Download and run the Linux installer
curl -O https://raw.githubusercontent.com/targuy/autocut-v2/main/scripts/install-linux.sh
chmod +x install-linux.sh
./install-linux.sh
```

## 📋 System Requirements

- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space for dependencies and temp files
- **GPU**: CUDA-compatible GPU recommended for faster processing
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)

## 🛠️ Manual Installation

If you prefer manual installation or need to customize the setup:

### Prerequisites
1. Install Python 3.8+ and conda/miniconda
2. Install FFmpeg (required for video processing)
3. Clone this repository

### Step-by-step Installation
```bash
# Clone the repository
git clone https://github.com/targuy/autocut-v2.git
cd autocut-v2

# Create and activate virtual environment
conda create -n autocut-v2 python=3.10
conda activate autocut-v2

# Install dependencies
pip install -e .

# Copy configuration template
cp config_template.yml config.yml
```

## ⚙️ Configuration Setup

To set up your configuration file for AutoCut v2, follow these steps:

1. **Locate the `config_template.yml` in the root directory.**  
   This file serves as the starting point for your configuration.

2. **Copy and rename it to `config.yml`.**  
   You can do this using the command line or your file explorer:
   ```bash
   cp config_template.yml config.yml
   ```

3. **Customize fields based on your needs.**  
   - **Input/Output Settings:**  
     - `input_video`: Path to your input video file or directory
     - `output_dir`: Directory where processed videos will be saved
   
   - **Execution Settings:**  
     - `device`: auto|cuda:0|mps|cpu (auto-detection recommended)
     - `num_workers`: Number of parallel processing threads
   
   - **Processing Criteria:**  
     - `nsfw`: Content filtering options
     - `face`: Face detection and analysis
     - `gender`: Gender-based filtering
     - `pose`: Head pose analysis
     - `visibility`: Face visibility requirements

4. **Save it in the root directory.**  
   Ensure that your `config.yml` file is saved in the root of your project directory.

## 🎯 Usage Examples

### Command Line Interface
```bash
# Basic usage
python -m autocut_v2 path/to/video.mp4

# With custom config
python -m autocut_v2 path/to/video.mp4 --config custom_config.yml

# Using predefined profiles
python -m autocut_v2 path/to/video.mp4 --profile face_focus

# Dry run (simulation mode)
python -m autocut_v2 path/to/video.mp4 --dry-run
```

### Python API
```python
from autocut_v2 import AutoCut

# Initialize with default config
processor = AutoCut()

# Process a single video
result = processor.process_video("path/to/video.mp4")

# Process with custom configuration
config = {"criteria": {"face": {"enabled": True}}}
processor = AutoCut(config)
result = processor.process_video("path/to/video.mp4")
```

## 🧪 Testing the Installation

Run the basic functionality test:
```bash
python test_basic.py
```

Expected output:
```
✓ Configuration tests passed
✓ Workflow optimizer tests passed  
✓ File handler tests passed
✓ AutoCut processor tests passed
✓ All tests passed! AutoCut v2 basic functionality is working.
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA/GPU Issues**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force CPU mode if GPU issues
# Set device: "cpu" in config.yml
```

**2. FFmpeg Not Found**
```bash
# Windows (with chocolatey)
choco install ffmpeg

# macOS (with homebrew)
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt update && sudo apt install ffmpeg
```

**3. Memory Issues**
- Reduce `num_workers` in config.yml
- Lower video resolution in normalization settings
- Enable `cleanup_temp: true` in config.yml

**4. Dependency Conflicts**
```bash
# Reset environment
conda env remove -n autocut-v2
# Re-run installation script
```

## 📊 Performance Optimization

### Hardware Recommendations
- **CPU**: Multi-core processor (8+ cores recommended)
- **GPU**: NVIDIA RTX series or equivalent with 8GB+ VRAM
- **RAM**: 16GB+ for processing large videos
- **Storage**: SSD for better I/O performance

### Configuration Tuning
```yaml
# For faster processing (lower quality)
normalize:
  target_width: 854
  target_height: 480
  target_fps: 15

# For better quality (slower processing)  
normalize:
  target_width: 1920
  target_height: 1080
  target_fps: 30
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV team for computer vision capabilities
- PyTorch team for deep learning framework
- MoviePy team for video processing utilities
- Hugging Face for transformer models
- Ultralytics for YOLO models

## 📞 Support

- 📧 Create an issue on GitHub
- 📖 Check the [documentation](docs/)
- 💬 Join our community discussions