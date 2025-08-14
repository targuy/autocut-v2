# AutoCut v2 Installation Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Manual Installation](#manual-installation)
4. [Platform-Specific Instructions](#platform-specific-instructions)
5. [Post-Installation Setup](#post-installation-setup)
6. [Troubleshooting](#troubleshooting)
7. [Verification](#verification)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **Internet**: Stable connection for downloads (~500MB)

### Recommended Requirements
- **CPU**: Multi-core processor (8+ cores)
- **GPU**: NVIDIA RTX series or equivalent with 8GB+ VRAM
- **RAM**: 16GB or more
- **Storage**: SSD for better performance

### Required System Software
- **FFmpeg**: Video processing library
- **Git**: Version control (recommended)
- **Python Package Manager**: pip (included with Python)

## Quick Installation

### Windows
```powershell
# Download and run the installer
irm https://raw.githubusercontent.com/targuy/autocut-v2/main/scripts/install-windows.ps1 | iex
```

### macOS
```bash
# Download and run the installer
curl -sSL https://raw.githubusercontent.com/targuy/autocut-v2/main/scripts/install-macos.sh | bash
```

### Linux
```bash
# Download and run the installer
curl -sSL https://raw.githubusercontent.com/targuy/autocut-v2/main/scripts/install-linux.sh | bash
```

## Manual Installation

If you prefer manual installation or need to customize the setup:

### 1. Prerequisites

#### Install Python 3.8+
- **Windows**: Download from [python.org](https://python.org)
- **macOS**: `brew install python@3.10`
- **Linux**: `sudo apt install python3 python3-pip` (Ubuntu/Debian)

#### Install FFmpeg
- **Windows**: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian)

#### Install Git (Optional but recommended)
- **Windows**: Download from [git-scm.com](https://git-scm.com)
- **macOS**: `brew install git`
- **Linux**: `sudo apt install git`

### 2. Download AutoCut v2

#### Using Git
```bash
git clone https://github.com/targuy/autocut-v2.git
cd autocut-v2
```

#### Using ZIP Download
```bash
# Download and extract
curl -L https://github.com/targuy/autocut-v2/archive/refs/heads/main.zip -o autocut-v2.zip
unzip autocut-v2.zip
cd autocut-v2-main
```

### 3. Setup Python Environment

#### Option A: Using Conda (Recommended)
```bash
# Create environment
conda create -n autocut-v2 python=3.10 -y
conda activate autocut-v2

# Install PyTorch (adjust for your system)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install -e .
```

#### Option B: Using venv
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### 4. Configuration Setup
```bash
# Copy configuration template
cp config_template.yml config.yml

# Create necessary directories
mkdir -p output temp cache metrics
```

## Platform-Specific Instructions

### Windows Specific

#### PowerShell Execution Policy
```powershell
# Allow script execution (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Visual Studio Code Integration
1. Install Python extension for VS Code
2. Open project folder in VS Code
3. Select Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Choose the AutoCut v2 environment

#### Chocolatey Package Manager (Optional)
```powershell
# Install Chocolatey for easier dependency management
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### macOS Specific

#### Homebrew Package Manager
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Xcode Command Line Tools
```bash
# Install development tools
xcode-select --install
```

#### Apple Silicon (M1/M2) Considerations
- PyTorch with MPS support is automatically selected
- Some dependencies may need Rosetta 2 for compatibility

### Linux Specific

#### Ubuntu/Debian Dependencies
```bash
# System dependencies
sudo apt update
sudo apt install -y python3-dev python3-venv build-essential
sudo apt install -y libopencv-dev libavcodec-dev libavformat-dev
sudo apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0
```

#### NVIDIA GPU Support
```bash
# Install NVIDIA drivers and CUDA toolkit
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
```

#### AMD GPU Support (ROCm)
```bash
# Add ROCm repository and install
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update && sudo apt install rocm-dev
```

## Post-Installation Setup

### 1. Environment Verification
```bash
# Test Python environment
python -c "import autocut_v2; print('✓ AutoCut v2 imported successfully')"

# Check PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run basic functionality test
python test_basic.py
```

### 2. Configuration Customization

Edit `config.yml` to match your setup:

```yaml
# Basic settings
input_video: "/path/to/your/videos"
output_dir: "./output"
device: "auto"  # or "cuda:0", "cpu", "mps"

# Performance tuning
num_workers: 4  # Adjust based on your CPU cores
```

### 3. Shell Integration (Optional)

#### Bash/Zsh
```bash
# Add alias to shell profile
echo 'alias autocut="cd /path/to/autocut-v2 && source activate.sh"' >> ~/.bashrc
source ~/.bashrc
```

#### Windows PowerShell
```powershell
# Add to PowerShell profile
echo 'function autocut { Set-Location "C:\path\to\autocut-v2"; .\activate.ps1 }' >> $PROFILE
```

## Troubleshooting

### Common Installation Issues

#### 1. Python Version Conflicts
```bash
# Check Python version
python --version
# Should be 3.8 or higher

# If using multiple Python versions, be explicit
python3.10 -m pip install -e .
```

#### 2. PyTorch Installation Failures
```bash
# Clear pip cache
pip cache purge

# Install CPU-only version if GPU issues
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 3. FFmpeg Not Found
```bash
# Check if FFmpeg is in PATH
ffmpeg -version

# Add to PATH or install system-wide
export PATH="/path/to/ffmpeg/bin:$PATH"
```

#### 4. Permission Errors (Linux/macOS)
```bash
# Fix permissions
chmod +x scripts/*.sh
sudo chown -R $USER:$USER /path/to/autocut-v2
```

#### 5. CUDA Version Mismatches
```bash
# Check NVIDIA driver version
nvidia-smi

# Install compatible PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error-Specific Solutions

#### ModuleNotFoundError
```bash
# Ensure environment is activated
# Reinstall in development mode
pip install -e .
```

#### OutOfMemoryError
```yaml
# Reduce workers in config.yml
num_workers: 1
# Reduce video resolution
normalize:
  target_width: 854
  target_height: 480
```

#### FFmpeg Codec Errors
```bash
# Install additional codecs (Linux)
sudo apt install ubuntu-restricted-extras

# Use different codec in config
video:
  codec: "libx264"
```

### Getting Help

1. **Check logs**: Look in `autocut.log` for detailed error messages
2. **Enable debug mode**: Set `logging.level: "DEBUG"` in config.yml
3. **Run tests**: Execute `python test_basic.py` to isolate issues
4. **Community support**: Create an issue on GitHub with full error details

## Verification

### Quick Test
```bash
# Basic import test
python -c "import autocut_v2; print('✅ Installation successful')"
```

### Comprehensive Test
```bash
# Run all basic tests
python test_basic.py

# Expected output:
# ✓ Configuration tests passed
# ✓ Workflow optimizer tests passed  
# ✓ File handler tests passed
# ✓ AutoCut processor tests passed
```

### Performance Benchmark
```bash
# Test with sample video (if available)
python -m autocut_v2 sample_video.mp4 --dry-run

# Check GPU utilization
nvidia-smi  # NVIDIA
rocm-smi    # AMD
```

### Environment Information
```bash
# Generate environment report
python -c "
import torch
import cv2
import autocut_v2
print(f'AutoCut v2: {autocut_v2.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
"
```

## Next Steps

After successful installation:

1. **Review configuration**: Customize `config.yml` for your needs
2. **Test with sample video**: Process a short test video
3. **Explore profiles**: Try different processing profiles
4. **Read documentation**: Check `docs/` folder for advanced usage
5. **Join community**: Follow project updates and discussions

For detailed usage instructions, see the main [README.md](../README.md) file.
