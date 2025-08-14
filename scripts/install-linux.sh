#!/bin/bash
# AutoCut v2 Linux Installation Script
# Bash script for Ubuntu/Debian and other Linux distributions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
SKIP_CONDA=false
USE_VENV=false
PYTHON_VERSION="3.10"
INSTALL_SYSTEM_DEPS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        --use-venv)
            USE_VENV=true
            shift
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --skip-system-deps)
            INSTALL_SYSTEM_DEPS=false
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}🎬 AutoCut v2 Linux Installation Script${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    elif command_exists lsb_release; then
        DISTRO=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
        VERSION=$(lsb_release -sr)
    else
        DISTRO="unknown"
        VERSION="unknown"
    fi
}

detect_distro
echo -e "${GREEN}✓ Detected OS: $DISTRO $VERSION${NC}"

# Install system dependencies
if [ "$INSTALL_SYSTEM_DEPS" = true ]; then
    echo -e "${CYAN}📦 Installing system dependencies...${NC}"
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y python3 python3-pip python3-venv python3-dev
            sudo apt install -y git curl wget build-essential
            sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev
            sudo apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0
            
            # Install additional dependencies for OpenCV
            sudo apt install -y libopencv-dev python3-opencv
            ;;
        fedora|centos|rhel)
            if command_exists dnf; then
                sudo dnf update -y
                sudo dnf install -y python3 python3-pip python3-devel
                sudo dnf install -y git curl wget gcc gcc-c++ make
                sudo dnf install -y ffmpeg ffmpeg-devel
                sudo dnf install -y opencv opencv-devel python3-opencv
            elif command_exists yum; then
                sudo yum update -y
                sudo yum install -y python3 python3-pip python3-devel
                sudo yum install -y git curl wget gcc gcc-c++ make
                sudo yum install -y epel-release
                sudo yum install -y ffmpeg ffmpeg-devel
            fi
            ;;
        arch|manjaro)
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm python python-pip git curl wget base-devel
            sudo pacman -S --noconfirm ffmpeg opencv python-opencv
            ;;
        *)
            echo -e "${YELLOW}⚠️ Unknown distribution. Please install dependencies manually:${NC}"
            echo "  - Python 3.8+"
            echo "  - Git, curl, wget"
            echo "  - FFmpeg"
            echo "  - Build tools (gcc, make, etc.)"
            echo "  - OpenCV development libraries"
            ;;
    esac
fi

# Check system requirements
echo -e "${CYAN}📋 Checking system requirements...${NC}"

# Check Python
if command_exists python3; then
    PYTHON_VER=$(python3 --version)
    echo -e "${GREEN}✓ Python found: $PYTHON_VER${NC}"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command_exists python; then
    PYTHON_VER=$(python --version)
    echo -e "${GREEN}✓ Python found: $PYTHON_VER${NC}"
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo -e "${RED}❌ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check Git
if command_exists git; then
    echo -e "${GREEN}✓ Git found${NC}"
else
    echo -e "${RED}❌ Git not found. Please install Git${NC}"
    exit 1
fi

# Check FFmpeg
if command_exists ffmpeg; then
    echo -e "${GREEN}✓ FFmpeg found${NC}"
else
    echo -e "${YELLOW}⚠️ FFmpeg not found. Some features may not work${NC}"
fi

# Check for GPU support
if command_exists nvidia-smi; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    GPU_SUPPORT="cuda"
elif command_exists rocm-smi; then
    echo -e "${GREEN}✓ AMD GPU detected${NC}"
    GPU_SUPPORT="rocm"
else
    echo -e "${YELLOW}⚠️ No GPU detected. Using CPU mode${NC}"
    GPU_SUPPORT="cpu"
fi

# Create project directory
PROJECT_DIR="$HOME/autocut-v2"
echo -e "${CYAN}📁 Creating project directory: $PROJECT_DIR${NC}"

if [ -d "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}⚠️ Directory already exists. Removing...${NC}"
    rm -rf "$PROJECT_DIR"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Download project
echo -e "${CYAN}⬇️ Downloading AutoCut v2...${NC}"

git clone https://github.com/targuy/autocut-v2.git .
echo -e "${GREEN}✓ Project downloaded${NC}"

# Setup Python environment
echo -e "${CYAN}🐍 Setting up Python environment...${NC}"

if [ "$SKIP_CONDA" = false ] && command_exists conda; then
    echo -e "${YELLOW}Using Conda environment...${NC}"
    
    # Remove existing environment if it exists
    conda env remove -n autocut-v2 -y 2>/dev/null || true
    
    # Create new environment
    conda create -n autocut-v2 python=$PYTHON_VERSION -y
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate autocut-v2
    
    echo -e "${YELLOW}Installing core dependencies...${NC}"
    
    # Install PyTorch with appropriate backend
    case $GPU_SUPPORT in
        cuda)
            conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
            ;;
        rocm)
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
            ;;
        *)
            conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
            ;;
    esac
    
    pip install opencv-python moviepy pillow click pyyaml ffmpeg-python numpy
    pip install ultralytics transformers huggingface-hub
    pip install scenedetect rich
    
    # Install development dependencies
    pip install pytest pytest-cov black flake8 mypy
    
    # Install project in development mode
    pip install -e .
    
elif [ "$USE_VENV" = true ] || ! command_exists conda; then
    echo -e "${YELLOW}Using Python venv...${NC}"
    
    # Create virtual environment
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    echo -e "${YELLOW}Installing dependencies...${NC}"
    
    # Install PyTorch with appropriate backend
    case $GPU_SUPPORT in
        cuda)
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            ;;
        rocm)
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
            ;;
        *)
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
    
    pip install opencv-python moviepy pillow click pyyaml ffmpeg-python numpy
    pip install ultralytics transformers huggingface-hub
    pip install scenedetect rich
    
    # Install development dependencies
    pip install pytest pytest-cov black flake8 mypy
    
    # Install project in development mode
    pip install -e .
fi

# Setup configuration
echo -e "${CYAN}⚙️ Setting up configuration...${NC}"

if [ -f "config_template.yml" ]; then
    cp config_template.yml config.yml
    
    # Update config for detected GPU
    if [ "$GPU_SUPPORT" = "cuda" ]; then
        sed -i 's/device: "auto"/device: "cuda:0"/' config.yml
    elif [ "$GPU_SUPPORT" = "rocm" ]; then
        sed -i 's/device: "auto"/device: "cuda:0"/' config.yml  # ROCm uses CUDA API
    else
        sed -i 's/device: "auto"/device: "cpu"/' config.yml
    fi
    
    echo -e "${GREEN}✓ Configuration file created${NC}"
else
    echo -e "${YELLOW}⚠️ config_template.yml not found${NC}"
fi

# Create necessary directories
mkdir -p output temp cache metrics
echo -e "${GREEN}✓ Project directories created${NC}"

# Set proper permissions
chmod +x scripts/*.sh 2>/dev/null || true

# Test installation
echo -e "${CYAN}🧪 Testing installation...${NC}"

if [ -f "test_basic.py" ]; then
    if $PYTHON_CMD test_basic.py; then
        echo -e "${GREEN}✓ Installation test passed!${NC}"
    else
        echo -e "${YELLOW}⚠️ Installation test failed${NC}"
    fi
else
    # Basic import test
    if $PYTHON_CMD -c "import autocut_v2; print('✓ AutoCut v2 imported successfully')"; then
        echo -e "${GREEN}✓ Basic import test passed!${NC}"
    else
        echo -e "${YELLOW}⚠️ Basic import test failed${NC}"
    fi
fi

# Create activation script
cat > activate.sh << 'EOF'
#!/bin/bash
# AutoCut v2 Activation Script
# Run this script to activate the environment and start working

if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate autocut-v2
    echo "✓ Conda environment activated"
else
    echo "⚠️ No environment found. Please check installation."
fi

echo ""
echo "🎬 AutoCut v2 Ready!"
echo "Usage examples:"
echo "  python -m autocut_v2 --help"
echo "  python -m autocut_v2 path/to/video.mp4"
echo "  python test_basic.py"
EOF

chmod +x activate.sh

# Update shell profile
SHELL_PROFILE=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -f "$HOME/.profile" ]; then
    SHELL_PROFILE="$HOME/.profile"
fi

if [ -n "$SHELL_PROFILE" ]; then
    echo "" >> "$SHELL_PROFILE"
    echo "# AutoCut v2 alias" >> "$SHELL_PROFILE"
    echo "alias autocut='cd $PROJECT_DIR && source activate.sh'" >> "$SHELL_PROFILE"
fi

# Create desktop entry (optional)
if [ -d "$HOME/.local/share/applications" ]; then
    cat > "$HOME/.local/share/applications/autocut-v2.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=AutoCut v2
Comment=Intelligent Video Processing and Cutting Tool
Exec=gnome-terminal --working-directory=$PROJECT_DIR -e "bash -c 'source activate.sh; bash'"
Icon=video-x-generic
Terminal=true
Categories=AudioVideo;Video;
EOF
    chmod +x "$HOME/.local/share/applications/autocut-v2.desktop"
    echo -e "${GREEN}✓ Desktop entry created${NC}"
fi

# Final message
echo ""
echo -e "${GREEN}🎉 Installation completed successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${CYAN}📁 Project directory: $PROJECT_DIR${NC}"
echo -e "${CYAN}🖥️ GPU support: $GPU_SUPPORT${NC}"
echo ""
echo -e "${YELLOW}🚀 To get started:${NC}"
echo "  1. cd \"$PROJECT_DIR\""
echo "  2. source activate.sh"
echo "  3. Edit config.yml to set your video paths"
echo "  4. python -m autocut_v2 path/to/your/video.mp4"
echo ""
echo -e "${YELLOW}💡 Pro tip: You can now use 'autocut' command from anywhere to start working${NC}"
echo ""
echo -e "${CYAN}📖 For more information, see README.md${NC}"
