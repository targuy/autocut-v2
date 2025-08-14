#!/bin/bash
# AutoCut v2 macOS Installation Script
# Bash script for macOS 10.15+

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
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}🎬 AutoCut v2 macOS Installation Script${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
echo -e "${CYAN}📋 Checking system requirements...${NC}"

# Check macOS version
MACOS_VERSION=$(sw_vers -productVersion)
echo -e "${GREEN}✓ macOS version: $MACOS_VERSION${NC}"

# Check if Xcode Command Line Tools are installed
if xcode-select -p >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Xcode Command Line Tools found${NC}"
else
    echo -e "${YELLOW}⚠️ Installing Xcode Command Line Tools...${NC}"
    xcode-select --install
    echo "Please complete the Xcode Command Line Tools installation and re-run this script."
    exit 1
fi

# Check Python
if command_exists python3; then
    PYTHON_VER=$(python3 --version)
    echo -e "${GREEN}✓ Python found: $PYTHON_VER${NC}"
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_VER=$(python --version)
    echo -e "${GREEN}✓ Python found: $PYTHON_VER${NC}"
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python not found. Installing via Homebrew...${NC}"
    if command_exists brew; then
        brew install python@3.10
        PYTHON_CMD="python3"
    else
        echo -e "${RED}❌ Homebrew not found. Please install Python 3.8+ manually.${NC}"
        exit 1
    fi
fi

# Check Git
if command_exists git; then
    echo -e "${GREEN}✓ Git found${NC}"
else
    echo -e "${YELLOW}⚠️ Git not found. Installing via Homebrew...${NC}"
    if command_exists brew; then
        brew install git
    else
        echo -e "${RED}❌ Homebrew not found. Please install Git manually.${NC}"
        exit 1
    fi
fi

# Check/Install Homebrew if needed
if ! command_exists brew; then
    echo -e "${YELLOW}⚠️ Homebrew not found. Installing...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for M1 Macs
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# Check/Install FFmpeg
if command_exists ffmpeg; then
    echo -e "${GREEN}✓ FFmpeg found${NC}"
else
    echo -e "${YELLOW}⚠️ FFmpeg not found. Installing via Homebrew...${NC}"
    brew install ffmpeg
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

if command_exists git; then
    git clone https://github.com/targuy/autocut-v2.git .
else
    # Download as ZIP
    curl -L https://github.com/targuy/autocut-v2/archive/refs/heads/main.zip -o autocut-v2.zip
    unzip -q autocut-v2.zip
    mv autocut-v2-main/* .
    rm -rf autocut-v2-main autocut-v2.zip
fi

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
    
    # Install PyTorch with appropriate backend for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        # Apple Silicon (M1/M2)
        conda install pytorch torchvision torchaudio -c pytorch -y
    else
        # Intel Mac
        conda install pytorch torchvision torchaudio -c pytorch -y
    fi
    
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
    if [[ $(uname -m) == "arm64" ]]; then
        # Apple Silicon (M1/M2) - use CPU version as MPS support varies
        pip install torch torchvision torchaudio
    else
        # Intel Mac
        pip install torch torchvision torchaudio
    fi
    
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
    echo -e "${GREEN}✓ Configuration file created${NC}"
else
    echo -e "${YELLOW}⚠️ config_template.yml not found${NC}"
fi

# Create necessary directories
mkdir -p output temp cache metrics
echo -e "${GREEN}✓ Project directories created${NC}"

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
if [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_PROFILE="$HOME/.bash_profile"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
fi

if [ -n "$SHELL_PROFILE" ]; then
    echo "# AutoCut v2 alias" >> "$SHELL_PROFILE"
    echo "alias autocut='cd $PROJECT_DIR && source activate.sh'" >> "$SHELL_PROFILE"
fi

# Final message
echo ""
echo -e "${GREEN}🎉 Installation completed successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${CYAN}📁 Project directory: $PROJECT_DIR${NC}"
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
