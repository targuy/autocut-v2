# AutoCut v2 Windows Installation Script
# PowerShell script for Windows 10/11

param(
    [switch]$SkipConda,
    [switch]$UseVenv,
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

Write-Host "🎬 AutoCut v2 Windows Installation Script" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Warning "Running without administrator privileges. Some features may not work optimally."
    Write-Host "For best experience, run PowerShell as Administrator" -ForegroundColor Yellow
    Write-Host ""
}

# Function to check if command exists
function Test-Command($cmdname) {
    try {
        Get-Command $cmdname -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Check system requirements
Write-Host "📋 Checking system requirements..." -ForegroundColor Cyan

# Check Python
if (Test-Command python) {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Error "❌ Python not found. Please install Python 3.8+ from https://python.org"
    exit 1
}

# Check Git
if (Test-Command git) {
    Write-Host "✓ Git found" -ForegroundColor Green
} else {
    Write-Warning "⚠️ Git not found. Will download as ZIP instead"
}

# Check FFmpeg
if (Test-Command ffmpeg) {
    Write-Host "✓ FFmpeg found" -ForegroundColor Green
} else {
    Write-Warning "⚠️ FFmpeg not found. Installing via chocolatey..."
    if (Test-Command choco) {
        try {
            choco install ffmpeg -y
            Write-Host "✓ FFmpeg installed" -ForegroundColor Green
        } catch {
            Write-Warning "Failed to install FFmpeg via chocolatey. Please install manually from https://ffmpeg.org"
        }
    } else {
        Write-Warning "Chocolatey not found. Please install FFmpeg manually from https://ffmpeg.org"
    }
}

# Create project directory
$projectDir = Join-Path $env:USERPROFILE "autocut-v2"
Write-Host "📁 Creating project directory: $projectDir" -ForegroundColor Cyan

if (Test-Path $projectDir) {
    Write-Host "⚠️ Directory already exists. Removing..." -ForegroundColor Yellow
    Remove-Item $projectDir -Recurse -Force
}

New-Item -Path $projectDir -ItemType Directory | Out-Null
Set-Location $projectDir

# Download project
Write-Host "⬇️ Downloading AutoCut v2..." -ForegroundColor Cyan

if (Test-Command git) {
    git clone https://github.com/targuy/autocut-v2.git .
} else {
    # Download as ZIP
    $zipUrl = "https://github.com/targuy/autocut-v2/archive/refs/heads/main.zip"
    $zipFile = "autocut-v2.zip"
    
    try {
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipFile
        Expand-Archive -Path $zipFile -DestinationPath . -Force
        
        # Move files from extracted folder
        $extractedFolder = "autocut-v2-main"
        if (Test-Path $extractedFolder) {
            Get-ChildItem $extractedFolder | Move-Item -Destination . -Force
            Remove-Item $extractedFolder -Recurse -Force
        }
        Remove-Item $zipFile
        
        Write-Host "✓ Project downloaded" -ForegroundColor Green
    } catch {
        Write-Error "❌ Failed to download project: $_"
        exit 1
    }
}

# Setup Python environment
Write-Host "🐍 Setting up Python environment..." -ForegroundColor Cyan

if (-not $SkipConda -and (Test-Command conda)) {
    Write-Host "Using Conda environment..." -ForegroundColor Yellow
    
    # Remove existing environment if it exists
    conda env remove -n autocut-v2 -y 2>$null
    
    # Create new environment
    conda create -n autocut-v2 python=$PythonVersion -y
    
    # Activate environment and install dependencies
    conda activate autocut-v2
    
    Write-Host "Installing core dependencies..." -ForegroundColor Yellow
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    pip install opencv-python moviepy pillow click pyyaml ffmpeg-python numpy
    pip install ultralytics transformers huggingface-hub
    pip install scenedetect rich
    
    # Install development dependencies
    pip install pytest pytest-cov black flake8 mypy
    
    # Install project in development mode
    pip install -e .
    
} elseif ($UseVenv -or -not (Test-Command conda)) {
    Write-Host "Using Python venv..." -ForegroundColor Yellow
    
    # Create virtual environment
    python -m venv venv
    
    # Activate virtual environment
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    } else {
        Write-Error "❌ Failed to create virtual environment"
        exit 1
    }
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install dependencies
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install opencv-python moviepy pillow click pyyaml ffmpeg-python numpy
    pip install ultralytics transformers huggingface-hub
    pip install scenedetect rich
    
    # Install development dependencies
    pip install pytest pytest-cov black flake8 mypy
    
    # Install project in development mode
    pip install -e .
}

# Setup configuration
Write-Host "⚙️ Setting up configuration..." -ForegroundColor Cyan

if (Test-Path "config_template.yml") {
    Copy-Item "config_template.yml" "config.yml"
    Write-Host "✓ Configuration file created" -ForegroundColor Green
} else {
    Write-Warning "⚠️ config_template.yml not found"
}

# Create necessary directories
New-Item -Path "output" -ItemType Directory -Force | Out-Null
New-Item -Path "temp" -ItemType Directory -Force | Out-Null
New-Item -Path "cache" -ItemType Directory -Force | Out-Null
New-Item -Path "metrics" -ItemType Directory -Force | Out-Null

Write-Host "✓ Project directories created" -ForegroundColor Green

# Test installation
Write-Host "🧪 Testing installation..." -ForegroundColor Cyan

try {
    if (Test-Path "test_basic.py") {
        python test_basic.py
        Write-Host "✓ Installation test passed!" -ForegroundColor Green
    } else {
        # Basic import test
        python -c "import autocut_v2; print('✓ AutoCut v2 imported successfully')"
        Write-Host "✓ Basic import test passed!" -ForegroundColor Green
    }
} catch {
    Write-Warning "⚠️ Installation test failed: $_"
    Write-Host "You may need to manually verify the installation" -ForegroundColor Yellow
}

# Create activation script
$activateScript = @"
# AutoCut v2 Activation Script
# Run this script to activate the environment and start working

if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} elseif (Get-Command conda -ErrorAction SilentlyContinue) {
    conda activate autocut-v2
    Write-Host "✓ Conda environment activated" -ForegroundColor Green
} else {
    Write-Warning "No environment found. Please check installation."
}

Write-Host ""
Write-Host "🎬 AutoCut v2 Ready!" -ForegroundColor Green
Write-Host "Usage examples:" -ForegroundColor Cyan
Write-Host "  python -m autocut_v2 --help"
Write-Host "  python -m autocut_v2 path/to/video.mp4"
Write-Host "  python test_basic.py"
"@

$activateScript | Out-File -FilePath "activate.ps1" -Encoding UTF8

# Final message
Write-Host ""
Write-Host "🎉 Installation completed successfully!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "📁 Project directory: $projectDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 To get started:" -ForegroundColor Yellow
Write-Host "  1. cd `"$projectDir`""
Write-Host "  2. .\activate.ps1"
Write-Host "  3. Edit config.yml to set your video paths"
Write-Host "  4. python -m autocut_v2 path/to/your/video.mp4"
Write-Host ""
Write-Host "📖 For more information, see README.md" -ForegroundColor Cyan
