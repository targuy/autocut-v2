# Platform Installation Scripts Permissions Setup

## Overview
This script ensures the installation scripts have proper permissions and can be executed on each platform.

## Windows (PowerShell)
```powershell
# Make scripts executable (already done by default on Windows)
# Set execution policy if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## macOS/Linux (Bash)
```bash
# Make scripts executable
chmod +x scripts/install-macos.sh
chmod +x scripts/install-linux.sh
chmod +x scripts/*.sh
```

## Testing the Scripts

### Local Testing (Development)
```bash
# Test each script with dry-run or help flags
scripts/install-windows.ps1 -WhatIf
scripts/install-macos.sh --help
scripts/install-linux.sh --help
```

### Remote Testing (End User)
```bash
# Test download and execution
curl -O https://raw.githubusercontent.com/targuy/autocut-v2/main/scripts/install-linux.sh
chmod +x install-linux.sh
./install-linux.sh --skip-system-deps
```

## Script Features Summary

### Windows (install-windows.ps1)
- ✅ Administrator privilege detection
- ✅ System requirement checks
- ✅ Conda/venv environment detection
- ✅ FFmpeg installation via Chocolatey
- ✅ GPU detection and PyTorch optimization
- ✅ VS Code integration hints
- ✅ Post-installation verification
- ✅ Activation script generation

### macOS (install-macos.sh)
- ✅ Xcode Command Line Tools check
- ✅ Homebrew installation and usage
- ✅ Apple Silicon (M1/M2) optimization
- ✅ FFmpeg installation
- ✅ Shell profile integration
- ✅ Environment activation scripts
- ✅ Desktop integration

### Linux (install-linux.sh)
- ✅ Distribution detection (Ubuntu/Debian/Fedora/Arch)
- ✅ System package manager integration
- ✅ NVIDIA/AMD GPU detection
- ✅ CUDA/ROCm support
- ✅ Desktop entry creation
- ✅ Shell alias creation
- ✅ Comprehensive dependency installation

## Error Handling
All scripts include:
- Command existence checking
- Error exit on failures
- Rollback capabilities
- Clear error messages
- Alternative fallback methods
