# AutoCut v2 Launch Script
# PowerShell script to easily launch AutoCut v2

param(
    [string]$VideoPath = "",  # Optional: override video from config.yml
    [string]$Config = "config.yml",
    [string]$ProcessingProfile = "",
    [string]$Device = "",
    [string]$Output = "",
    [switch]$DryRun,
    [switch]$Help
)

# Colors for output
$Green = "Green"
$Yellow = "Yellow"
$Red = "Red"
$Cyan = "Cyan"

if ($Help) {
    Write-Host "AutoCut v2 Launch Script" -ForegroundColor $Green
    Write-Host "=========================" -ForegroundColor $Green
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor $Cyan
    Write-Host "  .\launch.ps1                    # Use video from config.yml" -ForegroundColor $Yellow
    Write-Host "  .\launch.ps1 -VideoPath 'path\to\video.mp4'  # Override video from config" -ForegroundColor $Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor $Cyan
    Write-Host "  -VideoPath   : Path to video file (optional, overrides config.yml)"
    Write-Host "  -Config      : Configuration file (default: config.yml)"
    Write-Host "  -ProcessingProfile : Processing profile (safe_content|face_focus|custom)"
    Write-Host "  -Device      : Processing device (auto|cuda:0|cpu|mps)"
    Write-Host "  -Output      : Output directory (overrides config.yml)"
    Write-Host "  -DryRun      : Simulation mode without actual processing"
    Write-Host "  -Help        : Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor $Cyan
    Write-Host "  .\launch.ps1                    # Process video defined in config.yml"
    Write-Host "  .\launch.ps1 -DryRun           # Dry run with config.yml video"
    Write-Host "  .\launch.ps1 -VideoPath 'T:\other\video.mp4'"
    Write-Host "  .\launch.ps1 -ProcessingProfile face_focus -Device cuda:0"
    exit 0
}

Write-Host "🎬 AutoCut v2 Launcher" -ForegroundColor $Green
Write-Host "======================" -ForegroundColor $Green

# Check if video file exists or use config default
$ConfigOutputDir = ""
if ($VideoPath -eq "") {
    # Try to read video path from config file
    if (Test-Path $Config) {
        try {
            $configContent = Get-Content $Config -Raw
            if ($configContent -match 'input_video:\s*"([^"]+)"') {
                $VideoPath = $matches[1]
                Write-Host "📹 Using video from config: $VideoPath" -ForegroundColor $Cyan
            } elseif ($configContent -match "input_video:\s*'([^']+)'") {
                $VideoPath = $matches[1]
                Write-Host "📹 Using video from config: $VideoPath" -ForegroundColor $Cyan
            } elseif ($configContent -match 'input_video:\s*([^\s#]+)') {
                $VideoPath = $matches[1]
                Write-Host "📹 Using video from config: $VideoPath" -ForegroundColor $Cyan
            } else {
                Write-Host "❌ Error: No input_video found in config.yml and no -VideoPath provided" -ForegroundColor $Red
                Write-Host "Please either:" -ForegroundColor $Yellow
                Write-Host "  1. Set input_video in config.yml, or" -ForegroundColor $Yellow
                Write-Host "  2. Use -VideoPath parameter" -ForegroundColor $Yellow
                exit 1
            }
            
            # Also read output directory from config
            if ($configContent -match 'output_dir:\s*"([^"]+)"') {
                $ConfigOutputDir = $matches[1]
                Write-Host "📁 Output directory from config: $ConfigOutputDir" -ForegroundColor $Cyan
            } elseif ($configContent -match "output_dir:\s*'([^']+)'") {
                $ConfigOutputDir = $matches[1]
                Write-Host "📁 Output directory from config: $ConfigOutputDir" -ForegroundColor $Cyan
            } elseif ($configContent -match 'output_dir:\s*([^\s#]+)') {
                $ConfigOutputDir = $matches[1]
                Write-Host "📁 Output directory from config: $ConfigOutputDir" -ForegroundColor $Cyan
            }
        } catch {
            Write-Host "❌ Error reading config file: $_" -ForegroundColor $Red
            exit 1
        }
    } else {
        Write-Host "❌ Error: Config file not found and no -VideoPath provided" -ForegroundColor $Red
        exit 1
    }
} else {
    Write-Host "📹 Using video from parameter: $VideoPath" -ForegroundColor $Cyan
}

if (-not (Test-Path $VideoPath)) {
    Write-Host "❌ Error: Video file not found: $VideoPath" -ForegroundColor $Red
    exit 1
}

# Check if config exists
if (-not (Test-Path $Config)) {
    Write-Host "❌ Error: Config file not found: $Config" -ForegroundColor $Red
    exit 1
}

# Use output directory from config.yml if not overridden by parameter
$FinalOutput = $Output
if (-not $Output -and $ConfigOutputDir) {
    $FinalOutput = $ConfigOutputDir
}

if ($FinalOutput) {
    Write-Host "📁 Using output directory: $FinalOutput" -ForegroundColor $Cyan
}

if ($DryRun) {
    Write-Host "🧪 Running in DRY-RUN mode (simulation only)" -ForegroundColor $Yellow
}

# Check environment
Write-Host "🔍 Checking environment..." -ForegroundColor $Cyan

# First, try to detect if we're in the right conda environment
$condaEnvPath = "e:\DocumentsBenoit\pythonProject\autocut-v2\.conda"
$condaCommand = "C:/Users/benoi/miniforge3/condabin/conda.bat"

# Check if conda environment exists
$useCondaRun = $false
if (Test-Path $condaEnvPath) {
    Write-Host "✓ Conda environment found at: $condaEnvPath" -ForegroundColor $Green
    $useCondaRun = $true
    
    # Test the environment first
    try {
        $importTest = & "$condaCommand" run -p "$condaEnvPath" --no-capture-output python -c "import autocut_v2; print('✓ AutoCut v2 ready')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$importTest" -ForegroundColor $Green
        } else {
            Write-Host "❌ AutoCut v2 not properly installed in environment" -ForegroundColor $Red
            Write-Host "Error: $importTest" -ForegroundColor $Red
            Write-Host "Please run the installation again or check the environment" -ForegroundColor $Yellow
            exit 1
        }
    } catch {
        Write-Host "❌ Failed to test environment: $_" -ForegroundColor $Red
        exit 1
    }
} else {
    # Fallback: try regular python command
    Write-Host "⚠️ Conda environment not found, trying regular python..." -ForegroundColor $Yellow
    
    try {
        $importTest = python -c "import autocut_v2; print('✓ AutoCut v2 ready')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$importTest" -ForegroundColor $Green
        } else {
            Write-Host "❌ Environment not properly activated" -ForegroundColor $Red
            Write-Host "Please activate the conda environment first:" -ForegroundColor $Yellow
            Write-Host "`"C:/Users/benoi/miniforge3/condabin/conda.bat`" activate `"$condaEnvPath`"" -ForegroundColor $Yellow
            Write-Host "Or use the full conda command:" -ForegroundColor $Yellow
            Write-Host "`"$condaCommand`" activate `"$condaEnvPath`"" -ForegroundColor $Yellow
            exit 1
        }
    } catch {
        Write-Host "❌ Failed to check environment: $_" -ForegroundColor $Red
        exit 1
    }
}

Write-Host ""
Write-Host "🚀 Starting AutoCut v2..." -ForegroundColor $Green
Write-Host ""

# Execute the command
try {
    if ($useCondaRun) {
        # Use conda run with proper argument handling
        $condaArgs = @(
            "run", 
            "-p", $condaEnvPath,
            "--no-capture-output",
            "python", "-m", "autocut_v2"
        )
        
        # Add all the command arguments
        if ($VideoPath -and $VideoPath -ne "") {
            $condaArgs += $VideoPath
        }
        if ($Config) {
            $condaArgs += "--config", $Config
        }
        if ($ProcessingProfile) {
            $condaArgs += "--profile", $ProcessingProfile
        }
        if ($Device) {
            $condaArgs += "--device", $Device
        }
        if ($FinalOutput) {
            $condaArgs += "--output", $FinalOutput
        }
        $condaArgs += "--monitor", "term"
        if ($DryRun) {
            $condaArgs += "--dry-run"
        }
        
        Write-Host "📋 Executing with conda run:" -ForegroundColor $Cyan
        Write-Host "conda $($condaArgs -join ' ')" -ForegroundColor $Yellow
        Write-Host ""
        
        & "$condaCommand" @condaArgs
        
    } else {
        # Use regular python command
        $pythonArgs = @("-m", "autocut_v2")
        
        # Add all the command arguments
        if ($VideoPath -and $VideoPath -ne "") {
            $pythonArgs += $VideoPath
        }
        if ($Config) {
            $pythonArgs += "--config", $Config
        }
        if ($ProcessingProfile) {
            $pythonArgs += "--profile", $ProcessingProfile
        }
        if ($Device) {
            $pythonArgs += "--device", $Device
        }
        if ($FinalOutput) {
            $pythonArgs += "--output", $FinalOutput
        }
        $pythonArgs += "--monitor", "term"
        if ($DryRun) {
            $pythonArgs += "--dry-run"
        }
        
        Write-Host "📋 Executing with python:" -ForegroundColor $Cyan
        Write-Host "python $($pythonArgs -join ' ')" -ForegroundColor $Yellow
        Write-Host ""
        
        & python @pythonArgs
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ AutoCut v2 completed successfully!" -ForegroundColor $Green
        
        if (-not $DryRun) {
            Write-Host ""
            Write-Host "📁 Check your output directory for results:" -ForegroundColor $Cyan
            if ($FinalOutput) {
                Write-Host "$FinalOutput" -ForegroundColor $Yellow
                
                # Verify output directory and list contents
                Write-Host ""
                Write-Host "🔍 Verifying output..." -ForegroundColor $Cyan
                if (Test-Path $FinalOutput) {
                    Write-Host "✓ Output directory exists" -ForegroundColor $Green
                    $outputFiles = Get-ChildItem $FinalOutput -File
                    if ($outputFiles.Count -gt 0) {
                        Write-Host "✓ Found $($outputFiles.Count) file(s) in output directory:" -ForegroundColor $Green
                        foreach ($file in $outputFiles) {
                            $fileSize = [math]::Round($file.Length / 1MB, 2)
                            Write-Host "  - $($file.Name) ($fileSize MB)" -ForegroundColor $Yellow
                        }
                    } else {
                        Write-Host "⚠️ Output directory is empty" -ForegroundColor $Yellow
                        Write-Host "This suggests the processing was simulated only" -ForegroundColor $Yellow
                    }
                } else {
                    Write-Host "❌ Output directory was not created" -ForegroundColor $Red
                }
            } else {
                Write-Host "Output location determined by AutoCut v2" -ForegroundColor $Yellow
            }
        }
    } else {
        Write-Host ""
        Write-Host "❌ AutoCut v2 completed with errors (exit code: $LASTEXITCODE)" -ForegroundColor $Red
        Write-Host "Check the log file for details: autocut.log" -ForegroundColor $Yellow
    }
} catch {
    Write-Host "❌ Failed to execute AutoCut v2: $_" -ForegroundColor $Red
    exit 1
}
