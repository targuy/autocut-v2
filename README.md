# AutoCut v2

A powerful automated video editing and cutting tool that helps streamline video processing workflows.

## Features

- Automated video cutting and trimming
- Scene detection and segmentation
- Batch processing capabilities
- Multiple video format support
- Command-line interface for easy integration
- AI-powered criteria (NSFW, face, gender) with fallbacks

## Prerequisites (all OS)

- Python 3.8–3.11
- FFmpeg installed and on PATH
- Git (optional, recommended)

Install FFmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- Windows: `winget install Gyan.FFmpeg` or `choco install ffmpeg` (ensure `ffmpeg -version` works).

GPU (optional):
- Windows/Linux CUDA: follow https://pytorch.org/ to install CUDA wheels, e.g.
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- macOS Apple Silicon: MPS is supported by default in PyTorch 2.x.

## Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd autocut-v2

# Create and activate a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -e .

# Optional extras
pip install -e ".[dev]"                    # dev tools
pip install -e ".[monitoring,web,pose]"     # feature extras
```

### Wheel/Requirements installation
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Basic usage
autocut input_video.mp4 --output output_video.mp4
# or alias
autocut-v2 input_video.mp4 --output output_video.mp4
```

### Python API

```python
from autocut_v2.core.processor import AutoCut
from autocut_v2.utils.config import Config

config = Config('config.yml')
processor = AutoCut(config)
result = processor.process_video('input.mp4', 'output.mp4')
```

## Configuration

See `config_template.yml` for all options. Key sections:
- workflow auto-optimize, normalize, scenes, criteria fallbacks
- device: auto|cuda:0|mps|cpu
- output codecs: auto chooses h264_nvenc (NVIDIA), h264_videotoolbox (macOS), libx264 otherwise

## Testing

```bash
pytest -q
python test_basic.py
python tests/integration/test_pipeline.py
# Lightweight criteria test (no heavy downloads)
python test_criteria_simple.py
# Full criteria test (downloads models on first run)
python test_criteria.py
```

## Troubleshooting

- FFmpeg not found: install it and ensure it's on PATH (`ffmpeg -version`).
- Transformers model downloads are large: first run may take time; set `HF_HOME` to move cache, e.g. `export HF_HOME=~/.cache/hf`.
- CUDA (Windows/Linux): ensure matching CUDA/cuDNN per PyTorch docs; use correct index URL.
- MediaPipe install issues: upgrade pip and ensure Python <= 3.11 for certain versions.
- OpenCV GUI errors on servers: consider `opencv-python-headless`.

## Development

```bash
pip install -e ".[dev]"
flake8 src tests
black src tests
mypy src
pytest -q
```

## Project Structure

```
autocut-v2/
├── src/autocut_v2/
│   ├── cli.py
│   ├── core/
│   ├── processors/
│   └── utils/
├── tests/
├── README.md
├── pyproject.toml
└── requirements.txt
```

## License

MIT License.
