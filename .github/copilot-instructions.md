<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AutoCut v2 - Copilot Instructions

This is a Python-based video editing automation tool called AutoCut v2. When working on this project, please follow these guidelines:

## Project Context
- This is a video processing and automation tool
- Primary focus on automated video cutting, scene detection, and batch processing
- Uses OpenCV, MoviePy, and other video processing libraries
- Follows modern Python packaging standards with pyproject.toml

## Code Style and Standards
- Use Python 3.8+ features and type hints
- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 88)
- Include comprehensive docstrings for all public functions and classes
- Use meaningful variable and function names related to video processing

## Architecture Guidelines
- Keep video processing logic in the `core/` module
- Separate different processors in the `processors/` module
- Utility functions go in the `utils/` module
- CLI interface should be clean and user-friendly
- Support both programmatic API and command-line usage

## Video Processing Best Practices
- Always handle video file validation and error cases
- Consider memory efficiency when processing large video files
- Provide progress indicators for long-running operations
- Support multiple video formats (MP4, AVI, MOV, etc.)
- Implement proper cleanup of temporary files

## Testing
- Write unit tests for all core functionality
- Include integration tests for video processing workflows
- Use small test video files in tests/fixtures/
- Mock heavy video operations where appropriate

## Dependencies
- Prefer established video processing libraries (OpenCV, MoviePy)
- Keep dependencies minimal and well-maintained
- Document any platform-specific requirements

## Error Handling
- Provide clear error messages for common issues (file not found, unsupported format, etc.)
- Log processing steps for debugging
- Handle interruptions gracefully (Ctrl+C during processing)

When suggesting code, prioritize:
1. Performance and memory efficiency
2. Clear error messages and logging
3. Modular, testable code structure
4. User-friendly CLI experience
5. Comprehensive type hints and documentation
