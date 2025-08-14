"""
Command Line Interface for AutoCut v2.
"""

import click
import logging
from pathlib import Path
from typing import Optional

from .core.processor import AutoCut
from .utils.config import Config
from .utils.file_handler import FileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('video_input', type=click.Path(exists=True))
@click.option(
    '-c', '--config', 
    type=click.Path(exists=True),
    help='Path to YAML configuration file'
)
@click.option(
    '--profile',
    type=click.Choice(['safe_content', 'face_focus', 'custom']),
    default='safe_content',
    help='Predefined processing profile'
)
@click.option(
    '--device',
    type=click.Choice(['auto', 'cuda', 'mps', 'cpu']),
    default='auto',
    help='Processing device to use'
)
@click.option(
    '--monitor',
    type=click.Choice(['web', 'term', 'none']),
    default='term',
    help='Progress monitoring type'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Simulation mode without actual cutting'
)
@click.option(
    '--skip-normalize',
    is_flag=True,
    help='Force skip video normalization'
)
@click.option(
    '--force-normalize',
    is_flag=True,
    help='Force video normalization'
)
@click.option(
    '--skip-scenes',
    is_flag=True,
    help='Skip scene detection'
)
@click.option(
    '--enable-title-rename',
    is_flag=True,
    help='Rename clips using LLM-generated titles'
)
@click.option(
    '-o', '--output',
    type=click.Path(),
    help='Output directory path'
)
def main(
    video_input: str,
    config: Optional[str] = None,
    profile: str = 'safe_content',
    device: str = 'auto',
    monitor: str = 'term',
    dry_run: bool = False,
    skip_normalize: bool = False,
    force_normalize: bool = False,
    skip_scenes: bool = False,
    enable_title_rename: bool = False,
    output: Optional[str] = None
):
    """
    AutoCut v2 - Intelligent Video Processing and Cutting Tool
    
    Process VIDEO_INPUT with automated scene detection, content analysis,
    and intelligent cutting based on configurable criteria.
    """
    try:
        # Initialize configuration
        config_obj = Config()
        
        # Load config file if provided
        if config:
            config_obj.load_from_file(config)
        
        # Apply profile settings
        _apply_profile(config_obj, profile)
        
        # Override config with CLI options
        if device != 'auto':
            config_obj.set('device', device)
        if skip_normalize:
            config_obj.set('workflow.skip_normalize_if_conform', True)
        if force_normalize:
            config_obj.set('workflow.force_normalize', True)
        if skip_scenes:
            config_obj.set('scenes.enabled', False)
        if enable_title_rename:
            config_obj.set('describe.enable_title_rename', True)
        
        # Set output directory
        if output:
            config_obj.set('output_dir', output)
        else:
            input_path = Path(video_input)
            default_output = input_path.parent / f"{input_path.stem}_autocut"
            config_obj.set('output_dir', str(default_output))
        
        # Initialize processor
        processor = AutoCut(config_obj.to_dict())
        
        # Validate input
        file_handler = FileHandler()
        if Path(video_input).is_file():
            if not file_handler.validate_video_file(video_input):
                click.echo(f"Error: Invalid video file: {video_input}", err=True)
                return
            video_files = [video_input]
        elif Path(video_input).is_dir():
            video_files = file_handler.find_video_files(video_input)
            if not video_files:
                click.echo(f"Error: No video files found in: {video_input}", err=True)
                return
        else:
            click.echo(f"Error: Input path not found: {video_input}", err=True)
            return
        
        # Process videos
        click.echo(f"Processing {len(video_files)} video(s) with profile: {profile}")
        
        if dry_run:
            click.echo("DRY RUN MODE - No actual processing will occur")
            for video_file in video_files:
                click.echo(f"Would process: {video_file}")
            return
        
        # Process each video
        results = []
        for video_file in video_files:
            click.echo(f"Processing: {video_file}")
            
            output_path = Path(config_obj.get('output_dir')) / f"processed_{Path(video_file).name}"
            result = processor.process_video(video_file, str(output_path))
            results.append(result)
            
            if result['success']:
                click.echo(f"✓ Successfully processed: {result['output_path']}")
                
                # Show Step 1 analysis results if available
                if 'step1_analysis' in result:
                    analysis = result['step1_analysis']
                    click.echo(f"\n📊 Step 1 Analysis Results:")
                    if 'file_info' in analysis:
                        click.echo(f"  📁 File: {analysis['file_info']['size_mb']}MB")
                    if 'video_info' in analysis and 'error' not in analysis['video_info']:
                        video = analysis['video_info']
                        click.echo(f"  🎬 Video: {video.get('width', '?')}x{video.get('height', '?')} @ {video.get('fps', 0):.1f}fps")
                        click.echo(f"  ⏱️  Duration: {video.get('duration', 0):.1f}s")
                        click.echo(f"  🎥 Codec: {video.get('codec', 'unknown')}")
                
                # Show Step 2 scene detection results if available
                if 'step2_scenes' in result:
                    scenes = result['step2_scenes']
                    click.echo(f"\n🎬 Step 2 Scene Detection Results:")
                    click.echo(f"  📈 Method: {scenes.get('method', 'unknown')}")
                    scene_list = scenes.get('scenes', [])
                    click.echo(f"  🎞️  Scenes: {len(scene_list)} detected")
                    if scene_list and len(scene_list) <= 10:  # Show first 10 scenes
                        for scene in scene_list[:5]:  # Show first 5 for brevity
                            click.echo(f"    - Scene {scene['id']}: {scene['start_time']:.1f}s - {scene['end_time']:.1f}s ({scene['duration']:.1f}s)")
                        if len(scene_list) > 5:
                            click.echo(f"    ... and {len(scene_list) - 5} more scenes")
                
                # Show Step 3 content analysis results if available
                if 'step3_content' in result:
                    content = result['step3_content']
                    click.echo(f"\n🔍 Step 3 Content Analysis Results:")
                    click.echo(f"  📊 Scenes analyzed: {content.get('scenes_analyzed', 0)}")
                    click.echo(f"  ✅ Scenes kept: {content.get('scenes_kept', 0)}")
                    click.echo(f"  ❌ Scenes rejected: {content.get('scenes_rejected', 0)}")
                    click.echo(f"  📉 Rejection rate: {content.get('rejection_rate', 0):.1%}")
                    click.echo(f"  ⏱️  Duration kept: {content.get('duration_kept', 0):.1f}s / {content.get('duration_original', 0):.1f}s")
                    click.echo(f"  🖼️  Frames extracted: {content.get('frames_extracted', 0)}")
                    
                    # Show sample rate used
                    processing_info = content.get('processing_info', {})
                    sample_rate = processing_info.get('sample_rate_used', 'unknown')
                    click.echo(f"  📈 Sample rate: {sample_rate} fps")
                    
                    # Show criteria applied
                    criteria = content.get('criteria_applied', [])
                    if criteria:
                        click.echo(f"  📋 Criteria: {', '.join(criteria)}")
                
                # Show temp files info
                if 'temp_dir' in result:
                    click.echo(f"\n🗂️  Temp files: {result['temp_dir']}")
                if 'current_step' in result:
                    click.echo(f"📈 Progress: Step {result['current_step']}/{result['total_steps']}")
            else:
                click.echo(f"✗ Failed to process: {video_file} - {result.get('error', 'Unknown error')}", err=True)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        click.echo(f"\nProcessing complete: {successful}/{len(results)} videos processed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        click.echo(f"Error: {str(e)}", err=True)


def _apply_profile(config: Config, profile: str) -> None:
    """Apply predefined profile settings to configuration."""
    profiles = {
        'safe_content': {
            'criteria.nsfw.action': 'reject',
            'criteria.nsfw.mode': 'high',
            'criteria.face.min_confidence': 0.8,
            'criteria.gender.filter': 'any',
            'describe.enabled': True,
        },
        'face_focus': {
            'criteria.face.min_confidence': 0.9,
            'criteria.face.min_area_pct': 2.0,
            'criteria.gender.filter': 'female',
            'criteria.pose.enabled': True,
            'describe.enabled': True,
        },
        'custom': {
            # No defaults, use config file or CLI overrides
        }
    }
    
    profile_config = profiles.get(profile, {})
    for key, value in profile_config.items():
        config.set(key, value)


@click.group()
def cli():
    """AutoCut v2 - Modular video processing commands."""
    pass


@cli.command()
@click.argument('video', type=click.Path(exists=True))
@click.option('--target', default='720p', help='Target resolution')
def normalize(video: str, target: str):
    """Normalize video to standard format."""
    click.echo(f"Normalizing {video} to {target}")
    # Implementation would go here


@cli.command()
@click.argument('video', type=click.Path(exists=True))
@click.option('--threshold', default=0.3, help='Scene detection threshold')
def scenes(video: str, threshold: float):
    """Detect scenes in video."""
    click.echo(f"Detecting scenes in {video} with threshold {threshold}")
    # Implementation would go here


@cli.command()
@click.argument('video', type=click.Path(exists=True))
@click.option('-c', '--config', type=click.Path(exists=True))
def analyze(video: str, config: Optional[str]):
    """Analyze video content with criteria."""
    click.echo(f"Analyzing {video}")
    # Implementation would go here


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def validate_config(config_path: str):
    """Validate configuration file."""
    try:
        config = Config()
        config.load_from_file(config_path)
        click.echo(f"✓ Configuration file {config_path} is valid")
    except Exception as e:
        click.echo(f"✗ Configuration file {config_path} is invalid: {e}", err=True)


if __name__ == '__main__':
    main()
