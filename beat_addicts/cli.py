import torch
# Remove forced CPU mode - let auto-detection work

import sys
from pathlib import Path

# Add parent directory to path (safer implementation)
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from beat_addicts.song_builder import SongBuilder
from beat_addicts.melody_generator import get_optimal_device

@click.command()
@click.argument("prompt")
@click.option("--duration", default=15, 
             help="Duration in seconds", 
             type=click.IntRange(5, 120))
@click.option("--output", default="output",
             help="Output directory (will be created)")
@click.option("--device", default=None,
             help="Device to use (cpu/cuda/mps) - auto-detect if not specified",
             type=click.Choice(['cpu', 'cuda', 'mps'], case_sensitive=False))
@click.option("--model-size", default="small",
             help="Model size for generation",
             type=click.Choice(['small', 'medium', 'large'], case_sensitive=False))
@click.option("--use-jukebox", is_flag=True,
             help="Use Jukebox for higher quality (if available)")
@click.option("--benchmark", is_flag=True,
             help="Run benchmark before generation")
def generate_song(prompt, duration, output, device, model_size, use_jukebox, benchmark):
    """Generate complete songs with CPU/GPU support and Jukebox integration"""
    
    # Device detection
    actual_device = device if device else get_optimal_device()
    max_duration = 120 if actual_device != 'cpu' else 30
    duration = min(duration, max_duration)
    
    click.echo(f"\n� Beat Addicts | Device: {actual_device.upper()} | Model: {model_size}")
    click.echo(f"🎵 Generating {duration}s song: '{prompt}'")
    
    if use_jukebox:
        click.echo("🎼 Jukebox mode enabled for higher quality")
    
    try:
        builder = SongBuilder(
            output_dir=output, 
            device=actual_device,
            model_size=model_size,
            use_jukebox=use_jukebox
        )
        
        # Run benchmark if requested
        if benchmark:
            click.echo("\n🏃 Running performance benchmark...")
            bench_result = builder.melody_generator.benchmark_generation(10)
            click.echo(f"⏱️ Benchmark: {bench_result['generation_time']} ({bench_result['speed_ratio']} speed)")
        
        # Show device info
        device_info = builder.melody_generator.get_device_info()
        if device_info.get('cuda_available') and actual_device == 'cuda':
            click.echo(f"🔥 GPU: {device_info['cuda_device_name']} ({device_info['cuda_memory_total']})")
        
        with click.progressbar(length=100, label='Generating') as bar:
            def progress_callback(progress):
                bar.update(int(progress * 100) - bar.pos)
            
            result = builder.build_song(prompt, duration, progress_callback)
            bar.update(100 - bar.pos)
        
        click.secho("\n✅ Generation Complete!", fg="green")
        click.echo(f"🎵  Title: {result['title']}")
        click.echo(f"📜  Lyrics: {file_link(result['lyrics_path'])}")
        click.echo(f"🎶  Audio: {file_link(result['song_path'])}")
        
        # Clear GPU cache if used
        if actual_device == 'cuda':
            builder.melody_generator.clear_cache()
            click.echo("🧹 GPU cache cleared")
            
    except Exception as e:
        click.secho(f"\n❌ Error: {str(e)}", fg="red")
        if "memory" in str(e).lower() or "cuda" in str(e).lower():
            click.echo("💡 Try reducing --duration, using --device cpu, or using smaller --model-size")
        elif "jukebox" in str(e).lower():
            click.echo("💡 Try running without --use-jukebox flag")

def file_link(path):
    """Convert file path to clickable link in supported terminals"""
    return f"\033]8;;file://{Path(path).absolute()}\a{path}\033]8;;\a"

if __name__ == "__main__":
    generate_song()