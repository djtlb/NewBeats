import os
import time
import inspect
import torch
from typing import Optional, Callable

from beat_addicts.lyrics_generator import LyricsGenerator
from beat_addicts.melody_generator import MelodyGenerator

class SongBuilder:
    def __init__(self, output_dir="output", device: Optional[str] = None, 
                 model_size: str = "small", use_jukebox: bool = False):
        """Initialize SongBuilder with device and model options.
        
        Args:
            output_dir: Directory for output files
            device: Device to use (None for auto-detect)
            model_size: Model size for generation
            use_jukebox: Whether to use Jukebox for higher quality
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize generators with new options
        self.lyrics_gen = LyricsGenerator()
        self.melody_generator = MelodyGenerator(
            model_size=model_size,
            device=device,
            use_jukebox=use_jukebox
        )
        
        # Progress tracking
        self._progress = 0
        self._progress_steps = 100
        
        print(f"🏗️ SongBuilder initialized with {model_size} model on {self.melody_generator.device}")

    def _update_progress(self, step):
        """Internal progress handler"""
        self._progress = min(self._progress + step, self._progress_steps)
        return self._progress

    def build_song(self, prompt: str, duration: int = 15, 
                  progress_callback: Optional[Callable[[float], None]] = None,
                  use_jukebox: Optional[bool] = None):
        """Generate a complete song with progress tracking.
        
        Args:
            prompt: Text description for the song
            duration: Duration in seconds
            progress_callback: Optional callback for progress updates
            use_jukebox: Override instance setting for this generation
        """
        song_id = f"song_{int(time.time())}"
        song_dir = os.path.join(self.output_dir, song_id)
        os.makedirs(song_dir, exist_ok=True)
        
        print(f"🎵 Building song: '{prompt}' ({duration}s)")

        # Generate lyrics with progress support (40% of work)
        lyrics = self.lyrics_gen.generate(
            f"Song about {prompt}",
            progress_callback=lambda x: progress_callback(x*0.4) if progress_callback else None
        )
        if progress_callback:
            progress_callback(40)

        # Save lyrics
        lyrics_path = os.path.join(song_dir, f"{song_id}_lyrics.txt")
        with open(lyrics_path, "w", encoding='utf-8') as f:
            f.write(lyrics)
        if progress_callback:
            progress_callback(45)

        # Generate melody (55% of progress)
        melody_prompt = f"{prompt} - {lyrics[:100]}..."  # Include lyrics context
        melody = self.melody_generator.generate(
            melody_prompt,
            duration,
            progress_callback=lambda x: progress_callback(45 + x*0.55) if progress_callback else None,
            use_jukebox=use_jukebox
        )
        
        if progress_callback:
            progress_callback(100)

        # Save melody
        melody_path = os.path.join(song_dir, f"{song_id}.mp3")
        self.melody_generator.save(melody, melody_path[:-4])  # Remove .mp3 extension
        
        # Create a simple metadata file
        metadata_path = os.path.join(song_dir, f"{song_id}_metadata.txt")
        with open(metadata_path, "w", encoding='utf-8') as f:
            device_info = self.melody_generator.get_device_info()
            f.write(f"Title: {lyrics.split(chr(10))[0][:50]}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Duration: {duration}s\n")
            f.write(f"Device: {device_info['device']}\n")
            f.write(f"Model: {device_info.get('model_type', 'MusicGen')}\n")
            f.write(f"Jukebox: {device_info['using_jukebox']}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"✅ Song '{song_id}' completed successfully!")
        
        return {
            "title": lyrics.split("\n")[0][:50],  # Trim long titles
            "lyrics_path": lyrics_path,
            "song_path": melody_path,
            "metadata_path": metadata_path,
            "song_id": song_id
        }