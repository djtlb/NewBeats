import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import numpy as np
import sys
import os
from pathlib import Path
from typing import Optional, Callable, Union

# Add Jukebox to path
jukebox_path = Path("C:/Users/sally/OneDrive/Desktop/jukebox")
if jukebox_path.exists():
    sys.path.insert(0, str(jukebox_path))
    try:
        from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
        from jukebox.hparams import Hyperparams, setup_hparams
        from jukebox.sample import sample_single_window, _sample, sample_partial_window
        from jukebox.utils.dist_utils import setup_dist_from_mpi
        JUKEBOX_AVAILABLE = True
        print("✅ Jukebox successfully loaded")
    except ImportError as e:
        JUKEBOX_AVAILABLE = False
        print(f"⚠️ Jukebox not available: {e}")
else:
    JUKEBOX_AVAILABLE = False
    print("⚠️ Jukebox directory not found")

# Device detection and setup
def get_optimal_device():
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 CUDA detected: {torch.cuda.get_device_name()}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon
        print("🍎 Apple Metal Performance Shaders detected")
    else:
        device = 'cpu'
        print("💻 Using CPU")
        torch.set_num_threads(min(8, torch.get_num_threads()))  # Optimize CPU usage
    
    return device

# Global device setup
DEVICE = get_optimal_device()

class MelodyGenerator:
    def __init__(self, model_size: str = "small", device: Optional[str] = None, 
                 use_jukebox: bool = False):
        """Initialize the MusicGen model with specified size and device.
        
        Args:
            model_size: Model size ('small', 'medium', 'large')
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detect)
            use_jukebox: Whether to use Jukebox for high-quality generation
        """
        # Device setup
        self.device = device if device else DEVICE
        print(f"🎵 Initializing MelodyGenerator on {self.device}")
        
        # Model configurations
        self.available_models = {
            "small": "facebook/musicgen-small",
            "medium": "facebook/musicgen-medium", 
            "large": "facebook/musicgen-large"
        }
        
        if model_size not in self.available_models:
            raise ValueError(f"Invalid model size. Choose from {list(self.available_models.keys())}")
        
        # Initialize MusicGen
        print(f"📥 Loading MusicGen {model_size} model...")
        self.musicgen_model = MusicGen.get_pretrained(
            self.available_models[model_size],
            device=self.device
        )
        
        # Jukebox setup
        self.use_jukebox = use_jukebox and JUKEBOX_AVAILABLE
        self.jukebox_model = None
        
        if self.use_jukebox:
            print("🎼 Initializing Jukebox for high-quality generation...")
            self._setup_jukebox()
        
        self.sample_rate = 32000
        self.max_duration = 120 if self.device != 'cpu' else 30  # Longer on GPU
        self.current_melody: Optional[np.ndarray] = None
        
        print(f"✅ MelodyGenerator ready! Max duration: {self.max_duration}s")
    
    def _setup_jukebox(self):
        """Setup Jukebox model for high-quality generation."""
        try:
            # Use 5B model for best quality, 1B for speed
            model_size = "5b" if self.device == 'cuda' else "1b_lyrics"
            hps = Hyperparams()
            hps.sr = 22050
            hps.n_samples = 1
            hps.name = f'jukebox_{model_size}'
            
            self.jukebox_model = make_model(setup_hparams(model_size, dict()), self.device)
            print(f"✅ Jukebox {model_size} model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load Jukebox: {e}")
            self.use_jukebox = False

    def generate(self, description: str, duration: int = 15, 
                progress_callback: Optional[Callable[[float], None]] = None,
                use_jukebox: Optional[bool] = None) -> np.ndarray:
        """Generate melody from text description.
        
        Args:
            description: Text prompt for music generation
            duration: Duration in seconds
            progress_callback: Callback for progress updates
            use_jukebox: Override instance setting for this generation
            
        Returns:
            Generated audio as numpy array
        """
        duration = min(duration, self.max_duration)
        use_jb = use_jukebox if use_jukebox is not None else self.use_jukebox
        
        if use_jb and self.jukebox_model:
            return self._generate_with_jukebox(description, duration, progress_callback)
        else:
            return self._generate_with_musicgen(description, duration, progress_callback)
    
    def _generate_with_musicgen(self, description: str, duration: int,
                               progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Generate with MusicGen model."""
        print(f"🎵 Generating with MusicGen: '{description}' ({duration}s)")
        
        self.musicgen_model.set_generation_params(
            duration=duration,
            temperature=0.8,
            top_k=200,
            top_p=0.9,
            use_sampling=True,
            cfg_coef=3.0  # Classifier-free guidance
        )
        
        if progress_callback:
            progress_callback(0.1)
            
        try:
            with torch.no_grad():
                if self.device == 'cuda':
                    # Use mixed precision for faster GPU generation
                    with torch.cuda.amp.autocast():
                        wav = self.musicgen_model.generate([description])[0]
                else:
                    wav = self.musicgen_model.generate([description])[0]
                
                if progress_callback:
                    progress_callback(0.9)
                    
                self.current_melody = wav.cpu().squeeze().numpy()
                print("✅ MusicGen generation complete")
                return self.current_melody
                
        except Exception as e:
            print(f"❌ MusicGen generation failed: {e}")
            raise
    
    def _generate_with_jukebox(self, description: str, duration: int,
                              progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Generate with Jukebox model for higher quality."""
        print(f"🎼 Generating with Jukebox: '{description}' ({duration}s)")
        
        if progress_callback:
            progress_callback(0.1)
        
        try:
            # Jukebox generation parameters
            chunk_size = 8192  # Adjust based on memory
            total_length = int(duration * 22050)  # Jukebox sample rate
            
            # Generate with Jukebox
            with torch.no_grad():
                metas = [dict(artist="Various", genre="Electronic", total_length=total_length)]
                
                if progress_callback:
                    progress_callback(0.5)
                
                # This is a simplified version - full Jukebox integration requires more setup
                audio = sample_single_window(self.jukebox_model, metas[0], total_length, 
                                           temperature=0.8, chunk_size=chunk_size)
                
                if progress_callback:
                    progress_callback(0.9)
                
                # Convert to numpy and resample to match MusicGen
                audio_np = audio.cpu().squeeze().numpy()
                
                # Simple resampling (you might want to use librosa for better quality)
                if len(audio_np.shape) > 1:
                    audio_np = audio_np[0]  # Take first channel if stereo
                
                self.current_melody = audio_np
                print("✅ Jukebox generation complete")
                return self.current_melody
                
        except Exception as e:
            print(f"⚠️ Jukebox generation failed, falling back to MusicGen: {e}")
            return self._generate_with_musicgen(description, duration, progress_callback)
    
    def save(self, melody: np.ndarray, filename: str, format: str = "mp3") -> None:
        """Save generated melody to file.
        
        Args:
            melody: Audio data as numpy array
            filename: Output filename without extension
            format: Audio format ('mp3' or 'wav')
        """
        if format not in ["mp3", "wav"]:
            raise ValueError("Format must be 'mp3' or 'wav'")
            
        # Use torchaudio for better compatibility
        import torchaudio
        audio_tensor = torch.from_numpy(melody)
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
            
        torchaudio.save(
            f"{filename}.{format}",
            audio_tensor,
            self.sample_rate
        )
    
    def get_last_melody(self) -> Optional[np.ndarray]:
        """Get the last generated melody.
        
        Returns:
            Last generated melody or None if none exists
        """
        return self.current_melody
        
    def generate_with_melody(self, description: str, melody: np.ndarray, 
                           duration: int = 15) -> np.ndarray:
        """Generate music conditioned on both text and melody.
        
        Args:
            description: Text prompt
            melody: Reference melody as numpy array
            duration: Duration in seconds
            
        Returns:
            Generated audio as numpy array
        """
        duration = min(duration, self.max_duration)
        print(f"🎼 Generating melody-conditioned music: '{description}' ({duration}s)")
        
        self.musicgen_model.set_generation_params(
            duration=duration,
            temperature=0.7,
            top_k=150,
            top_p=0.85,
            cfg_coef=3.5
        )
        
        # Convert melody to tensor and add batch dimension
        melody_tensor = torch.from_numpy(melody).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                if self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        wav = self.musicgen_model.generate_with_chroma(
                            [description],
                            melody_tensor,
                            self.sample_rate
                        )[0]
                else:
                    wav = self.musicgen_model.generate_with_chroma(
                        [description],
                        melody_tensor,
                        self.sample_rate
                    )[0]
                
                self.current_melody = wav.cpu().squeeze().numpy()
                print("✅ Melody-conditioned generation complete")
                return self.current_melody
                
        except Exception as e:
            print(f"❌ Melody-conditioned generation failed: {e}")
            raise
    
    def get_device_info(self) -> dict:
        """Get information about the current device setup."""
        info = {
            "device": self.device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "jukebox_available": JUKEBOX_AVAILABLE,
            "using_jukebox": self.use_jukebox,
            "max_duration": self.max_duration
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
                "cuda_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
            })
        
        return info
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
    
    def benchmark_generation(self, duration: int = 10) -> dict:
        """Benchmark generation speed on current device."""
        import time
        
        prompt = "A simple electronic beat"
        print(f"🏃 Benchmarking generation speed ({duration}s)...")
        
        start_time = time.time()
        _ = self.generate(prompt, duration)
        generation_time = time.time() - start_time
        
        speed_ratio = duration / generation_time
        
        benchmark = {
            "duration": duration,
            "generation_time": f"{generation_time:.2f}s",
            "speed_ratio": f"{speed_ratio:.2f}x",
            "device": self.device,
            "model_type": "Jukebox" if self.use_jukebox else "MusicGen"
        }
        
        print(f"⏱️ Generated {duration}s in {generation_time:.2f}s ({speed_ratio:.2f}x speed)")
        return benchmark

# Example usage
if __name__ == "__main__":
    # Auto-detect best device
    generator = MelodyGenerator("small", use_jukebox=True)
    
    # Print device info
    print("\n📊 Device Information:")
    for key, value in generator.get_device_info().items():
        print(f"  {key}: {value}")
    
    # Benchmark performance
    print("\n🏃 Running benchmark...")
    benchmark = generator.benchmark_generation(10)
    
    # Simple generation
    print("\n🎵 Generating simple melody...")
    melody = generator.generate("A happy jazz tune with piano and drums", duration=10)
    generator.save(melody, "jazz_tune")
    
    # Try Jukebox if available
    if JUKEBOX_AVAILABLE:
        print("\n🎼 Generating with Jukebox...")
        jukebox_melody = generator.generate(
            "Electronic dance music with heavy bass", 
            duration=15, 
            use_jukebox=True
        )
        generator.save(jukebox_melody, "jukebox_edm")
    
    # Melody-conditioned generation
    print("\n🎼 Generating melody-conditioned music...")
    base_melody = generator.generate("Simple piano melody", duration=5)
    conditioned = generator.generate_with_melody(
        "Full orchestra arrangement of the piano melody",
        base_melody,
        duration=15
    )
    generator.save(conditioned, "orchestra_version")
    
    # Clear cache
    generator.clear_cache()
    print("\n✅ All generations complete!")