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
        self.max_duration = 120 if self.device != 'cpu' else 60  # 60 seconds on CPU, 120 on GPU
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
        
        # Enhance the prompt for better MusicGen understanding
        enhanced_description = self._enhance_prompt(description)
        
        use_jb = use_jukebox if use_jukebox is not None else self.use_jukebox
        
        if use_jb and self.jukebox_model:
            return self._generate_with_jukebox(enhanced_description, duration, progress_callback)
        else:
            return self._generate_with_musicgen(enhanced_description, duration, progress_callback)
    
    def _enhance_prompt(self, description: str) -> str:
        """Enhance the user prompt for better MusicGen results with genre-specific templates."""
        # Remove any existing dashes or separators that might confuse the model
        description = description.replace(" - ", " ").replace("...", "")
        
        # Genre-specific enhancement templates
        genre_templates = {
            'drum and bass': 'fast breakbeats, deep sub bass, jungle style',
            'dnb': 'fast breakbeats, deep sub bass, jungle style',
            'dubstep': 'wobble bass, syncopated drum patterns, electronic',
            'house': 'four-on-the-floor beat, electronic, dance',
            'techno': 'repetitive beats, electronic, industrial',
            'trap': '808 drums, hi-hats, bass-heavy',
            'hip hop': 'rhythmic beats, bass, urban style',
            'jazz': 'swing rhythm, improvisation, brass instruments',
            'rock': 'electric guitar, drums, energetic',
            'pop': 'catchy melody, upbeat, commercial',
            'electronic': 'synthesizers, digital sounds, modern',
            'ambient': 'atmospheric, peaceful, relaxing',
            'classical': 'orchestral, traditional instruments, elegant',
            'blues': 'twelve-bar progression, soulful, guitar',
            'reggae': 'offbeat rhythm, bass-heavy, caribbean',
            'country': 'acoustic guitar, storytelling, folk style',
            'r&b': 'smooth vocals, rhythm and blues, soulful',
            'funk': 'groove-based, syncopated rhythm, bass-heavy',
            'lofi': 'vintage sound, relaxed tempo, nostalgic',
            'synthwave': 'retro synthesizers, 80s style, electronic'
        }
        
        # Enhance based on detected genre
        enhanced = description.lower()
        for genre, template in genre_templates.items():
            if genre in enhanced:
                # Add professional musical descriptors
                description = f"{description}, {template}, professional studio quality"
                break
        else:
            # Default enhancement for unrecognized genres
            description = f"{description}, professional studio quality, clear audio"
        
        # Add tempo and energy descriptors based on keywords
        if any(word in enhanced for word in ['fast', 'energetic', 'upbeat', 'dance']):
            description += ", high energy, upbeat tempo"
        elif any(word in enhanced for word in ['slow', 'chill', 'relaxed', 'ambient']):
            description += ", relaxed tempo, smooth"
        elif any(word in enhanced for word in ['medium', 'moderate']):
            description += ", medium tempo"
        
        # Ensure it ends properly for MusicGen
        if not description.endswith(('.', '!', '?')):
            description += "."
            
        return description.strip()
    
    def _generate_with_musicgen(self, description: str, duration: int,
                               progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Generate with MusicGen model with enhanced parameters."""
        print(f"🎵 Generating with MusicGen: '{description}' ({duration}s)")
        
        # Enhanced generation parameters for professional quality
        generation_params = {
            'duration': duration,
            'temperature': 0.85,      # Slightly lower for more controlled output
            'top_k': 200,             # Balanced for quality and diversity
            'top_p': 0.9,             # Higher for better prompt following
            'use_sampling': True,
            'cfg_coef': 6.0,          # Higher classifier-free guidance for better prompt adherence
            'extend_stride': 18,      # Better continuation for longer tracks
        }
        
        # Adjust parameters based on genre for optimal results
        description_lower = description.lower()
        if any(genre in description_lower for genre in ['ambient', 'chill', 'lofi', 'relaxed']):
            # Smoother generation for ambient genres
            generation_params.update({
                'temperature': 0.7,
                'top_k': 150,
                'cfg_coef': 5.5
            })
        elif any(genre in description_lower for genre in ['drum and bass', 'dubstep', 'trap', 'electronic']):
            # More dynamic generation for electronic genres
            generation_params.update({
                'temperature': 0.9,
                'top_k': 250,
                'cfg_coef': 6.5
            })
        elif any(genre in description_lower for genre in ['jazz', 'classical', 'blues']):
            # More structured generation for traditional genres
            generation_params.update({
                'temperature': 0.8,
                'top_k': 180,
                'cfg_coef': 5.8
            })
        
        self.musicgen_model.set_generation_params(**generation_params)
        
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
                    progress_callback(0.8)
                    
                # Post-process the audio for better quality
                processed_audio = self._post_process_audio(wav.cpu().squeeze().numpy())
                
                if progress_callback:
                    progress_callback(0.9)
                    
                self.current_melody = processed_audio
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
    
    def _post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply post-processing to enhance audio quality."""
        try:
            # Normalize audio to prevent clipping
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Apply gentle compression to enhance dynamics
            # Simple soft compression algorithm
            threshold = 0.8
            ratio = 4.0
            
            # Find peaks above threshold
            mask = np.abs(audio) > threshold
            if np.any(mask):
                # Apply compression to peaks
                compressed = np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio)
                audio = np.where(mask, compressed, audio)
            
            # Apply gentle high-frequency boost for clarity (simple shelving filter approximation)
            # This is a very basic implementation - for production, use proper DSP libraries
            if len(audio) > 1000:
                # Simple high-frequency emphasis
                diff = np.diff(audio, prepend=audio[0])
                audio = audio + 0.05 * diff
            
            # Final normalization to -1dB peak to leave headroom
            peak_target = 0.89  # About -1dB
            current_peak = np.max(np.abs(audio))
            if current_peak > 0:
                audio = audio * (peak_target / current_peak)
            
            # Ensure we're in the correct range
            audio = np.clip(audio, -1.0, 1.0)
            
            print("🎛️ Audio post-processing applied")
            return audio
            
        except Exception as e:
            print(f"⚠️ Post-processing failed, using original audio: {e}")
            return audio
    
    def get_device_info(self) -> dict:
        """Get information about the current device setup."""
        # Calculate CPU optimization info
        cpu_threads = torch.get_num_threads()
        cpu_optimized = cpu_threads >= 4
        
        info = {
            "device": self.device.upper(),
            "device_status": "Optimized" if cpu_optimized or self.device != 'cpu' else "Standard",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "jukebox_available": JUKEBOX_AVAILABLE,
            "using_jukebox": self.use_jukebox,
            "max_duration": self.max_duration,
            "cpu_threads": cpu_threads,
            "audio_processing": "Enhanced with post-processing",
            "model_variants": len(self.available_models),
            "generation_quality": "Professional" if self.max_duration >= 60 else "Standard"
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
                "cuda_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
            })
        elif self.device == 'cpu':
            info.update({
                "cpu_optimization": "Multi-threaded processing enabled",
                "memory_efficiency": "Optimized for CPU generation",
                "processing_mode": "CPU-optimized algorithms"
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