import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Basic setup for CPU
torch.set_default_device('cpu')
torch.set_num_threads(4)

class SimpleMusicGen:
    def __init__(self):
        self.model = MusicGen.get_pretrained("facebook/musicgen-small")
        self.model.set_default_device('cpu')
        self.sample_rate = 32000

    def generate(self, description, duration=10):
        """Generate music from text description"""
        self.model.set_generation_params(
            duration=duration,
            use_sampling=True,
            top_k=250,
            top_p=0.0
        )
        
        with torch.no_grad():
            wav = self.model.generate([description])[0]
            return wav.cpu().squeeze()

    def save(self, audio, filename):
        """Save generated audio to MP3 file"""
        audio_write(
            filename,
            audio,
            self.sample_rate,
            format="mp3",
            loudness_compression=True
        )

# Example usage
if __name__ == "__main__":
    print("Generating music...")
    generator = SimpleMusicGen()
    audio = generator.generate("Happy acoustic guitar song", duration=5)
    generator.save(audio, "my_music")
    print("Done! Saved as my_music.mp3")