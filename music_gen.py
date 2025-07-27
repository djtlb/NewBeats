import sys
import torchaudio
from audiocraft.models import MusicGen

def generate(prompt, duration):
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.set_generation_params(duration=duration)
    audio = model.generate([prompt])
    torchaudio.save(sys.stdout.buffer, audio[0].cpu(), 32000, format="wav")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("-d", "--duration", type=int, default=10)
    args = parser.parse_args()
    generate(args.prompt, args.duration)
