import argparse
import torch
import soundfile as sf
from audiocraft.models import MusicGen

def main():
    parser = argparse.ArgumentParser(description='Generate music with MusicGen')
    parser.add_argument('description', type=str, help='Music description prompt')
    parser.add_argument('--duration', type=int, default=10)
    parser.add_argument('--output', type=str, default='output')
    args = parser.parse_args()

    print("Loading model...")
    model = MusicGen.get_pretrained("facebook/musicgen-small")

    print(f"Generating {args.duration}s audio...")
    model.set_generation_params(
        duration=min(args.duration, 30),
        use_sampling=True,
        top_k=250,
        top_p=0.0
    )
    
    with torch.no_grad():
        audio = model.generate([args.description])[0].cpu().numpy()

    print(f"Saving as {args.output}.wav...")
    sf.write(f"{args.output}.wav", audio.T, 32000)
    print("Done!")

if __name__ == "__main__":
    main()