import sys
import torchaudio
import numpy as np

def process_audio(low_gain=0, high_gain=0):
    waveform, sample_rate = torchaudio.load(sys.stdin.buffer)
    
    # Simple EQ (you'd replace this with proper DSP)
    if low_gain:
        waveform[:,:sample_rate//4] *= (1 + low_gain/10)
    if high_gain:
        waveform[:,sample_rate//2:] *= (1 + high_gain/10)
    
    torchaudio.save(sys.stdout.buffer, waveform, sample_rate, format="wav")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=float, default=0)
    parser.add_argument("--high", type=float, default=0)
    args = parser.parse_args()
    process_audio(args.low, args.high)
