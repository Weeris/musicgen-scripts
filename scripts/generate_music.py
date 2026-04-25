#!/usr/bin/env python3.11
"""
MusicGen Generator — Generate royalty-free AI music from text prompts.
Run on Apple Silicon Mac (MPS) or CPU.

Usage:
    python scripts/generate_music.py

Edit PROMPT, DURATION_SEC, and MODEL_NAME to customize output.
"""
import torch
from scipy.io import wavfile
import numpy as np
import os
import time
import argparse

from transformers import MusicgenForConditionalGeneration, MusicgenProcessor

# ============================================================
# CONFIG — Edit these to customize your generation
# ============================================================
OUTPUT_DIR = os.path.expanduser("~/.hermes/music_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "facebook/musicgen-small"  # small (3.3GB), medium (9.5GB), large (20GB)
DURATION_SEC = 60                       # how long each clip is
NUM_SEGMENTS = 1                        # number of clips to generate and stitch
CROSSFADE_SEC = 1.0                      # crossfade duration when stitching clips

PROMPT = "slow fingerstyle guitar, gentle, nostalgic, solo acoustic, peaceful, calm, soft fingerpicking, warm tone, mellow"
# ============================================================

DEVICE = "cpu"  # Force CPU — MPS has stability issues with MusicGen

def generate_clip(model, processor, prompt, duration_sec, device):
    """Generate a single audio clip."""
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_tokens = duration_sec * 50  # ~50 tokens/sec at 32kHz

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=gen_tokens,
        )

    audio_np = audio_values[0].cpu().numpy().T
    max_val = np.abs(audio_np).max()
    if max_val > 0.95:
        audio_np = audio_np / max_val * 0.95
    return audio_np


def crossfade_stitch(audio_list, crossfade_sec=1.0, sample_rate=32000):
    """Stitch audio arrays with crossfade for seamless looping."""
    crossfade_samples = int(crossfade_sec * sample_rate)

    if len(audio_list) == 1:
        return audio_list[0]

    output = audio_list[0]
    for segment in audio_list[1:]:
        fade_out = np.linspace(1.0, 0.0, crossfade_samples).reshape(-1, 1)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples).reshape(-1, 1)
        overlap = output[-crossfade_samples:] * fade_out + segment[:crossfade_samples] * fade_in
        output = np.concatenate([output[:-crossfade_samples], overlap, segment[crossfade_samples:]])
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate AI music with MusicGen")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Text prompt for music generation")
    parser.add_argument("--duration", type=int, default=DURATION_SEC, help="Duration per clip in seconds")
    parser.add_argument("--segments", type=int, default=NUM_SEGMENTS, help="Number of clips to generate and stitch")
    parser.add_argument("--crossfade", type=float, default=CROSSFADE_SEC, help="Crossfade duration in seconds")
    parser.add_argument("--output", type=str, default=None, help="Output filename (default: auto)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="MusicGen model name")
    args = parser.parse_args()

    print(f"🎵 MusicGen Generator")
    print(f"   Model:     {args.model}")
    print(f"   Prompt:    {args.prompt}")
    print(f"   Duration:  {args.duration}s × {args.segments} segment(s)")
    print(f"   Device:    {DEVICE}")
    print()

    # Load model
    print("📥 Loading model...")
    start = time.time()
    model = MusicgenForConditionalGeneration.from_pretrained(args.model)
    model = model.to(DEVICE)
    processor = MusicgenProcessor.from_pretrained(args.model)
    print(f"   ✅ Loaded in {time.time()-start:.1f}s")

    # Generate clips
    audio_clips = []
    for i in range(args.segments):
        print(f"🎼 Generating clip {i+1}/{args.segments}...")
        start = time.time()
        clip = generate_clip(model, processor, args.prompt, args.duration, DEVICE)
        print(f"   ✅ Generated in {time.time()-start:.1f}s ({(time.time()-start)/60:.1f} min)")
        audio_clips.append(clip)

    # Stitch
    if len(audio_clips) > 1:
        print(f"🔗 Stitching {len(audio_clips)} clips with {args.crossfade}s crossfade...")
        final_audio = crossfade_stitch(audio_clips, crossfade_sec=args.crossfade)
    else:
        final_audio = audio_clips[0]

    # Normalize
    max_val = np.abs(final_audio).max()
    if max_val > 0.95:
        final_audio = final_audio / max_val * 0.95

    # Save
    if args.output:
        out_path = os.path.join(OUTPUT_DIR, args.output)
    else:
        safe_name = args.prompt.split(",")[0].replace(" ", "_")[:30]
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{args.duration}s.wav")

    wavfile.write(out_path, rate=32000, data=final_audio.astype(np.float32))
    duration_min = len(final_audio) / 32000 / 60
    print(f"\n🎉 Done!")
    print(f"   File:     {out_path}")
    print(f"   Duration: {duration_min:.2f} min")
    print(f"   Size:     {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
