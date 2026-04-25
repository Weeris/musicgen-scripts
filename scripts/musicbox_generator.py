#!/usr/bin/env python3.11
"""
MusicBox Generator — Slow, nostalgic, loopable music box background music.
Uses Meta's MusicGen to generate royalty-free music for videos.

Usage:
    python scripts/musicbox_generator.py
"""
import torch
from scipy.io import wavfile
import numpy as np
import os
import time

from transformers import MusicgenForConditionalGeneration, MusicgenProcessor

OUTPUT_DIR = os.path.expanduser("~/.hermes/music_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "facebook/musicgen-small"
DURATION_SEC = 30

PROMPT = "slow music box melody, gentle, nostalgic, lullaby, solo, peaceful, calm, soft attack, warm tone"

DEVICE = "cpu"  # CPU is more stable than MPS on Apple Silicon


def main():
    print(f"🎵 MusicBox Generator")
    print(f"   Model:    {MODEL_NAME}")
    print(f"   Duration: {DURATION_SEC}s")
    print(f"   Device:  {DEVICE}")
    print()

    print("📥 Loading model...")
    start = time.time()
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    processor = MusicgenProcessor.from_pretrained(MODEL_NAME)
    print(f"   ✅ Loaded in {time.time()-start:.1f}s\n")

    print(f"🎼 Generating {DURATION_SEC}s music box clip...")
    start = time.time()
    inputs = processor(text=[PROMPT], padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    gen_tokens = DURATION_SEC * 50
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=gen_tokens,
        )

    elapsed = time.time() - start
    print(f"   ✅ Generated in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Convert to numpy [samples, channels]
    audio_np = audio_values[0].cpu().numpy().T

    # Normalize to prevent clipping
    max_val = np.abs(audio_np).max()
    if max_val > 0.95:
        audio_np = audio_np / max_val * 0.95

    out_path = os.path.join(OUTPUT_DIR, "musicbox_30s_loop.wav")
    wavfile.write(out_path, rate=32000, data=audio_np.astype(np.float32))

    print(f"\n🎉 Done!")
    print(f"   File:     {out_path}")
    print(f"   Duration: {len(audio_np)/32000:.1f}s")


if __name__ == "__main__":
    main()
