# 🎵 MusicGen Scripts

AI music generation using Meta's **MusicGen** — generate royalty-free background music locally on Apple Silicon Mac (or any machine with Python).

> No GPU needed — runs on CPU (with MPS fallback on Apple Silicon). Free forever, MIT licensed.

## Features

- Generate custom background music from text prompts
- Loopable clips for videos, streams, podcasts
- Multiple instrument styles: music box, guitar, piano, ambient, etc.
- Runs locally — no API keys, no subscription, no upload needed
- MIT licensed — use commercially with no attribution required

## Requirements

- Python 3.11+
- PyTorch 2.1.0 + torchaudio
- transformers + scipy
- Apple Silicon Mac (recommended) or any machine with Python 3.11

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Weeris/musicgen-scripts.git
cd musicgen-scripts

# 2. Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install torch==2.1.0 torchaudio==2.1.0
pip install numpy scipy transformers

# 4. Generate music!
python scripts/musicbox_generator.py
```

## Scripts

### `scripts/musicbox_generator.py`
Generates a slow, nostalgic music box melody. Perfect for calming videos, ASMR, or lo-fi backgrounds.

```bash
python scripts/musicbox_generator.py
```

### `scripts/generate_music.py`
**Main generator** — customize prompts and parameters.

```python
# Edit these for different styles:
PROMPT = "slow fingerstyle guitar, gentle, nostalgic, solo acoustic"
DURATION_SEC = 60
MODEL_NAME = "facebook/musicgen-small"  # or musicgen-medium
```

### `scripts/stitch_loops.py`
Stitch multiple generated clips into a long seamless loop with crossfade.

## Prompt Guide

| Style | Prompt |
|-------|--------|
| Music Box | `slow music box melody, gentle, nostalgic, lullaby, solo, peaceful` |
| Fingerstyle Guitar | `slow fingerstyle guitar, gentle, nostalgic, solo acoustic, warm tone` |
| Piano | `slow piano melody, gentle, romantic, solo, soft, contemplative` |
| Ambient | `ambient pad, atmospheric, peaceful, drone, minimal, ethereal` |
| Lo-fi | `lo-fi hip hop, chill, relaxed, boom bap, nostalgic` |

**Negative prompts** (what to avoid):
```
fast tempo, drums, electronic, distorted, loud, aggressive, percussion
```

## Generation Time (Apple Silicon M-series)

| Duration | Model | Time |
|----------|-------|------|
| 10s | small (300M) | ~4 min |
| 30s | small (300M) | ~14 min |
| 60s | small (300M) | ~29 min |

CPU only mode is slower but more stable.

## Models

- **`facebook/musicgen-small`** (default, 300M params, 3.3GB) — fastest, good quality
- **`facebook/musicgen-medium`** (1.5B params, 9.5GB) — better quality, slower
- **`facebook/musicgen-large`** (3.3B params, 20GB) — best quality, requires more RAM

## Troubleshooting

### MPS crashes with "Placeholder storage has not been allocated"
**Fix:** Use CPU mode by setting `device = "cpu"` in the script. Slower but stable.

### Out of memory
**Fix:** Use the small model (`facebook/musicgen-small`). On 16GB RAM Macs, CPU mode uses less memory than MPS.

### First run is slow
**Normal:** Model weights (~3GB) download on first use. Cached for subsequent runs.

## Samples

Generated samples are in `samples/` directory. MIT licensed — free to use anywhere.

## License

MIT — do whatever you want. No attribution required.
