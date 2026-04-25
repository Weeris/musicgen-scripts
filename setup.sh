#!/bin/bash
# Setup script for musicgen-scripts
# Run once to install all dependencies

set -e

echo "🎵 MusicGen Scripts — Setup"
echo

# Check Python version
PYTHON_VERSION=$(python3.11 --version 2>&1 | cut -d' ' -f2 || echo "not found")
if [[ "$PYTHON_VERSION" == "not found" ]]; then
    echo "❌ Python 3.11 not found."
    echo "   Install with: brew install python@3.11"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION found"

# Create venv
if [ ! -d ".venv" ]; then
    echo
    echo "📦 Creating virtual environment..."
    python3.11 -m venv .venv
fi
source .venv/bin/activate

echo "✅ Virtual environment activated"
echo

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first
echo
echo "📦 Installing PyTorch (CPU)..."
pip install torch==2.1.0 torchaudio==2.1.0

# Install other deps
echo
echo "📦 Installing transformers, scipy..."
pip install transformers==4.37.0 numpy scipy

echo
echo "✅ Setup complete!"
echo
echo "To generate music:"
echo "   source .venv/bin/activate"
echo "   python scripts/generate_music.py"
