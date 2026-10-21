#!/usr/bin/env bash
# One-time setup for the HQ worker on Linux / Google Colab (needs NVIDIA GPU, python3, ffmpeg).
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$DIR")"
cd "$DIR"

echo "=== Prompt2Rig HQ Worker Setup (Linux / Colab) ==="

PIP_CMD="pip"
if command -v uv &> /dev/null; then
    PIP_CMD="uv pip"
fi

echo "1/4. Installing build tools & Python dependencies..."
$PIP_CMD install pip setuptools wheel "numpy>=1.26"
$PIP_CMD install --no-build-isolation "chumpy==0.70"
$PIP_CMD install -r requirements.txt
$PIP_CMD install --no-deps "git+https://github.com/warmshao/WiLoR-mini"

echo "2/4. Setting up GVHMR checkpoints..."
export GVHMR_CHECKPOINTS="$DIR/weights"
export GVHMR_BODY_MODELS="$DIR/body_models"
mkdir -p "$DIR/weights"
mkdir -p "$DIR/body_models/smplx"

gvhmr download demo

echo "3/4. Setting up finger tracking (MediaPipe fallback)..."
python3 setup_hands.py --mediapipe

echo "4/4. Checking SMPL-X model..."
SMPLX_FILE="$DIR/body_models/smplx/SMPLX_NEUTRAL.npz"
if [ ! -f "$SMPLX_FILE" ]; then
    echo "Downloading SMPLX_NEUTRAL.npz from mirror..."
    curl -L --retry 3 -o "$SMPLX_FILE" "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz" || {
        echo "Warning: Auto-download failed. Please place SMPLX_NEUTRAL.npz into hq/body_models/smplx/ manually."
    }
fi

echo ""
echo "=== Setup complete! Start with: python3 hq/worker.py ==="
