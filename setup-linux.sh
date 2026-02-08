#!/bin/bash
# Linux installation helper for Voice Clone Studio
# This script helps with common Linux installation issues

set -e  # Exit on error

echo "========================================="
echo "Voice Clone Studio - Linux Setup Helper"
echo "========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"
echo ""
echo "Note: openai-whisper is not installed on Linux (compatibility issues)"
echo "   VibeVoice ASR will be used for transcription instead"
echo ""

# Ask all questions upfront so installation can run unattended
echo "========================================="
echo "Optional: Install LuxTTS voice cloning engine?"
echo "LuxTTS provides fast, high-quality voice cloning at 48kHz."
echo "Requires ~1GB disk space for model files."
echo "========================================="
echo ""
read -t 30 -p "Install LuxTTS? (y/N, default N in 30s): " INSTALL_LUXTTS
INSTALL_LUXTTS=${INSTALL_LUXTTS:-N}
echo ""
echo "All questions answered - installing now..."
echo ""

# Install system dependencies (Ubuntu/Debian)
if command -v apt >/dev/null 2>&1; then
  echo "Detected apt-based system (Ubuntu/Debian), installing system packages..."
  sudo apt update
  sudo apt install -y ffmpeg sox libsox-fmt-all
else
  echo "apt not found, skipping system package install (please install ffmpeg and sox manually)."
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists, skipping creation..."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA 13.0 support..."
echo "(This may take a while...)"
pip install torch==2.9.1 torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
echo ""
echo "Installing dependencies (using requirements.txt)..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found!"
    exit 1
fi

# Check for ONNX Runtime issues
echo ""
echo "Checking ONNX Runtime installation..."
if python3 -c "import onnxruntime" 2>/dev/null; then
    echo "✅ ONNX Runtime is working"
else
    echo "⚠️  ONNX Runtime import failed. Trying nightly build..."
    pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime
fi

# Optional modules
echo ""
echo "Installing optional modules..."
if [[ "$INSTALL_LUXTTS" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing LuxTTS prerequisites..."
    echo "[Step 1/3] Installing LinaCodec..."
    if pip install git+https://github.com/ysharma3501/LinaCodec.git; then
        echo "[Step 2/3] Installing piper-phonemize..."
        if pip install piper-phonemize --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html; then
            echo "[Step 3/3] Installing zipvoice (LuxTTS)..."
            if pip install "zipvoice @ git+https://github.com/ysharma3501/LuxTTS.git"; then
                echo "LuxTTS installed successfully!"
            else
                echo "zipvoice installation failed. LuxTTS will not be available."
            fi
        else
            echo "piper-phonemize installation failed. LuxTTS will not be available."
        fi
    else
        echo "LinaCodec installation failed. LuxTTS will not be available."
    fi
else
    echo "Skipping LuxTTS installation."
fi

echo ""
echo "========================================="
echo "✅ Setup complete!"
echo "========================================="
echo ""
echo "OPTIONAL: Install Flash Attention 2 for better performance"
echo ""
echo "Option 1 - Build from source (requires C++ compiler):"
echo "  pip install flash-attn --no-build-isolation"
echo ""
echo "Option 2 - Use prebuilt wheel (faster, no compiler needed):"
echo "  Download a wheel matching your Python version"
echo "  Then: pip install downloaded-wheel-file.whl"
echo ""
echo "  Possible source for wheels:"
echo "  https://huggingface.co/MonsterMMORPG/Wan_GGUF/tree/main"
echo "========================================="
echo ""
echo "To run the application:"
echo "  1. source venv/bin/activate"
echo "  2. python voice_clone_studio.py"
echo "  3. Or use: launch.sh"
echo ""
echo "NOTE: VibeVoice ASR is used for transcription on Linux."
echo ""
