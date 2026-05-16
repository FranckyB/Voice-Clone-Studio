#!/bin/bash
# Linux installation helper for Voice Clone Studio
# This script helps with common Linux installation issues

set -e  # Exit on error

echo "========================================="
echo "Voice Clone Studio - Linux Setup Helper"
echo "========================================="
echo ""

# Find a compatible Python version (3.10-3.12, 3.13+ not supported)
PYTHON_CMD=""
for PYVER in python3.11 python3.10 python3.12; do
    if command -v "$PYVER" >/dev/null 2>&1; then
        PYTHON_CMD="$PYVER"
        break
    fi
done

# Fall back to python3 if specific versions weren't found
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    else
        echo "ERROR: Python not found! Please install Python 3.10-3.12."
        echo "Install Python 3.12: https://www.python.org/downloads/"
        exit 1
    fi
fi

# Validate the version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MINOR" -lt 10 ] || [ "$PYTHON_MINOR" -gt 12 ]; then
    echo "ERROR: Python 3.10-3.12 is required. Detected: $PYTHON_VERSION"
    echo "Python 3.13+ is not supported due to dependency conflicts."
    echo "Install Python 3.12: https://www.python.org/downloads/"
    exit 1
fi
echo "Using: $PYTHON_CMD (Python $PYTHON_VERSION)"
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

echo "========================================="
echo "Optional: Install Qwen3 ASR speech recognition?"
echo "Qwen3 ASR provides high-quality multilingual speech recognition."
echo "Supports 52 languages with Small (0.6B) and Large (1.7B) models."
echo "Note: This will update transformers to 4.57.6+"
echo "========================================="
echo ""
read -t 30 -p "Install Qwen3 ASR? (y/N, default N in 30s): " INSTALL_QWEN3ASR
INSTALL_QWEN3ASR=${INSTALL_QWEN3ASR:-N}
echo ""

echo "========================================="
echo "Optional: Install llama.cpp for LLM prompt generation?"
echo "llama.cpp powers the Prompt Manager's local LLM feature."
echo "Lets you generate TTS and SFX prompts using Qwen3 models."
echo "========================================="
echo ""
read -t 30 -p "Install llama.cpp? (y/N, default N in 30s): " INSTALL_LLAMA
INSTALL_LLAMA=${INSTALL_LLAMA:-N}
echo ""

echo "========================================="
echo "Optional: Install DeepFilterNet for audio denoising?"
echo "DeepFilterNet removes background noise from voice samples."
echo "Note: Linux may require Rust compiler for source builds."
echo "========================================="
echo ""
read -t 30 -p "Install DeepFilterNet? (y/N, default N in 30s): " INSTALL_DEEPFILTER
INSTALL_DEEPFILTER=${INSTALL_DEEPFILTER:-N}
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
    $PYTHON_CMD -m venv venv
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
pip install torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
echo ""
echo "Installing dependencies (using requirements_linux.txt)..."
if [ -f "requirements_linux.txt" ]; then
    pip install -r requirements_linux.txt
else
    echo "⚠️  requirements_linux.txt not found!"
    exit 1
fi

# Fish Speech S2 codec (installed with --no-deps to avoid protobuf conflict)
echo "Installing Fish Speech S2 codec..."
pip install --no-deps descript-audio-codec descript-audiotools >/dev/null 2>&1 || true

# DeepFilterNet audio denoising (optional - requires Rust compiler for source build)
if [[ "$INSTALL_DEEPFILTER" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing DeepFilterNet (audio denoising)..."
    if pip install deepfilternet; then
        echo "DeepFilterNet installed successfully!"
    else
        echo "WARNING: DeepFilterNet installation failed (requires Rust compiler to build from source)."
        echo "Audio denoising will not be available, but all other features will work normally."
        echo "To install later: Install Rust from https://rustup.rs then run: pip install deepfilternet"
    fi
else
    echo "Skipping DeepFilterNet installation."
fi

# Faster Qwen3 TTS (CUDA graph acceleration, lightweight - auto-install)
echo ""
echo "Installing Faster Qwen3 TTS (CUDA graph acceleration)..."
if pip install faster-qwen3-tts; then
    echo "Faster Qwen3 TTS installed successfully!"
else
    echo "WARNING: Faster Qwen3 TTS installation failed. Standard engine will be used."
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

# Qwen3 ASR
if [[ "$INSTALL_QWEN3ASR" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing Qwen3 ASR..."
    pip install nagisa soynlp qwen-omni-utils pytz flask
    if pip install qwen-asr; then
        echo "Qwen3 ASR installed successfully!"
    else
        echo "Qwen3 ASR installation failed."
    fi
else
    echo "Skipping Qwen3 ASR installation."
fi

# llama.cpp
if [[ "$INSTALL_LLAMA" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing llama.cpp..."
    if command -v brew >/dev/null 2>&1; then
        if brew install llama.cpp; then
            echo "llama.cpp installed successfully!"
        else
            echo "llama.cpp installation via Homebrew failed."
            echo "You can install manually from: https://github.com/ggml-org/llama.cpp"
        fi
    elif command -v port >/dev/null 2>&1; then
        if sudo port install llama.cpp; then
            echo "llama.cpp installed successfully!"
        else
            echo "llama.cpp installation via MacPorts failed."
            echo "You can install manually from: https://github.com/ggml-org/llama.cpp"
        fi
    elif command -v nix >/dev/null 2>&1; then
        if nix profile install nixpkgs#llama-cpp; then
            echo "llama.cpp installed successfully!"
        else
            echo "llama.cpp installation via Nix failed."
            echo "You can install manually from: https://github.com/ggml-org/llama.cpp"
        fi
    else
        echo "No supported package manager found (brew, port, nix)."
        echo "Please install llama.cpp manually from: https://github.com/ggml-org/llama.cpp"
    fi
else
    echo "Skipping llama.cpp installation."
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
