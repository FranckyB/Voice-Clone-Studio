#!/bin/bash
# macOS installation helper for Voice Clone Studio
# Supports Apple Silicon (M1/M2/M3/M4) and Intel Macs
# Uses MPS (Metal Performance Shaders) for GPU acceleration on Apple Silicon

set -e  # Exit on error

echo "========================================="
echo "Voice Clone Studio - macOS Setup Helper"
echo "========================================="
echo ""

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "Detected: Apple Silicon ($ARCH) - MPS acceleration available"
else
    echo "Detected: Intel Mac ($ARCH) - CPU-only mode"
fi
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
        echo "Install with Homebrew: brew install python@3.12"
        exit 1
    fi
fi

# Validate the version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MINOR" -lt 10 ] || [ "$PYTHON_MINOR" -gt 12 ]; then
    echo "ERROR: Python 3.10-3.12 is required. Detected: $PYTHON_VERSION"
    echo "Python 3.13+ is not supported due to dependency conflicts."
    echo "Install with Homebrew: brew install python@3.12"
    exit 1
fi
echo "Using: $PYTHON_CMD (Python $PYTHON_VERSION)"
echo ""

echo "Note: Model training is not supported on macOS"
echo "   The Train Model tab will be automatically hidden"
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
echo "Optional: Install Whisper speech recognition?"
echo "OpenAI Whisper is an alternative transcription engine."
echo "Supports automatic splitting of clips with no length limit."
echo "========================================="
echo ""
read -t 30 -p "Install Whisper? (y/N, default N in 30s): " INSTALL_WHISPER
INSTALL_WHISPER=${INSTALL_WHISPER:-N}
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
echo "Note: installation may fail on some macOS environments."
echo "========================================="
echo ""
read -t 30 -p "Install DeepFilterNet? (y/N, default N in 30s): " INSTALL_DEEPFILTER
INSTALL_DEEPFILTER=${INSTALL_DEEPFILTER:-N}
echo ""

echo "All questions answered - installing now..."
echo ""

# Install system dependencies via Homebrew
if command -v brew >/dev/null 2>&1; then
    echo "Installing system dependencies via Homebrew..."
    brew install ffmpeg sox
    echo ""
else
    echo "WARNING: Homebrew not found!"
    echo "Install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "Then re-run this script, or install ffmpeg and sox manually."
    echo ""
    read -p "Continue without Homebrew? (y/N): " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        exit 1
    fi
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

# Install PyTorch (MPS support is built into default PyTorch builds)
echo ""
echo "Installing PyTorch with MPS support..."
echo "(This may take a while...)"
pip install torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1
echo ""

# Install requirements (skip GPU-specific packages)
echo "Installing dependencies..."
if [ -f "requirements_linux.txt" ]; then
    # Install requirements but skip packages that don't work on macOS
    # onnxruntime-gpu is CUDA-only, deepfilternet may have issues
    pip install -r requirements_linux.txt --no-deps 2>/dev/null || true

    # Install core packages explicitly (skip problematic ones)
    pip install qwen-tts librosa torchaudio soundfile sox einops huggingface_hub
    pip install gradio numpy diffusers markdown
    pip install open_clip_torch torchdiffeq timm colorlog opencv-python omegaconf

    # Install custom Gradio component
    if [ -f "wheel/gradio_filelister-0.4.0-py3-none-any.whl" ]; then
        pip install wheel/gradio_filelister-0.4.0-py3-none-any.whl
    fi

    # Install Chatterbox Voice Conversion dependencies
    echo ""
    echo "Installing Chatterbox Voice Conversion dependencies..."
    pip install s3tokenizer conformer

    # Fish Speech S2 codec (installed with --no-deps to avoid protobuf conflict)
    echo ""
    echo "Installing Fish Speech S2 codec..."
    pip install --no-deps descript-audio-codec descript-audiotools

    # Install onnxruntime (CPU version, not GPU)
    echo ""
    echo "Installing ONNX Runtime (CPU)..."
    pip install onnxruntime
    if python3 -c "import onnxruntime" 2>/dev/null; then
        echo "ONNX Runtime is working"
    else
        echo "WARNING: ONNX Runtime import failed."
        echo "Some features may not be available."
    fi

    # Try bitsandbytes (may not work on all Mac configurations)
    echo ""
    echo "Installing bitsandbytes (4-bit quantization)..."
    if pip install bitsandbytes 2>/dev/null; then
        echo "bitsandbytes installed"
    else
        echo "WARNING: bitsandbytes installation failed."
        echo "4-bit quantization will not be available."
    fi

    # Try deepfilternet (optional)
    if [[ "$INSTALL_DEEPFILTER" =~ ^[Yy]$ ]]; then
        echo ""
        echo "Installing DeepFilterNet (audio denoising)..."
        if pip install deepfilternet 2>/dev/null; then
            echo "DeepFilterNet installed"
        else
            echo "WARNING: DeepFilterNet installation failed."
            echo "Audio denoising will not be available."
        fi
    else
        echo ""
        echo "Skipping DeepFilterNet installation."
    fi
else
    echo "ERROR: requirements_linux.txt not found!"
    exit 1
fi

# Optional: LuxTTS
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

# Whisper
if [[ "$INSTALL_WHISPER" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing Whisper..."
    if pip install openai-whisper; then
        echo "Whisper installed successfully!"
    else
        echo "Whisper installation failed."
    fi
else
    echo "Skipping Whisper installation."
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
    else
        echo "Neither Homebrew nor MacPorts found."
        echo "Please install llama.cpp manually from: https://github.com/ggml-org/llama.cpp"
    fi
else
    echo "Skipping llama.cpp installation."
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "macOS Notes:"
echo "  - Apple Silicon Macs use MPS (Metal) for GPU acceleration"
echo "  - Intel Macs will run in CPU-only mode"
echo "  - Recommended attention mechanism: sdpa (set in Settings)"
echo "  - Model training is not supported on macOS"
echo "  - Flash Attention 2 is not available (CUDA only)"
echo ""
echo "To run the application:"
echo "  1. source venv/bin/activate"
echo "  2. python voice_clone_studio.py"
echo "  3. Or use: ./launch.sh"
echo ""
