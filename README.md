# Voice Clone Studio

A modular Gradio-based web UI for voice cloning, voice design, and multi-speaker conversation, powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [VibeVoice](https://github.com/microsoft/VibeVoice), and [LuxTTS](https://github.com/ysharma3501/LuxTTS).
Supports both Whisper and VibeVoice ASR for automatic transcription.

![Voice Clone Studio](https://img.shields.io/badge/Voice%20Clone%20Studio-v1.0-blue) ![Qwen3-TTS](https://img.shields.io/badge/Qwen3--TTS-Powered-blue) ![LuxTTS](https://img.shields.io/badge/LuxTTS-TTS-orange) ![VibeVoice](https://img.shields.io/badge/VibeVoice-TTS-green) ![VibeVoice](https://img.shields.io/badge/VibeVoice-ASR-green) 

## Architecture

Voice Clone Studio is fully modular. The main file dynamically loads self-contained tools as tabs. Each tool can be enabled or disabled from Settings without touching any code.  It supports multipe engine for voice cloning, as well as Model finetuning. More features are also planned.

## Features

### Voice Clone
Clone voices from your own audio samples. Provide a short reference audio clip with its transcript, and generate new speech in that voice.

- **Multiple engines** - Qwen3-TTS (0.6B/1.7B), VibeVoice (1.5B/Large/Large-4bit), or LuxTTS (Default)
- **Voice prompt caching** - First generation processes the sample, subsequent ones are instant
- **Seed control** - Reproducible results with saved seeds
- **Emotion presets** - 40+ emotion presets with adjustable intensity
- **Metadata tracking** - Each output saves generation info (sample, seed, text)

### Conversation
Create multi-speaker dialogues using either Qwen's premium voices or your own custom voice samples using VibeVoice:

**Choose Your Engine:**
- **Qwen** - Fast generation with 9 preset voices, optimized for their native languages
- **VibeVoice** - High-quality custom voices, up to 90 minutes continuous, perfect for podcasts/audiobooks

**Unified Script Format:**
Write scripts using `[N]:` format - works seamlessly with both engines:
```
[1]: Hey, how's it going?
[2]: I'm doing great, thanks for asking!
[3]: Mind if I join this conversation?
```

**Qwen Mode:**
- Mix any of the 9 premium speakers
- Adjustable pause duration between lines
- Fast generation with cached prompts

**Speaker Mapping:**
- [1] = Vivian, [2] = Serena, [3] = Uncle_Fu, [4] = Dylan, [5] = Eric
- [6] = Ryan, [7] = Aiden, [8] = Ono_Anna, [9] = Sohee

**VibeVoice Mode:**
- **Up to 90 minutes** of continuous speech
- **Up to 4 distinct speakers** using your own voice samples
- Cross-lingual support
- May spontaneously add background music/sounds for realism
- Numbers beyond 4 wrap around (5→1, 6→2, 7→3, 8→4, etc.)

Perfect for:
- Podcasts
- Audiobooks
- Long-form conversations
- Multi-speaker narratives

**Models:**
- **Small** - Faster generation (Qwen: 0.6B, VibeVoice: 1.5B)
- **Large** - Best quality (Qwen: 1.7B, VibeVoice: Large model)


### Voice Presets
Generate with premium pre-built voices with optional style instructions using Qwen3-TTS Custom Model:

| Speaker | Description | Language |
|---------|-------------|----------|
| Vivian | Bright, slightly edgy young female | Chinese |
| Serena | Warm, gentle young female | Chinese |
| Uncle_Fu | Seasoned male, low mellow timbre | Chinese |
| Dylan | Youthful Beijing male, clear natural | Chinese (Beijing) |
| Eric | Lively Chengdu male, husky brightness | Chinese (Sichuan) |
| Ryan | Dynamic male, strong rhythmic drive | English |
| Aiden | Sunny American male, clear midrange | English |
| Ono_Anna | Playful Japanese female, light nimble | Japanese |
| Sohee | Warm Korean female, rich emotion | Korean |

- Style instructions supported (emotion, tone, speed)
- Each speaker works best in native language but supports all

### Voice Design
Create voices from natural language descriptions - no audio needed, using Qwen3-TTS Voice Design Model:

- Describe age, gender, emotion, accent, speaking style
- Generate unique voices matching your description

### Train Custom Voices
Fine-tune your own custom voice models with your training data:

- **Dataset Management** - Organize training samples in the `datasets/` folder
- **Audio Preparation** - Auto-converts to 24kHz 16-bit mono format
- **Training Pipeline** - Complete 3-step workflow (validation → extract codes → train)
- **Epoch Selection** - Compare different training checkpoints
- **Live Progress** - Real-time training logs and loss monitoring
- **Voice Presets Integration** - Use trained models alongside premium speakers

**Requirements:**
- CUDA GPU required
- Multiple audio samples with transcripts
- Training time: ~10-30 minutes depending on dataset size

**Workflow:**
1. Prepare audio files (WAV/MP3) and organize in `datasets/YourSpeakerName/` folder
2. Use **Batch Transcribe** to automatically transcribe all files at once
3. Review and edit individual transcripts as needed
4. Configure training parameters (model size, epochs, learning rate)
5. Monitor training progress in real-time
6. Use trained model in Voice Presets tab

### Prep Audio
Unified audio preparation workspace for both voice samples and training datasets:

- **Trim** - Use waveform selection to cut audio
- **Normalize** - Balance audio levels
- **Convert to Mono** - Ensure single-channel audio
- **Denoise** - Clean audio with DeepFilterNet
- **Transcribe** - Whisper or VibeVoice ASR automatic transcription
- **Batch Transcribe** - Process entire folders of audio files at once
- **Save as Sample** - One-click sample creation
- **Dataset Management** - Organize and prepare training data

### Output History
View, play back, and manage your previously generated audio files. Multi-select for batch deletion, double-click to play.

### Settings
Centralized application configuration:

- **Model loading** - Attention mechanism, offline mode, low CPU memory usage
- **Folder paths** - Configurable directories for samples, output, datasets, models
- **Model downloads** - Download models directly to local storage
- **Visible Tools** - Enable or disable any tool tab (restart to apply)
- **Help Guide** - Built-in documentation for all tools

---
## Installation

### Prerequisites

- Python 3.10+ (recommended for all platforms)
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- **SOX**  (Sound eXchange) - Required for audio processing
- **FFMPEG** - Multimedia framework required for audio format conversion
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) (optional but recommended)

**Note for Linux users:** The Linux installation skips `openai-whisper` (compatibility issues). VibeVoice ASR is used for transcription instead.

### Setup

#### Quick Setup (Windows)

1. Clone the repository:
```bash
git clone https://github.com/FranckyB/Voice-Clone-Studio.git
cd Voice-Clone-Studio
```

2. Run the setup script:
```bash
setup-windows.bat
```

This will automatically:
- Install SOX (audio processing)
- Create virtual environment
- Install PyTorch with CUDA support
- Install all dependencies
- Display your Python version
- Show instructions for optional Flash Attention 2 installation

#### Quick Setup (Linux)

1. Clone the repository:
```bash
git clone https://github.com/FranckyB/Voice-Clone-Studio.git
cd Voice-Clone-Studio
```

2. Make the setup script executable and run it:
```bash
chmod +x setup-linux.sh
./setup-linux.sh
```

This will automatically:
- Detect your Python version
- Create virtual environment
- Install PyTorch with CUDA support
- Install all dependencies (using requirements file)
- Handle ONNX Runtime installation issues
- Warn about Whisper compatibility if needed

#### Manual Setup (All Platforms)

1. Clone the repository:
```bash
git clone https://github.com/FranckyB/Voice-Clone-Studio.git
cd Voice-Clone-Studio
```

2. Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/MacOs
source venv/bin/activate
```

3. (NVIDIA GPU) Install PyTorch with CUDA support:
```bash
# Linux/Windows
pip install torch==2.9.1 torchaudio --index-url https://download.pytorch.org/whl/cu130
```

4. Install dependencies:
```bash
# All platforms (Windows, Linux, macOS)
pip install -r requirements.txt
```

**Note:** The requirements file uses platform markers to automatically install the correct packages:
- Windows: Includes `openai-whisper` for transcription
- Linux/macOS: Excludes `openai-whisper` (uses VibeVoice ASR instead)

5. Install Sox

```bash
# Windows
winget install -e --id ChrisBagwell.SoX

# Linux
# Debian/Ubuntu
sudo apt install sox libsox-dev
# Fedora/RHEL
sudo dnf install sox sox-devel

# MacOs
brew install sox
```

6. Install ffmpeg

```bash
# Windows
winget install -e --id Gyan.FFmpeg

# Linux
# Debian/Ubuntu
sudo apt install ffmpeg
# Fedora/RHEL
sudo dnf install ffmpeg

# MacOs
brew install ffmpeg
```

7. (Optional) Install FlashAttention 2  for faster generation:
**Note:** The application automatically detects and uses the best available attention mechanism. Configure in Settings tab: `auto` (recommended) → `flash_attention_2` → `sdpa` → `eager`

## Troubleshooting
For troubleshooting solutions, see [docs/troubleshooting.md](docs/troubleshooting.md).

#### Docker Setup (Windows)

1. **Install NVIDIA Drivers (Windows Side)**
   - Install the latest standard NVIDIA driver (Game Ready or Studio) for Windows from the [NVIDIA Drivers page](https://www.nvidia.com/Download/index.aspx).
   - **Crucial:** Do *not* try to install NVIDIA drivers inside your WSL Linux terminal. It will conflict with the host driver.

2. **Update WSL 2**
   - Open **PowerShell** as Administrator and ensure your WSL kernel is up to date:
     ```powershell
     wsl --update
     ```
   - (If you don't have WSL installed yet, run `wsl --install` and restart your computer).

3. **Configure Docker Desktop**
   - Install the latest version of **Docker Desktop for Windows**.
   - Open Docker Desktop **Settings** (gear icon).
   - Under **General**, ensure **"Use the WSL 2 based engine"** is checked.
   - Under **Resources > WSL Integration**, ensure the switch is enabled for your default Linux distro (e.g., Ubuntu).

4. **Run with Docker Compose**
   - Run the following command in the repository root:
     ```powershell
     docker-compose up --build
     ```
   - The application will be accessible at `http://127.0.0.1:7860`.

### Running Tests (Docker)

To verify the installation and features (like the DeepFilterNet denoiser), runs the integration tests inside the container:

```powershell
# Run the Denoiser Integration Test
docker-compose exec voice-clone-studio python tests/integration_test_denoiser.py
```

## Usage

### Launch the UI

```bash
python voice_clone_studio.py
```

Or use the batch file (Windows):
```bash
launch.bat
```

The UI will open at `http://127.0.0.1:7860`

### Prepare Voice Samples

1. Go to the **Prep Audio** tab
2. Upload or record audio (3-10 seconds of clear speech)
3. Trim, normalize, and denoise as needed
4. Transcribe or manually enter the text
5. Save as a sample with a name

### Clone a Voice

1. Go to the **Voice Clone** tab
2. Select your sample from the dropdown
3. Enter the text you want to speak
4. Click Generate

### Design a Voice

1. Go to the **Voice Design** tab
2. Enter the text to speak
3. Describe the voice (e.g., "Young female, warm and friendly, slight British accent")
4. Click Generate

## Project Structure

```
Voice-Clone-Studio/
├── voice_clone_studio.py          # Main orchestrator (~230 lines)
├── config.json                    # User preferences & enabled tools
├── requirements.txt               # Python dependencies
├── launch.bat / launch.sh         # Launcher scripts
├── wheel/                         # Pre-built custom Gradio components
│   └── gradio_filelister-0.4.0-py3-none-any.whl
├── samples/                       # Voice samples (.wav + .json)
├── output/                        # Generated audio outputs
├── datasets/                      # Training datasets
├── models/                        # Downloaded & trained models
├── docs/                          # Documentation
│   ├── updates.md                 # Version history
│   ├── troubleshooting.md         # Troubleshooting guide
│   └── MODEL_MANAGEMENT_README.md # AI model manager docs
└── modules/
    ├── core_components/           # Core app code
    │   ├── tools/                 # All UI tools (tabs)
    │   │   ├── voice_clone.py
    │   │   ├── voice_presets.py
    │   │   ├── conversation.py
    │   │   ├── voice_design.py
    │   │   ├── prep_audio.py
    │   │   ├── output_history.py
    │   │   ├── train_model.py
    │   │   └── settings.py
    │   ├── ai_models/             # TTS & ASR model managers
    │   ├── ui_components/         # Modals, theme
    │   ├── gradio_filelister/     # Custom file browser component
    │   ├── constants.py           # Central constants
    │   ├── emotion_manager.py     # Emotion presets
    │   ├── audio_utils.py         # Audio processing
    │   └── help_page.py           # Help content
    ├── deepfilternet/             # Audio denoising
    ├── qwen_finetune/             # Training scripts
    ├── vibevoice_tts/             # VibeVoice TTS
    └── vibevoice_asr/             # VibeVoice ASR
```

## Models Used

Each tab lets you choose between model sizes:

| Model | Sizes | Use Case |
|-------|-------|----------|
| **Qwen3-TTS Base** | Small, Large | Voice cloning from samples |
| **Qwen3-TTS CustomVoice** | Small, Large | Premium speakers with style control |
| **Qwen3-TTS VoiceDesign** | 1.7B only | Voice design from descriptions |
| **VibeVoice-TTS** | Small, Large | Voice cloning & Long-form multi-speaker (up to 90 min) |
| **VibeVoice-ASR** | Large | Audio transcription |
| **Whisper** | Medium | Audio transcription |

- **Small** = Faster, less VRAM (Qwen: 0.6B ~4GB, VibeVoice: 1.5B)
- **Large** = Better quality, more expressive (Qwen: 1.7B ~8GB, VibeVoice: Large model)
- **4 Bit Quantized** version of the Large model is also included for VibeVoice.

Models are automatically downloaded on first use via HuggingFace.

## Tips

- **Reference Audio**: Use clear, noise-free recordings (3-10 seconds)
- **Transcripts**: Should exactly match what's spoken in the audio
- **Caching**: Voice prompts are cached - first generation is slow, subsequent ones are fast
- **Seeds**: Use the same seed to reproduce identical outputs

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

This project is based on and uses code from:
- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)**    - Apache 2.0 License (Alibaba)
- **[VibeVoice](https://github.com/microsoft/VibeVoice)** - MIT License
- **[Gradio](https://gradio.app/)**                       - Apache 2.0 License
- **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)** - MIT License

## Updates

For detailed version history and release notes, see [docs/updates.md](docs/updates.md).
