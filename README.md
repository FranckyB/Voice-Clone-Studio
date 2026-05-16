# Voice Clone Studio

Voice Clone Studio DramaBox edition is a modular Gradio app focused on local voice workflows:

- DramaBox for TTS voice cloning and finetuning
- MMAudio for sound effects
- Qwen3-ASR and Whisper for transcription
- llama.cpp or Ollama for prompt generation

<img src="https://img.shields.io/badge/DramaBox-TTS-blue" alt="DramaBox TTS"> <img src="https://img.shields.io/badge/Qwen3--ASR-ASR-blue" alt="Qwen3 ASR"> <img src="https://img.shields.io/badge/Whisper-ASR-yellow" alt="Whisper"> <img src="https://img.shields.io/badge/MMAudio-SFX-green" alt="MMAudio"> <img src="https://img.shields.io/badge/llama.cpp-LLM-orange" alt="llama.cpp">

<a href="docs/preview.png"><img src="docs/preview.png" alt="Voice Clone Studio Preview" width="600"></a>

## Features

### Voice Clone (DramaBox)
- Clone voices from your own samples
- Seeded generation for reproducible outputs
- Emotion preset controls
- Prompt Hub integration
- Output metadata tracking

### Train Model (DramaBox)
- Finetune DramaBox models from dataset folders
- Progress and log streaming in UI
- Configurable training options

### Prep Audio
- Trim, normalize, mono conversion, denoise
- Extract audio from video
- Batch transcription with Qwen3-ASR or Whisper
- Save processed samples and manage datasets

### Sound Effects
- Text-to-audio and video-to-audio generation with MMAudio
- Model and generation controls for SFX workflows

### Prompt Manager
- Save and reuse prompts
- Generate prompts locally with llama.cpp or Ollama

### Output History
- Browse, preview, and manage generated files

### Settings
- Enable or disable visible tools
- Configure output and folder paths
- Toggle model engine availability
- Download DramaBox and ASR models

## Installation

### Prerequisites
- Python 3.10-3.12
- Windows or Linux: CUDA GPU recommended for best performance
- macOS: Apple Silicon works with MPS (training can be restricted)
- SOX
- FFmpeg
- Optional: llama.cpp or Ollama

### Quick Setup

#### Windows
```bash
setup-windows.bat
```

#### Linux
```bash
chmod +x setup-linux.sh
./setup-linux.sh
```

#### macOS
```bash
chmod +x setup-mac.sh
./setup-mac.sh
```

### Manual Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux or macOS
source venv/bin/activate

pip install torch==2.9.1 torchaudio==2.9.1
pip install -r requirements_winodws.txt   # Windows
# or
pip install -r requirements_linux.txt      # Linux or macOS
```

## Usage

```bash
python voice_clone_studio.py
```

Or use launcher scripts:
- Windows: `launch.bat`
- Linux or macOS: `./launch.sh`

Default UI URL: `http://127.0.0.1:7860`

## Project Structure

```text
Voice-Clone-Studio/
+- voice_clone_studio.py
+- config.json
+- requirements_winodws.txt
+- requirements_linux.txt
+- launch.bat / launch.sh
+- setup-windows.bat / setup-linux.sh / setup-mac.sh
+- docs/
+- modules/
   +- core_components/
      +- tools/
         +- voice_clone.py
         +- prep_audio.py
         +- train_model.py
         +- sound_effects.py
         +- prompt_generator.py
         +- output_history.py
         +- settings.py
      +- ai_models/
      +- help_page.py
   +- mmaudio/
   +- qwen_finetune/
   +- vibevoice_asr/
```

## Notes
- The active TTS path is DramaBox only.
- Old TTS tool tabs (Voice Presets, Conversation, Voice Design, Voice Changer) are removed from the active app surface.

## License

This project is licensed under Apache 2.0. See [LICENSE](LICENSE).

Third-party projects used include:
- [DramaBox](https://github.com/Resemble-AI/DramaBox)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Whisper](https://github.com/openai/whisper)
- [MMAudio](https://github.com/hkchengrex/MMAudio)
- [Gradio](https://gradio.app/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
