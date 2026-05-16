# Voice Clone Studio — DramaBox Edition

> **This is a stripped-down version of [Voice Clone Studio](https://github.com/FranckyB/Voice-Clone-Studio), rebuilt around [DramaBox](https://github.com/Resemble-AI/DramaBox) as the sole TTS engine.**
>**Meant for Voice Cloning as well as Lora generation for ComfyUI**
>
> For the full-featured version with multiple TTS engines (Qwen3-TTS, VibeVoice, LuxTTS, and more), see the main repo: **https://github.com/FranckyB/Voice-Clone-Studio**

---

> ⚠️ **DramaBox is highly prone to hallucination.** It will frequently add, drop, or mangle words — especially on longer texts. **Expect to generate multiple times** before getting a clean output. Using a seed lets you reproduce good results once you find them.

---

A modular Gradio app for local voice workflows powered by DramaBox:

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
- LoRA support for finetuned speaker voices
- Paragraph splitting for long-form generation
- Prompt Hub integration
- Output metadata tracking

### Prep Audio
- Trim, normalize, mono conversion, denoise
- Extract audio from video
- Batch transcription with Qwen3-ASR or Whisper
- Save processed samples and manage datasets

### Train Model (DramaBox)
- Finetune DramaBox models from dataset folders
- Progress and log streaming in UI
- Configurable training options

### Voice Design
Create voices from natural language descriptions — no audio needed, using Qwen3-TTS Voice Design Model:
- Describe age, gender, emotion, accent, speaking style
- Generate unique voices matching your description

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

> **Note:** Manual installation is not recommended. The setup scripts handle vendor dependencies, custom wheels, and module patching that cannot be replicated with a plain `pip install`.

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

- The only active TTS engine is DramaBox. The multi-engine support from the main Voice Clone Studio repo (Qwen3-TTS, VibeVoice, LuxTTS) is not included here.
- **DramaBox hallucinates frequently.** Short, simple sentences work best. For longer texts, use "Split Audio by Paragraph" and review each segment.
- For the full-featured version: **https://github.com/FranckyB/Voice-Clone-Studio**

## License

This project is licensed under Apache 2.0. See [LICENSE](LICENSE).

Third-party projects used include:
- [DramaBox](https://github.com/Resemble-AI/DramaBox)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Whisper](https://github.com/openai/whisper)
- [MMAudio](https://github.com/hkchengrex/MMAudio)
- [Gradio](https://gradio.app/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
