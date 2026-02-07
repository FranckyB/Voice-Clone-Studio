# Version History


## February 7, 2026

#### Version 1.0.0 - Complete Modular Rewrite
- **LuxTTS Voice Clone Engine** - Added LuxTTS as a third voice cloning engine with 48 kHz output, caching, and UI controls
- **Full Modular Architecture** - Complete rewrite from a 6000+ line monolith into independent tool modules under `modules/core_components/tools/`
- **Tool System** - Each tab is now a self-contained tool with its own UI, events, and logic, loaded dynamically from a central registry
- **Enable/Disable Tools** - New "Visible Tools" section in Settings lets you toggle any tab on or off (persisted in config, takes effect on restart)
- **Simplified Prep Audio** - Formerly "Prep Samples", now serves dual purpose for both voice sample preparation and dataset creation in a single unified tool
- **Improved FileLister Component** - Custom Gradio component (v0.4.0) with multi-select for batch file deletion and double-click to instantly play audio
- **Help Guide in Settings** - Help documentation moved into the Settings tab as a sub-tab, keeping the main tab bar clean
- **Settings Tab Right-Aligned** - Settings gear icon pushed to the far right of the tab bar for quick access
- **Centralized Constants** - All model sizes, languages, speakers, and defaults defined once in `constants.py`
- **AI Model Managers** - Centralized TTS and ASR model management with automatic VRAM optimization and model switching
- **Shared State Architecture** - Tools receive configuration, utilities, and managers through a unified shared state, enabling independent testing
- **Cleaned Up Project** - Removed obsolete documentation, stale files, and migration artifacts

## January 30, 2026

#### Version 0.7.6 - Advanced Parameters & Emotion Presets
- Bug Fixes

## January 30, 2026

#### Version 0.7.5 - Advanced Parameters & Emotion Presets
- **Added option to save Emotions** - Improved Emotion system by allowing the user to create and save their own preset.
- **Moved the Emotion Manager to Modules** - Plans are to split the app into modules, the emotion manager is the first one.

#### Version 0.7.0 - Advanced Parameters & Emotion Presets
- **Advanced Parameter Controls** - Full access to sampling parameters (temperature, top_k, top_p, repetition_penalty, max_new_tokens) across all tabs
- **Voice Clone Tab** - Emotion presets with intensity slider for Qwen models
- **Voice Presets Tab** - Context-aware controls: style instructions for Premium Speakers, emotion presets for Trained Models
- **Conversation Tab** - Model-specific advanced parameters: VibeVoice diffusion controls (CFG, inference steps, LM sampling), Qwen sampling parameters
- **Voice Design Tab** - Advanced Qwen parameters for fine-tuning voice generation
- **Emotion Preset System** - Added emotion presets with intensity control - Inspired by [Qwen3-TTS Emotional Voice Clone](https://github.com/Dawizzer/ComfyUI-Qwen3TTS-Emotional)
- **Offline Mode** - Added offline mode toggle in Settings to use locally cached models without internet access
- **Model Download Tool** - Direct model download to `models/` folder with progress tracking in console
- **Training Script Fix** - Added attention mechanism fallback (flash_attention_2 → sdpa → eager) to prevent failures without flash-attn

## January 27, 2026

#### Version 0.6.5 - Improved Conversation Tool
- **Three Conversation Modes** - Qwen CustomVoice (9 preset speakers), Qwen Base (8 custom voice samples), and VibeVoice (4 custom voice samples)
- **VRAM Optimization** - Automatic model unloading when switching between conversation modes
- **Global Model Management** - Unload All Models button for manual VRAM cleanup
- **Voice Sample Persistence** - Dropdowns remember your selected voice samples across sessions
- **Improved Conversation Options** - Inspired by [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS)

#### Version 0.6.0 - Enhanced Model Support & Settings
- **VibeVoice Large 4-bit** - Added support for quantized 4-bit VibeVoice Large model for reduced VRAM usage
- **Settings Tab** - New centralized settings interface with configurable folder paths
- **Attention Mechanism Selection** - Choose between SAGE (fastest), Flash Attention 2, SDPA, or Eager with automatic fallback
- **Low CPU Memory Option** - Toggle to reduce CPU memory usage during model loading (all models)
- **UI Improvements** - Reorganized Voice Clone tab with conditional visibility and better layout
- **Refresh Voice Samples** - Added button in Conversation tab to refresh voice sample dropdowns

#### Version 0.5.5 - Platform Support & Requirements
- Consolidated requirements into single universal `requirements.txt` with platform markers
- Added platform-specific setup scripts (Windows/Linux)
- Enhanced setup automation

## January 26, 2026

#### Version 0.5.1 - Training Infrastructure
- Removed bundled Qwen3-TTS module (now using PyPI package)
- Added Qwen fine-tuning scripts for custom voice training
- Checkpoint management with configurable save intervals

#### Version 0.5.0 - UI Polish & Help System
- **Help Guide Tab** - Comprehensive in-app documentation with 8 topic sections
- **Modular Help System** - Extracted help content to separate `ui_help.py` module
- **Better Text Formatting** - Markdown rendering with scrollable containers
- gr.HTML styling improvements with container/padding support
- Label color matching enhancements

#### Version 0.4.0 - Custom Voice Training
- Added **Train Model** tab for fine-tuning custom voices
- Complete training pipeline with validation, data preparation, and model training
- **Batch Transcription** - Process 50-100+ audio files in one click
- Support for both 0.6B and 1.7B base models
- Real-time training progress monitoring with live loss values
- Checkpoint management - compare different training epochs
- Integration with Voice Presets tab for using trained models
- Dataset organization system with `datasets/` folder structure
- Automatic audio format conversion (24kHz 16-bit mono)

## January 25, 2026

#### Version 0.3.5 - Style Instructions
- Added Style Instructions support in Conversation for Qwen model (Unsupported by VibeVoice)

#### Version 0.3.0 - Enhanced Media Support
- **Video File Support** - Upload video files (.mp4, .mov, .avi, .mkv, etc.) to Prep Samples tab
- **Automatic Audio Extraction** - Uses ffmpeg to extract audio from video files for voice cloning
- **Improved Workflow** - Added Clear button to quickly reset the audio editor
- Enhanced media handling and file upload capabilities

## January 24, 2026

#### Version 0.2.0 - VibeVoice Integration
- Added **VibeVoice TTS** support for long-form multi-speaker generation (up to 90 minutes)
- Added **VibeVoice ASR** as alternative transcription engine alongside Whisper
- Conversation tab now supports both Qwen (9 preset voices) and VibeVoice (custom samples) engines
- Multi-speaker conversation support with up to 4 custom voices
- Added Output History management
- Removed Clone Design tab

## January 23, 2026

#### Version 0.1.0 - Initial Release
- Voice cloning with Qwen3-TTS (Base, CustomVoice, VoiceDesign models)
- Whisper-powered automatic transcription
- Sample preparation toolkit (trim, normalize, mono conversion)
- Voice prompt caching for faster generation
- Seed control for reproducible outputs
- 9 premium preset voices (Qwen CustomVoice)
- Voice Design from text descriptions
- Conversation mode with multi-speaker support
