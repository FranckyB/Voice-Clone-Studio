"""
Voice Clone Studio - Central Constants

Single source of truth for all constants used throughout the application.
Add new models, languages, or speakers here - changes automatically propagate everywhere.

## How to Add New Constants

### Adding a New Language:
1. Add to LANGUAGES list: `LANGUAGES = [..., "Italian", "Dutch"]`
2. That's it! All dropdowns update automatically.

### Adding a New Speaker:
1. Add to CUSTOM_VOICE_SPEAKERS: `CUSTOM_VOICE_SPEAKERS = [..., "NewSpeaker"]`
2. Done! Speaker appears in all relevant UI components.

### Adding a New Model Size:
1. Update appropriate MODEL_SIZES_* constant
2. Example: `MODEL_SIZES_VIBEVOICE = [..., "XLarge"]`
3. All tools and UI components get the new option.

### Adding Generation Defaults:
1. Update QWEN_GENERATION_DEFAULTS or VIBEVOICE_GENERATION_DEFAULTS
2. These defaults are used across all generation functions.

## Import in Your Code:
```python
from modules.core_components.constants import (
    LANGUAGES,
    CUSTOM_VOICE_SPEAKERS,
    MODEL_SIZES_CUSTOM,
    QWEN_GENERATION_DEFAULTS
)
```

Or via the package:
```python
from modules.core_components import LANGUAGES, CUSTOM_VOICE_SPEAKERS
```
"""

# ============================================================================
# MODEL SIZES & OPTIONS
# ============================================================================

MODEL_SIZES = ["Small", "Large"]  # Small=0.6B, Large=1.7B
MODEL_SIZES_BASE = ["Small", "Large"]  # Base model: Small=0.6B, Large=1.7B
MODEL_SIZES_CUSTOM = ["Small", "Large"]  # CustomVoice: Small=0.6B, Large=1.7B
MODEL_SIZES_DESIGN = ["1.7B"]  # VoiceDesign only has 1.7B
MODEL_SIZES_VIBEVOICE = [
    "Small",
    "Large (4-bit)",
    "Large",
]  # VibeVoice: 1.5B, 7B-4bit, 7B

# Voice Clone engine and model options
VOICE_CLONE_OPTIONS = [
    "Qwen3 - Small",
    "Qwen3 - Large",
    "VibeVoice - Small",
    "VibeVoice - Large (4-bit)",
    "VibeVoice - Large",
    "LuxTTS - Default",
]

# Default to Large models for better quality (static fallback)
DEFAULT_VOICE_CLONE_MODEL = "Qwen3 - Large"

# TTS Engines - master list of engines and which dropdown entries belong to each
# Format: engine_key -> { label, choices (subset of VOICE_CLONE_OPTIONS), default_enabled }
TTS_ENGINES = {
    "Qwen3": {
        "label": "Qwen3-TTS",
        "choices": ["Qwen3 - Small", "Qwen3 - Large"],
        "default_enabled": True,
    },
    "VibeVoice": {
        "label": "VibeVoice",
        "choices": [
            "VibeVoice - Small",
            "VibeVoice - Large (4-bit)",
            "VibeVoice - Large",
        ],
        "default_enabled": True,
    },
    "LuxTTS": {
        "label": "LuxTTS",
        "choices": ["LuxTTS - Default"],
        "default_enabled": True,
    },
}


def get_default_voice_clone_model(user_config=None):
    """Get the preferred default voice clone model, respecting engine visibility.

    Returns the largest model from the first enabled engine, falling back
    to DEFAULT_VOICE_CLONE_MODEL if no config is provided.
    """
    if user_config is None:
        return DEFAULT_VOICE_CLONE_MODEL

    engine_settings = user_config.get("enabled_engines", {})
    for engine_key, engine_info in TTS_ENGINES.items():
        if engine_settings.get(engine_key, engine_info.get("default_enabled", True)):
            # Return the last (largest) model from first enabled engine
            return engine_info["choices"][-1]

    # All engines disabled — return first option as absolute fallback
    return VOICE_CLONE_OPTIONS[0]


# ============================================================================
# LANGUAGES
# ============================================================================

LANGUAGES = [
    "Auto",
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# ============================================================================
# CUSTOM VOICE SPEAKERS (Qwen3 Presets)
# ============================================================================

CUSTOM_VOICE_SPEAKERS = [
    "Vivian",  # Bright young female (Chinese)
    "Serena",  # Warm gentle female (Chinese)
    "Uncle_Fu",  # Seasoned mellow male (Chinese)
    "Dylan",  # Youthful Beijing male (Chinese)
    "Eric",  # Lively Chengdu male (Chinese)
    "Ryan",  # Dynamic male (English)
    "Aiden",  # Sunny American male (English)
    "Ono_Anna",  # Playful female (Japanese)
    "Sohee",  # Warm female (Korean)
]

# ============================================================================
# SUPPORTED/BUILT-IN MODELS
# ============================================================================

SUPPORTED_MODELS = {
    # Qwen3-TTS models
    "qwen3-tts-12hz-1.7b-base",
    "qwen3-tts-12hz-1.7b-customvoice",
    "qwen3-tts-12hz-1.7b-voicedesign",
    "qwen3-tts-12hz-0.6b-base",
    "qwen3-tts-12hz-0.6b-customvoice",
    "qwen3-tts-0.6b-base",
    "qwen3-tts-0.6b-customvoice",
    "qwen3-tts-tokenizer-12hz",
    # VibeVoice models
    "vibevoice-tts-1.5b",
    "vibevoice-tts-4b",
    "vibevoice-asr",
    # LuxTTS models
    "luxtts",
    # Whisper models
    "whisper",
}

# ============================================================================
# AUDIO SPECIFICATIONS
# ============================================================================

SAMPLE_RATE = 24000  # Standard sample rate for TTS models (24kHz)
AUDIO_FORMAT = "wav"
AUDIO_DTYPE = "int16"
AUDIO_CHANNELS = 1  # Mono

# ============================================================================
# DEFAULT CONFIGURATION VALUES
# ============================================================================

DEFAULT_CONFIG = {
    "transcribe_model": "Whisper",
    "tts_base_size": "Large",
    "custom_voice_size": "Large",
    "language": "Auto",
    "conv_pause_duration": 0.5,
    "conv_pause_linebreak": 0.5,
    "conv_pause_period": 0.4,
    "conv_pause_comma": 0.2,
    "conv_pause_question": 0.8,
    "conv_pause_hyphen": 0.3,
    "whisper_language": "Auto-detect",
    "low_cpu_mem_usage": False,
    "attention_mechanism": "auto",
    "offline_mode": False,
    "browser_notifications": True,
    "samples_folder": "samples",
    "output_folder": "output",
    "datasets_folder": "datasets",
    "temp_folder": "temp",
    "models_folder": "models",
    "trained_models_folder": "models",
    "emotions": None,  # Initialized separately
    "luxtts_num_steps": 4,
    "luxtts_t_shift": 0.9,
    "luxtts_speed": 1.0,
    "luxtts_return_smooth": False,
    "luxtts_rms": 0.01,
    "luxtts_ref_duration": 5.0,
    "luxtts_device": "auto",
    "luxtts_threads": 2,
}

# ============================================================================
# GENERATION DEFAULTS
# ============================================================================

# Qwen TTS Generation Defaults
QWEN_GENERATION_DEFAULTS = {
    "do_sample": True,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
}

# VibeVoice TTS Generation Defaults
VIBEVOICE_GENERATION_DEFAULTS = {
    "do_sample": False,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "cfg_scale": 3.0,
    "num_steps": 20,
}

# LuxTTS Generation Defaults
LUXTTS_DEFAULTS = {
    "num_steps": 4,
    "t_shift": 0.9,
    "speed": 1.0,
    "return_smooth": False,
    "rms": 0.01,
    "ref_duration": 5.0,
    "device": "auto",
    "threads": 2,
}

# LuxTTS Generation Defaults
LUXTTS_GENERATION_DEFAULTS = {
    "num_steps": 4,
    "t_shift": 0.9,
    "speed": 1.0,
    "return_smooth": False,
    "rms": 0.01,
    "ref_duration": 5,
    "cpu_threads": 2,
}

# LuxTTS audio is 48kHz (higher quality than standard 24kHz)
LUXTTS_SAMPLE_RATE = 48000

# ============================================================================
# UI/UX CONSTANTS
# ============================================================================

APP_TITLE = "Voice Clone Studio"
APP_SUBTITLE = "Powered by Qwen3-TTS, VibeVoice, LuxTTS and Whisper"

# Port assignments for standalone tool testing
TOOL_PORTS = {
    "voice_design": 7861,
    "voice_clone": 7862,
    "voice_presets": 7863,
    "conversation": 7864,
    "prep_audio": 7865,
}
