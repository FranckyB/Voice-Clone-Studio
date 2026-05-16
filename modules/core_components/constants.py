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
MODEL_SIZES_VIBEVOICE = ["Small", "Large (4-bit)", "Large"]  # VibeVoice disabled in this build
MODEL_SIZES_QWEN3_ASR = ["Small", "Large"]  # Qwen3-ASR: 0.6B, 1.7B

# MMAudio Sound Effects model sizes - populated dynamically by FoleyManager
# Default built-in choices shown when manager hasn't scanned yet
MODEL_SIZES_MMAUDIO = ["Medium (44kHz)", "Large v2 (44kHz)"]

# MMAudio generation defaults
MMAUDIO_GENERATION_DEFAULTS = {
    "duration": 8.0,
    "num_steps": 25,
    "cfg_strength": 4.5,
    "seed": 42,
}

# Voice Clone engine and model options
VOICE_CLONE_OPTIONS = [
    "Qwen3 - Small",
    "Qwen3 - Large",
    "VibeVoice - Small",
    "VibeVoice - Large (4-bit)",
    "VibeVoice - Large",
    "LuxTTS - Default",
    "Chatterbox - Default",
    "Chatterbox - Multilingual",
    "Fish Speech - S2 Pro",
    "DramaBox - Default",
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
        "import_check": ("qwen_tts", "Qwen3TTSModel"),
    },
    "VibeVoice": {
        "label": "VibeVoice",
        "choices": ["VibeVoice - Small", "VibeVoice - Large (4-bit)", "VibeVoice - Large"],
        "default_enabled": True,
        "import_check": ("modules.vibevoice_tts.modular.modeling_vibevoice_inference", "VibeVoiceForConditionalGenerationInference"),
    },
    "LuxTTS": {
        "label": "LuxTTS",
        "choices": ["LuxTTS - Default"],
        "default_enabled": True,
        "import_check": ("zipvoice.luxvoice", "LuxTTS"),
    },
    "Chatterbox": {
        "label": "Chatterbox",
        "choices": ["Chatterbox - Default", "Chatterbox - Multilingual"],
        "default_enabled": True,
        "import_check": ("modules.chatterbox", "ChatterboxTTS"),
    },
    "FishSpeech": {
        "label": "Fish Speech S2",
        "choices": ["Fish Speech - S2 Pro"],
        "default_enabled": True,
        "import_check": ("fish_speech.models.text2semantic.inference", "init_model"),
    },
    "DramaBox": {
        "label": "DramaBox",
        "choices": ["DramaBox - Default"],
        "default_enabled": True,
        "import_check": None,
    },
}

# ASR Engines - transcription engine registry (matches TTS_ENGINES pattern)
# Format: engine_key -> { label, choices, default_enabled, show_language }
ASR_ENGINES = {
    "Qwen3 ASR": {
        "label": "Qwen3-ASR",
        "choices": ["Qwen3 ASR - Small", "Qwen3 ASR - Large"],
        "default_enabled": True,
        "show_language": True,
        "import_check": ("qwen_asr", "Qwen3ASRModel"),
    },
    "VibeVoice ASR": {
        "label": "VibeVoice ASR",
        "choices": ["VibeVoice ASR - Default"],
        "default_enabled": True,
        "show_language": False,
        "import_check": ("modules.vibevoice_asr.modular.modeling_vibevoice_asr", "VibeVoiceASRForConditionalGeneration"),
    },
    "Whisper": {
        "label": "Whisper",
        "choices": ["Whisper - Medium", "Whisper - Large"],
        "default_enabled": True,
        "show_language": True,
        "import_check": ("whisper", None),
    },
}

# All ASR dropdown options (derived from ASR_ENGINES)
ASR_OPTIONS = []
for _engine_info in ASR_ENGINES.values():
    ASR_OPTIONS.extend(_engine_info["choices"])

DEFAULT_ASR_MODEL = "Qwen3 ASR - Large"


def get_default_asr_model(user_config=None):
    """Get the preferred default ASR model, respecting engine visibility.

    Returns the last (largest) model from the first enabled ASR engine,
    falling back to DEFAULT_ASR_MODEL if no config is provided.
    """
    if user_config is None:
        return DEFAULT_ASR_MODEL

    asr_settings = user_config.get("enabled_asr_engines", {})
    for engine_key, engine_info in ASR_ENGINES.items():
        if asr_settings.get(engine_key, engine_info.get("default_enabled", True)):
            return engine_info["choices"][-1]

    # All engines disabled — return first option as absolute fallback
    return ASR_OPTIONS[0] if ASR_OPTIONS else DEFAULT_ASR_MODEL


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


def check_engine_availability(user_config, save_config_fn=None):
    """Check which enabled engines are actually importable and auto-disable missing ones.

    Only checks engines that are currently enabled. Already-disabled engines are skipped.
    Updates user_config in-place and optionally saves to disk.

    Returns dict with results: {engine_key: True/False} for all checked engines.
    """
    import importlib
    import warnings
    import logging
    import io
    import sys

    results = {}

    def _check_import(module_name, attr_name):
        """Try importing a module and optionally an attribute.

        Suppresses verbose warnings from libraries during import checks:
        - LuxTTS: k2 'Failed import' root logger WARNING
        - Qwen3-TTS: flash-attn 'not installed' print to stderr
        """
        # Suppress root logger warnings (catches k2 import warning from LuxTTS)
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.setLevel(logging.ERROR)

        # Capture stdout+stderr to suppress flash-attn print warnings from Qwen3
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                mod = importlib.import_module(module_name)
                if attr_name:
                    getattr(mod, attr_name)
                return True
        except (ImportError, AttributeError):
            return False
        finally:
            root_logger.setLevel(prev_level)
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # Cache: engines that passed last time are skipped (fast path).
    # Engines that failed are always re-checked so newly installed packages are detected.
    # Cache is stored in config.json under "_engine_check_cache".
    cache = user_config.get("_engine_check_cache", {})
    cache_changed = False

    # Prune stale cache entries for engines no longer in the registry
    known_keys = (
        {f"tts.{k}" for k in TTS_ENGINES} |
        {f"asr.{k}" for k in ASR_ENGINES}
    )
    stale = [k for k in list(cache) if k not in known_keys]
    for k in stale:
        del cache[k]
        cache_changed = True

    def _check_with_cache(engine_key, engine_info, cache_namespace):
        """Return (available, was_cached). Updates cache in-place."""
        cache_key = f"{cache_namespace}.{engine_key}"
        check = engine_info.get("import_check")
        if not check:
            results[engine_key] = True
            print(f"  {engine_info['label']:20s} [OK]")
            return True, False

        # Fast path: previously confirmed available — skip re-import
        if cache.get(cache_key) is True:
            results[engine_key] = True
            print(f"  {engine_info['label']:20s} [OK] (cached)")
            return True, False

        module_name, attr_name = check
        available = _check_import(module_name, attr_name)
        results[engine_key] = available

        if available:
            print(f"  {engine_info['label']:20s} [OK]")
            cache[cache_key] = True
        else:
            print(f"  {engine_info['label']:20s} [NOT FOUND] - auto-disabled")
            cache[cache_key] = False

        return available, True

    # --- Check ASR engines ---
    asr_settings = user_config.get("enabled_asr_engines", {})
    asr_changed = False

    for engine_key, engine_info in ASR_ENGINES.items():
        is_enabled = asr_settings.get(engine_key, engine_info.get("default_enabled", True))
        if not is_enabled:
            cache.pop(f"asr.{engine_key}", None)
            continue

        available, checked = _check_with_cache(engine_key, engine_info, "asr")
        if checked:
            cache_changed = True
        if not available:
            if "enabled_asr_engines" not in user_config:
                user_config["enabled_asr_engines"] = {}
            user_config["enabled_asr_engines"][engine_key] = False
            asr_changed = True

    # Persist updates
    if cache_changed:
        user_config["_engine_check_cache"] = cache
    if (asr_changed or cache_changed) and save_config_fn:
        if asr_changed:
            save_config_fn("enabled_asr_engines", user_config["enabled_asr_engines"])
        if cache_changed:
            save_config_fn("_engine_check_cache", cache)

    return results

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
    "Italian"
]

# ============================================================================
# CUSTOM VOICE SPEAKERS (Qwen3 Presets)
# ============================================================================

CUSTOM_VOICE_SPEAKERS = [
    "Vivian",        # Bright young female (Chinese)
    "Serena",        # Warm gentle female (Chinese)
    "Uncle_Fu",      # Seasoned mellow male (Chinese)
    "Dylan",         # Youthful Beijing male (Chinese)
    "Eric",          # Lively Chengdu male (Chinese)
    "Ryan",          # Dynamic male (English)
    "Aiden",         # Sunny American male (English)
    "Ono_Anna",      # Playful female (Japanese)
    "Sohee"          # Warm female (Korean)
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
    # VibeVoice Streaming model
    "vibevoice-streaming-0.5b",
    # Fish Speech models
    "fish-speech-s2-pro",
}

# ============================================================================
# VIBEVOICE STREAMING VOICES (0.5B Baked-In Speakers)
# ============================================================================

VIBEVOICE_STREAMING_VOICES = [
    "Carter",      # Male (English)
    "Davis",       # Male (English)
    "Emma",        # Female (English)
    "Frank",       # Male (English)
    "Grace",       # Female (English)
    "Mike",        # Male (English)
    "Samuel",      # Male (Indian English)
]

# Maps display name -> GitHub repo filename (without .pt extension)
VIBEVOICE_STREAMING_VOICE_FILES = {
    "Carter":  "en-Carter_man",
    "Davis":   "en-Davis_man",
    "Emma":    "en-Emma_woman",
    "Frank":   "en-Frank_man",
    "Grace":   "en-Grace_woman",
    "Mike":    "en-Mike_man",
    "Samuel":  "in-Samuel_man",
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
    "trained_models_folder": "trained_models"
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
    "max_new_tokens": 2048
}

# VibeVoice TTS Generation Defaults
VIBEVOICE_GENERATION_DEFAULTS = {
    "do_sample": False,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.1,
    "cfg_scale": 3.0,
    "num_steps": 20
}

# VibeVoice Streaming Generation Defaults
VIBEVOICE_STREAMING_GENERATION_DEFAULTS = {
    "cfg_scale": 1.5,
    "ddpm_steps": 20,
}

# Training Defaults — Qwen3
QWEN_TRAINING_DEFAULTS = {
    "batch_size": 1,
    "learning_rate": 2e-6,
    "num_epochs": 10,
    "save_interval": 5,
}

# Training Defaults — VibeVoice
DRAMABOX_TRAINING_DEFAULTS = {
    "batch_size": 1,
    "learning_rate": 3e-5,
    "num_epochs": 1000,
    "save_interval": 200,
    "gradient_accumulation": 4,
    "warmup_steps": 100,
    "lora_rank": 128,
    "lora_alpha": 128,
    "lora_dropout": 0.1,
    "lr_scheduler": "cosine",
    "base_model": "dev",
    "ref_ratio": 0.3,
    "text_dropout": 0.4,
    "seed": 42,
    "resume_lora": "",
}

# Keep alias for any code that still references the old name
VIBEVOICE_TRAINING_DEFAULTS = DRAMABOX_TRAINING_DEFAULTS

# LuxTTS Generation Defaults
LUXTTS_GENERATION_DEFAULTS = {
    "num_steps": 4,
    "t_shift": 0.9,
    "speed": 1.0,
    "return_smooth": False,
    "rms": 0.01,
    "ref_duration": 5,
    "cpu_threads": 2
}

# LuxTTS audio is 48kHz (higher quality than standard 24kHz)
LUXTTS_SAMPLE_RATE = 48000

# Fish Speech S2 Generation Defaults (matching official webui)
FISH_SPEECH_GENERATION_DEFAULTS = {
    "temperature": 0.9,
    "top_p": 0.9,
    "top_k": 30,
    "repetition_penalty": 1.1,
    "max_new_tokens": 0,  # 0 = auto (model max length)
    "chunk_length": 300,
}

# Fish Speech audio sample rate (determined by DAC codec, typically 24kHz)
FISH_SPEECH_SAMPLE_RATE = 24000

# Chatterbox Generation Defaults
CHATTERBOX_GENERATION_DEFAULTS = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
    "repetition_penalty": 1.2,
    "top_p": 1.0,
}

# Chatterbox supported languages (ISO codes from mtl_tts.py)
CHATTERBOX_LANGUAGES = [
    "Arabic", "Danish", "German", "Greek", "English", "Spanish",
    "Finnish", "French", "Hebrew", "Hindi", "Italian", "Japanese",
    "Korean", "Malay", "Dutch", "Norwegian", "Polish", "Portuguese",
    "Russian", "Swedish", "Swahili", "Turkish", "Chinese",
]

# Map display names to ISO codes for Chatterbox Multilingual
CHATTERBOX_LANG_TO_CODE = {
    "Arabic": "ar", "Danish": "da", "German": "de", "Greek": "el",
    "English": "en", "Spanish": "es", "Finnish": "fi", "French": "fr",
    "Hebrew": "he", "Hindi": "hi", "Italian": "it", "Japanese": "ja",
    "Korean": "ko", "Malay": "ms", "Dutch": "nl", "Norwegian": "no",
    "Polish": "pl", "Portuguese": "pt", "Russian": "ru", "Swedish": "sv",
    "Swahili": "sw", "Turkish": "tr", "Chinese": "zh",
}

# ============================================================================
# UI/UX CONSTANTS
# ============================================================================

APP_TITLE = "Voice Clone Studio"
APP_SUBTITLE = "Powered by DramaBox, MMAudio, Qwen3-ASR and Whisper"

# Port assignments for standalone tool testing
TOOL_PORTS = {
    "voice_clone": 7862,
    "prep_audio": 7865,
    "sound_effects": 7866,
    "train_model": 7867,
    "prompt_generator": 7868,
    "output_history": 7869,
    "settings": 7870,
}
