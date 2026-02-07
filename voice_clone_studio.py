"""
Voice Clone Studio - Main Application

Minimal orchestrator that loads modular tools and wires them together.
All tab implementations are in modules/core_components/tools/

ARCHITECTURE:
- Each tool is fully independent and self-contained
- Tools import get_tts_manager() / get_asr_manager() directly (singleton pattern)
- Tools implement their own generation logic, file I/O, progress updates
- This file only provides: directories, constants, shared utilities, modals
"""

import os
import sys
from pathlib import Path
import torch
import json
import random
import tempfile
import time

import gradio as gr

# Core imports
from modules.core_components import (
    CONFIRMATION_MODAL_CSS,
    CONFIRMATION_MODAL_HEAD,
    CONFIRMATION_MODAL_HTML,
    INPUT_MODAL_CSS,
    INPUT_MODAL_HEAD,
    INPUT_MODAL_HTML,
    CORE_EMOTIONS,
    show_confirmation_modal_js,
    show_input_modal_js,
    load_emotions_from_config,
    get_emotion_choices,
    calculate_emotion_values,
    handle_save_emotion,
    handle_delete_emotion,
)

# UI components
from modules.core_components.ui_components import (
    create_qwen_advanced_params,
    create_vibevoice_advanced_params,
    create_emotion_intensity_slider,
    create_pause_controls,
)

# AI Managers
from modules.core_components.ai_models import get_tts_manager, get_asr_manager
from modules.core_components.ai_models.model_utils import get_trained_models

# Modular tools
from modules.core_components.tools import (
    create_enabled_tools,
    setup_tool_events,
    load_config,
    save_config,
    build_shared_state,
    play_completion_beep,
    format_help_html,
    TRIGGER_HIDE_CSS,
    CONFIG_FILE,
)

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

# ============================================================================
# CONFIG & SETUP
# ============================================================================

# Load config (CONFIG_FILE imported from tools)
_user_config = load_config()
_active_emotions = load_emotions_from_config(_user_config)

# Ensure config has emotions key set (emotion_manager expects it)
if "emotions" not in _user_config or _user_config["emotions"] is None:
    _user_config["emotions"] = _active_emotions

# Initialize directories
SAMPLES_DIR = Path(__file__).parent / _user_config.get("samples_folder", "samples")
OUTPUT_DIR = Path(__file__).parent / _user_config.get("output_folder", "output")
DATASETS_DIR = Path(__file__).parent / _user_config.get("datasets_folder", "datasets")
TEMP_DIR = Path(__file__).parent / _user_config.get("temp_folder", "temp")
MODELS_DIR = Path(__file__).parent / _user_config.get("models_folder", "models")

for dir_path in [SAMPLES_DIR, OUTPUT_DIR, DATASETS_DIR, TEMP_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# CONSTANTS - Import from central location
# ============================================================================

from modules.core_components.constants import (
    MODEL_SIZES,
    MODEL_SIZES_BASE,
    MODEL_SIZES_CUSTOM,
    MODEL_SIZES_DESIGN,
    MODEL_SIZES_VIBEVOICE,
    VOICE_CLONE_OPTIONS,
    DEFAULT_VOICE_CLONE_MODEL,
    LUXTTS_DEFAULTS,
    LANGUAGES,
    CUSTOM_VOICE_SPEAKERS,
    SUPPORTED_MODELS,
    SAMPLE_RATE,
    DEFAULT_CONFIG as DEFAULT_CONFIG_TEMPLATE,
    QWEN_GENERATION_DEFAULTS,
    VIBEVOICE_GENERATION_DEFAULTS,
)

# ============================================================================
# GLOBAL MANAGERS - Tools access via shared_state
# ============================================================================
_tts_manager = None
_asr_manager = None

# ============================================================================
# UI CREATION
# ============================================================================


def create_ui():
    """Create the Gradio interface with modular tools."""

    # Initialize AI managers and make them available to wrapper functions
    global _tts_manager, _asr_manager
    _tts_manager = get_tts_manager(_user_config, SAMPLES_DIR)
    _asr_manager = get_asr_manager(_user_config)

    # CSS to hide trigger widgets (use imported TRIGGER_HIDE_CSS)
    custom_css = TRIGGER_HIDE_CSS

    with gr.Blocks(title="Voice Clone Studio") as app:
        # Modal HTML
        gr.HTML(CONFIRMATION_MODAL_HTML)
        gr.HTML(INPUT_MODAL_HTML)

        # Hidden triggers for modals
        confirm_trigger = gr.Textbox(
            label="Confirm Trigger", value="", elem_id="confirm-trigger"
        )
        input_trigger = gr.Textbox(
            label="Input Trigger", value="", elem_id="input-trigger"
        )

        # Header with unload button
        with gr.Row():
            with gr.Column(scale=20):
                gr.Markdown("""
                    # 🎙️ Voice Clone Studio
                    <p style="font-size: 0.9em; color: #ffffff; margin-top: -10px;">Powered by Qwen3-TTS, VibeVoice, LuxTTS and Whisper</p>
                    """)

            with gr.Column(scale=1, min_width=180):
                unload_all_btn = gr.Button("Clear VRAM", size="sm", variant="secondary")
                unload_status = gr.Markdown(" ", visible=True)

        # ============================================================
        # BUILD SHARED STATE - everything tools need
        # ============================================================
        shared_state = build_shared_state(
            user_config=_user_config,
            active_emotions=_active_emotions,
            directories={
                "OUTPUT_DIR": OUTPUT_DIR,
                "SAMPLES_DIR": SAMPLES_DIR,
                "DATASETS_DIR": DATASETS_DIR,
                "TEMP_DIR": TEMP_DIR,
            },
            constants={
                "MODEL_SIZES": MODEL_SIZES,
                "MODEL_SIZES_BASE": MODEL_SIZES_BASE,
                "MODEL_SIZES_CUSTOM": MODEL_SIZES_CUSTOM,
                "MODEL_SIZES_DESIGN": MODEL_SIZES_DESIGN,
                "MODEL_SIZES_VIBEVOICE": MODEL_SIZES_VIBEVOICE,
                "VOICE_CLONE_OPTIONS": VOICE_CLONE_OPTIONS,
                "DEFAULT_VOICE_CLONE_MODEL": DEFAULT_VOICE_CLONE_MODEL,
                "LUXTTS_DEFAULTS": LUXTTS_DEFAULTS,
                "LANGUAGES": LANGUAGES,
                "CUSTOM_VOICE_SPEAKERS": CUSTOM_VOICE_SPEAKERS,
            },
            managers={"tts_manager": _tts_manager, "asr_manager": _asr_manager},
            confirm_trigger=confirm_trigger,
            input_trigger=input_trigger,
        )

        # ============================================================
        # LOAD ALL MODULAR TOOLS
        # ============================================================
        with gr.Tabs(elem_id="main-tabs"):
            tool_components = create_enabled_tools(shared_state)
        setup_tool_events(tool_components, shared_state)

        # Wire up unload button
        def on_unload_all():
            _tts_manager.unload_all()
            _asr_manager.unload_all()
            return "VRAM freed."

        # Clear status after 3 seconds to keep UI tidy
        def clear_status():
            time.sleep(3)
            return " "

        unload_all_btn.click(on_unload_all, outputs=[unload_status]).then(
            fn=clear_status, inputs=[], outputs=[unload_status], show_progress="hidden"
        )

    return app


if __name__ == "__main__":
    theme = gr.themes.Base.load("modules/core_components/ui_components/theme.json")
    app = create_ui()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=TRIGGER_HIDE_CSS + CONFIRMATION_MODAL_CSS + INPUT_MODAL_CSS,
        head=CONFIRMATION_MODAL_HEAD + INPUT_MODAL_HEAD,
    )
