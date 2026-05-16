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
import warnings
import torch
import json
import random
import tempfile
import time
import logging

# Suppress benign Inductor warnings (e.g. online softmax split reduction)
warnings.filterwarnings("ignore", message=".*Online softmax.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

# ============================================================================
# RUNTIME PATCHES (Windows/Triton Compatibility)
# ============================================================================

def auto_patch_inductor():
    """Apply safety patch for Triton/Inductor cluster_dims AttributeError in Windows dev builds.

    Newer PyTorch uses get_first_attr(binary, "cluster_dims", "clusterDims") which
    raises AssertionError when neither attribute exists (common with triton-windows).
    We replace that call with a safe getattr chain that defaults to (1, 1, 1).
    """
    try:
        import torch._inductor.runtime.triton_heuristics as th
        patch_file = th.__file__
        if not os.path.exists(patch_file):
            return

        with open(patch_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Target: direct attribute access that causes crashes in dev builds on Windows
        # We replace them with safe getattr calls
        modified = False

        # --- Pattern A (older PyTorch): direct binary.cluster_dims ---
        unpatched = '(binary.num_ctas, *binary.cluster_dims)'
        patched = '(getattr(binary, "num_ctas", 1), *getattr(binary, "cluster_dims", getattr(binary, "clusterDims", (1, 1, 1))))'
        if unpatched in content:
            content = content.replace(unpatched, patched)
            modified = True

        # --- Pattern B (older PyTorch): binary.metadata.cluster_dims ---
        unpatched_meta = '(binary.metadata.num_ctas, *binary.metadata.cluster_dims)'
        patched_meta = '(getattr(binary.metadata, "num_ctas", 1), *getattr(binary.metadata, "cluster_dims", (1, 1, 1)))'
        if unpatched_meta in content:
            content = content.replace(unpatched_meta, patched_meta)
            modified = True

        # --- Pattern C (newer PyTorch): get_first_attr raises AssertionError ---
        # get_first_attr(binary, "cluster_dims", "clusterDims") fails when
        # triton-windows binaries lack both attributes.
        unpatched_gfa = '*get_first_attr(binary, "cluster_dims", "clusterDims")'
        patched_gfa = '*getattr(binary, "cluster_dims", getattr(binary, "clusterDims", (1, 1, 1)))'
        if unpatched_gfa in content:
            content = content.replace(unpatched_gfa, patched_gfa)
            modified = True

        if modified:
            with open(patch_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[SYSTEM] Triton/Inductor compatibility patch applied to {os.path.basename(patch_file)}")

    except Exception as e:
        # Silently fail if not applicable or already patched
        pass

# Execute early patch
auto_patch_inductor()

# ============================================================================
# TRITON & INDUCTOR CONFIGURATION (Windows)
# ============================================================================

# Configure Torch Inductor for Windows performance and persistence
try:
    import torch._inductor.config as inductor_config
    # Enable all caching and persistence mechanisms
    inductor_config.fx_graph_cache = True
    inductor_config.autotune_local_cache = True
    inductor_config.triton.autotune_at_compile_time = True
    # Maintain stability on Windows
    inductor_config.triton.unique_kernel_names = True
    # Avoid "c long" (int64) indexing overhead where 32-bit suffices
    inductor_config.force_bit32_indexing = True
except Exception:
    pass

# Suppress Gradio's noisy HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import gradio as gr

# Core imports
from modules.core_components import (
    CONFIRMATION_MODAL_CSS,
    CONFIRMATION_MODAL_HEAD,
    CONFIRMATION_MODAL_HTML,
    INPUT_MODAL_CSS,
    INPUT_MODAL_HEAD,
    INPUT_MODAL_HTML,
    show_confirmation_modal_js,
    show_input_modal_js
)

# UI components
from modules.core_components.ui_components import (
    create_qwen_advanced_params,
    create_vibevoice_advanced_params,
    create_pause_controls
)

# AI Managers
from modules.core_components.ai_models import (
    get_tts_manager,
    get_asr_manager,
    get_foley_manager
)
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
    CONFIG_FILE
)

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

# ============================================================================
# CONFIG & SETUP
# ============================================================================

# Load config (CONFIG_FILE imported from tools)
_user_config = load_config()

# Check which ASR engines are available
if not _user_config.get("skip_engine_check", False):
    from modules.core_components.constants import check_engine_availability
    check_engine_availability(
        _user_config,
        save_config_fn=lambda key, value: save_config(_user_config, key, value)
    )

# On macOS (MPS), training requires CUDA — auto-disable Train Model tab
import platform
if platform.system() == "Darwin":
    if "enabled_tools" not in _user_config:
        _user_config["enabled_tools"] = {}
    _user_config["enabled_tools"]["Train Model"] = False
    print("macOS detected: Train Model tab disabled (requires CUDA)")

# Initialize directories
SAMPLES_DIR = Path(__file__).parent / _user_config.get("samples_folder", "samples")
OUTPUT_DIR = Path(__file__).parent / _user_config.get("output_folder", "output")
DATASETS_DIR = Path(__file__).parent / _user_config.get("datasets_folder", "datasets")
TEMP_DIR = Path(__file__).parent / _user_config.get("temp_folder", "temp")
MODELS_DIR = Path(__file__).parent / _user_config.get("models_folder", "models")
TRAINED_MODELS_DIR = Path(__file__).parent / _user_config.get("trained_models_folder", "trained_models")

for dir_path in [SAMPLES_DIR, OUTPUT_DIR, DATASETS_DIR, TEMP_DIR, MODELS_DIR, TRAINED_MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Clean temp folder at startup
for f in TEMP_DIR.iterdir():
    try:
        if f.is_file():
            f.unlink()
    except Exception:
        pass

# ============================================================================
# CONSTANTS - Import from central location
# ============================================================================

from modules.core_components.constants import (
    MODEL_SIZES,
    MODEL_SIZES_BASE,
    MODEL_SIZES_CUSTOM,
    MODEL_SIZES_DESIGN,
    MODEL_SIZES_VIBEVOICE,
    MODEL_SIZES_QWEN3_ASR,
    VOICE_CLONE_OPTIONS,
    DEFAULT_VOICE_CLONE_MODEL,
    TTS_ENGINES,
    ASR_ENGINES,
    ASR_OPTIONS,
    DEFAULT_ASR_MODEL,
    LANGUAGES,
    CUSTOM_VOICE_SPEAKERS,
    SUPPORTED_MODELS,
    SAMPLE_RATE,
    DEFAULT_CONFIG as DEFAULT_CONFIG_TEMPLATE,
    QWEN_GENERATION_DEFAULTS,
    VIBEVOICE_GENERATION_DEFAULTS,
    VIBEVOICE_STREAMING_VOICES,
    MODEL_SIZES_MMAUDIO,
    MMAUDIO_GENERATION_DEFAULTS,
)

# ============================================================================
# GLOBAL MANAGERS - Tools access via shared_state
# ============================================================================
_tts_manager = None
_asr_manager = None
_foley_manager = None

# ============================================================================
# UI CREATION
# ============================================================================

def create_ui():
    """Create the Gradio interface with modular tools."""

    # Initialize AI managers and make them available to wrapper functions
    global _tts_manager, _asr_manager, _foley_manager
    _tts_manager = get_tts_manager(_user_config, SAMPLES_DIR)
    _asr_manager = get_asr_manager(_user_config)
    _foley_manager = get_foley_manager(_user_config, MODELS_DIR)

    # CSS to hide trigger widgets (use imported TRIGGER_HIDE_CSS)
    custom_css = TRIGGER_HIDE_CSS

    with gr.Blocks(title="Voice Clone Studio") as app:
        # Modal HTML
        gr.HTML(CONFIRMATION_MODAL_HTML)
        gr.HTML(INPUT_MODAL_HTML)

        # Hidden triggers for modals
        confirm_trigger = gr.Textbox(label="Confirm Trigger", value="", elem_id="confirm-trigger")
        input_trigger = gr.Textbox(label="Input Trigger", value="", elem_id="input-trigger")
        prompt_apply_trigger = gr.Textbox(label="Prompt Apply Trigger", value="", elem_id="prompt-apply-trigger")

        # Header with unload button
        with gr.Row():
            with gr.Column(scale=20):
                gr.Markdown("""
                    # 🎙️ Voice Clone Studio - DramaBox Edition
                    <p style="font-size: 0.9em; color: var(--body-text-color-subdued); margin-top: -10px;">Powered by DramaBox, MMAudio, Qwen3-ASR and Whisper</p>
                    """)

            with gr.Column(scale=1, min_width=180):
                unload_all_btn = gr.Button("Clear VRAM", size="sm", variant="secondary")
                unload_status = gr.Markdown(" ", visible=True)

        # ============================================================
        # BUILD SHARED STATE - everything tools need
        # ============================================================
        shared_state = build_shared_state(
            user_config=_user_config,
            directories={
                'OUTPUT_DIR': OUTPUT_DIR,
                'SAMPLES_DIR': SAMPLES_DIR,
                'DATASETS_DIR': DATASETS_DIR,
                'TEMP_DIR': TEMP_DIR
            },
            constants={
                'MODEL_SIZES': MODEL_SIZES,
                'MODEL_SIZES_BASE': MODEL_SIZES_BASE,
                'MODEL_SIZES_CUSTOM': MODEL_SIZES_CUSTOM,
                'MODEL_SIZES_DESIGN': MODEL_SIZES_DESIGN,
                'MODEL_SIZES_VIBEVOICE': MODEL_SIZES_VIBEVOICE,
                'MODEL_SIZES_QWEN3_ASR': MODEL_SIZES_QWEN3_ASR,
                'MODEL_SIZES_MMAUDIO': MODEL_SIZES_MMAUDIO,
                'VOICE_CLONE_OPTIONS': VOICE_CLONE_OPTIONS,
                'DEFAULT_VOICE_CLONE_MODEL': DEFAULT_VOICE_CLONE_MODEL,
                'TTS_ENGINES': TTS_ENGINES,
                'ASR_ENGINES': ASR_ENGINES,
                'ASR_OPTIONS': ASR_OPTIONS,
                'DEFAULT_ASR_MODEL': DEFAULT_ASR_MODEL,
                'LANGUAGES': LANGUAGES,
                'CUSTOM_VOICE_SPEAKERS': CUSTOM_VOICE_SPEAKERS,
                'VIBEVOICE_STREAMING_VOICES': VIBEVOICE_STREAMING_VOICES,
            },
            managers={
                'tts_manager': _tts_manager,
                'asr_manager': _asr_manager,
                'foley_manager': _foley_manager,
            },
            confirm_trigger=confirm_trigger,
            input_trigger=input_trigger,
            prompt_apply_trigger=prompt_apply_trigger
        )

        # ============================================================
        # LOAD ALL MODULAR TOOLS
        # ============================================================
        with gr.Tabs(elem_id="main-tabs") as main_tabs:
            tool_components = create_enabled_tools(shared_state)
        # Make main_tabs available to tools' setup_events for tab switching
        shared_state['main_tabs_component'] = main_tabs
        # Pass app reference so tools can register app.load() handlers for initial visibility
        shared_state['app'] = app
        setup_tool_events(tool_components, shared_state)

        # Wire up unload button
        def on_unload_all():
            import gc
            _tts_manager.unload_all()
            _asr_manager.unload_all()
            _foley_manager.unload_all()
            # Stop llama.cpp server if running
            try:
                from modules.core_components.tools.prompt_generator import _stop_server
                _stop_server()
            except Exception:
                pass
            gc.collect()
            from modules.core_components.ai_models.model_utils import empty_device_cache
            empty_device_cache()
            return "VRAM freed."

        # Clear status after 3 seconds to keep UI tidy
        def clear_status():
            time.sleep(3)
            return " "

        unload_all_btn.click(
            on_unload_all,
            outputs=[unload_status]
        ).then(
            fn=clear_status,
            inputs=[],
            outputs=[unload_status],
            show_progress="hidden"
        )

    return app


if __name__ == "__main__":
    # Load selected theme (v1 or v2, default v2)
    theme_choice = _user_config.get("ui_theme", "v2")
    if theme_choice == "v1":
        theme = gr.themes.Ocean(
            neutral_hue="gray",
            spacing_size=gr.themes.Size(lg="6px", md="4px", sm="2px", xl="9px", xs="1px", xxl="10px", xxs="1px"),
            primary_hue="orange",
            secondary_hue="orange",
            text_size="lg",
            radius_size="md",
        )
    else:
        theme = gr.themes.Ocean(
            neutral_hue=gr.themes.Color(c100="#f3f4f6", c200="#e5e7eb", c300="#d1d5db", c400="#9ca3af", c50="#f9fafb", c500="#6b7280", c600="hsl(215, 7%, 34%)", c700="hsl(217, 10%, 27%)", c800="hsl(215, 14%, 17%)", c900="hsl(221, 20%, 11%)", c950="hsl(223, 20%, 7%)"),
            spacing_size=gr.themes.Size(lg="6px", md="4px", sm="2px", xl="9px", xs="1px", xxl="10px", xxs="1px"),
            primary_hue="orange",
            secondary_hue="red",
            text_size="lg",
            radius_size="md",
        )

    app = create_ui()

    # Force dark mode JS snippet (conditional on user preference)
    dark_mode_js = "() => { document.body.classList.add('dark'); }" if _user_config.get("dark_mode_only", True) else None

    try:
        # Use 0.0.0.0 if user enabled network listening, otherwise localhost only
        network_mode = _user_config.get("listen_on_network", False)
        default_host = "0.0.0.0" if network_mode else "127.0.0.1"
        server_host = os.getenv("GRADIO_SERVER_NAME", default_host)

        # In network mode, don't auto-open browser — print the LAN address instead
        if network_mode or server_host == "0.0.0.0":
            import socket
            # Get actual LAN IP by checking which interface routes to external networks
            # This never sends data — just checks which local IP the OS would use
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("10.255.255.255", 1))
                lan_ip = s.getsockname()[0]
                s.close()
            except Exception:
                lan_ip = "<your-local-ip>"
            print()
            print("=" * 50)
            print("Network listening enabled")
            print("Local:   http://127.0.0.1:7860")
            print(f"Network: http://{lan_ip}:7860")
            print()
            print("NOTE: Only devices on your local network can")
            print("connect. Your router's firewall blocks outside")
            print("traffic unless you have port forwarding enabled.")
            print("Do NOT use this on public/untrusted WiFi.")
            print("=" * 50)
            print()

        app.launch(
            server_name=server_host,
            server_port=7860,
            share=False,
            inbrowser=not (network_mode or server_host == "0.0.0.0"),
            theme=theme,
            css=TRIGGER_HIDE_CSS + CONFIRMATION_MODAL_CSS + INPUT_MODAL_CSS,
            head=CONFIRMATION_MODAL_HEAD + INPUT_MODAL_HEAD,
            js=dark_mode_js
        )
    except OSError:
        print()
        print("=" * 50)
        print("Voice Clone Studio is already running!")
        if network_mode or server_host == "0.0.0.0":
            print("Local:   http://127.0.0.1:7860")
            print(f"Network: http://{lan_ip}:7860")
        else:
            print("Check your browser at http://127.0.0.1:7860")
        print("=" * 50)
        print()
