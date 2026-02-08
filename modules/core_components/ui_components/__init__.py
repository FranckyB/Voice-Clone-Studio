"""
Reusable UI Components for Voice Clone Studio

Provides modular, reusable Gradio components for model parameters and controls.
This eliminates code duplication across tabs and makes it easy to add support for new models.
"""

from pathlib import Path
import gradio as gr
from modules.core_components.emotion_manager import calculate_emotion_values, get_emotion_choices

# Load modal resources from files
_UI_DIR = Path(__file__).parent

# Confirmation Modal
CONFIRMATION_MODAL_CSS = (_UI_DIR / 'confirmation_modal.css').read_text(encoding='utf-8')
CONFIRMATION_MODAL_HEAD = '<script>\n' + (_UI_DIR / 'confirmation_modal.js').read_text(encoding='utf-8') + '\n</script>'
CONFIRMATION_MODAL_HTML = (_UI_DIR / 'confirmation_modal.html').read_text(encoding='utf-8')

# Input Modal
INPUT_MODAL_CSS = (_UI_DIR / 'input_modal.css').read_text(encoding='utf-8')
INPUT_MODAL_HEAD = '<script>\n' + (_UI_DIR / 'input_modal.js').read_text(encoding='utf-8') + '\n</script>'
INPUT_MODAL_HTML = (_UI_DIR / 'input_modal.html').read_text(encoding='utf-8')

# Import helper functions
from .modals import show_confirmation_modal_js, show_input_modal_js, create_confirmation_workflow


def create_qwen_advanced_params(
    emotions_dict=None,
    initial_do_sample=True,
    initial_temperature=0.9,
    initial_top_k=50,
    initial_top_p=1.0,
    initial_repetition_penalty=1.05,
    initial_max_new_tokens=2048,
    include_emotion=False,
    initial_emotion="",
    initial_intensity=1.0,
    visible=True,
    emotion_visible=True,
    shared_state=None
):
    """
    Reusable Qwen advanced parameters accordion.

    Each call creates independent component instances for the tab.

    Args:
        emotions_dict: Dictionary of emotion presets (optional, for initial choices only)
        initial_do_sample: Default sampling toggle
        initial_temperature: Default temperature value
        initial_top_k: Default top_k value
        initial_top_p: Default top_p value
        initial_repetition_penalty: Default penalty value
        initial_max_new_tokens: Default max tokens
        include_emotion: Show emotion preset controls
        initial_emotion: Pre-selected emotion
        initial_intensity: Starting intensity multiplier
        visible: Make accordion visible
        emotion_visible: Make emotion controls visible (only used if include_emotion=True)
        shared_state: Shared state dict (required if include_emotion=True, must have '_active_emotions' key)

    Returns:
        dict with component references and helper function for event binding
    """
    components = {}

    with gr.Accordion("Advanced Parameters", open=False, visible=visible) as accordion:
        # Emotion section (optional)
        if include_emotion:
            emotion_choices = get_emotion_choices(emotions_dict) if emotions_dict else []

            with gr.Row(visible=emotion_visible) as emotion_row:
                components['emotion_preset'] = gr.Dropdown(
                    choices=emotion_choices,
                    value=initial_emotion,
                    label="🎭 Emotion Preset",
                    info="Quick presets that adjust parameters for different emotions",
                    scale=3
                )
                components['emotion_intensity'] = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=initial_intensity,
                    step=0.1,
                    label="Intensity",
                    info="Emotion strength (0=none, 2=extreme)",
                    scale=1
                )

            components['emotion_row'] = emotion_row

            # Emotion management buttons
            with gr.Row(visible=emotion_visible) as emotion_buttons_row:
                components['save_emotion_btn'] = gr.Button("Save", size="sm", scale=1)
                components['delete_emotion_btn'] = gr.Button("Delete", size="sm", scale=1)

            components['emotion_buttons_row'] = emotion_buttons_row
            components['emotion_save_name'] = gr.Textbox(visible=False, value="")

        # Standard parameters
        with gr.Row():
            components['do_sample'] = gr.Checkbox(
                label="Enable Sampling",
                value=initial_do_sample,
                info="Qwen3 recommends sampling enabled (default: True)"
            )
            components['temperature'] = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=initial_temperature,
                step=0.05,
                label="Temperature",
                info="Sampling temperature"
            )

        with gr.Row():
            components['repetition_penalty'] = gr.Slider(
                minimum=1.0,
                maximum=2.0,
                value=initial_repetition_penalty,
                step=0.05,
                label="Repetition Penalty",
                info="Penalize repeated tokens"
            )

            components['top_p'] = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=initial_top_p,
                step=0.05,
                label="Top-P (Nucleus)",
                info="Cumulative probability threshold"
            )

        with gr.Row():
            components['top_k'] = gr.Slider(
                minimum=0,
                maximum=100,
                value=initial_top_k,
                step=1,
                label="Top-K",
                info="Keep only top K tokens"
            )

            components['max_new_tokens'] = gr.Slider(
                minimum=512,
                maximum=4096,
                value=initial_max_new_tokens,
                step=256,
                label="Max New Tokens",
                info="Maximum codec tokens to generate"
            )

    # Helper function to update sliders when emotion changes
    if include_emotion:
        if not shared_state or '_active_emotions' not in shared_state:
            raise ValueError("shared_state with '_active_emotions' key is required when include_emotion=True")

        def update_from_emotion(emotion_name, intensity):
            """Update slider values based on selected emotion."""
            # Read current emotions dynamically from shared_state
            current_emotions = shared_state['_active_emotions']
            temp, top_p, penalty, _ = calculate_emotion_values(
                current_emotions,
                emotion_name,
                intensity,
                baseline_temp=initial_temperature,
                baseline_top_p=initial_top_p,
                baseline_penalty=initial_repetition_penalty
            )
            return temp, top_p, penalty

        components['update_from_emotion'] = update_from_emotion

    # Store accordion reference for visibility toggling
    components['accordion'] = accordion

    return components


def create_vibevoice_advanced_params(
    initial_num_steps=20,
    initial_cfg_scale=3.0,
    initial_do_sample=False,
    initial_temperature=1.0,
    initial_top_k=50,
    initial_top_p=1.0,
    initial_repetition_penalty=1.0,
    visible=True
):
    """
    Reusable VibeVoice advanced parameters accordion.

    Args:
        initial_num_steps: Default inference steps
        initial_cfg_scale: Default CFG scale
        initial_do_sample: Default sampling toggle
        initial_temperature: Default temperature
        initial_top_k: Default top_k
        initial_top_p: Default top_p
        initial_repetition_penalty: Default penalty
        visible: Make accordion visible

    Returns:
        dict with component references
    """
    components = {}

    with gr.Accordion("Advanced Parameters", open=False, visible=visible):
        with gr.Row():
            components['num_steps'] = gr.Slider(
                minimum=5,
                maximum=50,
                value=initial_num_steps,
                step=1,
                label="Inference Steps",
                info="Number of diffusion steps"
            )
            components['cfg_scale'] = gr.Slider(
                minimum=1.0,
                maximum=5.0,
                value=initial_cfg_scale,
                step=0.1,
                label="CFG Scale",
                info="Controls audio adherence to voice prompt"
            )

        gr.Markdown("**Stochastic Sampling Parameters**")
        with gr.Row():
            components['do_sample'] = gr.Checkbox(
                label="Enable Sampling",
                value=initial_do_sample,
                info="Enable stochastic sampling (default: False)"
            )

        with gr.Row():
            components['repetition_penalty'] = gr.Slider(
                minimum=1.0,
                maximum=2.0,
                value=initial_repetition_penalty,
                step=0.05,
                label="Repetition Penalty",
                info="Penalize repeated tokens"
            )
            components['temperature'] = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=initial_temperature,
                step=0.05,
                label="Temperature",
                info="Sampling temperature"
            )

        with gr.Row():
            components['top_k'] = gr.Slider(
                minimum=0,
                maximum=100,
                value=initial_top_k,
                step=1,
                label="Top-K",
                info="Keep only top K tokens"
            )
            components['top_p'] = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=initial_top_p,
                step=0.05,
                label="Top-P (Nucleus)",
                info="Cumulative probability threshold"
            )

    return components


def create_qwen_emotion_controls(
    emotions_dict,
    initial_emotion="",
    initial_intensity=1.0,
    baseline_temp=0.9,
    baseline_top_p=1.0,
    baseline_penalty=1.05,
    visible=True
):
    """
    Standalone emotion preset + intensity controls that update sliders.

    Use this when you want emotion controls separate from advanced parameters.

    Args:
        emotions_dict: Dictionary of emotion presets
        initial_emotion: Pre-selected emotion
        initial_intensity: Starting intensity
        baseline_temp: Default temperature for calculations
        baseline_top_p: Default top_p for calculations
        baseline_penalty: Default penalty for calculations
        visible: Initial visibility

    Returns:
        dict with emotion components and update helper function
    """
    components = {}
    emotion_choices = get_emotion_choices(emotions_dict) if emotions_dict else []

    with gr.Row(visible=visible) as emotion_row:
        components['emotion_preset'] = gr.Dropdown(
            choices=emotion_choices,
            value=initial_emotion,
            label="🎭 Emotion Preset",
            info="Quick presets that adjust parameters for different emotions",
            scale=3
        )
        components['emotion_intensity'] = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=initial_intensity,
            step=0.1,
            label="Intensity",
            info="Emotion strength (0=none, 2=extreme)",
            scale=1
        )

    components['emotion_row'] = emotion_row

    # Emotion management buttons
    with gr.Row(visible=visible) as emotion_buttons_row:
        components['save_emotion_btn'] = gr.Button("Save", size="sm", scale=1)
        components['delete_emotion_btn'] = gr.Button("Delete", size="sm", scale=1)

    components['emotion_buttons_row'] = emotion_buttons_row
    components['emotion_save_name'] = gr.Textbox(visible=False, value="")

    # Helper function
    def update_from_emotion(emotion_name, intensity):
        """Update slider values based on selected emotion."""
        temp, top_p, penalty, _ = calculate_emotion_values(
            emotions_dict,
            emotion_name,
            intensity,
            baseline_temp=baseline_temp,
            baseline_top_p=baseline_top_p,
            baseline_penalty=baseline_penalty
        )
        return temp, top_p, penalty

    components['update_from_emotion'] = update_from_emotion

    return components


def create_emotion_intensity_slider(
    initial_intensity=1.0,
    label="Emotion Intensity",
    visible=True
):
    """
    Standalone emotion intensity slider (for auto-detected emotions).

    Use when emotion is auto-detected and user can only adjust intensity.

    Args:
        initial_intensity: Starting intensity value
        label: Slider label
        visible: Initial visibility

    Returns:
        gr.Slider component
    """
    return gr.Slider(
        minimum=0.0,
        maximum=3.0,
        value=initial_intensity,
        step=0.1,
        label=label,
        info="Strength multiplier for detected emotions (0=none, 3=extreme)",
        visible=visible
    )


def create_luxtts_advanced_params(
    initial_num_steps=4,
    initial_t_shift=0.5,
    initial_speed=1.0,
    initial_return_smooth=False,
    initial_rms=0.01,
    initial_ref_duration=30,
    initial_guidance_scale=3.0,
    visible=True
):
    """
    Reusable LuxTTS advanced parameters accordion.

    Args:
        initial_num_steps: Default sampling steps (3-4 recommended)
        initial_t_shift: Sampling parameter (higher = better quality but more pronunciation errors)
        initial_speed: Speed multiplier (lower=slower)
        initial_return_smooth: Smoother output (may reduce metallic artifacts)
        initial_rms: Loudness control (0.01 recommended)
        initial_ref_duration: Reference audio duration in seconds
        initial_guidance_scale: Classifier-free guidance scale (3.0 default)
        visible: Make accordion visible

    Returns:
        dict with component references
    """
    components = {}

    with gr.Accordion("LuxTTS Advanced Parameters", open=False, visible=visible) as accordion:
        with gr.Row():
            components['num_steps'] = gr.Slider(
                minimum=1,
                maximum=12,
                value=initial_num_steps,
                step=1,
                label="Steps (num_steps)",
                info="Sampling steps (3-4 best for efficiency)"
            )
            components['t_shift'] = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=initial_t_shift,
                step=0.05,
                label="t_shift",
                info="Higher = better quality but more pronunciation errors (0.5 default)"
            )

        with gr.Row():
            components['speed'] = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=initial_speed,
                step=0.05,
                label="Speed",
                info="Speed multiplier (lower=slower)"
            )
            components['guidance_scale'] = gr.Slider(
                minimum=0.5,
                maximum=6.0,
                value=initial_guidance_scale,
                step=0.1,
                label="Guidance Scale",
                info="Classifier-free guidance (3.0 default)"
            )

        with gr.Row():
            components['rms'] = gr.Slider(
                minimum=0.001,
                maximum=0.05,
                value=initial_rms,
                step=0.001,
                label="RMS (Loudness)",
                info="Higher = louder (0.01 recommended)"
            )
            components['ref_duration'] = gr.Slider(
                minimum=1,
                maximum=200,
                value=initial_ref_duration,
                step=1,
                label="Reference Duration (seconds)",
                info="How many seconds of reference audio to use (increase to Max if artifacts)"
            )

        with gr.Row():
            components['return_smooth'] = gr.Checkbox(
                value=initial_return_smooth,
                label="Return Smooth",
                info="Reduce metallic artifacts (may reduce clarity)"
            )

    components['accordion'] = accordion

    return components


def create_pause_controls(
    initial_linebreak=0.5,
    initial_period=0.4,
    initial_comma=0.2,
    initial_question=0.6,
    initial_hyphen=0.3,
    visible=True
):
    """
    Reusable pause control accordion for conversation tabs.

    Args:
        initial_linebreak: Default pause between lines
        initial_period: Default pause after period
        initial_comma: Default pause after comma
        initial_question: Default pause after question
        initial_hyphen: Default pause after hyphen
        visible: Make accordion visible

    Returns:
        dict with component references
    """
    components = {}

    with gr.Accordion("Pause Controls", open=False, visible=visible):
        with gr.Column():
            components['pause_linebreak'] = gr.Slider(
                minimum=0.0,
                maximum=3.0,
                value=initial_linebreak,
                step=0.1,
                label="Pause Between Lines",
                info="Silence between each speaker turn"
            )

            with gr.Row():
                components['pause_period'] = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=initial_period,
                    step=0.1,
                    label="After Period (.)",
                    info="Pause after periods"
                )
                components['pause_comma'] = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=initial_comma,
                    step=0.1,
                    label="After Comma (,)",
                    info="Pause after commas"
                )

            with gr.Row():
                components['pause_question'] = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=initial_question,
                    step=0.1,
                    label="After Question (?)",
                    info="Pause after questions"
                )
                components['pause_hyphen'] = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=initial_hyphen,
                    step=0.1,
                    label="After Hyphen (-)",
                    info="Pause after hyphens"
                )

    return components
