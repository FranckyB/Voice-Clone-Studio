"""
Reusable UI Components for Voice Clone Studio

Provides modular, reusable Gradio components for model parameters and controls.
This eliminates code duplication across tabs and makes it easy to add support for new models.
"""

from pathlib import Path
import gradio as gr

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
    initial_do_sample=True,
    initial_temperature=0.9,
    initial_top_k=50,
    initial_top_p=1.0,
    initial_repetition_penalty=1.05,
    initial_max_new_tokens=2048,
    visible=True
):
    """
    Reusable Qwen advanced parameters accordion.

    Each call creates independent component instances for the tab.

    Args:
        initial_do_sample: Default sampling toggle
        initial_temperature: Default temperature value
        initial_top_k: Default top_k value
        initial_top_p: Default top_p value
        initial_repetition_penalty: Default penalty value
        initial_max_new_tokens: Default max tokens
        visible: Make accordion visible

    Returns:
        dict with component references and helper function for event binding
    """
    components = {}

    with gr.Accordion("Advanced Parameters", open=False, visible=visible) as accordion:

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
                maximum=1.99,
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

    # Store accordion reference for visibility toggling
    components['accordion'] = accordion

    return components


def create_vibevoice_advanced_params(
    initial_num_steps=10,
    initial_cfg_scale=1.3,
    initial_do_sample=False,
    initial_temperature=1.0,
    initial_top_k=50,
    initial_top_p=1.0,
    initial_repetition_penalty=1.0,
    initial_paragraph_per_chunk=False,
    include_paragraph_per_chunk=False,
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
        initial_paragraph_per_chunk: Default paragraph per chunk toggle
        include_paragraph_per_chunk: Show paragraph per chunk control
        visible: Make accordion visible

    Returns:
        dict with component references
    """
    components = {}

    with gr.Accordion("VibeVoice Advanced Parameters", open=False, visible=visible) as accordion:
        with gr.Row():
            components['cfg_scale'] = gr.Slider(
                minimum=1.0,
                maximum=5.0,
                value=initial_cfg_scale,
                step=0.1,
                label="CFG Scale",
                info="Controls audio adherence to voice prompt"
            )
            components['num_steps'] = gr.Slider(
                minimum=5,
                maximum=50,
                value=initial_num_steps,
                step=1,
                label="Inference Steps",
                info="Number of diffusion steps"
            )

        with gr.Row():
            components['do_sample'] = gr.Checkbox(
                label="Enable Sampling",
                value=initial_do_sample,
                info="Enable stochastic sampling (default: False)"
            )
            if include_paragraph_per_chunk:
                components['paragraph_per_chunk'] = gr.Checkbox(
                    label="Paragraph per Chunk",
                    value=initial_paragraph_per_chunk,
                    info="Process text into chunks by paragraph for better quality. (Split using Enter key)"
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

    components['accordion'] = accordion

    return components


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


def create_chatterbox_advanced_params(
    initial_exaggeration=0.5,
    initial_cfg_weight=0.5,
    initial_temperature=0.8,
    initial_repetition_penalty=1.2,
    initial_top_p=1.0,
    visible=False
):
    """
    Reusable Chatterbox advanced parameters accordion.

    Args:
        initial_exaggeration: Emotion intensity (0-2)
        initial_cfg_weight: Classifier-free guidance weight
        initial_temperature: Sampling temperature
        initial_repetition_penalty: Repetition penalty
        initial_top_p: Top-p sampling
        visible: Make accordion visible

    Returns:
        dict with component references
    """
    components = {}

    with gr.Accordion("Chatterbox Advanced Parameters", open=False, visible=visible) as accordion:
        with gr.Row():
            components['exaggeration'] = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=initial_exaggeration,
                step=0.05,
                label="Exaggeration",
                info="Emotion intensity (0 = flat, 2 = very expressive)"
            )
            components['cfg_weight'] = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=initial_cfg_weight,
                step=0.05,
                label="CFG Weight",
                info="Classifier-free guidance (higher = more adherence to reference voice)"
            )

        with gr.Row():
            components['temperature'] = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=initial_temperature,
                step=0.05,
                label="Temperature",
                info="Sampling temperature"
            )
            components['repetition_penalty'] = gr.Slider(
                minimum=1.0,
                maximum=3.0,
                value=initial_repetition_penalty,
                step=0.05,
                label="Repetition Penalty",
                info="Higher = less repetition"
            )

        with gr.Row():
            components['top_p'] = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=initial_top_p,
                step=0.05,
                label="Top-p",
                info="Nucleus sampling threshold"
            )

    components['accordion'] = accordion

    return components


def create_fish_speech_advanced_params(
    initial_temperature=0.9,
    initial_top_p=0.9,
    initial_top_k=30,
    initial_repetition_penalty=1.1,
    initial_max_new_tokens=0,
    initial_chunk_length=300,
    visible=False
):
    """
    Reusable Fish Speech S2 advanced parameters accordion.

    Args:
        initial_temperature: Sampling temperature
        initial_top_p: Top-p nucleus sampling
        initial_top_k: Top-k sampling
        initial_repetition_penalty: Repetition penalty
        initial_max_new_tokens: Max tokens (0 = auto)
        initial_chunk_length: Bytes per batch
        visible: Make accordion visible

    Returns:
        dict with component references
    """
    components = {}

    with gr.Accordion("Fish Speech Advanced Parameters", open=False, visible=visible) as accordion:
        with gr.Row():
            components['temperature'] = gr.Slider(
                minimum=0.8,
                maximum=1.0,
                value=initial_temperature,
                step=0.01,
                label="Temperature",
                info="Sampling temperature (lower = more stable)"
            )
            components['top_p'] = gr.Slider(
                minimum=0.8,
                maximum=1.0,
                value=initial_top_p,
                step=0.01,
                label="Top-p",
                info="Nucleus sampling threshold"
            )

        with gr.Row():
            components['top_k'] = gr.Slider(
                minimum=1,
                maximum=100,
                value=initial_top_k,
                step=1,
                label="Top-k",
                info="Top-k sampling"
            )
            components['repetition_penalty'] = gr.Slider(
                minimum=1.0,
                maximum=1.2,
                value=initial_repetition_penalty,
                step=0.01,
                label="Repetition Penalty",
                info="Higher = less repetition"
            )

        with gr.Row():
            components['max_new_tokens'] = gr.Slider(
                minimum=0,
                maximum=4096,
                value=initial_max_new_tokens,
                step=64,
                label="Max New Tokens",
                info="0 = auto (model max length)"
            )
            components['chunk_length'] = gr.Slider(
                minimum=100,
                maximum=1000,
                value=initial_chunk_length,
                step=10,
                label="Chunk Length",
                info="Bytes per batch for long text generation"
            )

        gr.Markdown(
            "**Emotion tags:** Use inline `[tag]` syntax anywhere in your text to control prosody and emotion. "
            "Supports 15,000+ tags including free-form descriptions.\n\n"
            "**Common tags:** "
            "`[pause]` `[emphasis]` `[laughing]` `[excited]` `[angry]` `[whisper]` `[sad]` "
            "`[singing]` `[loud]` `[low voice]` `[sigh]` `[screaming]` `[shouting]` "
            "`[surprised]` `[delight]` `[clearing throat]` `[chuckle]` `[echo]`\n\n"
            "**Example:** `I can't believe it! [excited] This is amazing! [whisper] Don't tell anyone though.`",
            visible=True
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


def create_dramabox_advanced_params(
    initial_negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted, robotic voice, echo, background noise, off-sync audio, repetitive speech",
    initial_ref_duration=10.0,
    initial_gen_duration=0.0,
    initial_steps=30,
    initial_sampler="euler",
    initial_speed=1.0,
    initial_duration_multiplier=1.1,
    initial_cfg_scale=2.5,
    initial_stg_scale=1.5,
    initial_rescale_scale=-1.0,
    initial_id_guidance_scale=3.0,
    initial_no_watermark=False,
    visible=True
):
    """
    Reusable DramaBox advanced parameters accordion.

    Args:
        initial_*: Default values for each parameter
        visible: Make accordion visible

    Returns:
        dict with component references
    """
    components = {}

    with gr.Accordion("DramaBox Parameters", open=False, visible=visible) as accordion:
        components['negative_prompt'] = gr.Textbox(
            label="Negative Prompt",
            value=initial_negative_prompt,
            lines=3,
        )

        with gr.Row():
            components['sampler'] = gr.Dropdown(
                choices=["euler", "heun"],
                value=initial_sampler,
                label="Sampler",
                interactive=True
            )
            components['steps'] = gr.Slider(
                minimum=0,
                maximum=100,
                value=initial_steps,
                step=1,
                label="Steps (0 = auto)",
                info="Diffusion sampling steps"
            )

        with gr.Row():
            components['cfg_scale'] = gr.Slider(
                minimum=0.0,
                maximum=10.0,
                value=initial_cfg_scale,
                step=0.1,
                label="CFG Scale (0 = auto)",
                info="Classifier-free guidance strength"
            )
            components['stg_scale'] = gr.Slider(
                minimum=0.0,
                maximum=5.0,
                value=initial_stg_scale,
                step=0.1,
                label="STG Scale (0 = auto)",
                info="Style transfer guidance strength"
            )

        with gr.Row():
            components['rescale_scale'] = gr.Slider(
                minimum=-1.0,
                maximum=1.0,
                value=initial_rescale_scale,
                step=0.1,
                label="Rescale Scale (-1 = auto)",
                info="CFG rescaling factor (-1 = auto)"
            )
            components['id_guidance_scale'] = gr.Slider(
                minimum=0.0,
                maximum=10.0,
                value=initial_id_guidance_scale,
                step=0.1,
                label="ID Guidance Scale",
                info="Identity preservation strength"
            )
        with gr.Row():
            components['gen_duration'] = gr.Slider(
                minimum=0.0,
                maximum=60.0,
                value=initial_gen_duration,
                step=0.5,
                label="Gen Duration (0 = auto)",
                info="Target output duration in seconds (0 = auto)"
            )
            components['ref_duration'] = gr.Slider(
                minimum=1.0,
                maximum=30.0,
                value=initial_ref_duration,
                step=0.5,
                label="Ref Duration (seconds)",
                info="How many seconds of reference audio to use"
            )

        with gr.Row():
            components['speed'] = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=initial_speed,
                step=0.05,
                label="Speed",
                info="Speech speed multiplier"
            )
            components['duration_multiplier'] = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=initial_duration_multiplier,
                step=0.05,
                label="Duration Multiplier",
                info="Scale the predicted duration"
            )


        components['no_watermark'] = gr.Checkbox(
            label="Disable Watermark",
            value=initial_no_watermark
        )

    components['accordion'] = accordion
    return components

