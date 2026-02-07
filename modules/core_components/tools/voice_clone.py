"""
Voice Clone Tab

Clone voices from samples using Qwen3-TTS or VibeVoice.
"""

# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    # Also add modules directory for vibevoice_tts imports
    sys.path.insert(0, str(project_root / "modules"))

import gradio as gr
import soundfile as sf
import torch
import random
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager
from gradio_filelister import FileLister


class VoiceCloneTool(Tool):
    """Voice Clone tool implementation."""

    config = ToolConfig(
        name="Voice Clone",
        module_name="tool_voice_clone",
        description="Clone voices from voice samples",
        enabled=True,
        category="generation",
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Voice Clone tool UI."""
        components = {}

        # Get helper functions and config
        get_sample_choices = shared_state["get_sample_choices"]
        get_available_samples = shared_state["get_available_samples"]
        load_sample_details = shared_state["load_sample_details"]
        get_emotion_choices = shared_state["get_emotion_choices"]
        get_prompt_cache_path = shared_state["get_prompt_cache_path"]
        LANGUAGES = shared_state["LANGUAGES"]
        VOICE_CLONE_OPTIONS = shared_state["VOICE_CLONE_OPTIONS"]
        DEFAULT_VOICE_CLONE_MODEL = shared_state["DEFAULT_VOICE_CLONE_MODEL"]
        LUXTTS_DEFAULTS = shared_state.get("LUXTTS_DEFAULTS", {})
        _user_config = shared_state["_user_config"]
        _active_emotions = shared_state["_active_emotions"]
        show_input_modal_js = shared_state["show_input_modal_js"]
        show_confirmation_modal_js = shared_state["show_confirmation_modal_js"]
        save_emotion_handler = shared_state["save_emotion_handler"]
        delete_emotion_handler = shared_state["delete_emotion_handler"]
        save_preference = shared_state["save_preference"]
        refresh_samples = shared_state["refresh_samples"]
        confirm_trigger = shared_state["confirm_trigger"]
        input_trigger = shared_state["input_trigger"]

        with gr.TabItem("Voice Clone") as voice_clone_tab:
            components["voice_clone_tab"] = voice_clone_tab
            gr.Markdown(
                "Clone Voices from Samples. <small>(Use Prep Audio Samples to add samples)</small>"
            )
            with gr.Row():
                # Left column - Sample selection (1/3 width)
                with gr.Column(scale=1):
                    gr.Markdown("### Voice Sample")

                    components["sample_lister"] = FileLister(
                        value=get_sample_choices(),
                        height=200,
                        show_footer=False,
                        interactive=True,
                    )

                    with gr.Row():
                        components["refresh_samples_btn"] = gr.Button(
                            "Refresh", size="sm"
                        )

                    components["sample_audio"] = gr.Audio(
                        label="Sample Preview",
                        type="filepath",
                        interactive=False,
                        visible=True,
                        value=None,
                        elem_id="voice-clone-sample-audio",
                    )

                    components["sample_text"] = gr.Textbox(
                        label="Sample Text", interactive=False, max_lines=10, value=None
                    )

                    components["sample_info"] = gr.Textbox(
                        label="Info", interactive=False, max_lines=3, value=None
                    )

                # Right column - Generation (2/3 width)
                with gr.Column(scale=3):
                    gr.Markdown("### Generate Speech")

                    components["text_input"] = gr.Textbox(
                        label="Text to Generate",
                        placeholder="Enter the text you want to speak in the cloned voice...",
                        lines=6,
                    )

                    # Language dropdown (hidden for VibeVoice models)
                    selected_engine = _user_config.get(
                        "voice_clone_model", DEFAULT_VOICE_CLONE_MODEL
                    )
                    is_qwen_initial = "Qwen" in selected_engine
                    is_lux_initial = "LuxTTS" in selected_engine
                    components["language_row"] = gr.Row(visible=is_qwen_initial)
                    with components["language_row"]:
                        components["language_dropdown"] = gr.Dropdown(
                            choices=LANGUAGES,
                            value=_user_config.get("language", "Auto"),
                            label="Language",
                        )

                    with gr.Row():
                        components["clone_model_dropdown"] = gr.Dropdown(
                            choices=VOICE_CLONE_OPTIONS,
                            value=_user_config.get(
                                "voice_clone_model", DEFAULT_VOICE_CLONE_MODEL
                            ),
                            label="Engine & Model (Qwen3, VibeVoice, LuxTTS)",
                            scale=4,
                        )
                        components["seed_input"] = gr.Number(
                            label="Seed (-1 for random)", value=-1, precision=0, scale=1
                        )

                    # Qwen3 Advanced Parameters (create_qwen_advanced_params includes its own accordion)
                    is_qwen_initial = "Qwen" in selected_engine
                    create_qwen_advanced_params = shared_state[
                        "create_qwen_advanced_params"
                    ]

                    qwen_params = create_qwen_advanced_params(
                        emotions_dict=_active_emotions,
                        include_emotion=True,
                        initial_emotion="(None)",
                        initial_intensity=1.0,
                        visible=is_qwen_initial,
                        emotion_visible=True,
                        shared_state=shared_state,
                    )

                    # Store the accordion reference for toggling
                    components["qwen_params_accordion"] = qwen_params.get("accordion")

                    # Store references
                    components["qwen_params"] = qwen_params
                    components["qwen_emotion_preset"] = qwen_params["emotion_preset"]
                    components["qwen_emotion_intensity"] = qwen_params[
                        "emotion_intensity"
                    ]
                    components["qwen_save_emotion_btn"] = qwen_params.get(
                        "save_emotion_btn"
                    )
                    components["qwen_delete_emotion_btn"] = qwen_params.get(
                        "delete_emotion_btn"
                    )
                    components["qwen_emotion_save_name"] = qwen_params.get(
                        "emotion_save_name"
                    )
                    components["qwen_do_sample"] = qwen_params["do_sample"]
                    components["qwen_temperature"] = qwen_params["temperature"]
                    components["qwen_top_k"] = qwen_params["top_k"]
                    components["qwen_top_p"] = qwen_params["top_p"]
                    components["qwen_repetition_penalty"] = qwen_params[
                        "repetition_penalty"
                    ]
                    components["qwen_max_new_tokens"] = qwen_params["max_new_tokens"]

                    # VibeVoice Advanced Parameters
                    components["vv_params_accordion"] = gr.Accordion(
                        "VibeVoice Advanced Parameters",
                        open=False,
                        visible=("VibeVoice" in selected_engine),
                    )
                    with components["vv_params_accordion"]:
                        with gr.Row():
                            components["vv_cfg_scale"] = gr.Slider(
                                minimum=1.0,
                                maximum=5.0,
                                value=3.0,
                                step=0.1,
                                label="CFG Scale",
                                info="Controls audio adherence to voice prompt",
                            )
                            components["vv_num_steps"] = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=20,
                                step=1,
                                label="Inference Steps",
                                info="Number of diffusion steps",
                            )

                        gr.Markdown("Stochastic Sampling Parameters")
                        with gr.Row():
                            components["vv_do_sample"] = gr.Checkbox(
                                label="Enable Sampling",
                                value=False,
                                info="Enable stochastic sampling (default: False)",
                            )
                        with gr.Row():
                            components["vv_repetition_penalty"] = gr.Slider(
                                minimum=1.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                label="Repetition Penalty",
                                info="Penalize repeated tokens",
                            )

                            components["vv_temperature"] = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                label="Temperature",
                                info="Sampling temperature",
                            )

                        with gr.Row():
                            components["vv_top_k"] = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=50,
                                step=1,
                                label="Top-K",
                                info="Keep only top K tokens",
                            )
                            components["vv_top_p"] = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=1.0,
                                step=0.05,
                                label="Top-P (Nucleus)",
                                info="Cumulative probability threshold",
                            )

                    components["lux_params_accordion"] = gr.Accordion(
                        "LuxTTS Parameters", open=False, visible=is_lux_initial
                    )
                    with components["lux_params_accordion"]:
                        with gr.Row():
                            components["lux_device"] = gr.Dropdown(
                                choices=["auto", "cpu", "cuda"],
                                value=_user_config.get(
                                    "luxtts_device",
                                    LUXTTS_DEFAULTS.get("device", "auto"),
                                ),
                                label="Device",
                                info="Auto selects CUDA when available",
                            )
                            components["lux_threads"] = gr.Number(
                                label="CPU Threads",
                                value=_user_config.get(
                                    "luxtts_threads", LUXTTS_DEFAULTS.get("threads", 2)
                                ),
                                precision=0,
                            )

                        with gr.Row():
                            components["lux_num_steps"] = gr.Slider(
                                minimum=1,
                                maximum=12,
                                value=_user_config.get(
                                    "luxtts_num_steps",
                                    LUXTTS_DEFAULTS.get("num_steps", 4),
                                ),
                                step=1,
                                label="Inference Steps",
                            )
                            components["lux_t_shift"] = gr.Slider(
                                minimum=0.1,
                                maximum=1.5,
                                value=_user_config.get(
                                    "luxtts_t_shift",
                                    LUXTTS_DEFAULTS.get("t_shift", 0.9),
                                ),
                                step=0.05,
                                label="t_shift",
                            )

                        with gr.Row():
                            components["lux_speed"] = gr.Slider(
                                minimum=0.5,
                                maximum=1.5,
                                value=_user_config.get(
                                    "luxtts_speed", LUXTTS_DEFAULTS.get("speed", 1.0)
                                ),
                                step=0.05,
                                label="Speed",
                            )
                            components["lux_return_smooth"] = gr.Checkbox(
                                label="Return Smooth",
                                value=_user_config.get(
                                    "luxtts_return_smooth",
                                    LUXTTS_DEFAULTS.get("return_smooth", False),
                                ),
                            )

                        with gr.Row():
                            components["lux_rms"] = gr.Slider(
                                minimum=0.001,
                                maximum=0.05,
                                value=_user_config.get(
                                    "luxtts_rms", LUXTTS_DEFAULTS.get("rms", 0.01)
                                ),
                                step=0.001,
                                label="RMS",
                            )
                            components["lux_ref_duration"] = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=_user_config.get(
                                    "luxtts_ref_duration",
                                    LUXTTS_DEFAULTS.get("ref_duration", 5.0),
                                ),
                                step=0.5,
                                label="Reference Duration (s)",
                            )

                    components["generate_btn"] = gr.Button(
                        "Generate Audio", variant="primary", size="lg"
                    )

                    components["output_audio"] = gr.Audio(
                        label="Generated Audio", type="filepath"
                    )

                    components["clone_status"] = gr.Textbox(
                        label="Status", interactive=False, lines=2, max_lines=5
                    )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Clone tab events."""

        # Get helper functions and directories
        get_sample_choices = shared_state["get_sample_choices"]
        get_available_samples = shared_state["get_available_samples"]
        load_sample_details = shared_state["load_sample_details"]
        get_prompt_cache_path = shared_state["get_prompt_cache_path"]
        get_or_create_voice_prompt = shared_state["get_or_create_voice_prompt"]
        refresh_samples = shared_state["refresh_samples"]
        show_input_modal_js = shared_state["show_input_modal_js"]
        show_confirmation_modal_js = shared_state["show_confirmation_modal_js"]
        save_emotion_handler = shared_state["save_emotion_handler"]
        delete_emotion_handler = shared_state["delete_emotion_handler"]
        save_preference = shared_state["save_preference"]
        confirm_trigger = shared_state["confirm_trigger"]
        input_trigger = shared_state["input_trigger"]
        OUTPUT_DIR = shared_state["OUTPUT_DIR"]
        play_completion_beep = shared_state.get("play_completion_beep")

        # Get TTS manager (singleton)
        tts_manager = get_tts_manager()

        def get_selected_sample_name(lister_value):
            """Extract selected sample name from FileLister value (strips .wav extension)."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                from modules.core_components.tools import strip_sample_extension

                return strip_sample_extension(selected[0])
            return None

        def generate_audio_handler(
            sample_name,
            text_to_generate,
            language,
            seed,
            model_selection="Qwen3 - Small",
            qwen_do_sample=True,
            qwen_temperature=0.9,
            qwen_top_k=50,
            qwen_top_p=1.0,
            qwen_repetition_penalty=1.05,
            qwen_max_new_tokens=2048,
            vv_do_sample=False,
            vv_temperature=1.0,
            vv_top_k=50,
            vv_top_p=1.0,
            vv_repetition_penalty=1.0,
            vv_cfg_scale=3.0,
            vv_num_steps=20,
            lux_device="auto",
            lux_threads=2,
            lux_num_steps=4,
            lux_t_shift=0.9,
            lux_speed=1.0,
            lux_return_smooth=False,
            lux_rms=0.01,
            lux_ref_duration=5.0,
            progress=gr.Progress(),
        ):
            """Generate audio using voice cloning - supports Qwen, VibeVoice, and LuxTTS engines."""
            if not sample_name:
                return None, "❌ Please select a voice sample first."

            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate."

            # Parse model selection to determine engine and size
            if "LuxTTS" in model_selection:
                engine = "luxtts"
                model_size = "Default"
            elif "VibeVoice" in model_selection:
                engine = "vibevoice"
                if "Small" in model_selection:
                    model_size = "1.5B"
                elif "4-bit" in model_selection:
                    model_size = "Large (4-bit)"
                else:  # Large
                    model_size = "Large"
            else:  # Qwen3
                engine = "qwen"
                if "Small" in model_selection:
                    model_size = "0.6B"
                else:  # Large
                    model_size = "1.7B"

            # Find the selected sample
            samples = get_available_samples()
            sample = None
            for s in samples:
                if s["name"] == sample_name:
                    sample = s
                    break

            if not sample:
                return None, f"❌ Sample '{sample_name}' not found."

            try:
                # Set the seed for reproducibility
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                seed_msg = f"Seed: {seed}"

                if engine == "qwen":
                    # Qwen engine - uses cached prompts
                    progress(0.1, desc=f"Loading Qwen3 model ({model_size})...")

                    # Get or create the voice prompt (with caching)
                    model = tts_manager.get_qwen3_base(model_size)
                    prompt_items, was_cached = get_or_create_voice_prompt(
                        model=model,
                        sample_name=sample_name,
                        wav_path=sample["wav_path"],
                        ref_text=sample["ref_text"],
                        model_size=model_size,
                        progress_callback=progress,
                    )

                    cache_status = "cached" if was_cached else "newly processed"
                    progress(0.6, desc=f"Generating audio ({cache_status} prompt)...")

                    # Generate using manager method
                    audio_data, sr = tts_manager.generate_voice_clone_qwen(
                        text=text_to_generate,
                        language=language,
                        prompt_items=prompt_items,
                        seed=seed,
                        do_sample=qwen_do_sample,
                        temperature=qwen_temperature,
                        top_k=qwen_top_k,
                        top_p=qwen_top_p,
                        repetition_penalty=qwen_repetition_penalty,
                        max_new_tokens=qwen_max_new_tokens,
                        model_size=model_size,
                    )
                    wavs = [audio_data]

                    engine_display = f"Qwen3-{model_size}"

                elif engine == "vibevoice":
                    progress(0.1, desc=f"Loading VibeVoice model ({model_size})...")

                    # Generate using manager method
                    audio_data, sr = tts_manager.generate_voice_clone_vibevoice(
                        text=text_to_generate,
                        voice_sample_path=sample["wav_path"],
                        seed=seed,
                        do_sample=vv_do_sample,
                        temperature=vv_temperature,
                        top_k=vv_top_k,
                        top_p=vv_top_p,
                        repetition_penalty=vv_repetition_penalty,
                        cfg_scale=vv_cfg_scale,
                        num_steps=vv_num_steps,
                        model_size=model_size,
                        user_config=shared_state.get("_user_config", {}),
                    )
                    wavs = [audio_data]

                    engine_display = f"VibeVoice-{model_size}"
                    cache_status = "no caching (VibeVoice)"

                else:  # LuxTTS engine
                    progress(0.1, desc="Loading LuxTTS model...")

                    audio_data, sr, was_cached = (
                        tts_manager.generate_voice_clone_luxtts(
                            text=text_to_generate,
                            sample_name=sample_name,
                            voice_sample_path=sample["wav_path"],
                            num_steps=lux_num_steps,
                            t_shift=lux_t_shift,
                            speed=lux_speed,
                            return_smooth=lux_return_smooth,
                            rms=lux_rms,
                            ref_duration=lux_ref_duration,
                            device=lux_device,
                            threads=lux_threads,
                            progress_callback=progress,
                        )
                    )
                    wavs = [audio_data]
                    engine_display = "LuxTTS"
                    cache_status = "cached" if was_cached else "newly processed"

                progress(0.8, desc="Saving audio...")
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
                output_file = OUTPUT_DIR / f"{safe_name}_{timestamp}.wav"

                sf.write(str(output_file), wavs[0], sr)

                # Save metadata file
                metadata_file = output_file.with_suffix(".txt")
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Sample: {sample_name}
                    Engine: {engine_display}
                    Language: {language}
                    Seed: {seed}
                    Text: {text_to_generate.strip()}
                    """)
                metadata_file.write_text(metadata, encoding="utf-8")

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                return str(
                    output_file
                ), f"Generated using {engine_display}. {cache_status}\n{seed_msg}"

            except ImportError as e:
                return None, f"❌ Dependency error: {str(e)}"
            except Exception as e:
                import traceback

                traceback.print_exc()
                return None, f"❌ Error generating audio: {str(e)}"

        def load_sample_from_lister(lister_value):
            """Load audio, text, and info for the selected sample from FileLister."""
            sample_name = get_selected_sample_name(lister_value)
            if not sample_name:
                return None, "", ""
            return load_sample_details(sample_name)

        # Connect event handlers for Voice Clone tab
        components["sample_lister"].change(
            load_sample_from_lister,
            inputs=[components["sample_lister"]],
            outputs=[
                components["sample_audio"],
                components["sample_text"],
                components["sample_info"],
            ],
        )

        # Double-click = play sample audio
        components["sample_lister"].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#voice-clone-sample-audio .play-pause-button'); if (btn) btn.click(); }, 150); }",
        )

        components["refresh_samples_btn"].click(
            lambda: get_sample_choices(), outputs=[components["sample_lister"]]
        )

        # Wire up emotion preset handlers (same pattern as original)
        if "update_from_emotion" in components.get("qwen_params", {}):
            components["qwen_emotion_preset"].change(
                components["qwen_params"]["update_from_emotion"],
                inputs=[
                    components["qwen_emotion_preset"],
                    components["qwen_emotion_intensity"],
                ],
                outputs=[
                    components["qwen_temperature"],
                    components["qwen_top_p"],
                    components["qwen_repetition_penalty"],
                ],
            )

            components["qwen_emotion_intensity"].change(
                components["qwen_params"]["update_from_emotion"],
                inputs=[
                    components["qwen_emotion_preset"],
                    components["qwen_emotion_intensity"],
                ],
                outputs=[
                    components["qwen_temperature"],
                    components["qwen_top_p"],
                    components["qwen_repetition_penalty"],
                ],
            )

        # Emotion save button
        components["qwen_save_emotion_btn"].click(
            fn=None,
            inputs=[components["qwen_emotion_preset"]],
            outputs=None,
            js=show_input_modal_js(
                title="Save Emotion Preset",
                message="Enter a name for this emotion preset:",
                placeholder="e.g., Happy, Sad, Excited",
                context="qwen_emotion_",
            ),
        )

        # Emotion delete button
        components["qwen_delete_emotion_btn"].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Emotion Preset?",
                message="This will permanently delete this emotion preset from your configuration.",
                confirm_button_text="Delete",
                context="qwen_emotion_",
            ),
        )

        # Handler for emotion save from input modal
        def handle_qwen_emotion_input(input_value, intensity, temp, rep_pen, top_p):
            """Process input modal submission for Voice Clone emotion save."""
            if not input_value or not input_value.startswith("qwen_emotion_"):
                return gr.update(), gr.update()

            parts = input_value.split("_")
            if len(parts) >= 3:
                if parts[2] == "cancel":
                    return gr.update(), ""
                emotion_name = "_".join(parts[2:-1])

                # Use shared helper to process save result
                from modules.core_components.emotion_manager import (
                    process_save_emotion_result,
                )

                save_result = save_emotion_handler(
                    emotion_name, intensity, temp, rep_pen, top_p
                )
                return process_save_emotion_result(save_result, shared_state)

            return gr.update(), gr.update()

        input_trigger.change(
            handle_qwen_emotion_input,
            inputs=[
                input_trigger,
                components["qwen_emotion_intensity"],
                components["qwen_temperature"],
                components["qwen_repetition_penalty"],
                components["qwen_top_p"],
            ],
            outputs=[components["qwen_emotion_preset"], components["clone_status"]],
        )

        # Handler for emotion delete from confirmation modal
        def delete_qwen_emotion_wrapper(confirm_value, emotion_name):
            """Only process if context matches qwen_emotion_."""
            if not confirm_value or not confirm_value.startswith("qwen_emotion_"):
                return gr.update(), gr.update()

            # Call the delete handler and discard the clear_trigger (3rd value)
            from modules.core_components.emotion_manager import (
                process_delete_emotion_result,
            )

            delete_result = delete_emotion_handler(confirm_value, emotion_name)
            dropdown_update, status_msg, _clear = process_delete_emotion_result(
                delete_result, shared_state
            )
            return dropdown_update, status_msg

        confirm_trigger.change(
            delete_qwen_emotion_wrapper,
            inputs=[confirm_trigger, components["qwen_emotion_preset"]],
            outputs=[components["qwen_emotion_preset"], components["clone_status"]],
        )

        def generate_from_lister(lister_value, *args):
            """Extract sample name from lister and pass to generate."""
            return generate_audio_handler(get_selected_sample_name(lister_value), *args)

        components["generate_btn"].click(
            generate_from_lister,
            inputs=[
                components["sample_lister"],
                components["text_input"],
                components["language_dropdown"],
                components["seed_input"],
                components["clone_model_dropdown"],
                components["qwen_do_sample"],
                components["qwen_temperature"],
                components["qwen_top_k"],
                components["qwen_top_p"],
                components["qwen_repetition_penalty"],
                components["qwen_max_new_tokens"],
                components["vv_do_sample"],
                components["vv_temperature"],
                components["vv_top_k"],
                components["vv_top_p"],
                components["vv_repetition_penalty"],
                components["vv_cfg_scale"],
                components["vv_num_steps"],
                components["lux_device"],
                components["lux_threads"],
                components["lux_num_steps"],
                components["lux_t_shift"],
                components["lux_speed"],
                components["lux_return_smooth"],
                components["lux_rms"],
                components["lux_ref_duration"],
            ],
            outputs=[components["output_audio"], components["clone_status"]],
        )

        # Toggle language visibility based on model selection
        def toggle_language_visibility(model_selection):
            is_qwen = "Qwen" in model_selection
            return gr.update(visible=is_qwen)

        components["clone_model_dropdown"].change(
            toggle_language_visibility,
            inputs=[components["clone_model_dropdown"]],
            outputs=[components["language_row"]],
        )

        # Toggle accordion visibility based on engine
        def toggle_engine_params(model_selection):
            is_qwen = "Qwen" in model_selection
            is_vibe = "VibeVoice" in model_selection
            is_lux = "LuxTTS" in model_selection
            return (
                gr.update(visible=is_qwen),
                gr.update(visible=is_vibe),
                gr.update(visible=is_lux),
            )

        components["clone_model_dropdown"].change(
            toggle_engine_params,
            inputs=[components["clone_model_dropdown"]],
            outputs=[
                components["qwen_params_accordion"],
                components["vv_params_accordion"],
                components["lux_params_accordion"],
            ],
        )

        # Save voice clone model selection
        components["clone_model_dropdown"].change(
            lambda x: save_preference("voice_clone_model", x),
            inputs=[components["clone_model_dropdown"]],
            outputs=[],
        )

        # Save language selection
        components["language_dropdown"].change(
            lambda x: save_preference("language", x),
            inputs=[components["language_dropdown"]],
            outputs=[],
        )

        components["lux_device"].change(
            lambda x: save_preference("luxtts_device", x),
            inputs=[components["lux_device"]],
            outputs=[],
        )
        components["lux_threads"].change(
            lambda x: save_preference("luxtts_threads", int(x) if x is not None else 0),
            inputs=[components["lux_threads"]],
            outputs=[],
        )
        components["lux_num_steps"].change(
            lambda x: save_preference("luxtts_num_steps", int(x)),
            inputs=[components["lux_num_steps"]],
            outputs=[],
        )
        components["lux_t_shift"].change(
            lambda x: save_preference("luxtts_t_shift", float(x)),
            inputs=[components["lux_t_shift"]],
            outputs=[],
        )
        components["lux_speed"].change(
            lambda x: save_preference("luxtts_speed", float(x)),
            inputs=[components["lux_speed"]],
            outputs=[],
        )
        components["lux_return_smooth"].change(
            lambda x: save_preference("luxtts_return_smooth", bool(x)),
            inputs=[components["lux_return_smooth"]],
            outputs=[],
        )
        components["lux_rms"].change(
            lambda x: save_preference("luxtts_rms", float(x)),
            inputs=[components["lux_rms"]],
            outputs=[],
        )
        components["lux_ref_duration"].change(
            lambda x: save_preference("luxtts_ref_duration", float(x)),
            inputs=[components["lux_ref_duration"]],
            outputs=[],
        )

        # Refresh emotion dropdowns and auto-load first sample when tab is selected
        def on_tab_select(lister_value):
            """When tab is selected, refresh emotions and auto-load sample if not loaded."""
            emotion_update = gr.update(
                choices=shared_state["get_emotion_choices"](
                    shared_state["_active_emotions"]
                )
            )

            # Auto-load selected sample
            sample_name = get_selected_sample_name(lister_value)
            if sample_name:
                audio, text, info = load_sample_details(sample_name)
                return emotion_update, audio, text, info

            return emotion_update, None, "", ""

        components["voice_clone_tab"].select(
            on_tab_select,
            inputs=[components["sample_lister"]],
            outputs=[
                components["qwen_emotion_preset"],
                components["sample_audio"],
                components["sample_text"],
                components["sample_info"],
            ],
        )


# Export for tab registry
get_tool_class = lambda: VoiceCloneTool


if __name__ == "__main__":
    """Standalone testing of Voice Clone tool."""
    from modules.core_components.tools import run_tool_standalone

    run_tool_standalone(VoiceCloneTool, port=7862, title="Voice Clone - Standalone")
