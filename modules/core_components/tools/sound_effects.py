"""
Sound Effects Tab

Generate sound effects and foley audio using MMAudio.
Supports text-to-audio and video-to-audio synthesis.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
import soundfile as sf
import random
import time
from pathlib import Path

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.foley_manager import get_foley_manager
from modules.core_components.constants import MMAUDIO_GENERATION_DEFAULTS


class SoundEffectsTool(Tool):
    """Sound Effects tool implementation using MMAudio."""

    config = ToolConfig(
        name="Sound Effects",
        module_name="tool_sound_effects",
        description="Generate sound effects from text or video using MMAudio",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Sound Effects tool UI."""
        components = {}

        _user_config = shared_state.get('_user_config', {})
        OUTPUT_DIR = shared_state.get('OUTPUT_DIR')

        # Get foley manager to populate model choices
        foley_manager = get_foley_manager()
        model_choices = foley_manager.get_available_models() if foley_manager else [
            "Medium (44kHz)", "Large v2 (44kHz)"
        ]

        with gr.TabItem("SFX", id="tab_sound_effects") as sfx_tab:
            components['sfx_tab'] = sfx_tab
            gr.Markdown("Generate sound effects and foley audio from text prompts or video clips")
            # Mode toggle
            components['sfx_mode'] = gr.Radio(
                choices=["Text to Audio", "Video to Audio"],
                value="Text to Audio",
                label="Mode",
                show_label=False,
                container=False,
                interactive=True
            )

            with gr.Row():
                # Left column: Input controls (wider)
                with gr.Column(scale=2):
                    # Text prompt (always visible)
                    components['sfx_prompt'] = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the sound: e.g., 'thunder rumbling in the distance', 'glass shattering on a tile floor', 'birds chirping in a forest'...",
                        lines=4
                    )

                    components['sfx_negative_prompt'] = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to avoid: e.g., 'music, speech, noise'",
                        lines=2,
                        value="human sounds, music, speech, voices"
                    )

                    import modules.core_components.prompt_hub as _prompt_hub
                    components.update(_prompt_hub.create_prompt_loader("sfx", "Saved Prompts"))

                    with gr.Row():
                        components['sfx_model_size'] = gr.Dropdown(
                            choices=model_choices,
                            value=model_choices[-1] if model_choices else "Large v2 (44kHz)",
                            label="Model",
                            scale=2
                        )

                        components['sfx_seed'] = gr.Number(
                            label="Seed (-1 = random)",
                            value=-1,
                            precision=0,
                            scale=1
                        )

                    # Generation parameters
                    with gr.Accordion("Generation Settings", open=False):
                        with gr.Row():
                            components['sfx_duration'] = gr.Slider(
                                minimum=1.0,
                                maximum=30.0,
                                value=MMAUDIO_GENERATION_DEFAULTS["duration"],
                                step=0.5,
                                label="Duration (seconds)",
                                scale=2
                            )

                            components['sfx_steps'] = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=MMAUDIO_GENERATION_DEFAULTS["num_steps"],
                                step=1,
                                label="Steps (quality vs speed)",
                                scale=1
                            )
                            components['sfx_cfg_strength'] = gr.Slider(
                                minimum=1.0,
                                maximum=15.0,
                                value=MMAUDIO_GENERATION_DEFAULTS["cfg_strength"],
                                step=0.5,
                                label="Guidance Strength",
                                scale=1
                            )

                    # Generate button
                    components['sfx_generate_btn'] = gr.Button(
                        "Generate Sound Effect",
                        variant="primary",
                        size="lg"
                    )
                    components['sfx_status'] = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1,
                        max_lines=5
                    )

                # Right column: Video + Audio output (narrow)
                with gr.Column(scale=1):
                    # Video section — starts visible=True for DOM rendering; toggle_mode on tab.select hides it
                    with gr.Group(visible=True) as sfx_video_group:

                        # Radio to switch between source and result
                        components['sfx_video_toggle'] = gr.Radio(
                            choices=["Source", "Result"],
                            value="Source",
                            show_label=False,
                            interactive=True
                        )

                        # Source video input
                        components['sfx_video_input'] = gr.Video(
                            label="Source Video",
                            height=400
                        )

                        # Result video (source + generated audio muxed)
                        components['sfx_output_video'] = gr.Video(
                            label="Result (video + generated audio)",
                            visible=False,
                            height=400
                        )

                    components['sfx_video_group'] = sfx_video_group

                    # Audio output (always visible)
                    components['sfx_output_audio'] = gr.Audio(
                        label="Generated Audio",
                        type="filepath"
                    )

                    # Save button
                    components['sfx_save_btn'] = gr.Button(
                        "Save",
                        variant="primary",
                        interactive=False
                    )

                    # Hidden state for suggested filename, metadata, and existing files
                    components['sfx_suggested_name'] = gr.Textbox(
                        visible=False
                    )
                    components['sfx_metadata'] = gr.Textbox(
                        visible=False
                    )
                    components['sfx_existing_files_json'] = gr.Textbox(
                        visible=False
                    )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Sound Effects events."""

        OUTPUT_DIR = shared_state.get('OUTPUT_DIR')
        TEMP_DIR = shared_state.get('TEMP_DIR')
        play_completion_beep = shared_state.get('play_completion_beep')
        save_preference = shared_state.get('save_preference')
        show_input_modal_js = shared_state['show_input_modal_js']
        input_trigger = shared_state['input_trigger']
        convert_audio_format = shared_state['convert_audio_format']
        embed_metadata = shared_state['embed_metadata']
        user_config = shared_state.get('_user_config', {})

        foley_manager = get_foley_manager()

        # Mode toggle — show/hide video group, show/hide duration
        def toggle_mode(mode):
            is_video = mode == "Video to Audio"
            return (
                gr.update(visible=is_video),   # video group
                gr.update(
                    placeholder="Optional: describe expected sounds to guide generation..."
                    if is_video else
                    "Describe the sound: e.g., 'thunder rumbling in the distance', 'glass shattering on a tile floor', 'birds chirping in a forest'..."
                ),  # prompt placeholder
                gr.update(visible=not is_video),  # duration slider — hidden in video mode
            )

        components['sfx_mode'].change(
            toggle_mode,
            inputs=[components['sfx_mode']],
            outputs=[
                components['sfx_video_group'],
                components['sfx_prompt'],
                components['sfx_duration'],
            ]
        )

        # Set correct mode visibility on tab select
        components['sfx_tab'].select(
            toggle_mode,
            inputs=[components['sfx_mode']],
            outputs=[
                components['sfx_video_group'],
                components['sfx_prompt'],
                components['sfx_duration'],
            ]
        )

        # Source / Result radio — toggle which video is visible
        def toggle_video_preview(choice):
            return (
                gr.update(visible=choice == "Source"),   # source video
                gr.update(visible=choice == "Result"),   # result video
            )

        components['sfx_video_toggle'].change(
            toggle_video_preview,
            inputs=[components['sfx_video_toggle']],
            outputs=[
                components['sfx_video_input'],
                components['sfx_output_video'],
            ]
        )

        # Refresh model list when dropdown is clicked
        def refresh_model_choices():
            choices = foley_manager.get_available_models()
            return gr.update(choices=choices)

        # Generation handler
        def generate_sfx(mode, prompt, negative_prompt, video, model_size,
                         duration, seed, steps, cfg_strength,
                         progress=gr.Progress()):
            """Generate a sound effect."""
            if mode == "Text to Audio" and (not prompt or not prompt.strip()):
                return None, None, "Error: Please enter a text prompt.", "", "", gr.update(interactive=False), gr.update(), gr.update()

            if mode == "Video to Audio" and not video:
                return None, None, "Error: Please upload a video file.", "", "", gr.update(interactive=False), gr.update(), gr.update()

            try:
                # Handle seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)

                progress(0.05, desc="Loading MMAudio model...")

                # Load model
                foley_manager.load_model(
                    display_name=model_size,
                    progress_callback=lambda frac, desc: progress(frac * 0.5, desc=desc)
                )

                # Generate based on mode
                if mode == "Video to Audio":
                    progress(0.5, desc="Generating audio from video...")
                    # Duration is auto-detected from video — pass large value,
                    # load_video() truncates to actual video length
                    sr, audio_np = foley_manager.generate_video_to_audio(
                        video_path=video,
                        prompt=prompt.strip() if prompt else "",
                        negative_prompt=negative_prompt.strip() if negative_prompt else "",
                        duration=9999.0,
                        seed=seed,
                        num_steps=int(steps),
                        cfg_strength=cfg_strength,
                        progress_callback=lambda frac, desc: progress(0.5 + frac * 0.4, desc=desc)
                    )
                else:
                    progress(0.5, desc="Generating audio from text...")
                    sr, audio_np = foley_manager.generate_text_to_audio(
                        prompt=prompt.strip(),
                        negative_prompt=negative_prompt.strip() if negative_prompt else "",
                        duration=duration,
                        seed=seed,
                        num_steps=int(steps),
                        cfg_strength=cfg_strength,
                        progress_callback=lambda frac, desc: progress(0.5 + frac * 0.4, desc=desc)
                    )

                progress(0.9, desc="Saving preview...")

                # Save to temp dir (not output — user saves manually)
                # Build suggested name from first 8 words of prompt
                if prompt and prompt.strip():
                    words = prompt.strip().split()[:8]
                    safe_prompt = "_".join(words)
                    safe_prompt = safe_prompt.replace("/", "_").replace("\\", "_")
                    safe_prompt = "".join(c for c in safe_prompt if c.isalnum() or c in "_-")
                else:
                    safe_prompt = "sfx"
                if not safe_prompt:
                    safe_prompt = "sfx"

                suggested_name = f"sfx_{safe_prompt}"
                temp_filename = f"{suggested_name}.wav"
                temp_path = TEMP_DIR / temp_filename

                # audio_np shape is [channels, samples] — squeeze to mono if needed
                if audio_np.ndim > 1:
                    audio_data = audio_np[0]  # Take first channel
                else:
                    audio_data = audio_np

                # Compute actual duration from audio length
                actual_duration = round(len(audio_data) / sr, 2)

                sf.write(str(temp_path), audio_data, sr)

                # Build metadata text
                metadata_lines = [
                    f"Mode: {mode}",
                    f"Prompt: {' '.join(prompt.split()) if prompt else '(none)'}",
                ]
                if negative_prompt and negative_prompt.strip():
                    metadata_lines.append(f"Negative: {' '.join(negative_prompt.split())}")
                if mode == "Video to Audio" and video:
                    metadata_lines.append(f"Video: {Path(video).name}")
                metadata_lines.extend([
                    f"Model: {model_size}",
                    f"Duration: {actual_duration}s",
                    f"Steps: {int(steps)}",
                    f"CFG Strength: {cfg_strength}",
                    f"Seed: {seed}",
                    f"Sample Rate: {sr} Hz",
                    "Engine: MMAudio"
                ])
                metadata_text = "\n".join(metadata_lines)

                # If video mode, mux video + generated audio with ffmpeg
                combined_video_path = None
                if mode == "Video to Audio" and video:
                    try:
                        import subprocess
                        timestamp = int(time.time())
                        combined_filename = f"sfx_preview_{safe_prompt}_{timestamp}.mp4"
                        combined_path = TEMP_DIR / combined_filename
                        subprocess.run(
                            ["ffmpeg", "-y",
                             "-i", str(video),
                             "-i", str(temp_path),
                             "-c:v", "copy",
                             "-c:a", "aac", "-b:a", "192k",
                             "-map", "0:v:0", "-map", "1:a:0",
                             "-shortest",
                             "-loglevel", "error",
                             str(combined_path)],
                            check=True, timeout=60
                        )
                        combined_video_path = str(combined_path)
                    except Exception as mux_err:
                        print(f"Video mux failed: {mux_err}")

                progress(1.0, desc="Done!")
                play_completion_beep()

                status = f"Ready to save | Seed: {seed} | {actual_duration}s @ {sr}Hz"

                # In video mode, auto-switch preview to Result
                if combined_video_path:
                    return (
                        str(temp_path),
                        gr.update(value=combined_video_path, visible=True),
                        status,
                        suggested_name,
                        metadata_text,
                        gr.update(interactive=True),
                        gr.update(value="Result"),
                        gr.update(visible=False),
                    )

                return (
                    str(temp_path),
                    None,
                    status,
                    suggested_name,
                    metadata_text,
                    gr.update(interactive=True),
                    gr.update(),
                    gr.update(),
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, f"Error: {str(e)}", "", "", gr.update(interactive=False), gr.update(), gr.update()

        def _disable_sfx_btn():
            return gr.update(interactive=False)

        def _enable_sfx_btn():
            return gr.update(interactive=True)

        components['sfx_generate_btn'].click(
            _disable_sfx_btn, outputs=[components['sfx_generate_btn']]
        ).then(
            generate_sfx,
            inputs=[
                components['sfx_mode'],
                components['sfx_prompt'],
                components['sfx_negative_prompt'],
                components['sfx_video_input'],
                components['sfx_model_size'],
                components['sfx_duration'],
                components['sfx_seed'],
                components['sfx_steps'],
                components['sfx_cfg_strength'],
            ],
            outputs=[
                components['sfx_output_audio'],
                components['sfx_output_video'],
                components['sfx_status'],
                components['sfx_suggested_name'],
                components['sfx_metadata'],
                components['sfx_save_btn'],
                components['sfx_video_toggle'],
                components['sfx_video_input'],
            ]
        ).then(
            _enable_sfx_btn, outputs=[components['sfx_generate_btn']]
        )

        # --- Save button: open input modal with suggested filename ---
        save_sfx_modal_js = show_input_modal_js(
            title="Save Sound Effect",
            message="Enter a filename for this sound effect:",
            placeholder="e.g., thunder_rumble, glass_shatter",
            context="save_sfx_"
        )

        def get_sfx_existing_files():
            """Return JSON list of existing file stems in output dir for overwrite detection."""
            import json as json_mod
            names = set()
            for ext in ('*.wav', '*.flac', '*.mp3'):
                names.update(f.stem for f in OUTPUT_DIR.glob(ext))
            return json_mod.dumps(sorted(names))

        save_sfx_js = f"""
        (existingFilesJson, suggestedName) => {{
            try {{
                window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
            }} catch(e) {{
                window.inputModalExistingFiles = [];
            }}
            const openModal = {save_sfx_modal_js};
            openModal(suggestedName);
        }}
        """

        components['sfx_save_btn'].click(
            fn=get_sfx_existing_files,
            inputs=[],
            outputs=[components['sfx_existing_files_json']],
        ).then(
            fn=None,
            inputs=[components['sfx_existing_files_json'], components['sfx_suggested_name']],
            js=save_sfx_js
        )

        # --- Input modal handler: save file from temp to output ---
        def handle_sfx_input_modal(input_value, audio_path, metadata_text):
            """Process input modal result for saving sound effects."""
            no_update = gr.update(), gr.update()

            if not input_value or not input_value.startswith("save_sfx_"):
                return no_update

            parts = input_value.split("_", 3)
            # parts: ['save', 'sfx', 'cancel'] or ['save', 'sfx', '<user_name>', '<uuid>']
            if len(parts) >= 3 and parts[2] == "cancel":
                return no_update

            # Extract user-chosen name (everything between 'save_sfx_' and last '_uuid')
            raw_name = input_value[len("save_sfx_"):]
            # Remove trailing UUID (last segment after _)
            name_parts = raw_name.rsplit("_", 1)
            chosen_name = name_parts[0] if len(name_parts) > 1 else raw_name

            if not chosen_name.strip():
                return "Error: No filename provided", gr.update()

            if not audio_path:
                return "Error: No audio to save", gr.update()

            # Clean filename
            clean_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in chosen_name).strip()
            clean_name = clean_name.replace(" ", "_")
            if not clean_name:
                return "Error: Invalid filename", gr.update()

            try:
                source_path = Path(audio_path)
                if not source_path.exists():
                    return "Error: Audio file not found", gr.update()

                output_format = user_config.get("output_format", "wav")
                output_path = OUTPUT_DIR / f"{clean_name}.{output_format}"
                convert_audio_format(source_path, output_path, output_format)

                # Embed metadata in audio file
                if metadata_text:
                    metadata_out = '\n'.join(line.lstrip() for line in metadata_text.lstrip().splitlines())
                    embed_metadata(output_path, metadata_out)

                return f"Saved: {output_path.name}", gr.update(interactive=False)

            except Exception as e:
                return f"Error saving: {str(e)}", gr.update()

        input_trigger.change(
            handle_sfx_input_modal,
            inputs=[input_trigger, components['sfx_output_audio'],
                    components['sfx_metadata']],
            outputs=[components['sfx_status'], components['sfx_save_btn']]
        )

        # --- Cross-tab prompt routing ---
        import modules.core_components.prompt_hub as _prompt_hub
        _prompt_hub.wire_prompt_loader(components, "sfx", {"sound_effects.prompt": components['sfx_prompt']})

        prompt_apply_trigger = shared_state.get('prompt_apply_trigger')
        if prompt_apply_trigger is not None:
            import modules.core_components.prompt_hub as _prompt_hub

            def _apply_sfx_prompt(raw_value, current):
                parsed = _prompt_hub.parse_apply_payload(raw_value)
                if not parsed or parsed['target_id'] != 'sfx.prompt':
                    return gr.update()
                return gr.update(value=_prompt_hub.merge_text(current, parsed['text'], parsed['mode']))

            prompt_apply_trigger.change(
                _apply_sfx_prompt,
                inputs=[prompt_apply_trigger, components['sfx_prompt']],
                outputs=[components['sfx_prompt']],
            )

            def _apply_sfx_negative(raw_value, current):
                parsed = _prompt_hub.parse_apply_payload(raw_value)
                if not parsed or parsed['target_id'] != 'sfx.negative':
                    return gr.update()
                return gr.update(value=_prompt_hub.merge_text(current, parsed['text'], parsed['mode']))

            prompt_apply_trigger.change(
                _apply_sfx_negative,
                inputs=[prompt_apply_trigger, components['sfx_negative_prompt']],
                outputs=[components['sfx_negative_prompt']],
            )

        # Set correct initial visibility on page load (tab.select doesn't fire for the first tab)
        app = shared_state.get('app')
        if app:
            app.load(
                toggle_mode,
                inputs=[components['sfx_mode']],
                outputs=[
                    components['sfx_video_group'],
                    components['sfx_prompt'],
                    components['sfx_duration'],
                ]
            )


# Export for registry
get_tool_class = lambda: SoundEffectsTool


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(SoundEffectsTool, port=7870, title="Sound Effects - Standalone")
