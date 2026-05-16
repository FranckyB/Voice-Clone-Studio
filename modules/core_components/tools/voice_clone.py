"""
Voice Clone Tab

Clone voices from samples using DramaBox.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
import random
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager
from gradio_filelister import FileLister


class VoiceCloneTool(Tool):
    """Voice Clone tool — DramaBox only."""

    config = ToolConfig(
        name="Voice Clone",
        module_name="tool_voice_clone",
        description="Clone voices from voice samples using DramaBox",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Voice Clone tool UI."""
        components = {}

        get_sample_choices = shared_state['get_sample_choices']
        load_sample_details = shared_state['load_sample_details']
        _user_config = shared_state['_user_config']
        create_dramabox_advanced_params = shared_state['create_dramabox_advanced_params']

        with gr.TabItem("Voice Clone", id="tab_voice_clone") as voice_clone_tab:
            components['voice_clone_tab'] = voice_clone_tab
            gr.Markdown("Clone Voices from Samples using DramaBox. <small>(Use Prep Samples to add samples)</small>")

            with gr.Row():
                # Left column — sample selection
                with gr.Column(scale=1):
                    gr.Markdown("### Voice Sample")

                    components['clear_sample_btn'] = gr.Button(
                        "Clear Sample (LoRA Only)", size="sm", variant="secondary"
                    )

                    components['sample_lister'] = FileLister(
                        value=get_sample_choices(),
                        height=200,
                        show_footer=False,
                        interactive=True,
                    )

                    components['sample_audio'] = gr.Audio(
                        label="Sample Preview",
                        type="filepath",
                        interactive=False,
                        value=None,
                        elem_id="voice-clone-sample-audio"
                    )

                    components['sample_text'] = gr.Textbox(
                        label="Sample Text",
                        interactive=False,
                        max_lines=10,
                        value=None
                    )

                    components['sample_info'] = gr.Textbox(
                        label="Info",
                        interactive=False,
                        max_lines=10,
                        value=None
                    )

                    components['lora_dropdown'] = gr.Dropdown(
                        choices=["(None)"],
                        value="(None)",
                        label="LoRA (Optional)",
                        info="Choose a trained LoRA from Train Model, or keep (None)",
                        interactive=True
                    )
                    components['lora_path_map'] = gr.State(value={})

                # Right column — generation
                with gr.Column(scale=3):
                    gr.Markdown("### Generate Speech")

                    components['text_input'] = gr.Textbox(
                        label="Text to Generate",
                        placeholder="Enter the text you want to speak in the cloned voice...",
                        lines=6
                    )

                    import modules.core_components.prompt_hub as _prompt_hub
                    components.update(_prompt_hub.create_prompt_loader("vc", "Saved Prompts"))

                    components['seed_input'] = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0,
                    )

                    # Create with hardcoded defaults; saved values are restored on tab.select
                    _db_params = create_dramabox_advanced_params()
                    components['dramabox_params_accordion'] = _db_params['accordion']
                    components['dramabox_negative_prompt'] = _db_params['negative_prompt']
                    components['dramabox_ref_duration'] = _db_params['ref_duration']
                    components['dramabox_gen_duration'] = _db_params['gen_duration']
                    components['dramabox_steps'] = _db_params['steps']
                    components['dramabox_sampler'] = _db_params['sampler']
                    components['dramabox_speed'] = _db_params['speed']
                    components['dramabox_duration_multiplier'] = _db_params['duration_multiplier']
                    components['dramabox_cfg_scale'] = _db_params['cfg_scale']
                    components['dramabox_stg_scale'] = _db_params['stg_scale']
                    components['dramabox_rescale_scale'] = _db_params['rescale_scale']
                    components['dramabox_id_guidance_scale'] = _db_params['id_guidance_scale']
                    components['dramabox_no_watermark'] = _db_params['no_watermark']

                    components['generate_btn'] = gr.Button(
                        "Generate Audio", variant="primary", size="lg"
                    )

                    components['output_audio'] = gr.Audio(
                        label="Generated Audio", type="filepath"
                    )

                    manual_save = _user_config.get("manual_save", False)
                    components['save_result_btn'] = gr.Button(
                        "Save to Output", variant="primary", size="lg",
                        visible=manual_save, interactive=False
                    )
                    components['_result_metadata'] = gr.Textbox(visible=False)

                    components['clone_status'] = gr.Textbox(
                        label="Status", interactive=False, lines=2, max_lines=5
                    )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Clone tab events."""

        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        load_sample_details = shared_state['load_sample_details']
        save_preference = shared_state['save_preference']
        OUTPUT_DIR = shared_state['OUTPUT_DIR']
        TEMP_DIR = shared_state['TEMP_DIR']
        play_completion_beep = shared_state.get('play_completion_beep')
        save_result_to_output = shared_state['save_result_to_output']
        _user_config = shared_state['_user_config']

        tts_manager = get_tts_manager()

        wire_param_persistence = shared_state['wire_param_persistence']
        param_map = {
            'dramabox': [
                ('seed_input', 'seed'),
                ('dramabox_negative_prompt', 'negative_prompt'),
                ('dramabox_ref_duration', 'ref_duration'),
                ('dramabox_gen_duration', 'gen_duration'),
                ('dramabox_steps', 'steps'),
                ('dramabox_sampler', 'sampler'),
                ('dramabox_speed', 'speed'),
                ('dramabox_duration_multiplier', 'duration_multiplier'),
                ('dramabox_cfg_scale', 'cfg_scale'),
                ('dramabox_stg_scale', 'stg_scale'),
                ('dramabox_rescale_scale', 'rescale_scale'),
                ('dramabox_id_guidance_scale', 'id_guidance_scale'),
                ('dramabox_no_watermark', 'no_watermark'),
            ],
        }
        wire_param_persistence(components, _user_config, param_map)

        create_param_restore_handler = shared_state['create_param_restore_handler']
        restore_fn, restore_outputs = create_param_restore_handler(components, _user_config, param_map)

        def list_trained_loras():
            trained_models_folder = _user_config.get("trained_models_folder", "loras")
            trained_root = OUTPUT_DIR.parent / trained_models_folder
            lora_items = []
            if not trained_root.exists():
                return lora_items
            for speaker_dir in sorted(
                [d for d in trained_root.iterdir() if d.is_dir()],
                key=lambda d: d.name.lower()
            ):
                # New naming: slug_dramabox_*.safetensors (periodic + best)
                for ckpt in sorted(speaker_dir.glob("*_dramabox_*.safetensors"), key=lambda p: p.name, reverse=True):
                    lora_items.append((f"{speaker_dir.name} / {ckpt.name}", str(ckpt)))
                # Legacy naming fallback
                adapter_path = speaker_dir / "adapter_model.safetensors"
                if adapter_path.exists():
                    lora_items.append((f"{speaker_dir.name} / adapter_model.safetensors", str(adapter_path)))
                for lora_step in sorted(speaker_dir.glob("lora_step_*.safetensors"), key=lambda p: p.name, reverse=True):
                    lora_items.append((f"{speaker_dir.name} / {lora_step.name}", str(lora_step)))
                for best_step in sorted(speaker_dir.glob("best_step_*.safetensors"), key=lambda p: p.name, reverse=True):
                    lora_items.append((f"{speaker_dir.name} / {best_step.name}", str(best_step)))
            return lora_items

        def refresh_lora_choices(current_choice):
            lora_items = list_trained_loras()
            lora_map = {label: path for label, path in lora_items}
            choices = ["(None)"] + [label for label, _ in lora_items]
            selected = current_choice if current_choice in choices else "(None)"
            return gr.update(choices=choices, value=selected), lora_map

        def get_selected_sample_name(lister_value):
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                from modules.core_components.tools import strip_sample_extension
                return strip_sample_extension(selected[0])
            return None

        def generate_audio_handler(
            lister_value, text_to_generate, seed,
            lora_selection, lora_path_map,
            db_negative_prompt, db_ref_duration, db_gen_duration, db_steps,
            db_sampler, db_speed, db_duration_multiplier,
            db_cfg_scale, db_stg_scale, db_rescale_scale,
            db_id_guidance_scale, db_no_watermark,
            progress=gr.Progress()
        ):
            if not text_to_generate or not text_to_generate.strip():
                return None, "Please enter text to generate.", "", gr.update()

            sample_name = get_selected_sample_name(lister_value)
            sample_wav = None
            if sample_name:
                samples = get_available_samples()
                sample = next((s for s in samples if s["name"] == sample_name), None)
                sample_wav = sample["wav_path"] if sample else None

            lora_path = None
            if lora_selection and lora_selection != "(None)":
                lora_path = (lora_path_map or {}).get(lora_selection)
                if not lora_path:
                    return None, "❌ Selected LoRA path could not be resolved. Refresh and try again.", "", gr.update()

            try:
                actual_seed = int(seed) if seed is not None else -1
                if actual_seed < 0:
                    actual_seed = random.randint(0, 2147483647)

                progress(0.1, desc="Preparing DramaBox generation...")

                sample_tag = sample_name or "no_sample"
                stem = f"dramabox_{sample_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                temp_path = TEMP_DIR / f"{stem}.wav"

                def _opt_float(v):
                    try: return float(v) if v is not None else None
                    except Exception: return None

                def _opt_int(v):
                    try: return int(v) if v is not None else None
                    except Exception: return None

                dramabox_params = {
                    "sampler": db_sampler if db_sampler in ("euler", "heun") else "euler",
                    "ref_duration": _opt_float(db_ref_duration),
                    "speed": _opt_float(db_speed),
                    "duration_multiplier": _opt_float(db_duration_multiplier),
                    "negative_prompt": (db_negative_prompt or "").strip(),
                    "id_guidance_scale": _opt_float(db_id_guidance_scale),
                    "no_watermark": bool(db_no_watermark),
                    "gen_duration": None,
                    "steps": None,
                    "cfg_scale": None,
                    "stg_scale": None,
                    "rescale_scale": None,
                }

                gen_dur = _opt_float(db_gen_duration)
                if gen_dur is not None and gen_dur > 0:
                    dramabox_params["gen_duration"] = gen_dur
                steps_val = _opt_int(db_steps)
                if steps_val is not None and steps_val > 0:
                    dramabox_params["steps"] = steps_val
                cfg = _opt_float(db_cfg_scale)
                if cfg is not None and cfg > 0:
                    dramabox_params["cfg_scale"] = cfg
                stg = _opt_float(db_stg_scale)
                if stg is not None and stg > 0:
                    dramabox_params["stg_scale"] = stg
                rsc = _opt_float(db_rescale_scale)
                if rsc is not None and rsc >= 0:
                    dramabox_params["rescale_scale"] = rsc

                progress(0.3, desc="Running DramaBox inference...")
                cpu_offload = _user_config.get("dramabox_cpu_offload", False)
                tts_manager.generate_dramabox_to_file(
                    prompt=text_to_generate.strip(),
                    output_path=str(temp_path),
                    voice_sample=str(sample_wav) if sample_wav else None,
                    seed=int(actual_seed),
                    lora_path=str(lora_path) if lora_path else None,
                    cpu_offload=cpu_offload,
                    dramabox_params=dramabox_params,
                )

                progress(0.85, desc="Finalizing output...")
                metadata_lines = [
                    f"Generated: {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    f"Sample: {sample_name or '(None)'}",
                    f"LoRA: {lora_selection if lora_path else '(None)'}",
                    f"Engine: DramaBox",
                    f"Seed: {actual_seed}",
                    f"Text: {' '.join(text_to_generate.split())}",
                ]
                metadata_out = "\n".join(metadata_lines)

                manual_save = _user_config.get("manual_save", False)
                if manual_save:
                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    return (
                        str(temp_path),
                        f"Generated with DramaBox. Seed: {actual_seed}\nClick 'Save to Output' to keep this result.",
                        metadata_out,
                        gr.update(interactive=True)
                    )

                output_format = _user_config.get("output_format", "wav")
                output_path = save_result_to_output(temp_path, OUTPUT_DIR, output_format, metadata_out)
                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                return str(output_path), f"Generated with DramaBox. Seed: {actual_seed}", "", gr.update()

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Error generating audio: {str(e)}", "", gr.update()

        def load_sample_from_lister(lister_value):
            sample_name = get_selected_sample_name(lister_value)
            if not sample_name:
                return None, "", ""
            return load_sample_details(sample_name)

        def refresh_samples_keep_selection(lister_value):
            new_files = get_sample_choices()
            prev_selected = []
            if lister_value:
                prev = lister_value.get("selected", [])
                new_names = set(new_files)
                prev_selected = [s for s in prev if s in new_names]
            return {"files": [{"name": f, "date": ""} for f in new_files], "selected": prev_selected}

        # Event wiring
        components['sample_lister'].change(
            load_sample_from_lister,
            inputs=[components['sample_lister']],
            outputs=[components['sample_audio'], components['sample_text'], components['sample_info']]
        )

        components['sample_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#voice-clone-sample-audio .play-pause-button'); if (btn) btn.click(); }, 150); }"
        )

        def clear_sample_selection():
            """Reset FileLister to no selection and clear preview fields."""
            return gr.update(value=get_sample_choices()), None, None, None

        components['clear_sample_btn'].click(
            clear_sample_selection,
            outputs=[
                components['sample_lister'],
                components['sample_audio'],
                components['sample_text'],
                components['sample_info'],
            ]
        )

        components['voice_clone_tab'].select(
            refresh_samples_keep_selection,
            inputs=[components['sample_lister']],
            outputs=[components['sample_lister']]
        )

        components['voice_clone_tab'].select(
            refresh_lora_choices,
            inputs=[components['lora_dropdown']],
            outputs=[components['lora_dropdown'], components['lora_path_map']]
        )

        components['voice_clone_tab'].select(
            restore_fn,
            inputs=[],
            outputs=restore_outputs
        )

        components['generate_btn'].click(
            generate_audio_handler,
            inputs=[
                components['sample_lister'],
                components['text_input'],
                components['seed_input'],
                components['lora_dropdown'],
                components['lora_path_map'],
                components['dramabox_negative_prompt'],
                components['dramabox_ref_duration'],
                components['dramabox_gen_duration'],
                components['dramabox_steps'],
                components['dramabox_sampler'],
                components['dramabox_speed'],
                components['dramabox_duration_multiplier'],
                components['dramabox_cfg_scale'],
                components['dramabox_stg_scale'],
                components['dramabox_rescale_scale'],
                components['dramabox_id_guidance_scale'],
                components['dramabox_no_watermark'],
            ],
            outputs=[
                components['output_audio'],
                components['clone_status'],
                components['_result_metadata'],
                components['save_result_btn'],
            ]
        )

        if _user_config.get("manual_save", False):
            def save_result_handler(audio_path, metadata_text):
                if not audio_path:
                    return None, "❌ No audio to save.", gr.update(interactive=False)
                output_format = _user_config.get("output_format", "wav")
                output_path = save_result_to_output(audio_path, OUTPUT_DIR, output_format, metadata_text)
                return str(output_path), "Saved to output folder.", gr.update(interactive=False)

            components['save_result_btn'].click(
                save_result_handler,
                inputs=[components['output_audio'], components['_result_metadata']],
                outputs=[components['output_audio'], components['clone_status'], components['save_result_btn']]
            )

        app = shared_state.get('app')
        if app:
            app.load(
                refresh_lora_choices,
                inputs=[components['lora_dropdown']],
                outputs=[components['lora_dropdown'], components['lora_path_map']]
            )


get_tool_class = lambda: VoiceCloneTool
