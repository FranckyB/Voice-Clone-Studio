"""
Conversation Tab

Multi-speaker conversation generation using DramaBox.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import re
import os
import random
import soundfile as sf
import numpy as np
import gradio as gr
from datetime import datetime
from pathlib import Path

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager


NUM_SPEAKERS = 4

DRAMABOX_TIPS_HTML = """
<div style="background: #1e1e2e; border: 1px solid #444; border-radius: 8px; padding: 14px; font-size: 13px; color: #cdd6f4; line-height: 1.7;">
  <b>DramaBox Conversation Tips</b>
  <ul style="margin: 8px 0 0 18px; padding: 0;">
    <li>Label each line with a speaker number: <code>[1]: Hello</code>, <code>[2]: Hi there</code></li>
    <li>Speakers without a label default to speaker 1</li>
    <li>Assign a voice sample to each active speaker number</li>
    <li>Pause Linebreak adds silence between every speaker exchange</li>
    <li>Style instructions in parentheses e.g. <code>(whispering)</code> are stripped before generation</li>
  </ul>
</div>
"""


def preprocess_conversation_script(script_text):
    """Ensure every line has a [N]: speaker label. Defaults unlabelled lines to [1]:."""
    if not script_text:
        return ""
    lines = script_text.strip().split("\n")
    processed = []
    pattern = re.compile(r"^\s*\[(\d+)\]\s*:")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            processed.append("")
            continue
        if pattern.match(stripped):
            processed.append(stripped)
        else:
            processed.append(f"[1]: {stripped}")
    return "\n".join(processed)


def extract_style_instructions(text):
    """Remove parenthetical style markers like (whispering) from text."""
    if not text:
        return text
    cleaned = re.sub(r"\([^)]*\)", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def prepare_voice_samples_dict(get_available_samples, *voice_names):
    """
    Build {speaker_index: wav_path} from voice dropdown values.
    voice_names is a sequence of (sample_name or None), indexed 1..N.
    """
    samples = get_available_samples()
    samples_by_name = {s["name"]: s["wav_path"] for s in samples}
    result = {}
    for i, name in enumerate(voice_names, start=1):
        if name and name != "(None)" and name in samples_by_name:
            result[i] = samples_by_name[name]
        else:
            result[i] = None
    return result


def finalize_conversation_output(segment_paths, pause_secs, TEMP_DIR, OUTPUT_DIR, _user_config, save_result_to_output, metadata_text):
    """Concatenate audio segments with optional silence pauses and return final path."""
    if not segment_paths:
        raise RuntimeError("No audio segments to concatenate.")

    SAMPLE_RATE = 24000
    frames_list = []
    for path in segment_paths:
        data, sr = sf.read(str(path))
        if sr != SAMPLE_RATE:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        if data.ndim > 1:
            data = data[:, 0]
        frames_list.append(data)
        if pause_secs > 0:
            pause_frames = np.zeros(int(SAMPLE_RATE * pause_secs), dtype=np.float32)
            frames_list.append(pause_frames)

    combined = np.concatenate(frames_list).astype(np.float32)
    stem = f"conv_dramabox_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    temp_out = TEMP_DIR / stem
    sf.write(str(temp_out), combined, SAMPLE_RATE)

    manual_save = _user_config.get("manual_save", False)
    if manual_save:
        return str(temp_out)

    output_format = _user_config.get("output_format", "wav")
    final = save_result_to_output(temp_out, OUTPUT_DIR, output_format, metadata_text)
    return str(final)


class ConversationTool(Tool):
    """Multi-speaker Conversation tool — DramaBox only."""

    config = ToolConfig(
        name="Conversation",
        module_name="tool_conversation",
        description="Generate multi-speaker conversations using DramaBox",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Conversation tool UI."""
        components = {}

        get_sample_choices = shared_state['get_sample_choices']
        _user_config = shared_state['_user_config']
        create_dramabox_advanced_params = shared_state['create_dramabox_advanced_params']

        voice_choices = ["(None)"] + get_sample_choices()

        with gr.TabItem("Conversation", id="tab_conversation") as conversation_tab:
            components['conversation_tab'] = conversation_tab
            gr.Markdown("Generate multi-speaker conversations using DramaBox. Label each line with <code>[1]:</code>, <code>[2]:</code> etc.")

            with gr.Row():
                # Left column — script + voice assignment
                with gr.Column(scale=2):
                    gr.Markdown("### Conversation Script")

                    components['conv_script'] = gr.Textbox(
                        label="Script",
                        placeholder="[1]: Hello, how are you?\n[2]: I'm doing great, thanks!\n[1]: That's wonderful to hear.",
                        lines=18,
                    )

                    import modules.core_components.prompt_hub as _prompt_hub
                    components.update(_prompt_hub.create_prompt_loader("conv", "Saved Prompts"))

                    gr.Markdown("### Voice Assignment")
                    with gr.Row():
                        with gr.Column():
                            components['conv_voice_1'] = gr.Dropdown(
                                choices=voice_choices, value="(None)",
                                label="Speaker 1 Voice", interactive=True
                            )
                            components['conv_lora_dropdown_1'] = gr.Dropdown(
                                choices=["(None)"], value="(None)",
                                label="Speaker 1 LoRA", interactive=True
                            )
                        with gr.Column():
                            components['conv_voice_2'] = gr.Dropdown(
                                choices=voice_choices, value="(None)",
                                label="Speaker 2 Voice", interactive=True
                            )
                            components['conv_lora_dropdown_2'] = gr.Dropdown(
                                choices=["(None)"], value="(None)",
                                label="Speaker 2 LoRA", interactive=True
                            )
                    with gr.Row():
                        with gr.Column():
                            components['conv_voice_3'] = gr.Dropdown(
                                choices=voice_choices, value="(None)",
                                label="Speaker 3 Voice", interactive=True
                            )
                            components['conv_lora_dropdown_3'] = gr.Dropdown(
                                choices=["(None)"], value="(None)",
                                label="Speaker 3 LoRA", interactive=True
                            )
                        with gr.Column():
                            components['conv_voice_4'] = gr.Dropdown(
                                choices=voice_choices, value="(None)",
                                label="Speaker 4 Voice", interactive=True
                            )
                            components['conv_lora_dropdown_4'] = gr.Dropdown(
                                choices=["(None)"], value="(None)",
                                label="Speaker 4 LoRA", interactive=True
                            )
                    components['conv_lora_path_map'] = gr.State(value={})

                # Right column — DramaBox settings + generate
                with gr.Column(scale=1):

                    gr.Markdown("### Generation Settings")
                    components['conv_seed'] = gr.Number(
                        label="Seed (-1 for random)", value=-1, precision=0
                    )

                    components['db_conv_pause_linebreak'] = gr.Slider(
                        label="Pause Between Exchanges (seconds)",
                        minimum=0.0, maximum=3.0, step=0.05, value=0.3
                    )

                    # Create with hardcoded defaults; saved values are restored on tab.select
                    _db_params = create_dramabox_advanced_params()
                    components['db_conv_accordion'] = _db_params['accordion']
                    components['db_conv_negative_prompt'] = _db_params['negative_prompt']
                    components['db_conv_ref_duration'] = _db_params['ref_duration']
                    components['db_conv_gen_duration'] = _db_params['gen_duration']
                    components['db_conv_steps'] = _db_params['steps']
                    components['db_conv_sampler'] = _db_params['sampler']
                    components['db_conv_speed'] = _db_params['speed']
                    components['db_conv_duration_multiplier'] = _db_params['duration_multiplier']
                    components['db_conv_cfg_scale'] = _db_params['cfg_scale']
                    components['db_conv_stg_scale'] = _db_params['stg_scale']
                    components['db_conv_rescale_scale'] = _db_params['rescale_scale']
                    components['db_conv_id_guidance_scale'] = _db_params['id_guidance_scale']
                    components['db_conv_no_watermark'] = _db_params['no_watermark']

                    components['generate_btn'] = gr.Button(
                        "Generate Conversation", variant="primary", size="lg"
                    )
                    components['output_audio'] = gr.Audio(
                        label="Generated Conversation", type="filepath"
                    )

                    manual_save = _user_config.get("manual_save", False)
                    components['save_result_btn'] = gr.Button(
                        "Save to Output", variant="primary", size="lg",
                        visible=manual_save, interactive=False
                    )
                    components['_result_metadata'] = gr.Textbox(visible=False)

                    components['conv_status'] = gr.Textbox(
                        label="Status", interactive=False, lines=3, max_lines=8
                    )

            components['dramabox_tips'] = gr.HTML(value=DRAMABOX_TIPS_HTML)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Conversation tab events."""

        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
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
                ('conv_seed', 'seed'),
                ('db_conv_ref_duration', 'ref_duration'),
                ('db_conv_gen_duration', 'gen_duration'),
                ('db_conv_steps', 'steps'),
                ('db_conv_sampler', 'sampler'),
                ('db_conv_speed', 'speed'),
                ('db_conv_duration_multiplier', 'duration_multiplier'),
                ('db_conv_cfg_scale', 'cfg_scale'),
                ('db_conv_stg_scale', 'stg_scale'),
                ('db_conv_rescale_scale', 'rescale_scale'),
                ('db_conv_id_guidance_scale', 'id_guidance_scale'),
                ('db_conv_pause_linebreak', 'pause_linebreak'),
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

        def refresh_all_lora_dropdowns(c1, c2, c3, c4):
            lora_items = list_trained_loras()
            lora_map = {label: path for label, path in lora_items}
            choices = ["(None)"] + [label for label, _ in lora_items]
            def _pick(cur):
                return cur if cur in choices else "(None)"
            return (
                gr.update(choices=choices, value=_pick(c1)),
                gr.update(choices=choices, value=_pick(c2)),
                gr.update(choices=choices, value=_pick(c3)),
                gr.update(choices=choices, value=_pick(c4)),
                lora_map,
            )

        def refresh_voice_choices():
            new_choices = ["(None)"] + get_sample_choices()
            return (
                gr.update(choices=new_choices),
                gr.update(choices=new_choices),
                gr.update(choices=new_choices),
                gr.update(choices=new_choices),
            )

        def generate_dramabox_conversation_handler(
            script, seed,
            voice_1, voice_2, voice_3, voice_4,
            lora_1, lora_2, lora_3, lora_4, lora_path_map,
            pause_linebreak,
            db_negative_prompt, db_ref_duration, db_gen_duration, db_steps,
            db_sampler, db_speed, db_duration_multiplier,
            db_cfg_scale, db_stg_scale, db_rescale_scale,
            db_id_guidance_scale, db_no_watermark,
            progress=gr.Progress()
        ):
            if not script or not script.strip():
                return None, "Please enter a conversation script.", "", gr.update()

            processed = preprocess_conversation_script(script)
            lines = [l for l in processed.split("\n") if l.strip()]
            if not lines:
                return None, "Script has no speakable lines after processing.", "", gr.update()

            voice_samples = prepare_voice_samples_dict(
                get_available_samples, voice_1, voice_2, voice_3, voice_4
            )

            lora_selections = [lora_1, lora_2, lora_3, lora_4]
            lora_paths = {}
            for i, sel in enumerate(lora_selections, start=1):
                if sel and sel != "(None)":
                    path = (lora_path_map or {}).get(sel)
                    if path:
                        lora_paths[i] = path

            try:
                actual_seed = int(seed) if seed is not None else -1
                if actual_seed < 0:
                    actual_seed = random.randint(0, 2147483647)

                def _opt_float(v):
                    try: return float(v) if v is not None else None
                    except Exception: return None

                def _opt_int(v):
                    try: return int(v) if v is not None else None
                    except Exception: return None

                def build_db_params():
                    p = {
                        "sampler": db_sampler if db_sampler in ("euler", "heun") else "euler",
                        "ref_duration": _opt_float(db_ref_duration),
                        "speed": _opt_float(db_speed),
                        "duration_multiplier": _opt_float(db_duration_multiplier),
                        "negative_prompt": (db_negative_prompt or "").strip(),
                        "id_guidance_scale": _opt_float(db_id_guidance_scale),
                        "no_watermark": bool(db_no_watermark),
                        "gen_duration": None, "steps": None,
                        "cfg_scale": None, "stg_scale": None, "rescale_scale": None,
                    }
                    gen_dur = _opt_float(db_gen_duration)
                    if gen_dur is not None and gen_dur > 0:
                        p["gen_duration"] = gen_dur
                    steps_val = _opt_int(db_steps)
                    if steps_val is not None and steps_val > 0:
                        p["steps"] = steps_val
                    cfg = _opt_float(db_cfg_scale)
                    if cfg is not None and cfg > 0:
                        p["cfg_scale"] = cfg
                    stg = _opt_float(db_stg_scale)
                    if stg is not None and stg > 0:
                        p["stg_scale"] = stg
                    rsc = _opt_float(db_rescale_scale)
                    if rsc is not None and rsc >= 0:
                        p["rescale_scale"] = rsc
                    return p

                pattern = re.compile(r"^\[(\d+)\]\s*:\s*(.+)$")
                segment_paths = []
                cpu_offload = _user_config.get("dramabox_cpu_offload", False)
                db_params = build_db_params()

                for idx, line in enumerate(lines):
                    m = pattern.match(line.strip())
                    if not m:
                        continue
                    speaker_idx = int(m.group(1))
                    text_raw = m.group(2).strip()
                    text_clean = extract_style_instructions(text_raw)
                    if not text_clean:
                        continue

                    progress((idx + 1) / len(lines), desc=f"Generating line {idx + 1}/{len(lines)} (Speaker {speaker_idx})...")

                    wav_path = voice_samples.get(speaker_idx)
                    lora_path = lora_paths.get(speaker_idx)

                    seg_name = f"conv_seg_{idx:04d}.wav"
                    seg_path = TEMP_DIR / seg_name

                    tts_manager.generate_dramabox_to_file(
                        prompt=text_clean,
                        output_path=str(seg_path),
                        voice_sample=str(wav_path) if wav_path else None,
                        seed=actual_seed + idx,
                        lora_path=str(lora_path) if lora_path else None,
                        cpu_offload=cpu_offload,
                        dramabox_params=db_params,
                    )
                    segment_paths.append(seg_path)

                if not segment_paths:
                    return None, "❌ No segments generated. Check your script format.", "", gr.update()

                progress(0.9, desc="Concatenating segments...")

                metadata_lines = [
                    f"Generated: {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    f"Engine: DramaBox",
                    f"Seed: {actual_seed}",
                    f"Segments: {len(segment_paths)}",
                    "Speakers: " + ", ".join(
                        "{}:{}".format(i, v or "(None)")
                        for i, v in enumerate([voice_1, voice_2, voice_3, voice_4], 1)
                    ),
                    f"Script: {' '.join(script.split())[:200]}",
                ]
                metadata_out = "\n".join(metadata_lines)

                pause_secs = float(pause_linebreak) if pause_linebreak else 0.3
                final_path = finalize_conversation_output(
                    segment_paths, pause_secs, TEMP_DIR, OUTPUT_DIR,
                    _user_config, save_result_to_output, metadata_out
                )

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()

                manual_save = _user_config.get("manual_save", False)
                status = f"Generated {len(segment_paths)} segments with DramaBox. Seed: {actual_seed}"
                if manual_save:
                    status += "\nClick 'Save to Output' to keep this result."
                    return final_path, status, metadata_out, gr.update(interactive=True)

                return final_path, status, "", gr.update()

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Error generating conversation: {str(e)}", "", gr.update()

        # Tab select — refresh samples and LoRAs
        components['conversation_tab'].select(
            refresh_voice_choices,
            inputs=[],
            outputs=[
                components['conv_voice_1'],
                components['conv_voice_2'],
                components['conv_voice_3'],
                components['conv_voice_4'],
            ]
        )

        components['conversation_tab'].select(
            refresh_all_lora_dropdowns,
            inputs=[
                components['conv_lora_dropdown_1'],
                components['conv_lora_dropdown_2'],
                components['conv_lora_dropdown_3'],
                components['conv_lora_dropdown_4'],
            ],
            outputs=[
                components['conv_lora_dropdown_1'],
                components['conv_lora_dropdown_2'],
                components['conv_lora_dropdown_3'],
                components['conv_lora_dropdown_4'],
                components['conv_lora_path_map'],
            ]
        )

        components['conversation_tab'].select(
            restore_fn,
            inputs=[],
            outputs=restore_outputs
        )

        components['generate_btn'].click(
            generate_dramabox_conversation_handler,
            inputs=[
                components['conv_script'],
                components['conv_seed'],
                components['conv_voice_1'],
                components['conv_voice_2'],
                components['conv_voice_3'],
                components['conv_voice_4'],
                components['conv_lora_dropdown_1'],
                components['conv_lora_dropdown_2'],
                components['conv_lora_dropdown_3'],
                components['conv_lora_dropdown_4'],
                components['conv_lora_path_map'],
                components['db_conv_pause_linebreak'],
                components['db_conv_negative_prompt'],
                components['db_conv_ref_duration'],
                components['db_conv_gen_duration'],
                components['db_conv_steps'],
                components['db_conv_sampler'],
                components['db_conv_speed'],
                components['db_conv_duration_multiplier'],
                components['db_conv_cfg_scale'],
                components['db_conv_stg_scale'],
                components['db_conv_rescale_scale'],
                components['db_conv_id_guidance_scale'],
                components['db_conv_no_watermark'],
            ],
            outputs=[
                components['output_audio'],
                components['conv_status'],
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
                outputs=[components['output_audio'], components['conv_status'], components['save_result_btn']]
            )

        app = shared_state.get('app')
        if app:
            app.load(
                refresh_all_lora_dropdowns,
                inputs=[
                    components['conv_lora_dropdown_1'],
                    components['conv_lora_dropdown_2'],
                    components['conv_lora_dropdown_3'],
                    components['conv_lora_dropdown_4'],
                ],
                outputs=[
                    components['conv_lora_dropdown_1'],
                    components['conv_lora_dropdown_2'],
                    components['conv_lora_dropdown_3'],
                    components['conv_lora_dropdown_4'],
                    components['conv_lora_path_map'],
                ]
            )

        prompt_apply_trigger = shared_state.get('prompt_apply_trigger')
        if prompt_apply_trigger is not None:
            import modules.core_components.prompt_hub as _prompt_hub

            def _apply_conv_script(raw_value, current):
                parsed = _prompt_hub.parse_apply_payload(raw_value)
                if not parsed or parsed['target_id'] != 'conversation.script':
                    return gr.update()
                return gr.update(value=_prompt_hub.merge_text(current, parsed['text'], parsed['mode']))

            prompt_apply_trigger.change(
                _apply_conv_script,
                inputs=[prompt_apply_trigger, components['conv_script']],
                outputs=[components['conv_script']],
            )


get_tool_class = lambda: ConversationTool
