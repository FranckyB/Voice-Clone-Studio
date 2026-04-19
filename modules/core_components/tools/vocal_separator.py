"""
Vocal Separator Tab

Isolate vocals (and instrumental) from any song using UVR/Demucs/MDX-Net models.
Designed for the "separate → voice-change" workflow: separate vocals, then feed
them into the Voice Changer tab with your target voice.

Standalone testing:
    python -m modules.core_components.tools.vocal_separator
"""
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "modules"))

import gradio as gr
import shutil
from datetime import datetime
from pathlib import Path

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.vocal_separator_manager import (
    is_available as uvr_is_available,
    scan_models,
    separate_vocals,
)

# Models the user is most likely to have — shown as hints in the UI
_RECOMMENDED = [
    "Kim_Vocal_2.onnx — MDX-Net  (best quality/speed, recommended)",
    "htdemucs_ft.yaml — Demucs   (highest quality, slower)",
    "5_HP-Karaoke-UVR.pth — VR Arch (karaoke-style separation)",
]


class VocalSeparatorTool(Tool):
    """Vocal Separator tool — isolates vocals from music."""

    config = ToolConfig(
        name="Vocal Separator",
        module_name="tool_vocal_separator",
        description="Isolate vocals and instrumental from a song (UVR / Demucs / MDX-Net)",
        enabled=True,
        category="utility",
    )

    @classmethod
    def create_tool(cls, shared_state):
        components = {}

        user_config = shared_state.get('_user_config', {})
        DEEPFILTER_AVAILABLE = shared_state['DEEPFILTER_AVAILABLE']
        UVR_AVAILABLE = uvr_is_available()

        default_model_dir = user_config.get('uvr_models_dir', '')

        with gr.TabItem("Vocal Separator") as tab:
            components['vocal_sep_tab'] = tab

            gr.Markdown(
                "Extract **vocals** and **instrumental** from any song. "
                "Use the vocals as *Source Audio* in the **Voice Changer** tab to clone your voice into the track. "
                "<small>(Requires [audio-separator](https://github.com/nomadkaraoke/python-audio-separator))</small>"
            )

            if not UVR_AVAILABLE:
                gr.Markdown(
                    "> ⚠ **audio-separator not installed.**  \n"
                    "> Run `pip install audio-separator[gpu]` then restart the app."
                )

            # ── Top row: model config ──────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=3):
                    components['model_dir'] = gr.Textbox(
                        label="UVR Models Directory",
                        value=default_model_dir,
                        placeholder=r"e.g. C:\UVR\models   or   /home/user/uvr/models",
                        info="Folder that contains your .onnx / .pth / .yaml model files (subfolders are scanned).",
                    )
                with gr.Column(scale=2):
                    components['model_dropdown'] = gr.Dropdown(
                        label="Model",
                        choices=scan_models(default_model_dir),
                        value=None,
                        interactive=True,
                        info="Best for vocals: Kim_Vocal_2.onnx (fast) or htdemucs_ft (quality).",
                    )
                with gr.Column(scale=1, min_width=120):
                    components['refresh_btn'] = gr.Button("Refresh", size="sm")

            # ── Source audio ───────────────────────────────────────────────────
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Source Song")
                    components['source_audio'] = gr.Audio(
                        label="Upload a song (MP3, WAV, FLAC, …)",
                        type="filepath",
                        sources=["upload"],
                    )
                    with gr.Row():
                        components['sep_clear_btn'] = gr.Button("Clear", size="sm", scale=1)

            components['separate_btn'] = gr.Button(
                "Separate Vocals", variant="primary", size="lg", interactive=UVR_AVAILABLE
            )

            # ── Outputs ────────────────────────────────────────────────────────
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Vocals")
                    components['vocals_audio'] = gr.Audio(
                        label="Vocals",
                        type="filepath",
                        interactive=False,
                    )
                    with gr.Row():
                        components['denoise_btn'] = gr.Button(
                            "AI Denoise", size="sm", scale=2,
                            visible=DEEPFILTER_AVAILABLE,
                        )
                        components['normalize_vocals_btn'] = gr.Button(
                            "Normalize", size="sm", scale=2,
                        )
                    components['save_vocals_btn'] = gr.Button(
                        "Save Vocals to Output", variant="primary",
                        interactive=False,
                    )

                with gr.Column():
                    gr.Markdown("#### Instrumental")
                    components['instrumental_audio'] = gr.Audio(
                        label="Instrumental",
                        type="filepath",
                        interactive=False,
                    )
                    with gr.Row():
                        components['normalize_inst_btn'] = gr.Button(
                            "Normalize", size="sm", scale=2,
                        )
                    components['save_inst_btn'] = gr.Button(
                        "Save Instrumental to Output", variant="primary",
                        interactive=False,
                    )

            components['sep_status'] = gr.Textbox(
                label="Status", interactive=False, lines=2, max_lines=4,
            )

            # Hidden state
            components['vocals_temp_path'] = gr.State(value=None)
            components['inst_temp_path'] = gr.State(value=None)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        OUTPUT_DIR = shared_state['OUTPUT_DIR']
        TEMP_DIR = shared_state['TEMP_DIR']
        DEEPFILTER_AVAILABLE = shared_state['DEEPFILTER_AVAILABLE']
        user_config = shared_state.get('_user_config', {})
        save_preference = shared_state.get('save_preference')
        normalize_audio = shared_state['normalize_audio']
        clean_audio = shared_state['clean_audio']
        show_input_modal_js = shared_state['show_input_modal_js']
        input_trigger = shared_state['input_trigger']
        convert_audio_format = shared_state['convert_audio_format']
        embed_metadata = shared_state.get('embed_metadata')
        play_completion_beep = shared_state.get('play_completion_beep')

        # ── Model directory change → save config + rescan ─────────────────────
        def on_model_dir_change(model_dir):
            stripped = (model_dir or '').strip()
            if save_preference:
                save_preference('uvr_models_dir', stripped)
            user_config['uvr_models_dir'] = stripped
            models = scan_models(stripped)
            return gr.update(choices=models, value=models[0] if models else None)

        components['model_dir'].change(
            on_model_dir_change,
            inputs=[components['model_dir']],
            outputs=[components['model_dropdown']],
        )

        def on_refresh(model_dir):
            models = scan_models((model_dir or '').strip())
            return gr.update(choices=models, value=models[0] if models else None)

        components['refresh_btn'].click(
            on_refresh,
            inputs=[components['model_dir']],
            outputs=[components['model_dropdown']],
        )

        # ── Separation ────────────────────────────────────────────────────────
        def do_separate(source_audio, model_dir, model_name, progress=gr.Progress()):
            if source_audio is None:
                return (None, None,
                        gr.update(interactive=False), gr.update(interactive=False),
                        None, None,
                        "Please upload a song first.")
            if not model_name:
                return (None, None,
                        gr.update(interactive=False), gr.update(interactive=False),
                        None, None,
                        "Please select a model.")

            vocals, instrumental, status = separate_vocals(
                audio_path=source_audio,
                model_name=model_name,
                model_dir=(model_dir or '').strip(),
                output_dir=str(TEMP_DIR),
                progress=progress,
            )

            ok = vocals is not None or instrumental is not None
            if ok and play_completion_beep:
                play_completion_beep()

            return (
                vocals,
                instrumental,
                gr.update(interactive=vocals is not None),
                gr.update(interactive=instrumental is not None),
                vocals,
                instrumental,
                status,
            )

        components['separate_btn'].click(
            do_separate,
            inputs=[
                components['source_audio'],
                components['model_dir'],
                components['model_dropdown'],
            ],
            outputs=[
                components['vocals_audio'],
                components['instrumental_audio'],
                components['save_vocals_btn'],
                components['save_inst_btn'],
                components['vocals_temp_path'],
                components['inst_temp_path'],
                components['sep_status'],
            ],
        )

        # ── Clear ─────────────────────────────────────────────────────────────
        components['sep_clear_btn'].click(
            lambda: (None, ""),
            outputs=[components['source_audio'], components['sep_status']],
        )

        # ── Post-processing (vocals) ──────────────────────────────────────────
        components['normalize_vocals_btn'].click(
            normalize_audio,
            inputs=[components['vocals_audio']],
            outputs=[components['vocals_audio'], components['sep_status']],
        )

        components['normalize_inst_btn'].click(
            normalize_audio,
            inputs=[components['instrumental_audio']],
            outputs=[components['instrumental_audio'], components['sep_status']],
        )

        if DEEPFILTER_AVAILABLE:
            components['denoise_btn'].click(
                clean_audio,
                inputs=[components['vocals_audio']],
                outputs=[components['vocals_audio'], components['sep_status']],
            )

        # ── Save vocals ───────────────────────────────────────────────────────
        save_vocals_modal_js = show_input_modal_js(
            title="Save Vocals",
            message="Enter a filename for the isolated vocals:",
            placeholder="e.g. my_song_vocals",
            context="save_sep_vocals_",
        )

        components['save_vocals_btn'].click(
            fn=None,
            inputs=[],
            outputs=None,
            js=f"() => {{ const open = {save_vocals_modal_js}; open('separated_vocals'); }}",
        )

        # ── Save instrumental ─────────────────────────────────────────────────
        save_inst_modal_js = show_input_modal_js(
            title="Save Instrumental",
            message="Enter a filename for the instrumental track:",
            placeholder="e.g. my_song_instrumental",
            context="save_sep_inst_",
        )

        components['save_inst_btn'].click(
            fn=None,
            inputs=[],
            outputs=None,
            js=f"() => {{ const open = {save_inst_modal_js}; open('separated_instrumental'); }}",
        )

        def _copy_to_output(src_path, base_name, user_config):
            """Copy a temp file to OUTPUT_DIR in the configured format."""
            if not src_path or not Path(src_path).exists():
                return gr.update(interactive=False), "❌ Temp file not found. Please separate again."
            fmt = user_config.get("output_format", "wav")
            dest = OUTPUT_DIR / f"{base_name}.{fmt}"
            counter = 1
            while dest.exists():
                dest = OUTPUT_DIR / f"{base_name}_{counter}.{fmt}"
                counter += 1
            convert_audio_format(src_path, dest, fmt)
            if embed_metadata:
                embed_metadata(dest, f"Source: Vocal Separator\nFile: {Path(src_path).name}")
            return gr.update(interactive=False), f"Saved as {dest.name}"

        def handle_modal(input_value, vocals_path, inst_path):
            if not input_value:
                return gr.update(), gr.update(), gr.update()

            if input_value.startswith("save_sep_vocals_"):
                raw = input_value[len("save_sep_vocals_"):]
                parts = raw.rsplit("_", 1)
                if len(parts) >= 2 and parts[0] == "cancel":
                    return gr.update(), gr.update(), gr.update()
                name = (parts[0] if len(parts) > 1 else raw).strip().replace(" ", "_") or "vocals"
                save_btn_update, status = _copy_to_output(vocals_path, name, user_config)
                return save_btn_update, gr.update(), status

            if input_value.startswith("save_sep_inst_"):
                raw = input_value[len("save_sep_inst_"):]
                parts = raw.rsplit("_", 1)
                if len(parts) >= 2 and parts[0] == "cancel":
                    return gr.update(), gr.update(), gr.update()
                name = (parts[0] if len(parts) > 1 else raw).strip().replace(" ", "_") or "instrumental"
                save_btn_update, status = _copy_to_output(inst_path, name, user_config)
                return gr.update(), save_btn_update, status

            return gr.update(), gr.update(), gr.update()

        input_trigger.change(
            handle_modal,
            inputs=[
                input_trigger,
                components['vocals_temp_path'],
                components['inst_temp_path'],
            ],
            outputs=[
                components['save_vocals_btn'],
                components['save_inst_btn'],
                components['sep_status'],
            ],
        )

        # ── Refresh model list when tab is selected ───────────────────────────
        components['vocal_sep_tab'].select(
            on_refresh,
            inputs=[components['model_dir']],
            outputs=[components['model_dropdown']],
        )


get_tool_class = lambda: VocalSeparatorTool


if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(VocalSeparatorTool, port=7867, title="Vocal Separator - Standalone")
