"""
Voice Design Tab

Create new voices from natural language descriptions using Qwen3-TTS VoiceDesign.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
import gradio as gr
import soundfile as sf
import shutil
import json
import torch
import random
import tempfile
from datetime import datetime
from pathlib import Path
from modules.core_components.ai_models.model_utils import set_seed

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager


class VoiceDesignTool(Tool):
    """Voice Design tool implementation."""

    config = ToolConfig(
        name="Voice Design",
        module_name="tool_voice_design",
        description="Create voices from natural language descriptions",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Voice Design tool UI."""
        components = {}

        # Get shared utilities
        LANGUAGES = shared_state.get('LANGUAGES', ['Auto'])
        _user_config = shared_state.get('_user_config', {})
        create_qwen_advanced_params = shared_state.get('create_qwen_advanced_params')

        with gr.TabItem("Voice Design", id="tab_voice_design") as voice_design_tab:
            components['voice_design_tab'] = voice_design_tab
            gr.Markdown("Create new voices from natural language descriptions")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Create Design")

                    components['design_instruct_input'] = gr.Textbox(
                        label="Voice Design Instructions",
                        placeholder="Describe the voice: e.g., 'Young female voice, bright and cheerful, slightly breathy' or 'Deep male voice with a warm, comforting tone, speak slowly'",
                        lines=3,
                        max_lines=10
                    )

                    import modules.core_components.prompt_hub as _prompt_hub
                    components.update(_prompt_hub.create_prompt_loader("vd", "Saved Prompts"))

                    with gr.Row():
                        components['design_language'] = gr.Dropdown(
                            choices=LANGUAGES,
                            value=_user_config.get("language", "Auto"),
                            label="Language",
                            scale=2
                        )
                        components['design_seed'] = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                            scale=1
                        )

                    # Qwen Advanced Parameters
                    design_params = create_qwen_advanced_params(visible=True)
                    components.update(design_params)

                    components['design_generate_btn'] = gr.Button("Generate Voice", variant="primary", size="lg")

                    components['design_text_input'] = gr.Textbox(
                        label="Reference Text",
                        placeholder="Enter the text for the voice design (this will be spoken in the designed voice)...",
                        lines=3,
                        max_lines=6,
                        value="Thank you for listening to this voice design sample. This sentence is intentionally a bit long so you can hear the full range and quality of the generated voice."
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Preview & Save")
                    components['design_output_audio'] = gr.Audio(
                        label="Generated Audio",
                        type="filepath"
                    )

                    components['design_save_btn'] = gr.Button("Save Sample", variant="primary", interactive=False)
                    components['design_status'] = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Design events."""

        # Get shared utilities and directories
        show_input_modal_js = shared_state.get('show_input_modal_js')
        save_preference = shared_state.get('save_preference')
        input_trigger = shared_state.get('input_trigger')
        SAMPLES_DIR = shared_state.get('SAMPLES_DIR')
        OUTPUT_DIR = shared_state.get('OUTPUT_DIR')
        TEMP_DIR = shared_state.get('TEMP_DIR')
        play_completion_beep = shared_state.get('play_completion_beep')

        # Get TTS manager (singleton)
        tts_manager = get_tts_manager()

        # Wire param persistence (auto-save on change)
        wire_param_persistence = shared_state['wire_param_persistence']
        _user_config = shared_state.get('_user_config', {})
        param_map = {
            'qwen_design': [
                ('do_sample', 'do_sample'),
                ('temperature', 'temperature'),
                ('top_k', 'top_k'),
                ('top_p', 'top_p'),
                ('repetition_penalty', 'repetition_penalty'),
                ('max_new_tokens', 'max_new_tokens'),
            ],
        }
        wire_param_persistence(components, _user_config, param_map)

        # Create restore handler for applying saved params on tab select
        create_param_restore_handler = shared_state['create_param_restore_handler']
        restore_fn, restore_outputs = create_param_restore_handler(components, _user_config, param_map)

        # Restore saved params when accordion is opened
        components['accordion'].expand(restore_fn, outputs=restore_outputs)

        def generate_voice_design_handler(text_to_generate, language, instruct, seed, do_sample, temperature,
                                          top_k, top_p, repetition_penalty, max_new_tokens, progress=gr.Progress()):
            """Generate audio using voice design with natural language instructions."""
            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate.", gr.update()

            if not instruct or not instruct.strip():
                return None, "❌ Please enter voice design instructions.", gr.update()

            try:
                # Set the seed for reproducibility
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)

                set_seed(seed)
                seed_msg = f"Seed: {seed}"

                progress(0.1, desc="Loading VoiceDesign model...")

                # Call manager to generate
                audio_data, sr = tts_manager.generate_voice_design(
                    text=text_to_generate.strip(),
                    language=language if language != "Auto" else "Auto",
                    instruct=instruct.strip(),
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens
                )

                progress(0.8, desc="Saving audio temp...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Ensure audio is numpy array (move to CPU if tensor)
                if hasattr(audio_data, "cpu"):
                    audio_data = audio_data.cpu().numpy()
                elif hasattr(audio_data, "numpy"):
                    audio_data = audio_data.numpy()

                out_file = TEMP_DIR / f"temp_voice_design_{timestamp}.wav"

                # Ensure directory exists
                TEMP_DIR.mkdir(parents=True, exist_ok=True)
                sf.write(str(out_file), audio_data, sr)

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                return str(out_file), f"Voice design generated. Save to samples to keep.\n{seed_msg}", gr.update(interactive=True)

            except Exception as e:
                return None, f"❌ Error generating audio: {str(e)}", gr.update(interactive=False)

        def save_designed_voice(audio, design_name, ref_text, instruct, seed):
            """Save a designed voice as a sample (tool-specific implementation)."""
            if not audio:
                return "❌ No audio to save"

            if not design_name or not design_name.strip():
                return "❌ Please enter a name"

            try:
                # Clean the name
                safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in design_name.strip())

                # Copy audio file to samples folder
                wav_path = SAMPLES_DIR / f"{safe_name}.wav"
                json_path = SAMPLES_DIR / f"{safe_name}.json"

                # If audio is a path, copy it
                if isinstance(audio, str):
                    shutil.copy(audio, wav_path)
                else:
                    # If audio is data, save it
                    sf.write(str(wav_path), audio, 24000)

                # Create metadata
                metadata = {
                    "Name": safe_name,
                    "Text": ref_text if ref_text else "Voice design sample",
                    "Type": "Voice Design",
                    "Instruct": instruct if instruct else "",
                    "Seed": seed if seed is not None else -1,
                    "Created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

                return f"Saved as sample: {safe_name}"

            except Exception as e:
                return f"❌ Error saving: {str(e)}"

        def _disable_gen_btn():
            return gr.update(interactive=False)

        def _enable_gen_btn():
            return gr.update(interactive=True)

        # Wire generate button (restore saved params first, then generate)
        components['design_generate_btn'].click(
            _disable_gen_btn, outputs=[components['design_generate_btn']]
        ).then(
            restore_fn, outputs=restore_outputs
        ).then(
            generate_voice_design_handler,
            inputs=[components['design_text_input'], components['design_language'], components['design_instruct_input'],
                    components['design_seed'], components['do_sample'], components['temperature'], components['top_k'],
                    components['top_p'], components['repetition_penalty'], components['max_new_tokens']],
            outputs=[components['design_output_audio'], components['design_status'], components['design_save_btn']]
        ).then(
            _enable_gen_btn, outputs=[components['design_generate_btn']]
        )

        # Save designed voice - show modal (pre-fill label from JSON instruct if available)
        save_design_modal_js = show_input_modal_js(
            title="Save Designed Voice",
            message="Enter a name for this voice design:",
            placeholder="e.g., Bright-Female, Deep-Male, Cheerful-Voice",
            context="save_design_"
        )

        components['design_save_btn'].click(
            fn=None,
            inputs=[components['design_instruct_input']],
            outputs=None,
            js=f"""
            (instruct) => {{
                let defaultName = '';
                if (instruct && instruct.trim()) {{
                    try {{
                        const parsed = JSON.parse(instruct.trim());
                        if (parsed && parsed.label) {{
                            defaultName = parsed.label;
                        }}
                    }} catch(e) {{}}
                }}
                const openModal = {save_design_modal_js};
                openModal(defaultName);
            }}
            """
        )

        # Handle save designed voice input modal submission
        if input_trigger:
            def handle_save_design_input(input_value, audio, ref_text, instruct, seed):
                """Process input modal submission for saving designed voice."""
                # Context filtering: only process if this is our context
                if not input_value or not input_value.startswith("save_design_"):
                    return gr.update(), gr.update()

                # Extract design name from context prefix
                # Format: "save_design_<name>_<timestamp>"
                parts = input_value.split("_")
                if len(parts) >= 3:
                    # Remove context prefix and timestamp (last part)
                    design_name = "_".join(parts[2:-1])
                    status = save_designed_voice(audio, design_name, ref_text, instruct, seed)
                    return status, gr.update(interactive=False)

                return gr.update(), gr.update()

            input_trigger.change(
                handle_save_design_input,
                inputs=[input_trigger, components['design_output_audio'], components['design_text_input'],
                        components['design_instruct_input'], components['design_seed']],
                outputs=[components['design_status'], components['design_save_btn']]
            )

        # Save language preference
        if save_preference:
            components['design_language'].change(
                lambda x: save_preference("language", x),
                inputs=[components['design_language']],
                outputs=[]
            )

        # --- Cross-tab prompt routing ---
        import modules.core_components.prompt_hub as _prompt_hub
        _prompt_hub.wire_prompt_loader(components, "vd", {"voice_design.instructions": components['design_instruct_input']})

        prompt_apply_trigger = shared_state.get('prompt_apply_trigger')
        if prompt_apply_trigger is not None:
            import modules.core_components.prompt_hub as _prompt_hub

            def _apply_vd_reference(raw_value, current):
                parsed = _prompt_hub.parse_apply_payload(raw_value)
                if not parsed or parsed['target_id'] != 'voice_design.reference':
                    return gr.update()
                return gr.update(value=_prompt_hub.merge_text(current, parsed['text'], parsed['mode']))

            prompt_apply_trigger.change(
                _apply_vd_reference,
                inputs=[prompt_apply_trigger, components['design_text_input']],
                outputs=[components['design_text_input']],
            )

            def _apply_vd_instructions(raw_value, current):
                parsed = _prompt_hub.parse_apply_payload(raw_value)
                if not parsed or parsed['target_id'] != 'voice_design.instructions':
                    return gr.update()
                return gr.update(value=_prompt_hub.merge_text(current, parsed['text'], parsed['mode']))

            prompt_apply_trigger.change(
                _apply_vd_instructions,
                inputs=[prompt_apply_trigger, components['design_instruct_input']],
                outputs=[components['design_instruct_input']],
            )


# Export for registry
get_tool_class = lambda: VoiceDesignTool


if __name__ == "__main__":
    """Standalone testing of Voice Design tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(VoiceDesignTool, port=7861, title="Voice Design - Standalone")
