"""
Settings Tab

Configure global application settings.

Standalone testing:
    python -m modules.core_components.tools.settings
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
import torch as _torch
from pathlib import Path
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.help_page import (
    show_voice_clone_help, show_prep_audio_help, show_dataset_help,
    show_train_help, show_tips_help
)

# Tools that can be toggled (everything except Settings)
# Format: (config_key, display_label)
TOGGLEABLE_TOOLS = [
    ("Voice Clone", "Voice Clone"),
    # ("Conversation", "Conversation"),
    ("Voice Design", "Voice Design"),
    ("Prep Samples", "Prep Samples"),
    ("Train Model", "Train Model"),
    ("Sound Effects", "Sound Effects"),
    ("Prompt Manager", "Prompt Manager"),
    ("Output History", "Output History"),
]


class SettingsTool(Tool):
    """Settings tool implementation."""

    config = ToolConfig(
        name="Settings",
        module_name="tool_settings",
        description="Application settings and preferences",
        enabled=True,
        category="utility"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Settings tool UI."""
        components = {}

        # Extract needed items from shared_state
        _user_config = shared_state.get('_user_config', {})
        format_help_html = shared_state.get('format_help_html')

        with gr.TabItem("⚙️"):
            with gr.Tabs():
                with gr.TabItem("Settings"):
                    gr.Markdown("# ⚙️ Settings")
                    gr.Markdown("Configure global application settings")

                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Displayed Tools")

                                # Get current enabled_tools from config
                                tool_settings = _user_config.get("enabled_tools", {})

                                with gr.Column():
                                    for key, label in TOGGLEABLE_TOOLS:
                                        is_enabled = tool_settings.get(key, True)
                                        components[f'tool_toggle_{key}'] = gr.Checkbox(
                                            label=label,
                                            value=is_enabled,
                                            interactive=True
                                        )
                                gr.Markdown("### Misc")
                                components['settings_dark_mode_only'] = gr.Checkbox(
                                    label="Dark Mode Only",
                                    value=_user_config.get("dark_mode_only", True),
                                    info="Force dark mode regardless of browser/OS setting."
                                )
                                components['settings_audio_notifications'] = gr.Checkbox(
                                    label="Enable Audio Notifications",
                                    value=_user_config.get("browser_notifications", True),
                                    info="Play sound when audio generation completes"
                                )
                                components['settings_listen_on_network'] = gr.Checkbox(
                                    label="Listen on Network",
                                    value=_user_config.get("listen_on_network", False),
                                    info="Allow other devices on your local network to connect. (Restart required)"
                                )

                            with gr.Column():
                                gr.Markdown("### LLM Backend (Prompt Manager)")

                                _current_llm_backend = _user_config.get("llm_backend", "llama.cpp")
                                components['settings_llm_backend'] = gr.Radio(
                                    label="LLM Backend",
                                    choices=["llama.cpp", "Ollama"],
                                    value=_current_llm_backend,
                                    info="llama.cpp runs local GGUF models. Ollama uses a running Ollama instance."
                                )

                                with gr.Group(visible=_current_llm_backend == "llama.cpp") as llama_cpp_settings_group:
                                    components['llama_cpp_settings_group'] = llama_cpp_settings_group
                                    components['settings_llama_cpp_path'] = gr.Textbox(
                                        label="llama.cpp Location",
                                        value=_user_config.get("llama_cpp_path", ""),
                                        info="Path to the folder containing llama-server. Leave empty to use system PATH.",
                                        placeholder="e.g. C:\\llama.cpp\\build\\bin"
                                    )
                                    components['reset_llama_cpp_path_btn'] = gr.Button("Reset", size="sm")

                                    components['settings_llama_models_path'] = gr.Textbox(
                                        label="Additional LLM Models Folder",
                                        value=_user_config.get("llama_models_path", ""),
                                        info="Extra folder to scan for .gguf models (in addition to models/llama/). Downloads go here if set.",
                                        placeholder="e.g. D:\\models\\gguf"
                                    )
                                    components['reset_llama_models_path_btn'] = gr.Button("Reset", size="sm")

                                with gr.Group(visible=_current_llm_backend == "Ollama") as ollama_settings_group:
                                    components['ollama_settings_group'] = ollama_settings_group
                                    components['settings_ollama_url'] = gr.Textbox(
                                        label="Ollama Base URL",
                                        value=_user_config.get("llm_ollama_url", "http://127.0.0.1:11434"),
                                        info="URL of your running Ollama instance. Models must be pre-pulled with 'ollama pull'.",
                                        placeholder="http://127.0.0.1:11434"
                                    )
                                    components['reset_ollama_url_btn'] = gr.Button("Reset", size="sm")

                                gr.Markdown("### Output")
                                components['settings_output_format'] = gr.Dropdown(
                                    label="Output Format",
                                    choices=["wav", "flac", "mp3"],
                                    value=_user_config.get("output_format", "wav"),
                                    info="Format for saved audio files. MP3 uses 320kbps."
                                )
                                components['settings_manual_save'] = gr.Checkbox(
                                    label="Review Before Saving",
                                    value=_user_config.get("manual_save", False),
                                    info="Results stay in temp until you click Save.\nLets you keep only the ones you like. (Restart required)"
                                )

                            with gr.Column():

                                gr.Markdown("### Model Optimizations")
                                components['settings_low_cpu_mem'] = gr.Checkbox(
                                    label="Low CPU Memory Usage (Slower loading time)",
                                    value=_user_config.get("low_cpu_mem_usage", False),
                                    info="Reduces CPU RAM usage when loading models."
                                )

                                # components['settings_attention_mechanism'] = gr.Dropdown(
                                #     label="Choose Attention Mechanism",
                                #     choices=["auto", "flash_attention_2", "sdpa", "eager"],
                                #     value=_user_config.get("attention_mechanism", "auto"),
                                #     info="Auto = fastest available."
                                # )

                                components['settings_dramabox_cpu_offload'] = gr.Checkbox(
                                    label="DramaBox CPU Offloading",
                                    value=_user_config.get("dramabox_cpu_offload", False),
                                    info="Saves VRAM but slower — disable for fast warm-server mode.",
                                    interactive=True
                                )

                                components['settings_skip_engine_check'] = gr.Checkbox(
                                    label="Skip Engine Check at Startup",
                                    value=_user_config.get("skip_engine_check", False),
                                    info="Assumes all engines are available. Faster launch. (Restart required)"
                                )

                                gr.Markdown("### Model Downloading")
                                components['settings_offline_mode'] = gr.Checkbox(
                                    label="Offline Mode",
                                    value=_user_config.get("offline_mode", False),
                                    info="When enabled, only uses models found in models folder"
                                )

                                components['model_select'] = gr.Dropdown(
                                    label="Select Model to Download to Models Folder",
                                    info="Needed to work in offline mode.\nFor Whisper models, copy local files to models folder",
                                    choices=[
                                        "--- DramaBox ---",
                                        "DramaBox DiT v1",
                                        "DramaBox Audio Components",
                                        "DramaBox Gemma Text Encoder",
                                        "--- ASR ---",
                                        "Qwen3-ASR-Small",
                                        "Qwen3-ASR-Large",
                                    ],
                                    value="DramaBox DiT v1"
                                )
                                components['download_btn'] = gr.Button("Download Model", scale=1)

                                # Mapping from display names to (repo_id, filename) tuple,
                                # repo_id string (full snapshot), or (repo_id, None) for snapshot
                                components['MODEL_ID_MAP'] = {
                                    "DramaBox DiT v1": ("ResembleAI/Dramabox", "dramabox-dit-v1.safetensors"),
                                    "DramaBox Audio Components": ("ResembleAI/Dramabox", "dramabox-audio-components.safetensors"),
                                    "DramaBox Gemma Text Encoder": ("unsloth/gemma-3-12b-it-bnb-4bit", None),
                                    "Qwen3-ASR-Small": "Qwen/Qwen3-ASR-0.6B",
                                    "Qwen3-ASR-Large": "Qwen/Qwen3-ASR-1.7B",
                                }

                        with gr.Row():
                            with gr.Column():
                                if _torch.cuda.is_available():
                                    gr.Markdown("")

                                # GPU Assignment (only shown when multiple CUDA GPUs available)
                                from modules.core_components.ai_models.model_utils import get_available_gpus
                                available_gpus = get_available_gpus()
                                has_multi_gpu = len(available_gpus) > 1
                                if has_multi_gpu:
                                    gpu_choices = [f"GPU {i}: {name}" for i, name in available_gpus]

                                    gr.Markdown("### GPU Assignment")
                                    with gr.Row():
                                        components['settings_tts_gpu'] = gr.Dropdown(
                                            label="TTS GPU",
                                            choices=gpu_choices,
                                            value=gpu_choices[int(_user_config.get("tts_gpu", 0))],
                                            info="GPU used for text-to-speech models"
                                        )
                                        components['settings_asr_gpu'] = gr.Dropdown(
                                            label="ASR GPU",
                                            choices=gpu_choices,
                                            value=gpu_choices[int(_user_config.get("asr_gpu", 0))],
                                            info="GPU used for speech recognition models"
                                        )
                                        components['settings_llama_gpu'] = gr.Dropdown(
                                            label="Llama.cpp GPU",
                                            choices=gpu_choices,
                                            value=gpu_choices[int(_user_config.get("llama_gpu", 0))],
                                            info="GPU used for LLM prompt generation"
                                        )


                        gr.Markdown("Configure where files are stored. Changes apply after clicking **Apply Changes**.")
                        # Default folder paths
                        default_folders = {
                            "samples": "samples",
                            "output": "output",
                            "datasets": "datasets",
                            "temp": "temp",
                            "models": "models",
                            "loras": "loras"
                        }
                        components['default_folders'] = default_folders

                        # Row 1: Samples, Datasets and Output folders
                        with gr.Row():
                            with gr.Column():
                                components['settings_samples_folder'] = gr.Textbox(
                                    label="Voice Samples Folder",
                                    value=_user_config.get("samples_folder", default_folders["samples"]),
                                    info="Folder for voice sample files (.wav + .json)"
                                )
                                components['reset_samples_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                components['settings_output_folder'] = gr.Textbox(
                                    label="Output Folder",
                                    value=_user_config.get("output_folder", default_folders["output"]),
                                    info="Folder for generated audio files"
                                )
                                components['reset_output_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                components['settings_datasets_folder'] = gr.Textbox(
                                    label="Datasets Folder",
                                    value=_user_config.get("datasets_folder", default_folders["datasets"]),
                                    info="Folder for training/finetuning datasets"
                                )
                                components['reset_datasets_btn'] = gr.Button("Reset", size="sm")

                        with gr.Row():
                            with gr.Column():
                                components['settings_models_folder'] = gr.Textbox(
                                    label="Downloaded Models Folder",
                                    value=_user_config.get("models_folder", default_folders["models"]),
                                    info="Folder for downloaded model files"
                                )
                                components['reset_models_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                components['settings_trained_models_folder'] = gr.Textbox(
                                    label="Trained Loras Folder",
                                    value=_user_config.get("trained_models_folder", default_folders["loras"]),
                                    info="Folder for your custom trained Loras"
                                )
                                components['reset_trained_models_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                gr.Markdown("")

                    with gr.Column():
                        components['apply_folders_btn'] = gr.Button("Apply Changes", variant="primary", size="lg")
                        components['settings_status'] = gr.Textbox(
                            label="Status",
                            interactive=False,
                            max_lines=10
                        )

                with gr.TabItem("Help Guide"):
                    gr.Markdown("# Voice Clone Studio - Help & Guide")

                    components['help_topic'] = gr.Radio(
                        choices=[
                            "Voice Clone",
                            "Prep Samples",
                            "Prep Dataset",
                            "Train Model",
                            "Tips & Tricks"
                        ],
                        value="Voice Clone",
                        show_label=False,
                        interactive=True,
                        container=False
                    )

                    components['help_content'] = gr.HTML(
                        value=format_help_html(show_voice_clone_help()),
                        container=True,
                        padding=True
                    )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Settings tab events."""

        # Extract needed items from shared_state
        _user_config = shared_state.get('_user_config', {})
        save_preference = shared_state.get('save_preference')
        download_model_from_huggingface = shared_state.get('download_model_from_huggingface')
        format_help_html = shared_state.get('format_help_html')

        # Lazy import to avoid circular dependency
        from modules.core_components.tools import save_config

        components['settings_dramabox_cpu_offload'].change(
            lambda x: save_preference("dramabox_cpu_offload", x),
            inputs=[components['settings_dramabox_cpu_offload']],
            outputs=[]
        )

        components['settings_low_cpu_mem'].change(
            lambda x: save_preference("low_cpu_mem_usage", x),
            inputs=[components['settings_low_cpu_mem']],
            outputs=[]
        )

        if 'settings_attention_mechanism' in components:
            components['settings_attention_mechanism'].change(
                lambda x: save_preference("attention_mechanism", x),
                inputs=[components['settings_attention_mechanism']],
                outputs=[]
            )

        # Save offline mode setting
        components['settings_offline_mode'].change(
            lambda x: save_preference("offline_mode", x),
            inputs=[components['settings_offline_mode']],
            outputs=[]
        )

        # Save skip engine check setting
        components['settings_skip_engine_check'].change(
            lambda x: save_preference("skip_engine_check", x),
            inputs=[components['settings_skip_engine_check']],
            outputs=[]
        )

        # Save CUDA graphs acceleration setting
        if 'settings_cuda_graphs' in components:
            components['settings_cuda_graphs'].change(
                lambda x: save_preference("cuda_graphs", x),
                inputs=[components['settings_cuda_graphs']],
                outputs=[]
            )

        # Save GPU assignment settings (only if multi-GPU dropdowns exist)
        if 'settings_tts_gpu' in components:
            def extract_gpu_index(choice):
                """Extract GPU index from dropdown choice string like 'GPU 0: NVIDIA ...'"""
                try:
                    return int(choice.split(":")[0].replace("GPU ", ""))
                except (ValueError, AttributeError, IndexError):
                    return 0

            components['settings_tts_gpu'].change(
                lambda x: save_preference("tts_gpu", extract_gpu_index(x)),
                inputs=[components['settings_tts_gpu']],
                outputs=[]
            )
            components['settings_asr_gpu'].change(
                lambda x: save_preference("asr_gpu", extract_gpu_index(x)),
                inputs=[components['settings_asr_gpu']],
                outputs=[]
            )
            components['settings_llama_gpu'].change(
                lambda x: save_preference("llama_gpu", extract_gpu_index(x)),
                inputs=[components['settings_llama_gpu']],
                outputs=[]
            )

        # Save theme setting
        if 'settings_theme' in components:
            components['settings_theme'].change(
                lambda x: (save_preference("ui_theme", x), "Restart the app to apply the new theme.")[1],
                inputs=[components['settings_theme']],
                outputs=[components['settings_status']]
            )

        # Save dark mode only setting (applies live via JS)
        components['settings_dark_mode_only'].change(
            lambda x: (save_preference("dark_mode_only", x), "Restart the app to apply changes.")[1],
            inputs=[components['settings_dark_mode_only']],
            outputs=[components['settings_status']],
            js="(checked) => { if (checked) { document.body.classList.add('dark'); } return checked; }"
        )

        # Save audio notifications setting
        components['settings_audio_notifications'].change(
            lambda x: save_preference("browser_notifications", x),
            inputs=[components['settings_audio_notifications']],
            outputs=[]
        )

        # Save output format setting
        components['settings_output_format'].change(
            lambda x: save_preference("output_format", x),
            inputs=[components['settings_output_format']],
            outputs=[]
        )

        # Save manual save setting
        components['settings_manual_save'].change(
            lambda x: save_preference("manual_save", x),
            inputs=[components['settings_manual_save']],
            outputs=[]
        )

        # Save listen on network setting
        components['settings_listen_on_network'].change(
            lambda x: save_preference("listen_on_network", x),
            inputs=[components['settings_listen_on_network']],
            outputs=[]
        )

        # Tool toggle handlers
        def toggle_tool(tool_name, enabled):
            """Save tool visibility to config."""
            if "enabled_tools" not in _user_config:
                _user_config["enabled_tools"] = {}
            _user_config["enabled_tools"][tool_name] = enabled
            # Use save_preference to persist (saves full config)
            save_preference("enabled_tools", _user_config["enabled_tools"])
            return "Restart the app to apply changes."

        for key, label in TOGGLEABLE_TOOLS:
            comp = components[f'tool_toggle_{key}']
            comp.change(
                lambda enabled, k=key: toggle_tool(k, enabled),
                inputs=[comp],
                outputs=[components['settings_status']]
            )

        # Reset button handlers
        def reset_folder(folder_key):
            return components['default_folders'][folder_key]

        components['reset_samples_btn'].click(
            lambda: reset_folder("samples"),
            outputs=[components['settings_samples_folder']]
        )

        components['reset_output_btn'].click(
            lambda: reset_folder("output"),
            outputs=[components['settings_output_folder']]
        )

        components['reset_datasets_btn'].click(
            lambda: reset_folder("datasets"),
            outputs=[components['settings_datasets_folder']]
        )

        components['reset_models_btn'].click(
            lambda: reset_folder("models"),
            outputs=[components['settings_models_folder']]
        )

        components['reset_trained_models_btn'].click(
            lambda: reset_folder("loras"),
            outputs=[components['settings_trained_models_folder']]
        )

        components['reset_llama_cpp_path_btn'].click(
            lambda: "",
            outputs=[components['settings_llama_cpp_path']]
        )

        components['reset_llama_models_path_btn'].click(
            lambda: "",
            outputs=[components['settings_llama_models_path']]
        )

        components['reset_ollama_url_btn'].click(
            lambda: "http://127.0.0.1:11434",
            outputs=[components['settings_ollama_url']]
        )

        def on_llm_backend_change(backend):
            show_llama = backend == "llama.cpp"
            _user_config["llm_backend"] = backend
            save_config(_user_config)
            return gr.update(visible=show_llama), gr.update(visible=not show_llama)

        components['settings_llm_backend'].change(
            on_llm_backend_change,
            inputs=[components['settings_llm_backend']],
            outputs=[components['llama_cpp_settings_group'], components['ollama_settings_group']]
        )

        def on_ollama_url_change(url):
            _user_config["llm_ollama_url"] = url.strip()
            save_config(_user_config)

        components['settings_ollama_url'].change(
            on_ollama_url_change,
            inputs=[components['settings_ollama_url']],
            outputs=[]
        )

        def download_model_clicked(model_display_name):
            if not model_display_name or model_display_name.startswith("---"):
                return "❌ Please select an actual model (not a category header)"
            entry = components['MODEL_ID_MAP'].get(model_display_name, model_display_name)

            # DramaBox entries — download directly to the models folder (no HF cache)
            if isinstance(entry, tuple):
                repo_id, filename = entry
                try:
                    import os
                    import shutil
                    from huggingface_hub import hf_hub_download, snapshot_download

                    base_dir = Path(__file__).parent.parent.parent.parent
                    models_dir = base_dir / _user_config.get("models_folder", "models")
                    dramabox_dir = models_dir / "dramabox"
                    dramabox_dir.mkdir(parents=True, exist_ok=True)

                    token = os.environ.get("HF_TOKEN")

                    if filename is not None:
                        dest = dramabox_dir / Path(filename).name
                        if dest.exists():
                            return f"Already downloaded: {dest.name}"
                        print(f"Downloading {filename} to {dramabox_dir}...")
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            local_dir=str(dramabox_dir),
                            token=token,
                        )
                        print(f"Done: {dest}")
                        return f"Downloaded: {dest.name}"
                    else:
                        # Full repo (Gemma) — snapshot to a named subfolder
                        repo_name = repo_id.split("/")[-1]
                        dest_dir = models_dir / repo_name
                        if dest_dir.exists() and any(dest_dir.iterdir()):
                            return f"Already downloaded: {repo_name}"
                        print(f"Downloading {repo_id} to {dest_dir}...")
                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=str(dest_dir),
                            token=token,
                        )
                        print(f"Done: {dest_dir}")
                        return f"Downloaded: {repo_name}"
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"❌ Download failed: {e}"

            success, message, path = download_model_from_huggingface(entry, progress=None)
            return f"Downloaded: {message}" if success else f"❌ {message}"

        # Apply folder changes
        def apply_folder_changes(samples, output, datasets, models, trained_models, llama_cpp_path, llama_models_path, llm_backend, ollama_url):
            try:
                # Get project root directory
                base_dir = Path(__file__).parent.parent.parent.parent

                # Update paths
                new_samples = base_dir / samples
                new_output = base_dir / output
                new_datasets = base_dir / datasets
                new_models = base_dir / models
                new_trained_models = base_dir / trained_models

                # Create directories if they don't exist
                new_samples.mkdir(exist_ok=True)
                new_output.mkdir(exist_ok=True)
                new_datasets.mkdir(exist_ok=True)
                new_models.mkdir(exist_ok=True)
                new_trained_models.mkdir(exist_ok=True)

                # Set HuggingFace cache environment variable
                import os
                os.environ['HF_HOME'] = str(new_models)

                # Save to config
                _user_config["samples_folder"] = samples
                _user_config["output_folder"] = output
                _user_config["datasets_folder"] = datasets
                _user_config["models_folder"] = models
                _user_config["trained_models_folder"] = trained_models
                _user_config["llama_cpp_path"] = llama_cpp_path.strip()
                _user_config["llama_models_path"] = llama_models_path.strip()
                _user_config["llm_backend"] = llm_backend
                _user_config["llm_ollama_url"] = ollama_url.strip()
                save_config(_user_config)

                status_lines = [
                    "Folder paths updated successfully!",
                    f"\nSamples: {new_samples}",
                    f"Output: {new_output}",
                    f"Datasets: {new_datasets}",
                    f"Downloaded Models: {new_models}",
                    f"Trained Loras: {new_trained_models}",
                ]
                if llama_cpp_path.strip():
                    status_lines.append(f"llama.cpp: {llama_cpp_path.strip()}")
                if llama_models_path.strip():
                    status_lines.append(f"LLM Models: {llama_models_path.strip()}")
                if llm_backend == "Ollama":
                    status_lines.append(f"Ollama URL: {ollama_url.strip()}")
                status_lines.append("\nNote: Restart the app to fully apply changes to all components.")
                return "\n".join(status_lines)

            except Exception as e:
                return f"❌ Error applying changes: {str(e)}"

        components['download_btn'].click(
            fn=lambda: (gr.update(interactive=False, value="Downloading..."), "Downloading model... (check console for progress)"),
            outputs=[components['download_btn'], components['settings_status']]
        ).then(
            fn=download_model_clicked,
            inputs=[components['model_select']],
            outputs=[components['settings_status']]
        ).then(
            fn=lambda: gr.update(interactive=True, value="Download Model"),
            outputs=[components['download_btn']]
        )

        components['apply_folders_btn'].click(
            apply_folder_changes,
            inputs=[
                components['settings_samples_folder'], components['settings_output_folder'],
                components['settings_datasets_folder'], components['settings_models_folder'],
                components['settings_trained_models_folder'],
                components['settings_llama_cpp_path'], components['settings_llama_models_path'],
                components['settings_llm_backend'], components['settings_ollama_url']
            ],
            outputs=[components['settings_status']]
        )

        # Help Guide topic selector
        def show_help(topic):
            help_map = {
                "Voice Clone": show_voice_clone_help,
                "Prep Samples": show_prep_audio_help,
                "Prep Dataset": show_dataset_help,
                "Train Model": show_train_help,
                "Tips & Tricks": show_tips_help
            }
            return format_help_html(help_map[topic]())

        components['help_topic'].change(
            fn=show_help,
            inputs=components['help_topic'],
            outputs=components['help_content']
        )


# Export for tab registry
get_tool_class = lambda: SettingsTool


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(SettingsTool, port=7870, title="Settings - Standalone")
