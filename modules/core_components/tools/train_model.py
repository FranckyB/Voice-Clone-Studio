"""
Train Model Tab

Train custom voice models using finetuning datasets.
"""

import gradio as gr
from textwrap import dedent
from gradio_filelister import FileLister
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.constants import (
    DRAMABOX_TRAINING_DEFAULTS,
)


class TrainModelTool(Tool):
    """Train Model tool implementation."""

    config = ToolConfig(
        name="Train Model",
        module_name="tool_train_model",
        description="Train custom voice models",
        enabled=True,
        category="training"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Train Model tool UI.

        Sliders are initialized with default values from constants.
        Saved user preferences are applied on tab select via
        create_param_restore_handler (same pattern as voice_clone).
        """
        components = {}

        format_help_html = shared_state['format_help_html']
        get_dataset_folders = shared_state['get_dataset_folders']
        _user_config = shared_state['_user_config']

        vv = DRAMABOX_TRAINING_DEFAULTS

        with gr.TabItem("Train Model", id="tab_train_model") as train_tab:
            components['train_tab'] = train_tab
            gr.Markdown("Train a custom voice model using DramaBox finetuning")
            with gr.Row():
                # Left column - Dataset selection
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Selection")

                    components['model_type_radio'] = gr.State(value="DramaBox")

                    components['train_folder_dropdown'] = gr.Dropdown(
                        choices=["(Select Dataset)"] + get_dataset_folders(),
                        value="(Select Dataset)",
                        label="Training Dataset",
                        info="Select prepared subfolder",
                        interactive=True
                    )

                    components['refresh_train_folder_btn'] = gr.Button("Refresh Datasets", size="sm", visible=False)

                    # Legacy Qwen section kept hidden in this build
                    with gr.Group(visible=False) as qwen_ref_section:
                        components['qwen_ref_section'] = qwen_ref_section

                        components['ref_audio_lister'] = FileLister(
                            value=[],
                            height=150,
                            show_footer=False,
                            interactive=True,
                        )

                        components['ref_audio_preview'] = gr.Audio(
                            label="Preview",
                            type="filepath",
                            interactive=False,
                            elem_id="train-ref-audio-preview"
                        )

                    with gr.Row():
                        components['start_training_btn'] = gr.Button("Start Training", variant="primary", size="lg")
                        components['stop_training_btn'] = gr.Button("Stop Training", variant="stop", size="lg", interactive=False)

                # Right column - Training configuration
                with gr.Column(scale=1):
                    gr.Markdown("### Training Configuration")

                    with gr.Accordion("Training Settings", open=False) as train_accordion:
                        components['train_accordion'] = train_accordion

                        # Legacy Qwen section kept hidden in this build
                        q = {"batch_size": 1, "learning_rate": 3e-5, "num_epochs": 10, "save_interval": 1}
                        with gr.Group(visible=False) as qwen_params_section:
                            components['qwen_params_section'] = qwen_params_section

                            with gr.Row():
                                components['qwen_batch_size'] = gr.Slider(
                                    minimum=1, maximum=10, value=q["batch_size"], step=1,
                                    label="Batch Size",
                                    info="Reduce if you get out of memory errors"
                                )

                                components['qwen_learning_rate'] = gr.Slider(
                                    minimum=1e-6, maximum=1e-3, value=q["learning_rate"],
                                    label="Learning Rate",
                                    info="Default: 2e-6"
                                )

                            with gr.Row():
                                components['qwen_num_epochs'] = gr.Slider(
                                    minimum=1, maximum=100, value=q["num_epochs"], step=1,
                                    label="Number of Epochs",
                                    info="How many times to train on the full dataset"
                                )

                                components['qwen_save_interval'] = gr.Slider(
                                    minimum=0, maximum=10, value=q["save_interval"], step=1,
                                    label="Save Interval (Epochs)",
                                    info="Save checkpoint every N epochs (0 = no intermediate saves)"
                                )

                        # --- DramaBox training parameters ---
                        with gr.Group(visible=True) as vv_params_section:
                            components['vv_params_section'] = vv_params_section

                            with gr.Row():
                                components['vv_batch_size'] = gr.Slider(
                                    minimum=1, maximum=16, value=vv["batch_size"], step=1,
                                    label="Batch Size",
                                    info="Reduce if you get out of memory errors"
                                )

                                components['vv_learning_rate'] = gr.Slider(
                                    minimum=1e-6, maximum=1e-3, value=vv["learning_rate"],
                                    label="Learning Rate",
                                    info="Default: 3e-5"
                                )

                            with gr.Row():
                                components['vv_num_epochs'] = gr.Slider(
                                    minimum=0, maximum=6000, value=vv["num_epochs"], step=100,
                                    label="Steps",
                                    info="Total number of training steps"
                                )

                                components['vv_save_interval'] = gr.Slider(
                                    minimum=100, maximum=5000, value=vv["save_interval"], step=100,
                                    label="Save Every N Steps",
                                    info="Save checkpoint every N steps"
                                )

                            with gr.Row():
                                components['vv_gradient_accumulation'] = gr.Slider(
                                    minimum=1, maximum=32, value=vv["gradient_accumulation"], step=1,
                                    label="Gradient Accumulation",
                                    info="Effective batch size = batch_size * grad_accum"
                                )

                                components['vv_gradient_checkpointing'] = gr.Checkbox(
                                    value=bool(vv.get("gradient_checkpointing", 1)),
                                    label="Gradient Checkpointing",
                                    info="Saves VRAM at the cost of speed. Disable if you have headroom."
                                )

                            with gr.Row():
                                components['vv_warmup_steps'] = gr.Slider(
                                    minimum=0, maximum=1000, value=vv["warmup_steps"], step=10,
                                    label="Warmup Steps",
                                    info="Linearly ramp up LR at the start"
                                )

                                components['vv_num_workers'] = gr.Slider(
                                    minimum=0, maximum=8, value=vv.get("num_workers", 2), step=1,
                                    label="DataLoader Workers",
                                    info="CPU threads prefetching batches. Try 0 if GPU utilisation is low on Windows"
                                )

                            gr.Markdown("#### LoRA Settings")

                            with gr.Row():
                                components['vv_lora_rank'] = gr.Slider(
                                    minimum=16, maximum=256, value=vv["lora_rank"], step=16,
                                    label="LoRA Rank",
                                    info="Higher = more capacity, more VRAM"
                                )

                                components['vv_lora_alpha'] = gr.Slider(
                                    minimum=16, maximum=256, value=vv["lora_alpha"], step=16,
                                    label="LoRA Alpha",
                                    info="Scaling factor; typically equals rank"
                                )

                            with gr.Row():
                                components['vv_lora_dropout'] = gr.Slider(
                                    minimum=0.0, maximum=0.5, value=vv["lora_dropout"], step=0.05,
                                    label="LoRA Dropout",
                                    info="Regularisation (0 = off)"
                                )

                                components['vv_lr_scheduler'] = gr.Radio(
                                    choices=["cosine", "linear", "constant"],
                                    value=vv["lr_scheduler"],
                                    label="LR Scheduler"
                                )

                            gr.Markdown("#### Advanced")

                            with gr.Row():
                                components['vv_base_model'] = gr.State(value="dev")

                                components['vv_seed'] = gr.Number(
                                    value=vv["seed"], precision=0,
                                    label="Seed",
                                    info="Random seed for reproducibility"
                                )

                            with gr.Row():
                                components['vv_ref_ratio'] = gr.Slider(
                                    minimum=0.1, maximum=0.9, value=vv["ref_ratio"], step=0.05,
                                    label="Ref Ratio",
                                    info="Fraction of sequence used as reference audio"
                                )

                                components['vv_text_dropout'] = gr.Slider(
                                    minimum=0.0, maximum=0.5, value=vv["text_dropout"], step=0.05,
                                    label="Text Dropout",
                                    info="Drop text conditioning during training (CFG training)"
                                )

                            with gr.Row():
                                components['vv_resume_lora'] = gr.Textbox(
                                    value=vv["resume_lora"],
                                    label="Resume from LoRA checkpoint (optional)",
                                    placeholder="Path to .safetensors checkpoint",
                                    info="Leave blank to start fresh"
                                )

                    components['training_status'] = gr.Textbox(
                        label="Status",
                        lines=3,
                        max_lines=20,
                        interactive=False
                    )

            with gr.Accordion("Convert LoRA for LTX Inference", open=False):
                gr.Markdown(
                    "Convert a trained DramaBox LoRA from PEFT format to the LTX-compatible format "
                    "expected by the inference pipeline. Training auto-converts on completion — "
                    "use this to manually convert existing checkpoints."
                )
                with gr.Column():
                    components['convert_lora_input'] = gr.Textbox(
                        label="LoRA file path",
                        placeholder="e.g. trained_models/MyVoice/lora_best_00100.safetensors",
                        scale=4
                    )
                    components['convert_lora_btn'] = gr.Button("Convert", variant="primary", scale=1)

            with gr.Row():
                train_quick_guide = dedent("""\
                    **Quick Guide:**
                    1. Select dataset folder
                    2. Configure parameters as needed
                    3. Start training & Enter Name
                """)
                gr.HTML(
                    value=format_help_html(train_quick_guide),
                    container=True,
                    padding=True)


        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Train Model tab events."""

        convert_dramabox_lora_to_ltx = shared_state['convert_dramabox_lora_to_ltx']

        get_dataset_files = shared_state['get_dataset_files']
        get_dataset_folders = shared_state['get_dataset_folders']
        get_trained_model_names = shared_state['get_trained_model_names']
        train_model = shared_state['train_model']
        train_dramabox_model = shared_state['train_dramabox_model']
        stop_training = shared_state['stop_training']
        tts_manager = shared_state.get('tts_manager')
        asr_manager = shared_state.get('asr_manager')
        input_trigger = shared_state['input_trigger']
        show_input_modal_js = shared_state['show_input_modal_js']
        DATASETS_DIR = shared_state['DATASETS_DIR']
        save_preference = shared_state['save_preference']
        _user_config = shared_state['_user_config']
        wire_param_persistence = shared_state['wire_param_persistence']
        create_param_restore_handler = shared_state['create_param_restore_handler']

        def get_selected_ref_filename(lister_value):
            """Extract selected filename from FileLister value."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                return selected[0]
            return None



        # --- Folder change: update ref audio lister ---
        def update_ref_audio_lister(folder):
            """Update reference audio lister when folder changes."""
            files = get_dataset_files(folder)
            return files, None

        components['train_folder_dropdown'].change(
            update_ref_audio_lister,
            inputs=[components['train_folder_dropdown']],
            outputs=[components['ref_audio_lister'], components['ref_audio_preview']]
        )

        # Auto-refresh datasets when tab is selected (preserve selection)
        def refresh_datasets_keep_selection(current_folder):
            """Refresh dataset list while preserving the current selection."""
            folder_choices = ["(Select Dataset)"] + get_dataset_folders()
            if current_folder and current_folder in folder_choices:
                return gr.update(choices=folder_choices, value=current_folder)
            return gr.update(choices=folder_choices, value="(Select Dataset)")

        # --- Ref audio preview on selection ---
        def load_ref_audio_preview(lister_value, folder):
            """Load reference audio preview from FileLister selection."""
            filename = get_selected_ref_filename(lister_value)
            if not folder or not filename or folder in ("(No folders)", "(Select Dataset)"):
                return None
            audio_path = DATASETS_DIR / folder / filename
            if audio_path.exists():
                return str(audio_path)
            return None

        components['ref_audio_lister'].change(
            load_ref_audio_preview,
            inputs=[components['ref_audio_lister'], components['train_folder_dropdown']],
            outputs=[components['ref_audio_preview']]
        )

        # Double-click = play preview
        components['ref_audio_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#train-ref-audio-preview .play-pause-button'); if (btn) btn.click(); }, 150); }"
        )

        # --- Settings persistence (same pattern as voice_clone) ---
        # Auto-save & restore training parameters per engine
        param_map = {
            'training_vv': [
                ('vv_batch_size', 'batch_size'),
                ('vv_learning_rate', 'learning_rate'),
                ('vv_num_epochs', 'num_epochs'),
                ('vv_save_interval', 'save_interval'),
                ('vv_gradient_accumulation', 'gradient_accumulation'),
                ('vv_gradient_checkpointing', 'gradient_checkpointing'),
                ('vv_num_workers', 'num_workers'),
                ('vv_warmup_steps', 'warmup_steps'),
                ('vv_lora_rank', 'lora_rank'),
                ('vv_lora_alpha', 'lora_alpha'),
                ('vv_lora_dropout', 'lora_dropout'),
                ('vv_lr_scheduler', 'lr_scheduler'),
                ('vv_ref_ratio', 'ref_ratio'),
                ('vv_text_dropout', 'text_dropout'),
                ('vv_seed', 'seed'),
            ],
        }

        wire_param_persistence(components, _user_config, param_map)

        restore_fn, restore_outputs = create_param_restore_handler(
            components, _user_config, param_map
        )

        # Restore saved params when accordion is opened
        components['train_accordion'].expand(restore_fn, outputs=restore_outputs)

        # --- Start Training: 2-step modal with dynamic validation ---
        # Hidden JSON to pass existing model names to JS for validation
        components['existing_models_json'] = gr.JSON(value=[], visible=False)
        # Hidden state to pass model type to modal handler
        components['train_model_type_state'] = gr.State(value="DramaBox")

        def fetch_existing_models_and_type():
            """Fetch current model list before opening modal."""
            return get_trained_model_names(), "DramaBox"

        # Build the base modal JS using show_input_modal_js
        base_modal_js = show_input_modal_js(
            title="Start Training",
            message="Enter a name for this trained voice model:",
            placeholder="e.g., MyVoice, Female-Narrator, John-Doe",
            submit_button_text="Start Training",
            context="train_model_"
        )

        # Wrap to inject validation and existing-model overwrite confirmation
        open_modal_js = f"""
        (existingModels) => {{
            window.inputModalValidation = (value) => {{
                if (!value || value.trim().length === 0) {{
                    return 'Please enter a model name';
                }}
                return null;
            }};
            window.inputModalExistingFiles = existingModels || [];
            const openModal = {base_modal_js};
            openModal('');
        }}
        """

        # Apply saved params then open modal
        components['start_training_btn'].click(
            fn=restore_fn,
            outputs=restore_outputs
        ).then(
            fn=fetch_existing_models_and_type,
            inputs=[],
            outputs=[components['existing_models_json'], components['train_model_type_state']]
        ).then(
            fn=None,
            inputs=[components['existing_models_json']],
            outputs=None,
            js=open_modal_js
        )

        # --- Handle training modal submission ---
        def activate_stop_btn(input_value):
            """Enable stop button when training is about to start."""
            if not input_value or not input_value.startswith("train_model_"):
                return gr.update(), gr.update()
            # Disable Start, enable Stop
            return gr.update(interactive=False), gr.update(interactive=True)

        def handle_train_model_input(input_value, model_type, folder, ref_lister,
                                     qwen_batch_size, qwen_lr, qwen_epochs, qwen_save_interval,
                                     vv_batch_size, vv_lr, vv_epochs, vv_save_interval,
                                     vv_gradient_accumulation, vv_gradient_checkpointing, vv_num_workers, vv_warmup_steps,
                                     vv_lora_rank, vv_lora_alpha, vv_lora_dropout,
                                     vv_lr_scheduler, vv_base_model, vv_ref_ratio,
                                     vv_text_dropout, vv_seed, vv_resume_lora,
                                     progress=gr.Progress()):
            """Process input modal submission for training."""
            if not input_value or not input_value.startswith("train_model_"):
                return gr.update()

            # Format: "train_model_SpeakerName_timestamp"
            parts = input_value.split("_")
            if len(parts) < 3:
                return gr.update()

            speaker_name = "_".join(parts[2:-1])

            # Unload all models to free VRAM before training
            if tts_manager:
                tts_manager.unload_all()
            if asr_manager:
                asr_manager.unload_all()

            if model_type == "DramaBox":
                return train_dramabox_model(
                    folder, speaker_name, vv_batch_size, vv_lr, vv_epochs,
                    vv_save_interval, vv_gradient_accumulation, 1 if vv_gradient_checkpointing else 0, int(vv_num_workers or 2), vv_warmup_steps,
                    vv_lora_rank, vv_lora_alpha, vv_lora_dropout,
                    vv_lr_scheduler, vv_base_model, vv_ref_ratio,
                    vv_text_dropout, int(vv_seed) if vv_seed is not None else 42,
                    str(vv_resume_lora).strip() if vv_resume_lora else "",
                    progress
                )
            else:
                return "Error: Qwen3 finetuning is disabled in this DramaBox build."

        def deactivate_stop_btn():
            """Re-enable Start, disable Stop after training finishes."""
            return gr.update(interactive=True), gr.update(interactive=False)

        input_trigger.change(
            activate_stop_btn,
            inputs=[input_trigger],
            outputs=[components['start_training_btn'], components['stop_training_btn']]
        ).then(
            handle_train_model_input,
            inputs=[
                input_trigger,
                components['train_model_type_state'],
                components['train_folder_dropdown'],
                components['ref_audio_lister'],
                # Qwen3 params
                components['qwen_batch_size'],
                components['qwen_learning_rate'],
                components['qwen_num_epochs'],
                components['qwen_save_interval'],
                # VV params
                components['vv_batch_size'],
                components['vv_learning_rate'],
                components['vv_num_epochs'],
                components['vv_save_interval'],
                # DramaBox-specific
                components['vv_gradient_accumulation'],
                components['vv_gradient_checkpointing'],
                components['vv_num_workers'],
                components['vv_warmup_steps'],
                components['vv_lora_rank'],
                components['vv_lora_alpha'],
                components['vv_lora_dropout'],
                components['vv_lr_scheduler'],
                components['vv_base_model'],
                components['vv_ref_ratio'],
                components['vv_text_dropout'],
                components['vv_seed'],
                components['vv_resume_lora'],
            ],
            outputs=[components['training_status']]
        ).then(
            deactivate_stop_btn,
            outputs=[components['start_training_btn'], components['stop_training_btn']]
        )

        # --- Stop Training button ---
        def handle_stop_training():
            """Stop the active training subprocess."""
            stop_training()
            return "Stopping training... please wait."

        components['stop_training_btn'].click(
            handle_stop_training,
            outputs=[components['training_status']]
        )

        # --- Convert LoRA button ---
        def handle_convert_lora(input_path):
            """Convert a PEFT DramaBox LoRA to LTX-compatible format."""
            if not input_path or not input_path.strip():
                return "❌ Error: Please enter a LoRA file path."
            from pathlib import Path
            from modules.core_components.tools import PROJECT_ROOT
            p = Path(input_path.strip())
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if not p.exists():
                return f"❌ Error: File not found: {p}"
            result = convert_dramabox_lora_to_ltx(p)
            if result is None:
                return "❌ Error: Conversion failed. Check that the file is a valid safetensors LoRA."
            return f"Converted: {result}"

        components['convert_lora_btn'].click(
            handle_convert_lora,
            inputs=[components['convert_lora_input']],
            outputs=[components['training_status']]
        )

        # Tab select: refresh datasets and set correct section visibility
        components['train_tab'].select(
            refresh_datasets_keep_selection,
            inputs=[components['train_folder_dropdown']],
            outputs=[components['train_folder_dropdown']]
        )


# Export for tab registry
get_tool_class = lambda: TrainModelTool

if __name__ == "__main__":
    """Standalone testing of Train Model tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(TrainModelTool, port=7863, title="Train Model - Standalone")