"""
Prep Samples Tab

Unified tool for preparing voice samples and managing finetuning datasets.
Supports two modes via radio toggle: Samples (voice cloning) and Datasets (finetuning).
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "modules"))

import gradio as gr
import re
import os
import shutil
import soundfile as sf
from pathlib import Path
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.asr_manager import get_asr_manager
from gradio_filelister import FileLister

# --- Console colors (ANSI) ---
# Enable ANSI on Windows
if os.name == 'nt':
    os.system('')

_CYAN = '\033[96m'
_GREEN = '\033[92m'
_YELLOW = '\033[93m'
_RED = '\033[91m'
_DIM = '\033[2m'
_BOLD = '\033[1m'
_RESET = '\033[0m'


class PrepSamplesTool(Tool):
    """Unified Prep Samples + Dataset tool."""

    config = ToolConfig(
        name="Prep Samples",
        module_name="tool_prep_audio",
        description="Prepare and manage voice samples and datasets",
        enabled=True,
        category="preparation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Prep Samples tool UI."""
        components = {}

        format_help_html = shared_state['format_help_html']
        get_sample_choices = shared_state['get_sample_choices']
        get_dataset_folders = shared_state['get_dataset_folders']
        get_dataset_files = shared_state['get_dataset_files']
        _user_config = shared_state['_user_config']
        LANGUAGES = shared_state['LANGUAGES']
        ASR_ENGINES = shared_state.get('ASR_ENGINES', {})
        ASR_OPTIONS = shared_state.get('ASR_OPTIONS', [])
        DEFAULT_ASR_MODEL = shared_state.get('DEFAULT_ASR_MODEL', 'Qwen3 ASR - Large')
        WHISPER_AVAILABLE = shared_state['WHISPER_AVAILABLE']
        QWEN3_ASR_AVAILABLE = shared_state['QWEN3_ASR_AVAILABLE']
        DEEPFILTER_AVAILABLE = shared_state['DEEPFILTER_AVAILABLE']

        # Let's hide dataset if train model is off
        train_model_enabled = _user_config.get("enabled_tools", {}).get("Train Model", True)

        with gr.TabItem("Prep Samples") as prep_tab:
            components['prep_tab'] = prep_tab
            if train_model_enabled is True:
                gr.Markdown("Prepare audio samples for voice cloning or finetuning datasets.")
            else:
                gr.Markdown("Prepare audio samples for voice cloning.")

            components['prep_data_type'] = gr.Radio(
                choices=['Samples', 'Datasets'],
                value='Samples',
                show_label=False,
                container=False,
                visible=train_model_enabled,
            )

            with gr.Row():
                # Left column - File browser
                with gr.Column(scale=1):
                    # --- Samples mode ---
                    with gr.Column(visible=True) as samples_col:
                        gr.Markdown("### Audio Samples")
                        components['sample_lister'] = FileLister(
                            value=get_sample_choices(),
                            height=250,
                            show_footer=False,
                            interactive=True,
                        )
                    components['samples_col'] = samples_col

                    # --- Datasets mode (starts visible=True for DOM rendering; toggled on tab.select) ---
                    with gr.Column(visible=True) as datasets_col:
                        gr.Markdown("### Dataset Files")
                        components['finetune_folder_dropdown'] = gr.Dropdown(
                            choices=["(Select Dataset)"] + get_dataset_folders(),
                            value="(Select Dataset)",
                            label="Dataset Folder",
                            info="Subfolders in datasets",
                            interactive=True,
                        )
                        with gr.Row():
                            components['create_folder_btn'] = gr.Button("New Folder", size="sm")
                            components['delete_folder_btn'] = gr.Button("Delete Folder", size="sm", variant="stop")
                        components['dataset_lister'] = FileLister(
                            value=[],
                            height=250,
                            show_footer=False,
                            interactive=True,
                        )
                    components['datasets_col'] = datasets_col

                    # --- Shared buttons ---
                    with gr.Row():
                        pass

                    with gr.Row():
                        components['clear_cache_btn'] = gr.Button("Clear Cache", size="sm")
                        components['delete_btn'] = gr.Button("Delete", size="sm", variant="stop")

                    components['existing_sample_info'] = gr.Textbox(
                        label="Info",
                        interactive=False,
                        lines=3
                    )

                    gr.Markdown("### Transcription Settings")

                    # Build ASR dropdown from enabled engines (same pattern as TTS)
                    asr_settings = _user_config.get("enabled_asr_engines", {})
                    visible_asr_options = []
                    for engine_key, engine_info in ASR_ENGINES.items():
                        if asr_settings.get(engine_key, engine_info.get("default_enabled", True)):
                            # Check availability for engines that need it
                            if engine_key == "Qwen3 ASR" and not QWEN3_ASR_AVAILABLE:
                                continue
                            if engine_key == "Whisper" and not WHISPER_AVAILABLE:
                                continue
                            visible_asr_options.extend(engine_info["choices"])
                    if not visible_asr_options:
                        visible_asr_options = ["VibeVoice ASR - Default"]

                    from modules.core_components.constants import get_default_asr_model
                    default_asr = _user_config.get("transcribe_model", get_default_asr_model(_user_config))
                    if default_asr not in visible_asr_options:
                        default_asr = visible_asr_options[0]

                    with gr.Row():
                        components['transcribe_model'] = gr.Dropdown(
                            choices=visible_asr_options,
                            value=default_asr,
                            label="Transcription Engine",
                        )

                        # Language dropdown (shown for Qwen3 ASR and Whisper)
                        show_lang = any(k in default_asr for k in ("Qwen3 ASR", "Whisper"))
                        components['whisper_language'] = gr.Dropdown(
                            choices=["Auto-detect"] + LANGUAGES[1:],
                            value=_user_config.get("whisper_language", "Auto-detect"),
                            label="Language",
                            visible=show_lang
                        )

                # Right column - Audio editing
                with gr.Column(scale=2):
                    components['editor_heading'] = gr.Markdown("### Add or Edit Audio <small>(Drag and drop audio or video files)</small>")

                    components['prep_file_input'] = gr.File(
                        label="Audio or Video File",
                        type="filepath",
                        file_types=["audio", "video"],
                        interactive=True
                    )

                    components['prep_audio_editor'] = gr.Audio(
                        label="Audio Editor (Use Trim icon to edit)",
                        type="filepath",
                        interactive=True,
                        visible=False,
                        elem_id="prep-audio-editor"
                    )

                    with gr.Row():
                        components['clear_btn'] = gr.Button("Clear", scale=1, size="sm")
                        components['clean_btn'] = gr.Button("AI Denoise", scale=2, size="sm", visible=DEEPFILTER_AVAILABLE)
                        components['normalize_btn'] = gr.Button("Normalize Volume", scale=2, size="sm")
                        components['mono_btn'] = gr.Button("Convert to Mono", scale=2, size="sm")

                    with gr.Accordion("Split Settings", open=False,
                                      visible=True) as auto_split_accordion:
                        with gr.Row():
                            components['split_min'] = gr.Slider(
                                minimum=1, maximum=15, value=_user_config.get("split_min", 5), step=0.5,
                                label="Minimum clip duration (seconds)",
                                info="Sentences shorter than this will be merged with the next one"
                            )
                            components['split_max'] = gr.Slider(
                                minimum=10, maximum=60, value=_user_config.get("split_max", 20), step=1,
                                label="Maximum clip duration (seconds)",
                                info="Clips longer than this will split at commas too"
                            )
                        with gr.Row():
                            components['silence_trim'] = gr.Slider(
                                minimum=0, maximum=10, value=_user_config.get("silence_trim", 1), step=0.5,
                                label="Max silence to keep (seconds)",
                                info="Silence gaps longer than this are trimmed down"
                            )
                            components['discard_under'] = gr.Slider(
                                minimum=0, maximum=5, value=_user_config.get("discard_under", 1), step=0.5,
                                label="Discard clips under (seconds)",
                                info="Clips shorter than this are discarded (0 = keep all)"
                            )
                    components['auto_split_accordion'] = auto_split_accordion

                    components['auto_split_btn'] = gr.Button(
                        "Auto-Split Audio",
                        variant="primary",
                        visible=True
                    )

                    gr.Markdown("### Reference Text")
                    components['transcription_output'] = gr.Textbox(
                        label="Text",
                        lines=4,
                        max_lines=10,
                        interactive=True,
                        placeholder="Transcription will appear here, or enter/edit text manually..."
                    )

                    with gr.Row():
                        components['transcribe_btn'] = gr.Button("Transcribe Audio", variant="primary")
                        components['save_btn'] = gr.Button("Save Sample", variant="primary")

                    # Dataset-only: batch transcribe + auto-split
                    with gr.Column(visible=True) as batch_col:
                        components['batch_transcribe_btn'] = gr.Button(
                            "Batch Transcribe All Clips",
                            variant="primary", size="lg"
                        )
                        components['batch_replace_existing'] = gr.Checkbox(
                            label="Replace existing transcripts",
                            value=False
                        )
                    components['batch_col'] = batch_col

                    components['prep_status'] = gr.Textbox(label="Status", interactive=False,
                                                           lines=1, max_lines=15)

                    # Hidden state for passing existing file names to JS
                    components['existing_files_json'] = gr.Textbox(visible=False)

            return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Prep Samples tab events."""

        # Config
        _user_config = shared_state['_user_config']

        # Shared utilities
        get_sample_choices = shared_state['get_sample_choices']
        load_sample_details = shared_state['load_sample_details']
        get_prompt_cache_path = shared_state['get_prompt_cache_path']
        play_completion_beep = shared_state.get('play_completion_beep')
        save_preference = shared_state['save_preference']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        show_input_modal_js = shared_state['show_input_modal_js']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        SAMPLES_DIR = shared_state['SAMPLES_DIR']
        DATASETS_DIR = shared_state['DATASETS_DIR']
        get_dataset_folders = shared_state['get_dataset_folders']
        get_dataset_files = shared_state['get_dataset_files']

        # Audio utilities
        is_audio_file = shared_state['is_audio_file']
        is_video_file = shared_state['is_video_file']
        extract_audio_from_video = shared_state['extract_audio_from_video']
        get_audio_duration = shared_state['get_audio_duration']
        format_time = shared_state['format_time']
        normalize_audio = shared_state['normalize_audio']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']

        # ASR manager (singleton)
        asr_manager = get_asr_manager()

        # ===== Helpers =====

        def is_dataset_mode(data_type):
            return data_type == "Datasets"

        def parse_asr_model(model_str):
            """Parse unified ASR dropdown value into (engine, size).

            Examples:
                'Qwen3 ASR - Small' -> ('Qwen3 ASR', 'Small')
                'Qwen3 ASR - Large' -> ('Qwen3 ASR', 'Large')
                'VibeVoice ASR - Default' -> ('VibeVoice ASR', None)
                'Whisper - Medium' -> ('Whisper', 'Medium')
                'Whisper - Large' -> ('Whisper', 'Large')
            """
            if " - " in model_str:
                engine, size = model_str.rsplit(" - ", 1)
                return engine, (size if size != "Default" else None)
            return model_str, None

        def get_selected_sample_name(lister_value):
            """Extract selected sample name (strips .wav extension)."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                from modules.core_components.tools import strip_sample_extension
                return strip_sample_extension(selected[0])
            return None

        def get_selected_filename(lister_value):
            """Extract selected filename from FileLister value."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                return selected[0]
            return None

        def get_dataset_dir(folder):
            """Get directory for a dataset folder."""
            if folder and folder not in ("(No folders)", "(Select Dataset)"):
                return DATASETS_DIR / folder
            return DATASETS_DIR

        # ===== Mode switching =====

        def on_mode_change(data_type, transcribe_model):
            """Switch between Samples and Datasets mode."""
            is_ds = is_dataset_mode(data_type)
            heading = "### Add or Edit Audio <small>(Drag and drop audio or video files)</small>"
            engine, _ = parse_asr_model(transcribe_model)
            show_auto_split = is_ds and engine in ("Qwen3 ASR", "Whisper")
            return (
                gr.update(visible=not is_ds),    # samples_col
                gr.update(visible=is_ds),        # datasets_col
                gr.update(visible=not is_ds),    # clear_cache_btn
                gr.update(visible=is_ds),        # batch_col
                gr.update(visible=True, value=None),  # prep_file_input (visible, cleared)
                gr.update(visible=False),        # prep_audio_editor visibility
                None,                            # prep_audio_editor value
                "",                              # transcription_output
                "",                              # prep_status
                gr.update(visible=not is_ds, value=""),  # existing_sample_info
                heading,                         # editor_heading
                gr.update(visible=show_auto_split),  # auto_split_accordion
                gr.update(visible=show_auto_split),  # auto_split_btn
            )

        # ===== Selection handlers =====

        def on_sample_selection_change(lister_value):
            """Handle sample selection - load into editor and transcription."""
            sample_name = get_selected_sample_name(lister_value)
            if not sample_name:
                return gr.update(visible=False), gr.update(visible=True), "", ""

            audio_path, ref_text, info_text = load_sample_details(sample_name)
            return gr.update(visible=True, value=audio_path), gr.update(visible=False, value=None), ref_text, info_text

        def on_dataset_selection_change(folder, lister_value):
            """Load audio and transcript for selected dataset item."""
            filename = get_selected_filename(lister_value)
            if not filename:
                return gr.update(visible=False), gr.update(visible=True), "", ""

            base_dir = get_dataset_dir(folder)
            audio_path = base_dir / filename
            txt_path = audio_path.with_suffix(".txt")

            transcript = ""
            if txt_path.exists():
                try:
                    transcript = txt_path.read_text(encoding="utf-8")
                except Exception:
                    pass

            if audio_path.exists():
                return gr.update(visible=True, value=str(audio_path)), gr.update(visible=False, value=None), transcript, ""
            return gr.update(visible=True, value=None), gr.update(visible=False, value=None), transcript, ""

        # ===== File load (samples mode) =====

        def on_prep_audio_load_handler(audio_file):
            """When audio/video is loaded via file input."""
            if audio_file is None:
                return gr.update(), ""
            try:
                if is_video_file(audio_file):
                    print(f"{_DIM}Video file detected: {Path(audio_file).name}{_RESET}")
                    audio_path, message = extract_audio_from_video(audio_file)
                    if audio_path:
                        duration = get_audio_duration(audio_path)
                        info = f"[VIDEO] Audio extracted\nDuration: {format_time(duration)} ({duration:.2f}s)"
                        return audio_path, info
                    return None, message
                elif is_audio_file(audio_file):
                    duration = get_audio_duration(audio_file)
                    return audio_file, f"Duration: {format_time(duration)} ({duration:.2f}s)"
                else:
                    return None, "Unsupported file type"
            except Exception as e:
                return None, f"Error: {str(e)}"

        # ===== Delete handler (unified) =====

        def delete_handler(action, data_type, sample_lister, dataset_lister, folder):
            """Delete selected items or folder based on current mode."""
            no_update = gr.update(), gr.update(), gr.update(), gr.update()
            if not action or not action.strip():
                return no_update

            # --- Delete entire dataset folder ---
            if action.startswith("delete_folder_") and "confirm" in action:
                if not folder or folder in ("(No folders)", "(Select Dataset)"):
                    return "No folder selected", gr.update(), gr.update(), gr.update()

                folder_path = DATASETS_DIR / folder
                if not folder_path.exists():
                    return f"Folder '{folder}' not found", gr.update(), gr.update(), gr.update()

                try:
                    shutil.rmtree(str(folder_path))
                    updated_folders = ["(Select Dataset)"] + get_dataset_folders()
                    return (f"Deleted folder '{folder}'",
                            gr.update(),
                            gr.update(value=[]),
                            gr.update(choices=updated_folders, value="(Select Dataset)"))
                except Exception as e:
                    return f"Error deleting folder: {str(e)}", gr.update(), gr.update(), gr.update()

            if is_dataset_mode(data_type):
                if not action.startswith("finetune_") or "confirm" not in action:
                    return no_update

                if not dataset_lister or not dataset_lister.get("selected"):
                    return "No file(s) selected", gr.update(), gr.update(), gr.update()

                base_dir = get_dataset_dir(folder)
                deleted_count = 0
                errors = []

                for filename in dataset_lister["selected"]:
                    try:
                        audio_path = base_dir / filename
                        txt_path = audio_path.with_suffix(".txt")
                        if audio_path.exists():
                            audio_path.unlink()
                        if txt_path.exists():
                            txt_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        errors.append(f"{filename}: {str(e)}")

                updated = get_dataset_files(folder)
                if errors:
                    msg = f"Deleted {deleted_count} file(s), {len(errors)} error(s): {'; '.join(errors)}"
                else:
                    msg = f"Deleted {deleted_count} file(s)"
                return msg, gr.update(), updated, gr.update()

            else:
                if not action.startswith("sample_") or "confirm" not in action:
                    return no_update

                if not sample_lister or not sample_lister.get("selected"):
                    return "No sample selected", gr.update(), gr.update(), gr.update()

                import os
                deleted_count = 0
                errors = []

                for display_name in sample_lister["selected"]:
                    from modules.core_components.tools import strip_sample_extension
                    sample_name = strip_sample_extension(display_name)
                    try:
                        wav_path = SAMPLES_DIR / f"{sample_name}.wav"
                        if wav_path.exists():
                            os.remove(wav_path)
                        json_path = SAMPLES_DIR / f"{sample_name}.json"
                        if json_path.exists():
                            os.remove(json_path)
                        for model_size in ["0.6B", "1.7B"]:
                            cache_path = get_prompt_cache_path(sample_name, model_size)
                            if cache_path.exists():
                                os.remove(cache_path)
                        # Also delete LuxTTS cache
                        luxtts_cache = SAMPLES_DIR / f"{sample_name}_luxtts.pt"
                        if luxtts_cache.exists():
                            os.remove(luxtts_cache)
                        deleted_count += 1
                    except Exception as e:
                        errors.append(f"{sample_name}: {str(e)}")

                updated = get_sample_choices()
                if errors:
                    msg = f"Deleted {deleted_count} sample(s), {len(errors)} error(s): {'; '.join(errors)}"
                else:
                    msg = f"Deleted {deleted_count} sample(s)"
                return msg, updated, gr.update(), gr.update()

        # ===== Cache clear (samples only) =====

        def clear_sample_cache_handler(lister_value):
            """Clear voice prompt cache for a sample."""
            sample_name = get_selected_sample_name(lister_value)
            if not sample_name:
                return "No sample selected", gr.update()
            try:
                import os
                deleted = []
                for model_size in ["0.6B", "1.7B"]:
                    cache_path = get_prompt_cache_path(sample_name, model_size)
                    if cache_path.exists():
                        os.remove(cache_path)
                        deleted.append(model_size)
                # Also clear LuxTTS cache
                luxtts_cache = SAMPLES_DIR / f"{sample_name}_luxtts.pt"
                if luxtts_cache.exists():
                    os.remove(luxtts_cache)
                    deleted.append("LuxTTS")
                if deleted:
                    _, _, new_info = load_sample_details(sample_name)
                    return f"Cache cleared for: {', '.join(deleted)}", new_info
                return "No cache files found", gr.update()
            except Exception as e:
                return f"Error clearing cache: {str(e)}", gr.update()

        # ===== Transcribe (shared) =====

        def transcribe_audio_handler(audio_file, whisper_language, transcribe_model, progress=gr.Progress()):
            """Transcribe audio using Whisper, Qwen3 ASR, or VibeVoice ASR."""
            if audio_file is None:
                return "Please load an audio file first.", ""
            try:
                engine, size = parse_asr_model(transcribe_model)
                if engine == "Qwen3 ASR":
                    progress(0.2, desc=f"Loading Qwen3 ASR ({size})...")
                    model = asr_manager.get_qwen3_asr(size=size)
                    progress(0.4, desc="Transcribing...")
                    options = {}
                    if whisper_language and whisper_language != "Auto-detect":
                        options["language"] = whisper_language
                    result = model.transcribe(audio_file, **options)
                elif engine == "VibeVoice ASR":
                    progress(0.2, desc="Loading VibeVoice ASR...")
                    model = asr_manager.get_vibevoice_asr()
                    progress(0.4, desc="Transcribing...")
                    result = model.transcribe(audio_file)
                else:
                    if not asr_manager.whisper_available:
                        return "Whisper not available. Use VibeVoice ASR instead.", ""
                    progress(0.2, desc=f"Loading Whisper ({size or 'Medium'})...")
                    model = asr_manager.get_whisper(size=size or "Medium")
                    progress(0.4, desc="Transcribing...")
                    options = {}
                    if whisper_language and whisper_language != "Auto-detect":
                        lang_code = {
                            "English": "en", "Chinese": "zh", "Japanese": "ja",
                            "Korean": "ko", "German": "de", "French": "fr",
                            "Russian": "ru", "Portuguese": "pt", "Spanish": "es",
                            "Italian": "it"
                        }.get(whisper_language, None)
                        if lang_code:
                            options["language"] = lang_code
                    result = model.transcribe(audio_file, **options)

                progress(1.0, desc="Done!")
                print(f"\n{_CYAN}--- Transcription ({engine}) ---{_RESET}")
                print(f"{_DIM}{result['text']}{_RESET}")
                transcription = result["text"].strip()

                if engine == "VibeVoice ASR":
                    transcription = re.sub(r'\[.*?\]\s*:', '', transcription)
                    transcription = re.sub(r'\[.*?\]', '', transcription)
                    transcription = ' '.join(transcription.split())

                if play_completion_beep:
                    play_completion_beep()

                return transcription, "Transcription complete"

            except Exception as e:
                import traceback
                print(f"{_RED}Error in transcribe:\n{traceback.format_exc()}{_RESET}")
                return f"Error transcribing: {str(e)}", ""

        # ===== Save handlers =====
        def _do_save_sample(clean_name, audio, transcription):
            """Perform the actual sample save (shared by initial save and overwrite)."""
            import json, os

            cleaned_text = re.sub(r'\[.*?\]\s*', '', transcription).strip() if transcription else ""

            audio_path = Path(audio).resolve()
            original_path = (SAMPLES_DIR / f"{clean_name}.wav").resolve()
            audio_unmodified = audio_path == original_path

            meta = {"Type": "Sample", "Text": cleaned_text}
            json_path = SAMPLES_DIR / f"{clean_name}.json"
            json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            if audio_unmodified:
                return (f"Sample '{clean_name}' updated (text only)",
                        get_sample_choices(), gr.update(), gr.update())

            audio_data, sr = sf.read(audio)
            sf.write(str(SAMPLES_DIR / f"{clean_name}.wav"), audio_data, sr)

            cleared = []
            for model_size in ["0.6B", "1.7B"]:
                cache_path = get_prompt_cache_path(clean_name, model_size)
                if cache_path.exists():
                    os.remove(cache_path)
                    cleared.append(model_size)

            cache_msg = f", cache cleared ({', '.join(cleared)})" if cleared else ""
            return (f"Sample '{clean_name}' saved (audio + text{cache_msg})",
                    get_sample_choices(), gr.update(), gr.update())

        def _do_save_dataset(clean_name, audio, transcription, folder):
            """Perform the actual dataset clip save (shared by initial save and overwrite)."""
            base_dir = get_dataset_dir(folder)
            base_dir.mkdir(parents=True, exist_ok=True)
            results = []

            filename = f"{clean_name}.wav"
            if audio:
                try:
                    audio_path = base_dir / filename
                    if isinstance(audio, str):
                        if Path(audio).resolve() != audio_path.resolve():
                            shutil.copy(audio, str(audio_path))
                            results.append(f"Audio saved as {filename}")
                        else:
                            results.append("Audio unchanged")
                    else:
                        sr, audio_data = audio
                        sf.write(str(audio_path), audio_data, sr, subtype='PCM_16')
                        results.append(f"Audio saved as {filename}")
                except Exception as e:
                    results.append(f"Error saving audio: {str(e)}")

            if transcription and transcription.strip():
                try:
                    txt_path = (base_dir / filename).with_suffix(".txt")
                    txt_path.write_text(transcription.strip(), encoding="utf-8")
                    results.append("Transcript saved")
                except Exception as e:
                    results.append(f"Error saving transcript: {str(e)}")

            msg = " | ".join(results) if results else "Nothing to save"
            updated_files = get_dataset_files(folder)
            generate_dramabox_manifest(base_dir)
            return msg, gr.update(), updated_files, gr.update()

        def handle_input_modal(input_value, audio, transcription, folder, language, transcribe_model,
                               split_min, split_max, silence_trim, discard_under):
            """Process input modal results for save sample/dataset, auto-split, and create folder."""
            # 4 outputs: prep_status, sample_lister, dataset_lister, folder_dropdown
            no_update = gr.update(), gr.update(), gr.update(), gr.update()

            if not input_value:
                return no_update

            # --- Save sample (samples mode) ---
            if input_value.startswith("save_sample_"):
                parts = input_value.split("_")
                if len(parts) >= 3:
                    if parts[2] == "cancel":
                        return no_update

                    sample_name = "_".join(parts[2:-1])

                    if not audio:
                        return "No audio file to save", gr.update(), gr.update(), gr.update()

                    clean_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in sample_name).strip()
                    clean_name = clean_name.replace(" ", "_")
                    if not clean_name:
                        return "Invalid sample name", gr.update(), gr.update(), gr.update()

                    try:
                        return _do_save_sample(clean_name, audio, transcription)
                    except Exception as e:
                        return f"Error saving sample: {str(e)}", gr.update(), gr.update(), gr.update()

                return no_update

            # --- Save dataset clip ---
            if input_value.startswith("save_dataset_"):
                parts = input_value.split("_")
                if len(parts) >= 3:
                    if parts[2] == "cancel":
                        return no_update

                    clip_name = "_".join(parts[2:-1])

                    if not audio:
                        return "No audio file to save", gr.update(), gr.update(), gr.update()

                    clean_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in clip_name).strip()
                    clean_name = clean_name.replace(" ", "_")
                    if not clean_name:
                        return "Invalid clip name", gr.update(), gr.update(), gr.update()

                    if not folder or folder in ("(No folders)", "(Select Dataset)"):
                        return "Select a dataset folder first", gr.update(), gr.update(), gr.update()

                    try:
                        return _do_save_dataset(clean_name, audio, transcription, folder)
                    except Exception as e:
                        return f"Error saving: {str(e)}", gr.update(), gr.update(), gr.update()

                return no_update

            # --- Auto-split audio ---
            if input_value.startswith("auto_split_"):
                parts = input_value.split("_")
                if len(parts) >= 3:
                    if parts[2] == "cancel":
                        return no_update

                    raw_name = "_".join(parts[2:-1])
                    clip_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_name).strip("_")
                    if not clip_prefix:
                        return "Invalid clip name", gr.update(), gr.update(), gr.update()

                    engine, asr_size = parse_asr_model(transcribe_model)
                    status, files = auto_split_audio_handler(
                        clip_prefix, audio, folder, language, engine, asr_size or "Large",
                        split_min, split_max, silence_trim, discard_under
                    )
                    return status, gr.update(), files, gr.update()

                return no_update

            # --- Create dataset folder ---
            if input_value.startswith("create_folder_"):
                parts = input_value.split("_")
                if len(parts) >= 3:
                    if parts[2] == "cancel":
                        return no_update

                    raw_name = "_".join(parts[2:-1])
                    folder_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in raw_name).strip()
                    if not folder_name:
                        return "Invalid folder name", gr.update(), gr.update(), gr.update()

                    folder_path = DATASETS_DIR / folder_name
                    if folder_path.exists():
                        return (f"Folder '{folder_name}' already exists",
                                gr.update(), gr.update(), gr.update())

                    try:
                        folder_path.mkdir(parents=True, exist_ok=True)
                        updated_folders = ["(Select Dataset)"] + get_dataset_folders()
                        return (f"Created folder '{folder_name}'",
                                gr.update(),
                                gr.update(),
                                gr.update(choices=updated_folders, value=folder_name))
                    except Exception as e:
                        return f"Error creating folder: {str(e)}", gr.update(), gr.update(), gr.update()

                return no_update

            return no_update

        # ===== Batch transcribe (dataset mode only) =====

        def generate_dramabox_manifest(base_dir):
            """Generate train_dramabox.jsonl from all WAV+TXT pairs in the dataset folder.

            Format: {"audio_filepath": "<abs_path>", "text": "<transcript>"}
            This is the manifest format DramaBox's preprocess.py expects.
            Returns the number of entries written.
            """
            import json as _json
            entries = []
            for wav in sorted(base_dir.glob("*.wav")):
                txt = wav.with_suffix(".txt")
                if txt.exists():
                    text = txt.read_text(encoding="utf-8").strip()
                    if text:
                        try:
                            info = sf.info(str(wav))
                            duration = info.frames / info.samplerate
                        except Exception:
                            duration = 0.0
                        entries.append({"audio_filepath": str(wav), "text": text, "duration": round(duration, 3)})
            if not entries:
                return 0
            manifest_path = base_dir / "train_dramabox.jsonl"
            with open(manifest_path, "w", encoding="utf-8") as f:
                for e in entries:
                    _json.dump(e, f, ensure_ascii=False)
                    f.write("\n")
            return len(entries)

        def batch_transcribe_handler(folder, replace_existing, language, transcribe_model, progress=gr.Progress()):
            """Batch transcribe all audio files in a dataset folder."""
            if not folder or folder in ("(No folders)", "(Select Dataset)"):
                return "Please select a dataset folder first."

            try:
                engine, asr_size = parse_asr_model(transcribe_model)
                base_dir = DATASETS_DIR / folder
                if not base_dir.exists():
                    return f"Folder not found: {folder}"

                audio_files = sorted(list(base_dir.glob("*.wav")) + list(base_dir.glob("*.mp3")))
                if not audio_files:
                    return f"No audio files found in {folder}"

                files_to_process = [f for f in audio_files
                                    if not f.with_suffix(".txt").exists() or replace_existing]
                if not files_to_process:
                    return (f"All {len(audio_files)} files already have transcripts. "
                            "Check 'Replace existing' to re-transcribe.")

                print(f"\n{_CYAN}{'=' * 55}")
                print(f" Batch Transcribe  |  Engine: {engine}")
                print(f" Folder: {folder}  |  {len(files_to_process)} file(s)")
                print(f"{'=' * 55}{_RESET}")

                status_log = [
                    f"Batch transcribing folder: {folder}",
                    f"Found {len(audio_files)} audio files ({len(files_to_process)} to process)",
                    ""
                ]

                # Load model once
                options = {}
                if engine == "Qwen3 ASR":
                    progress(0.05, desc=f"Loading Qwen3 ASR model ({asr_size})...")
                    try:
                        model = asr_manager.get_qwen3_asr(size=asr_size)
                        status_log.append("Loaded Qwen3 ASR model")
                    except Exception as e:
                        return f"Qwen3 ASR not available: {str(e)}"
                    if language and language != "Auto-detect":
                        options["language"] = language
                elif engine == "VibeVoice ASR":
                    progress(0.05, desc="Loading VibeVoice ASR model...")
                    try:
                        model = asr_manager.get_vibevoice_asr()
                        status_log.append("Loaded VibeVoice ASR model")
                    except Exception as e:
                        return f"VibeVoice ASR not available: {str(e)}"
                else:
                    if not asr_manager.whisper_available:
                        return "Whisper not available. Use VibeVoice ASR instead."
                    progress(0.05, desc=f"Loading Whisper ({asr_size or 'Medium'})...")
                    try:
                        model = asr_manager.get_whisper(size=asr_size or "Medium")
                        status_log.append("Loaded Whisper model")
                    except ImportError as e:
                        return f"Error: {str(e)}"
                    if language and language != "Auto-detect":
                        lang_code = {
                            "English": "en", "Chinese": "zh", "Japanese": "ja",
                            "Korean": "ko", "German": "de", "French": "fr",
                            "Russian": "ru", "Portuguese": "pt", "Spanish": "es",
                            "Italian": "it"
                        }.get(language, None)
                        if lang_code:
                            options["language"] = lang_code

                status_log.extend(["", "=" * 60])
                transcribed_count = 0
                skipped_count = 0
                error_count = 0

                for i, audio_file in enumerate(audio_files):
                    txt_file = audio_file.with_suffix(".txt")

                    if txt_file.exists() and not replace_existing:
                        status_log.append(f"Skipped: {audio_file.name} (transcript exists)")
                        skipped_count += 1
                        continue

                    progress_val = 0.1 + (0.9 * i / len(audio_files))
                    progress(progress_val,
                             desc=f"Transcribing {i + 1}/{len(audio_files)}: {audio_file.name[:30]}...")

                    try:
                        if engine == "VibeVoice ASR":
                            result = model.transcribe(str(audio_file))
                        else:
                            result = model.transcribe(str(audio_file), **options)

                        print(f"  {_GREEN}>{_RESET} {audio_file.name}  {_DIM}{result['text'][:60]}{'...' if len(result['text']) > 60 else ''}{_RESET}")
                        text = result["text"].strip()
                        if engine == "VibeVoice ASR":
                            text = re.sub(r'\[.*?\]\s*:', '', text)
                            text = re.sub(r'\[.*?\]', '', text)
                            text = ' '.join(text.split())

                        txt_file.write_text(text, encoding="utf-8")
                        status_log.append(f"{audio_file.name} -> {len(text)} chars")
                        transcribed_count += 1

                    except Exception as e:
                        status_log.append(f"Error: {audio_file.name} - {str(e)}")
                        error_count += 1

                status_log.extend([
                    "=" * 60, "",
                    "Summary:",
                    f"   Transcribed: {transcribed_count}",
                    f"   Skipped: {skipped_count}",
                    f"   Errors: {error_count}",
                    f"   Total: {len(audio_files)}"
                ])

                progress(1.0, desc="Batch transcription complete!")
                if play_completion_beep:
                    play_completion_beep()

                count = generate_dramabox_manifest(base_dir)
                if count > 0:
                    status_log.append("")
                    status_log.append(f"Generated train_dramabox.jsonl ({count} entries)")

                return "\n".join(status_log)

            except Exception as e:
                return f"Error during batch transcription: {str(e)}"

        # ===== Auto-Split Audio =====

        def split_into_segments(full_text, word_timestamps, min_duration=4.0, max_duration=20.0,
                                silence_trim=1.0, discard_under=1.0):
            """Split transcribed text into segments using word timestamps and silence detection.

            Pipeline:
            1. Map sentences to word timestamp ranges
            2. Detect silence gaps between words — force-cut at gaps > silence_trim
            3. Group small pieces by sentences up to min_duration
            4. Break oversized segments (> max_duration) at commas
            5. Trim leading/trailing silence from each segment
            6. Discard segments shorter than discard_under

            Args:
                full_text: Original transcription with punctuation from ASR
                word_timestamps: List of timestamp objects with .text, .start_time, .end_time
                min_duration: Minimum clip duration in seconds
                max_duration: Maximum clip duration — commas become split points above this
                silence_trim: Silence gaps longer than this cause a forced cut (seconds)
                discard_under: Discard final clips shorter than this (seconds, 0 = keep all)

            Returns:
                List of (start_time, end_time, text) tuples
            """
            if not word_timestamps or not full_text:
                return []

            all_words = list(word_timestamps)

            # ── Punctuation source detection ──
            # Whisper includes punctuation in word.text ("Hello," "world.")
            # Qwen3 ForcedAligner strips it ("Hello" "world") but full_text keeps it.
            # When word timestamps lack punctuation, use full_text tokens as the
            # punctuated parallel array (1:1 mapping — aligner produces one timestamp
            # per whitespace-delimited token in the input text).
            text_tokens = full_text.split()
            sample = all_words[:min(30, len(all_words))]
            has_punct_in_words = any(re.search(r'[.!?,]', w.text) for w in sample)

            if has_punct_in_words or len(text_tokens) != len(all_words):
                # Whisper path (or length mismatch fallback) — word.text has punctuation
                def get_word_text(i):
                    return all_words[i].text.strip()
            else:
                # Qwen3 path — use full_text tokens which preserve punctuation
                def get_word_text(i):
                    return text_tokens[i]

            # ── Step 1: Group word timestamps into sentences by punctuation ──
            # Detect sentence boundaries from punctuated word text (. ! ?).
            # Uses get_word_text() so it works whether punctuation comes from
            # word.text (Whisper) or full_text tokens (Qwen3 ForcedAligner).
            sentence_ranges = []  # (word_start_idx, word_end_idx, text)
            sent_start = 0

            for i, w in enumerate(all_words):
                is_last = i == len(all_words) - 1
                word_text = get_word_text(i)
                ends_sentence = bool(re.search(r'[.!?][\"\')]?$', word_text))

                if ends_sentence or is_last:
                    sent_words = [get_word_text(j) for j in range(sent_start, i + 1)]
                    sent_text = " ".join(sent_words)
                    sentence_ranges.append((sent_start, i, sent_text))
                    sent_start = i + 1

            if not sentence_ranges:
                return []

            # ── Step 2: Detect silence gaps and mark forced cut points ──
            # A silence gap is between word[i].end_time and word[i+1].start_time
            silence_cuts = set()  # word indices AFTER which we force-cut
            if silence_trim > 0:
                for i in range(len(all_words) - 1):
                    gap = all_words[i + 1].start_time - all_words[i].end_time
                    if gap > silence_trim:
                        silence_cuts.add(i)

            # ── Step 3: Group sentences into segments, respecting silence cuts ──
            # A sentence that contains a silence cut within it gets split at the gap.
            segments = []
            group_texts = []
            group_word_start = sentence_ranges[0][0]

            for si, (s_start, s_end, s_text) in enumerate(sentence_ranges):
                is_last_sentence = si == len(sentence_ranges) - 1

                # Check if any silence cut falls WITHIN this sentence's word range
                cuts_in_sentence = sorted(
                    c for c in silence_cuts if s_start <= c < s_end
                )

                if cuts_in_sentence:
                    # Split this sentence at the silence gaps
                    # First, finalize any accumulated group before this sentence
                    if group_texts:
                        grp_start = all_words[group_word_start].start_time
                        grp_end = all_words[s_start - 1].end_time if s_start > 0 else grp_start
                        combined = " ".join(group_texts)
                        if combined.strip():
                            segments.append((grp_start, grp_end, combined.strip()))
                        group_texts = []

                    # Now split this sentence at each silence gap
                    # We'll collect word-index boundaries for sub-chunks
                    chunk_boundaries = [s_start] + [c + 1 for c in cuts_in_sentence] + [s_end + 1]

                    for cb_i in range(len(chunk_boundaries) - 1):
                        chunk_w_start = chunk_boundaries[cb_i]
                        chunk_w_end = chunk_boundaries[cb_i + 1] - 1
                        if chunk_w_end < chunk_w_start:
                            continue
                        # Gather text for these words
                        chunk_word_texts = [get_word_text(w) for w in range(chunk_w_start, chunk_w_end + 1)]
                        chunk_text = " ".join(chunk_word_texts)
                        t_start = all_words[chunk_w_start].start_time
                        t_end = all_words[chunk_w_end].end_time
                        if chunk_text.strip():
                            segments.append((t_start, t_end, chunk_text.strip()))

                    # Next sentence starts a fresh group
                    if not is_last_sentence:
                        group_word_start = sentence_ranges[si + 1][0]
                    continue

                # Check if there's a silence cut between this sentence and the previous group
                if group_texts and s_start > 0:
                    prev_word_idx = s_start - 1
                    if prev_word_idx in silence_cuts:
                        # Force-cut: finalize current group before this sentence
                        grp_start = all_words[group_word_start].start_time
                        grp_end = all_words[prev_word_idx].end_time
                        combined = " ".join(group_texts)
                        if combined.strip():
                            segments.append((grp_start, grp_end, combined.strip()))
                        group_texts = []
                        group_word_start = s_start

                # Add this sentence to the group
                group_texts.append(s_text)
                group_end_idx = s_end

                grp_start_time = all_words[group_word_start].start_time
                grp_end_time = all_words[min(group_end_idx, len(all_words) - 1)].end_time
                grp_duration = grp_end_time - grp_start_time

                # Also check: would continuing to next sentence cross a silence gap?
                next_crosses_silence = False
                if not is_last_sentence:
                    next_s_start = sentence_ranges[si + 1][0]
                    # Check if there's a silence gap between current end and next start
                    for w in range(group_end_idx, next_s_start):
                        if w in silence_cuts:
                            next_crosses_silence = True
                            break

                if grp_duration >= min_duration or is_last_sentence or next_crosses_silence:
                    combined = " ".join(group_texts)
                    if combined.strip():
                        segments.append((grp_start_time, grp_end_time, combined.strip()))
                    group_texts = []
                    if not is_last_sentence:
                        group_word_start = sentence_ranges[si + 1][0]

            # ── Step 4: Break oversized segments at commas ──
            # Uses word timestamp text directly to find comma positions,
            # avoiding word-count mismatches from text parsing.
            if max_duration and max_duration > 0:
                final_segments = []
                for seg_start, seg_end, seg_text in segments:
                    seg_duration = seg_end - seg_start
                    if seg_duration <= max_duration:
                        final_segments.append((seg_start, seg_end, seg_text))
                        continue

                    # Find word indices belonging to this segment
                    seg_word_indices = [i for i, w in enumerate(all_words)
                                        if w.start_time >= seg_start - 0.01
                                        and w.end_time <= seg_end + 0.01]

                    if not seg_word_indices:
                        final_segments.append((seg_start, seg_end, seg_text))
                        continue

                    # Find comma split points (words whose text ends with ',')
                    comma_indices = [wi for wi in seg_word_indices
                                     if get_word_text(wi).endswith(',')]

                    if not comma_indices:
                        final_segments.append((seg_start, seg_end, seg_text))
                        continue

                    # Accumulate words, cutting at commas when duration thresholds met
                    sub_start_idx = seg_word_indices[0]

                    for ci, comma_wi in enumerate(comma_indices):
                        is_last_comma = ci == len(comma_indices) - 1
                        sub_start_time = all_words[sub_start_idx].start_time
                        sub_end_time = all_words[comma_wi].end_time
                        sub_dur = sub_end_time - sub_start_time

                        should_cut = sub_dur >= min_duration and (sub_dur >= max_duration or is_last_comma)
                        if should_cut:
                            sub_text = " ".join(
                                get_word_text(j)
                                for j in range(sub_start_idx, comma_wi + 1)
                            )
                            if sub_text.strip():
                                final_segments.append((sub_start_time, sub_end_time, sub_text.strip()))
                            sub_start_idx = comma_wi + 1

                    # Remaining words after last cut
                    last_word_idx = seg_word_indices[-1]
                    if sub_start_idx <= last_word_idx:
                        sub_text = " ".join(
                            get_word_text(j)
                            for j in range(sub_start_idx, last_word_idx + 1)
                        )
                        sub_start_time = all_words[sub_start_idx].start_time
                        sub_end_time = all_words[last_word_idx].end_time
                        if sub_text.strip():
                            final_segments.append((sub_start_time, sub_end_time, sub_text.strip()))

                segments = final_segments

            # ── Step 5: Trim leading/trailing silence on each segment ──
            pad = 0.15  # small padding to keep around speech
            trimmed = []
            for seg_start, seg_end, seg_text in segments:
                seg_words = [w for w in all_words
                             if w.start_time >= seg_start - 0.05
                             and w.end_time <= seg_end + 0.05]
                if seg_words:
                    first_start = seg_words[0].start_time
                    last_end = seg_words[-1].end_time
                    if first_start - seg_start > silence_trim:
                        seg_start = max(0, first_start - pad)
                    if seg_end - last_end > silence_trim:
                        seg_end = last_end + pad
                trimmed.append((seg_start, seg_end, seg_text))
            segments = trimmed

            # ── Step 6: Discard clips that are too short ──
            if discard_under and discard_under > 0:
                kept = []
                discarded = 0
                for seg_start, seg_end, seg_text in segments:
                    if seg_end - seg_start >= discard_under:
                        kept.append((seg_start, seg_end, seg_text))
                    else:
                        discarded += 1
                if discarded:
                    print(f"  {_DIM}Discarded {discarded} clip(s) under {discard_under}s{_RESET}")
                segments = kept

            return segments

        def auto_split_audio_handler(clip_prefix, audio_file, folder, language, engine, asr_size,
                                     split_min=4.0, split_max=20.0, silence_trim=1.0,
                                     discard_under=1.0, progress=gr.Progress()):
            """Auto-split a long audio file into training clips using word-level timestamps.

            Supports two engines:
            - Qwen3 ASR: Transcribes with Qwen3 ASR, then aligns with Qwen3-ForcedAligner
            - Whisper: Transcribes with word_timestamps=True (built-in alignment)

            Splits at natural sentence boundaries and saves numbered clips with transcripts.
            """
            if audio_file is None:
                return "Load an audio file into the editor first (drag and drop).", gr.update()
            if not folder or folder in ("(No folders)", "(Select Dataset)"):
                return "Select a dataset folder first.", gr.update()
            if not clip_prefix:
                return "No clip name provided.", gr.update()

            try:
                import numpy as np

                # Check audio duration (ForcedAligner recommended limit: 5 minutes)
                info = sf.info(audio_file)
                duration = info.duration
                bypass_limit = _user_config.get("bypass_split_limit", False)
                if engine == "Qwen3 ASR" and duration > 300:
                    if not bypass_limit:
                        return ("Audio exceeds the recommended 5 minute limit for Qwen3 ASR auto-split. "
                                "You can enable extended splitting in Settings, or use Whisper instead."), gr.update()
                    else:
                        print(f"{_YELLOW}[!] WARNING: Audio is {duration:.0f}s — exceeds 5 min recommended limit{_RESET}")
                        print(f"{_YELLOW}    Extended splitting enabled — alignment accuracy may be reduced{_RESET}")
                if duration < 3:
                    return "Audio too short for auto-splitting (minimum 3 seconds).", gr.update()

                lang_option = language if language and language != "Auto-detect" else None

                if engine == "Whisper":
                    # --- Whisper path: single call with word_timestamps ---
                    if not asr_manager.whisper_available:
                        return "Whisper not available. Use Qwen3 ASR instead.", gr.update()

                    progress(0.05, desc=f"Loading Whisper ({asr_size or 'Medium'})...")
                    model = asr_manager.get_whisper(size=asr_size or "Medium")

                    progress(0.15, desc="Transcribing with word timestamps...")
                    whisper_options = {"word_timestamps": True}
                    if lang_option:
                        lang_code = {
                            "English": "en", "Chinese": "zh", "Japanese": "ja",
                            "Korean": "ko", "German": "de", "French": "fr",
                            "Russian": "ru", "Portuguese": "pt", "Spanish": "es",
                            "Italian": "it"
                        }.get(lang_option, None)
                        if lang_code:
                            whisper_options["language"] = lang_code

                    result = model.transcribe(audio_file, **whisper_options)
                    full_text = result["text"].strip()

                    if not full_text:
                        return "No speech detected in the audio.", gr.update()

                    print(f"\n{_CYAN}{'=' * 55}")
                    print(" Auto-Split  |  Engine: Whisper")
                    print(f"{'=' * 55}{_RESET}")
                    print(f"{_DIM}Transcript: {full_text[:150]}{'...' if len(full_text) > 150 else ''}{_RESET}")

                    # Extract word timestamps from Whisper segments
                    # Whisper returns segments[].words[] with {word, start, end}
                    class WhisperWordTimestamp:
                        def __init__(self, word, start, end):
                            self.text = word
                            self.start_time = start
                            self.end_time = end

                    word_timestamps = []
                    for seg in result.get("segments", []):
                        for w in seg.get("words", []):
                            word_timestamps.append(WhisperWordTimestamp(
                                w["word"].strip(), w["start"], w["end"]
                            ))

                    if not word_timestamps:
                        return "Whisper produced no word timestamps.", gr.update()

                    print(f"{_GREEN}[OK]{_RESET} {len(word_timestamps)} word timestamps")
                    for j, wt in enumerate(word_timestamps[:5]):
                        print(f"  {_DIM}{j + 1}. '{wt.text}' {wt.start_time:.2f}s - {wt.end_time:.2f}s{_RESET}")
                    if len(word_timestamps) > 5:
                        print(f"  {_DIM}... and {len(word_timestamps) - 5} more{_RESET}")

                else:
                    # --- Qwen3 ASR path: transcribe + forced aligner ---
                    progress(0.05, desc=f"Loading Qwen3 ASR ({asr_size})...")
                    try:
                        model = asr_manager.get_qwen3_asr(size=asr_size)
                    except Exception as e:
                        return f"Failed to load Qwen3 ASR: {str(e)}", gr.update()

                    progress(0.10, desc="Transcribing audio...")
                    asr_result = model.transcribe(audio_file, language=lang_option)
                    full_text = asr_result["text"].strip()
                    detected_language = asr_result.get("language", None)

                    if not full_text:
                        return "No speech detected in the audio.", gr.update()

                    print(f"\n{_CYAN}{'=' * 55}")
                    print(f" Auto-Split  |  Engine: Qwen3 ASR ({asr_size})")
                    print(f"{'=' * 55}{_RESET}")
                    print(f"{_DIM}Transcript: {full_text[:150]}{'...' if len(full_text) > 150 else ''}{_RESET}")
                    if detected_language:
                        print(f"{_DIM}Language: {detected_language}{_RESET}")

                    # Use detected language for alignment if user didn't specify one
                    align_language = lang_option or detected_language

                    # Load forced aligner and get word timestamps
                    progress(0.25, desc="Loading forced aligner...")
                    try:
                        aligner = asr_manager.get_qwen3_forced_aligner()
                    except Exception as e:
                        return f"Failed to load forced aligner: {str(e)}", gr.update()

                    progress(0.35, desc="Aligning words to audio...")
                    try:
                        align_results = aligner.align(
                            audio=audio_file,
                            text=full_text,
                            language=align_language,
                        )
                        word_timestamps = align_results[0] if align_results else []
                    except Exception as e:
                        asr_manager.unload_forced_aligner()
                        return f"Alignment failed: {str(e)}", gr.update()

                    if not word_timestamps:
                        asr_manager.unload_forced_aligner()
                        return "Alignment produced no word timestamps.", gr.update()

                    print(f"{_GREEN}[OK]{_RESET} {len(word_timestamps)} word timestamps")
                    for j, wt in enumerate(word_timestamps[:5]):
                        print(f"  {_DIM}{j + 1}. '{wt.text}' {wt.start_time:.2f}s - {wt.end_time:.2f}s{_RESET}")
                    if len(word_timestamps) > 5:
                        print(f"  {_DIM}... and {len(word_timestamps) - 5} more{_RESET}")

                # Step 3: Split into segments at sentence boundaries
                progress(0.50, desc="Splitting into segments...")
                segments = split_into_segments(full_text, word_timestamps,
                                               min_duration=split_min,
                                               max_duration=split_max,
                                               silence_trim=silence_trim,
                                               discard_under=discard_under)

                if not segments:
                    if engine == "Qwen3 ASR":
                        asr_manager.unload_forced_aligner()
                    return "Could not identify any segments to split.", gr.update()

                print(f"{_GREEN}[OK]{_RESET} Split into {len(segments)} segments")
                print()

                # Step 4: Read source audio and save clips
                progress(0.55, desc="Reading source audio...")
                data, sr = sf.read(audio_file, dtype='float32')

                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                base_dir = DATASETS_DIR / folder
                base_dir.mkdir(parents=True, exist_ok=True)

                # Collect existing stems to avoid name collisions
                existing_stems = {f.stem for f in base_dir.glob("*.wav")}

                def _make_clip_name(prefix, text):
                    """Build a clip name from the first few words of the transcript.

                    Rules: strip punctuation, take words until 5 words or 25 chars
                    is reached, join with underscores, prepend prefix if given.
                    """
                    words = re.sub(r"[^\w\s]", "", text).split()
                    name_part = ""
                    for w in words:
                        candidate = f"{name_part}_{w}" if name_part else w
                        if len(candidate) > 25 or len(candidate.split("_")) > 5:
                            break
                        name_part = candidate
                    if not name_part:
                        name_part = "clip"
                    return f"{prefix}_{name_part}" if prefix else name_part

                saved_count = 0
                for i, (start_time, end_time, text) in enumerate(segments):
                    base_name = _make_clip_name(clip_prefix, text)
                    clip_name = base_name
                    n = 2
                    while clip_name in existing_stems:
                        clip_name = f"{base_name}_{n}"
                        n += 1
                    existing_stems.add(clip_name)

                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)

                    # Clamp to valid range
                    start_sample = max(0, start_sample)
                    end_sample = min(len(data), end_sample)

                    if end_sample <= start_sample:
                        continue

                    clip_data = data[start_sample:end_sample]

                    # Save audio clip (WAV 16-bit PCM)
                    clip_path = base_dir / f"{clip_name}.wav"
                    sf.write(str(clip_path), clip_data, sr, subtype='PCM_16')

                    # Save transcript
                    txt_path = base_dir / f"{clip_name}.txt"
                    txt_path.write_text(text.strip(), encoding="utf-8")

                    saved_count += 1
                    clip_dur = end_time - start_time
                    print(f"  {_GREEN}>{_RESET} {clip_name}  {_DIM}{clip_dur:.1f}s{_RESET}  {text[:55]}{'...' if len(text) > 55 else ''}")
                    progress(0.55 + (0.40 * (i + 1) / len(segments)),
                             desc=f"Saving clip {i + 1}/{len(segments)}...")

                # Step 5: Unload aligner to free VRAM (Qwen3 only)
                if engine == "Qwen3 ASR":
                    asr_manager.unload_forced_aligner()

                print(f"\n{_GREEN}{'=' * 55}")
                print(f" Done! {saved_count} clips saved to {folder}/")
                print(f"{'=' * 55}{_RESET}\n")

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()

                # Refresh file list
                updated_files = get_dataset_files(folder)
                status_msg = (f"Created {saved_count} clips from {duration:.1f}s audio "
                              f"(saved to {folder}/)")
                if engine == "Qwen3 ASR" and duration > 300:
                    status_msg += ("\n⚠ Extended split mode: audio exceeded the recommended "
                                   "5 min limit. Review clips for alignment accuracy.")
                return status_msg, updated_files

            except Exception as e:
                import traceback
                print(f"{_RED}[Auto-Split] Error:\n{traceback.format_exc()}{_RESET}")
                # Clean up aligner on error (Qwen3 only)
                try:
                    if engine == "Qwen3 ASR":
                        asr_manager.unload_forced_aligner()
                except Exception:
                    pass
                return f"Error: {str(e)}", gr.update()

        # ====================================================
        # Wire up events
        # ====================================================

        # --- Mode switching ---
        components['prep_data_type'].change(
            on_mode_change,
            inputs=[components['prep_data_type'], components['transcribe_model']],
            outputs=[
                components['samples_col'],
                components['datasets_col'],
                components['clear_cache_btn'],
                components['batch_col'],
                components['prep_file_input'],
                components['prep_audio_editor'],    # visibility
                components['prep_audio_editor'],    # value (clear)
                components['transcription_output'],
                components['prep_status'],
                components['existing_sample_info'],
                components['editor_heading'],
                components['auto_split_accordion'],
                components['auto_split_btn'],
            ]
        )

        # --- Sample lister events ---
        components['sample_lister'].change(
            on_sample_selection_change,
            inputs=[components['sample_lister']],
            outputs=[components['prep_audio_editor'], components['prep_file_input'],
                     components['transcription_output'], components['existing_sample_info']]
        )

        components['sample_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#prep-audio-editor .play-pause-button'); if (btn) btn.click(); }, 150); }"
        )

        # --- Dataset lister events ---
        components['finetune_folder_dropdown'].change(
            lambda folder: get_dataset_files(folder),
            inputs=[components['finetune_folder_dropdown']],
            outputs=[components['dataset_lister']]
        )

        # --- Create dataset folder ---
        create_folder_modal_js = show_input_modal_js(
            title="Create Dataset Folder",
            message="Enter a name for the new dataset folder:",
            placeholder="e.g., Interview, ReadingSession, Podcast",
            context="create_folder_"
        )
        components['create_folder_btn'].click(
            fn=None,
            js=f"() => {{ const openModal = {create_folder_modal_js}; openModal(''); }}"
        )

        # --- Delete dataset folder ---
        delete_folder_modal_js = show_confirmation_modal_js(
            title="Delete Dataset Folder?",
            message="This will permanently delete the selected folder and ALL its contents (audio files and transcripts). This action cannot be undone.",
            confirm_button_text="Delete",
            context="delete_folder_"
        )
        components['delete_folder_btn'].click(
            fn=None,
            js=f"() => {{ const fn = {delete_folder_modal_js}; return fn(); }}"
        )

        components['dataset_lister'].change(
            on_dataset_selection_change,
            inputs=[components['finetune_folder_dropdown'], components['dataset_lister']],
            outputs=[components['prep_audio_editor'], components['prep_file_input'],
                     components['transcription_output'], components['existing_sample_info']]
        )

        components['dataset_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#prep-audio-editor .play-pause-button'); if (btn) btn.click(); }, 150); }"
        )

        # --- Auto-refresh when tab is selected (preserve selections) ---
        def on_prep_tab_select(data_type, folder, sample_lister_value, dataset_lister_value):
            """Refresh all file lists when Prep Audio tab is selected."""
            # Refresh samples with preserved selection
            new_sample_files = get_sample_choices()
            prev_sample_selected = []
            if sample_lister_value:
                prev = sample_lister_value.get("selected", [])
                new_names = set(new_sample_files)
                prev_sample_selected = [s for s in prev if s in new_names]
            sample_update = {"files": [{"name": f, "date": ""} for f in new_sample_files], "selected": prev_sample_selected}

            # Refresh folder dropdown (preserve current folder selection)
            folder_choices = ["(Select Dataset)"] + get_dataset_folders()
            if folder and folder in folder_choices:
                folder_update = gr.update(choices=folder_choices, value=folder)
            else:
                folder_update = gr.update(choices=folder_choices)

            # Refresh dataset files with preserved selection
            if is_dataset_mode(data_type) and folder and folder not in ("(No folders)", "(Select Dataset)"):
                new_dataset_files = get_dataset_files(folder)
                prev_ds_selected = []
                if dataset_lister_value:
                    prev_ds = dataset_lister_value.get("selected", [])
                    new_ds_names = set(new_dataset_files) if isinstance(new_dataset_files, list) else set()
                    prev_ds_selected = [s for s in prev_ds if s in new_ds_names]
                if isinstance(new_dataset_files, list):
                    dataset_update = {"files": [{"name": f, "date": ""} for f in new_dataset_files], "selected": prev_ds_selected}
                else:
                    dataset_update = new_dataset_files
            else:
                dataset_update = gr.update()
            return sample_update, folder_update, dataset_update

        # Set correct mode visibility on tab select (all sections start visible=True for DOM rendering)
        def toggle_prep_mode_visibility(data_type, transcribe_model):
            """Set correct visibility for mode-dependent sections without clearing editor state."""
            is_ds = is_dataset_mode(data_type)
            engine, _ = parse_asr_model(transcribe_model)
            show_auto_split = is_ds and engine in ("Qwen3 ASR", "Whisper")
            return (
                gr.update(visible=not is_ds),        # samples_col
                gr.update(visible=is_ds),            # datasets_col
                gr.update(visible=is_ds),            # batch_col
                gr.update(visible=show_auto_split),  # auto_split_accordion
                gr.update(visible=show_auto_split),  # auto_split_btn
            )

        components['prep_tab'].select(
            on_prep_tab_select,
            inputs=[components['prep_data_type'], components['finetune_folder_dropdown'],
                    components['sample_lister'], components['dataset_lister']],
            outputs=[components['sample_lister'], components['finetune_folder_dropdown'], components['dataset_lister']]
        ).then(
            toggle_prep_mode_visibility,
            inputs=[components['prep_data_type'], components['transcribe_model']],
            outputs=[
                components['samples_col'], components['datasets_col'],
                components['batch_col'],
                components['auto_split_accordion'], components['auto_split_btn'],
            ]
        )

        # --- Delete (mode-aware JS context) ---
        sample_delete_js = show_confirmation_modal_js(
            title="Delete Sample(s)?",
            message="This will permanently delete the selected sample(s), metadata, and cached files. This action cannot be undone.",
            confirm_button_text="Delete",
            context="sample_"
        )
        dataset_delete_js = show_confirmation_modal_js(
            title="Delete Dataset Item(s)?",
            message="This will permanently delete the selected audio file(s) and transcript(s). This action cannot be undone.",
            confirm_button_text="Delete",
            context="finetune_"
        )
        delete_js = f"""
        (dataType) => {{
            if (dataType === 'Datasets') {{
                const fn = {dataset_delete_js};
                return fn();
            }} else {{
                const fn = {sample_delete_js};
                return fn();
            }}
        }}
        """

        components['delete_btn'].click(
            fn=None,
            inputs=[components['prep_data_type']],
            js=delete_js
        )

        confirm_trigger.change(
            delete_handler,
            inputs=[confirm_trigger, components['prep_data_type'],
                    components['sample_lister'], components['dataset_lister'],
                    components['finetune_folder_dropdown']],
            outputs=[components['prep_status'], components['sample_lister'],
                     components['dataset_lister'], components['finetune_folder_dropdown']]
        )

        # --- Clear cache (samples only) ---
        components['clear_cache_btn'].click(
            clear_sample_cache_handler,
            inputs=[components['sample_lister']],
            outputs=[components['prep_status'], components['existing_sample_info']]
        )

        # --- Clear button ---
        components['clear_btn'].click(
            lambda: (gr.update(visible=True, value=None), gr.update(visible=False), None, ""),
            outputs=[components['prep_file_input'], components['prep_audio_editor'],
                     components['prep_audio_editor'], components['prep_status']]
        )

        # --- File input (load new audio/video - works in both modes) ---
        components['prep_file_input'].change(
            on_prep_audio_load_handler,
            inputs=[components['prep_file_input']],
            outputs=[components['prep_audio_editor'], components['prep_status']]
        ).then(
            lambda audio: (
                gr.update(visible=True),
                gr.update(visible=False)
            ) if audio is not None else (
                gr.update(),
                gr.update()
            ),
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor'], components['prep_file_input']]
        )

        # --- Audio processing (shared) ---
        components['normalize_btn'].click(
            normalize_audio,
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor'], components['prep_status']]
        )

        components['mono_btn'].click(
            convert_to_mono,
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor'], components['prep_status']]
        )

        components['clean_btn'].click(
            clean_audio,
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor'], components['prep_status']]
        )

        # --- Transcribe (shared) ---
        components['transcribe_btn'].click(
            transcribe_audio_handler,
            inputs=[components['prep_audio_editor'], components['whisper_language'],
                    components['transcribe_model']],
            outputs=[components['transcription_output'], components['prep_status']]
        )

        # --- Save button (unified for both modes) ---
        # JS always opens input modal with mode-appropriate context
        sample_modal_js = show_input_modal_js(
            title="Save Voice Sample",
            message="Enter a name for this voice sample:",
            placeholder="e.g., MyVoice, Female-Accent, John-Doe",
            context="save_sample_"
        )
        dataset_modal_js = show_input_modal_js(
            title="Save Dataset Clip",
            message="Enter a filename for this clip:",
            placeholder="e.g., clip_001, interview_01",
            context="save_dataset_"
        )

        def get_existing_files(data_type, folder):
            """Return JSON list of existing file names (without extension) for overwrite detection."""
            import json as json_mod
            if is_dataset_mode(data_type):
                base_dir = get_dataset_dir(folder)
                if base_dir.exists():
                    names = [f.stem for f in base_dir.glob("*.wav")]
                    return json_mod.dumps(names)
                return "[]"
            else:
                names = [f.stem for f in SAMPLES_DIR.glob("*.wav")]
                return json_mod.dumps(names)

        save_js = f"""
        (existingFilesJson, dataType, audio, transcription, folder, datasetLister, sampleLister, fileInput) => {{
            // Set existing files for overwrite detection
            try {{
                window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
            }} catch(e) {{
                window.inputModalExistingFiles = [];
            }}

            let name = '';
            // Prefer dragged-in filename if present
            if (fileInput) {{
                let fname = '';
                if (typeof fileInput === 'string') {{
                    fname = fileInput;
                }} else if (fileInput.orig_name) {{
                    fname = fileInput.orig_name;
                }} else if (fileInput.name) {{
                    fname = fileInput.name;
                }}
                if (fname) {{
                    name = fname.split(/[\\\\/]/).pop().replace(/\\.[^.]+$/, '');
                }}
            }}
            if (dataType === 'Samples') {{
                // Fall back to selected sample name
                if (!name && sampleLister && sampleLister.selected && sampleLister.selected.length === 1) {{
                    name = sampleLister.selected[0].replace(/\\.wav$/i, '');
                }}
                const openModal = {sample_modal_js};
                openModal(name);
            }} else {{
                // Fall back to selected dataset filename
                if (!name && datasetLister && datasetLister.selected && datasetLister.selected.length === 1) {{
                    name = datasetLister.selected[0].replace(/\\.[^.]+$/, '');
                }}
                const openModal = {dataset_modal_js};
                openModal(name);
            }}
        }}
        """

        components['save_btn'].click(
            fn=get_existing_files,
            inputs=[components['prep_data_type'], components['finetune_folder_dropdown']],
            outputs=[components['existing_files_json']],
        ).then(
            fn=None,
            inputs=[components['existing_files_json'],
                    components['prep_data_type'], components['prep_audio_editor'],
                    components['transcription_output'], components['finetune_folder_dropdown'],
                    components['dataset_lister'], components['sample_lister'],
                    components['prep_file_input']],
            js=save_js
        )

        # Input modal handlers (save sample/dataset, auto-split, create folder)
        input_trigger.change(
            handle_input_modal,
            inputs=[input_trigger, components['prep_audio_editor'],
                    components['transcription_output'],
                    components['finetune_folder_dropdown'],
                    components['whisper_language'],
                    components['transcribe_model'],
                    components['split_min'],
                    components['split_max'],
                    components['silence_trim'],
                    components['discard_under']],
            outputs=[components['prep_status'], components['sample_lister'],
                     components['dataset_lister'], components['finetune_folder_dropdown']]
        )

        # --- Batch transcribe (dataset mode only) ---
        components['batch_transcribe_btn'].click(
            batch_transcribe_handler,
            inputs=[components['finetune_folder_dropdown'],
                    components['batch_replace_existing'],
                    components['whisper_language'], components['transcribe_model']],
            outputs=[components['prep_status']]
        )

        # --- Auto-split audio (dataset mode, Qwen3 ASR only) ---
        auto_split_modal_js = show_input_modal_js(
            title="Auto-Split Audio",
            message="Enter a name for the clips (e.g., Interview, Reading, Podcast):",
            placeholder="e.g., Interview, Reading, Podcast",
            context="auto_split_"
        )
        components['auto_split_btn'].click(
            fn=None,
            js=f"() => {{ const openModal = {auto_split_modal_js}; openModal(''); }}"
        )

        # --- Save preferences ---
        def on_transcribe_model_change(model, data_type):
            save_preference("transcribe_model", model)
            engine, _ = parse_asr_model(model)
            is_ds = is_dataset_mode(data_type)
            show_auto_split = is_ds and engine in ("Qwen3 ASR", "Whisper")
            show_lang = engine in ("Qwen3 ASR", "Whisper")
            return (
                gr.update(visible=show_lang),
                gr.update(visible=show_auto_split),
                gr.update(visible=show_auto_split),
            )

        components['transcribe_model'].change(
            on_transcribe_model_change,
            inputs=[components['transcribe_model'], components['prep_data_type']],
            outputs=[components['whisper_language'],
                     components['auto_split_accordion'], components['auto_split_btn']]
        )

        components['whisper_language'].change(
            lambda x: save_preference("whisper_language", x),
            inputs=[components['whisper_language']],
            outputs=[]
        )

        # --- Save split settings ---
        components['split_min'].change(
            lambda x: save_preference("split_min", x),
            inputs=[components['split_min']], outputs=[]
        )
        components['split_max'].change(
            lambda x: save_preference("split_max", x),
            inputs=[components['split_max']], outputs=[]
        )
        components['silence_trim'].change(
            lambda x: save_preference("silence_trim", x),
            inputs=[components['silence_trim']], outputs=[]
        )
        components['discard_under'].change(
            lambda x: save_preference("discard_under", x),
            inputs=[components['discard_under']], outputs=[]
        )

        # Set correct initial visibility on page load (tab.select doesn't fire for the first tab)
        app = shared_state.get('app')
        if app:
            app.load(
                toggle_prep_mode_visibility,
                inputs=[components['prep_data_type'], components['transcribe_model']],
                outputs=[
                    components['samples_col'], components['datasets_col'],
                    components['batch_col'],
                    components['auto_split_accordion'], components['auto_split_btn'],
                ]
            )


# Export for tab registry
get_tool_class = lambda: PrepSamplesTool


if __name__ == "__main__":
    """Standalone testing of Prep Samples tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(PrepSamplesTool, port=7865, title="Prep Samples - Standalone")
