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
import shutil
import soundfile as sf
from pathlib import Path
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.asr_manager import get_asr_manager
from gradio_filelister import FileLister


class PrepSamplesTool(Tool):
    """Unified Prep Samples + Dataset tool."""

    config = ToolConfig(
        name="Prep Samples",
        module_name="tool_prep_audio",
        description="Prepare and manage voice samples and datasets",
        enabled=True,
        category="preparation",
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Prep Samples tool UI."""
        components = {}

        format_help_html = shared_state["format_help_html"]
        get_sample_choices = shared_state["get_sample_choices"]
        get_dataset_folders = shared_state["get_dataset_folders"]
        get_dataset_files = shared_state["get_dataset_files"]
        _user_config = shared_state["_user_config"]
        LANGUAGES = shared_state["LANGUAGES"]
        WHISPER_AVAILABLE = shared_state["WHISPER_AVAILABLE"]
        DEEPFILTER_AVAILABLE = shared_state["DEEPFILTER_AVAILABLE"]

        # Let's hide dataset if train model is off
        train_model_enabled = _user_config.get("enabled_tools", {}).get(
            "Train Model", True
        )

        with gr.TabItem("Prep Audio Samples"):
            if train_model_enabled is True:
                gr.Markdown(
                    "Prepare audio samples for voice cloning or finetuning datasets."
                )
            else:
                gr.Markdown("Prepare audio samples for voice cloning.")

            with gr.Row():
                # Left column - File browser

                with gr.Column(scale=1):
                    if train_model_enabled is True:
                        gr.Markdown("### Select Samples or Dataset")

                    components["prep_data_type"] = gr.Radio(
                        choices=["Samples", "Datasets"],
                        value="Samples",
                        show_label=False,
                        visible=train_model_enabled,
                    )

                    # --- Samples mode ---
                    with gr.Column(visible=True) as samples_col:
                        gr.Markdown("### Audio Samples")
                        components["sample_lister"] = FileLister(
                            value=get_sample_choices(),
                            height=250,
                            show_footer=False,
                            interactive=True,
                        )
                    components["samples_col"] = samples_col

                    # --- Datasets mode ---
                    with gr.Column(visible=False) as datasets_col:
                        gr.Markdown("### Dataset Files")
                        components["finetune_folder_dropdown"] = gr.Dropdown(
                            choices=["(Select Dataset)"] + get_dataset_folders(),
                            value="(Select Dataset)",
                            label="Dataset Folder",
                            info="Subfolders in datasets",
                            interactive=True,
                        )
                        components["refresh_folder_btn"] = gr.Button(
                            "Refresh Folders", size="sm"
                        )
                        components["dataset_lister"] = FileLister(
                            value=[],
                            height=250,
                            show_footer=False,
                            interactive=True,
                        )
                    components["datasets_col"] = datasets_col

                    # --- Shared buttons ---
                    with gr.Row():
                        components["refresh_preview_btn"] = gr.Button(
                            "Refresh", size="sm"
                        )

                    with gr.Row():
                        components["clear_cache_btn"] = gr.Button(
                            "Clear Cache", size="sm"
                        )
                        components["delete_btn"] = gr.Button(
                            "Delete", size="sm", variant="stop"
                        )

                    components["existing_sample_info"] = gr.Textbox(
                        label="Info", interactive=False, lines=3
                    )

                    gr.Markdown("### Transcription Settings")

                    available_models = ["VibeVoice ASR"]
                    if WHISPER_AVAILABLE:
                        available_models.append("Whisper")

                    default_model = _user_config.get(
                        "transcribe_model", "VibeVoice ASR"
                    )
                    if default_model not in available_models:
                        default_model = available_models[0]

                    components["transcribe_model"] = gr.Radio(
                        choices=available_models,
                        value=default_model,
                        show_label=False,
                        info="Choose transcription engine",
                    )

                    components["whisper_language"] = gr.Dropdown(
                        choices=["Auto-detect"] + LANGUAGES[1:],
                        value=_user_config.get("whisper_language", "Auto-detect"),
                        label="Language",
                        visible=(default_model == "Whisper"),
                    )

                # Right column - Audio editing
                with gr.Column(scale=2):
                    components["editor_heading"] = gr.Markdown(
                        "### Add or Edit Audio Sample <small>(Drag and drop audio or video files)</small>"
                    )

                    components["prep_file_input"] = gr.File(
                        label="Audio or Video File",
                        type="filepath",
                        file_types=["audio", "video"],
                        interactive=True,
                    )

                    components["prep_audio_editor"] = gr.Audio(
                        label="Audio Editor (Use Trim icon to edit)",
                        type="filepath",
                        interactive=True,
                        visible=False,
                        elem_id="prep-audio-editor",
                    )

                    with gr.Row():
                        components["clear_btn"] = gr.Button("Clear", scale=1, size="sm")
                        components["clean_btn"] = gr.Button(
                            "AI Denoise",
                            scale=2,
                            size="sm",
                            variant="secondary",
                            visible=DEEPFILTER_AVAILABLE,
                        )
                        components["normalize_btn"] = gr.Button(
                            "Normalize Volume", scale=2, size="sm"
                        )
                        components["mono_btn"] = gr.Button(
                            "Convert to Mono", scale=2, size="sm"
                        )

                    gr.Markdown("### Reference Text")
                    components["transcription_output"] = gr.Textbox(
                        label="Text",
                        lines=4,
                        max_lines=10,
                        interactive=True,
                        placeholder="Transcription will appear here, or enter/edit text manually...",
                    )

                    with gr.Row():
                        components["transcribe_btn"] = gr.Button(
                            "Transcribe Audio", variant="primary"
                        )
                        components["save_btn"] = gr.Button(
                            "Save Sample", variant="primary"
                        )

                    # Dataset-only: batch transcribe
                    with gr.Column(visible=False) as batch_col:
                        components["batch_transcribe_btn"] = gr.Button(
                            "Batch Transcribe All Clips", variant="primary", size="lg"
                        )
                        components["batch_replace_existing"] = gr.Checkbox(
                            label="Replace existing transcripts", value=False
                        )
                    components["batch_col"] = batch_col

                    components["prep_status"] = gr.Textbox(
                        label="Status", interactive=False, lines=1, max_lines=15
                    )

            return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Prep Samples tab events."""

        # Shared utilities
        get_sample_choices = shared_state["get_sample_choices"]
        load_sample_details = shared_state["load_sample_details"]
        get_prompt_cache_path = shared_state["get_prompt_cache_path"]
        get_luxtts_prompt_cache_files = shared_state.get(
            "get_luxtts_prompt_cache_files"
        )
        play_completion_beep = shared_state.get("play_completion_beep")
        save_preference = shared_state["save_preference"]
        show_confirmation_modal_js = shared_state["show_confirmation_modal_js"]
        show_input_modal_js = shared_state["show_input_modal_js"]
        confirm_trigger = shared_state["confirm_trigger"]
        input_trigger = shared_state["input_trigger"]
        SAMPLES_DIR = shared_state["SAMPLES_DIR"]
        DATASETS_DIR = shared_state["DATASETS_DIR"]
        get_dataset_folders = shared_state["get_dataset_folders"]
        get_dataset_files = shared_state["get_dataset_files"]
        tts_manager = shared_state.get("tts_manager")

        # Audio utilities
        is_audio_file = shared_state["is_audio_file"]
        is_video_file = shared_state["is_video_file"]
        extract_audio_from_video = shared_state["extract_audio_from_video"]
        get_audio_duration = shared_state["get_audio_duration"]
        format_time = shared_state["format_time"]
        normalize_audio = shared_state["normalize_audio"]
        convert_to_mono = shared_state["convert_to_mono"]
        clean_audio = shared_state["clean_audio"]

        # ASR manager (singleton)
        asr_manager = get_asr_manager()

        # ===== Helpers =====

        def is_dataset_mode(data_type):
            return data_type == "Datasets"

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

        def on_mode_change(data_type):
            """Switch between Samples and Datasets mode."""
            is_ds = is_dataset_mode(data_type)
            heading = (
                "### Adjust Audio Sample"
                if is_ds
                else "### Add or Edit Audio Sample <small>(Drag and drop audio or video files)</small>"
            )
            return (
                gr.update(visible=not is_ds),  # samples_col
                gr.update(visible=is_ds),  # datasets_col
                gr.update(visible=not is_ds),  # clear_btn
                gr.update(visible=not is_ds),  # clear_cache_btn
                gr.update(visible=not is_ds),  # refresh_preview_btn
                gr.update(visible=is_ds),  # batch_col
                gr.update(visible=not is_ds),  # prep_file_input
                gr.update(visible=is_ds),  # prep_audio_editor visibility
                None,  # prep_audio_editor value
                "",  # transcription_output
                "",  # prep_status
                "",  # existing_sample_info
                heading,  # editor_heading
            )

        # ===== Selection handlers =====

        def on_sample_selection_change(lister_value):
            """Handle sample selection - load into editor and transcription."""
            sample_name = get_selected_sample_name(lister_value)
            if not sample_name:
                return gr.update(visible=False), gr.update(visible=True), None, "", ""

            audio_path, ref_text, info_text = load_sample_details(sample_name)
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                audio_path,
                ref_text,
                info_text,
            )

        def on_dataset_selection_change(folder, lister_value):
            """Load audio and transcript for selected dataset item."""
            filename = get_selected_filename(lister_value)
            if not filename:
                return gr.update(visible=True), None, "", ""

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
                return gr.update(visible=True), str(audio_path), transcript, ""
            return gr.update(visible=True), None, transcript, ""

        # ===== File load (samples mode) =====

        def on_prep_audio_load_handler(audio_file):
            """When audio/video is loaded via file input."""
            if audio_file is None:
                return None, "No file loaded"
            try:
                if is_video_file(audio_file):
                    print(f"Video file detected: {Path(audio_file).name}")
                    audio_path, message = extract_audio_from_video(audio_file)
                    if audio_path:
                        duration = get_audio_duration(audio_path)
                        info = f"[VIDEO] Audio extracted\nDuration: {format_time(duration)} ({duration:.2f}s)"
                        return audio_path, info
                    return None, message
                elif is_audio_file(audio_file):
                    duration = get_audio_duration(audio_file)
                    return (
                        audio_file,
                        f"Duration: {format_time(duration)} ({duration:.2f}s)",
                    )
                else:
                    return None, "Unsupported file type"
            except Exception as e:
                return None, f"Error: {str(e)}"

        # ===== Delete handler (unified) =====

        def delete_handler(action, data_type, sample_lister, dataset_lister, folder):
            """Delete selected items based on current mode."""
            if not action or not action.strip():
                return gr.update(), gr.update(), gr.update()

            if is_dataset_mode(data_type):
                if not action.startswith("finetune_") or "confirm" not in action:
                    return gr.update(), gr.update(), gr.update()

                if not dataset_lister or not dataset_lister.get("selected"):
                    return "No file(s) selected", gr.update(), gr.update()

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
                return msg, gr.update(), updated

            else:
                if not action.startswith("sample_") or "confirm" not in action:
                    return gr.update(), gr.update(), gr.update()

                if not sample_lister or not sample_lister.get("selected"):
                    return "No sample selected", gr.update(), gr.update()

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
                        if get_luxtts_prompt_cache_files:
                            for lux_cache in get_luxtts_prompt_cache_files(sample_name):
                                lux_cache.unlink(missing_ok=True)
                        if tts_manager:
                            tts_manager.clear_prompt_cache_for_sample(sample_name)
                        deleted_count += 1
                    except Exception as e:
                        errors.append(f"{sample_name}: {str(e)}")

                updated = get_sample_choices()
                if errors:
                    msg = f"Deleted {deleted_count} sample(s), {len(errors)} error(s): {'; '.join(errors)}"
                else:
                    msg = f"Deleted {deleted_count} sample(s)"
                return msg, updated, gr.update()

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
                lux_deleted = 0
                if get_luxtts_prompt_cache_files:
                    for lux_cache in get_luxtts_prompt_cache_files(sample_name):
                        lux_cache.unlink(missing_ok=True)
                        lux_deleted += 1
                if tts_manager:
                    tts_manager.clear_prompt_cache_for_sample(sample_name)
                if deleted:
                    _, _, new_info = load_sample_details(sample_name)
                    lux_msg = f", LuxTTS ({lux_deleted})" if lux_deleted else ""
                    return f"Cache cleared for: {', '.join(deleted)}{lux_msg}", new_info
                if lux_deleted:
                    _, _, new_info = load_sample_details(sample_name)
                    return f"Cache cleared for: LuxTTS ({lux_deleted})", new_info
                return "No cache files found", gr.update()
            except Exception as e:
                return f"Error clearing cache: {str(e)}", gr.update()

        # ===== Transcribe (shared) =====

        def transcribe_audio_handler(
            audio_file, whisper_language, transcribe_model, progress=gr.Progress()
        ):
            """Transcribe audio using Whisper or VibeVoice ASR."""
            if audio_file is None:
                return "Please load an audio file first.", ""
            try:
                if transcribe_model == "VibeVoice ASR":
                    progress(0.2, desc="Loading VibeVoice ASR...")
                    model = asr_manager.get_vibevoice_asr()
                    progress(0.4, desc="Transcribing...")
                    result = model.transcribe(audio_file)
                else:
                    if not asr_manager.whisper_available:
                        return "Whisper not available. Use VibeVoice ASR instead.", ""
                    progress(0.2, desc="Loading Whisper...")
                    model = asr_manager.get_whisper()
                    progress(0.4, desc="Transcribing...")
                    options = {}
                    if whisper_language and whisper_language != "Auto-detect":
                        lang_code = {
                            "English": "en",
                            "Chinese": "zh",
                            "Japanese": "ja",
                            "Korean": "ko",
                            "German": "de",
                            "French": "fr",
                            "Russian": "ru",
                            "Portuguese": "pt",
                            "Spanish": "es",
                            "Italian": "it",
                        }.get(whisper_language, None)
                        if lang_code:
                            options["language"] = lang_code
                    result = model.transcribe(audio_file, **options)

                progress(1.0, desc="Done!")
                transcription = result["text"].strip()

                if transcribe_model == "VibeVoice ASR":
                    transcription = re.sub(r"\[.*?\]\s*:", "", transcription)
                    transcription = re.sub(r"\[.*?\]", "", transcription)
                    transcription = " ".join(transcription.split())

                if play_completion_beep:
                    play_completion_beep()

                return transcription, "Transcription complete"

            except Exception as e:
                import traceback

                print(f"Error in transcribe:\n{traceback.format_exc()}")
                return f"Error transcribing: {str(e)}", ""

        # ===== Save handlers =====

        def save_dataset_item(folder, lister_value, audio, transcription):
            """Save audio and/or transcript for a dataset item."""
            filename = get_selected_filename(lister_value)
            if not filename:
                return "No file selected"

            results = []
            base_dir = get_dataset_dir(folder)

            # Save audio if present
            if audio:
                try:
                    audio_path = base_dir / filename
                    if isinstance(audio, str):
                        # Check if it's the same file (unmodified)
                        if Path(audio).resolve() != audio_path.resolve():
                            shutil.copy(audio, str(audio_path))
                            results.append(f"Audio saved to {filename}")
                    else:
                        sr, audio_data = audio
                        sf.write(str(audio_path), audio_data, sr, subtype="PCM_16")
                        results.append(f"Audio saved to {filename}")
                except Exception as e:
                    results.append(f"Error saving audio: {str(e)}")

            # Save transcript
            if transcription and transcription.strip():
                try:
                    txt_path = (base_dir / filename).with_suffix(".txt")
                    txt_path.write_text(transcription.strip(), encoding="utf-8")
                    results.append("Transcript saved")
                except Exception as e:
                    results.append(f"Error saving transcript: {str(e)}")

            return " | ".join(results) if results else "Nothing to save"

        def save_btn_handler(
            data_type, audio, transcription, folder, dataset_lister, sample_lister
        ):
            """Handle save button click - only acts in dataset mode."""
            if not is_dataset_mode(data_type):
                # Samples mode: handled by JS modal + input_trigger
                return gr.update()

            return save_dataset_item(folder, dataset_lister, audio, transcription)

        def handle_save_sample_input(input_value, audio, transcription):
            """Process save sample modal input (samples mode only)."""
            if not input_value or not input_value.startswith("save_sample_"):
                return gr.update(), gr.update()

            parts = input_value.split("_")
            if len(parts) >= 3:
                if parts[2] == "cancel":
                    return gr.update(), gr.update()

                sample_name = "_".join(parts[2:-1])

                if not audio:
                    return "No audio file to save", gr.update()

                clean_name = "".join(
                    c if c.isalnum() or c in "-_ " else "" for c in sample_name
                ).strip()
                clean_name = clean_name.replace(" ", "_")
                if not clean_name:
                    return "Invalid sample name", gr.update()

                try:
                    import json

                    cleaned_text = (
                        re.sub(r"\[.*?\]\s*", "", transcription).strip()
                        if transcription
                        else ""
                    )

                    audio_path = Path(audio).resolve()
                    original_path = (SAMPLES_DIR / f"{clean_name}.wav").resolve()
                    audio_unmodified = audio_path == original_path

                    meta = {"Type": "Sample", "Text": cleaned_text}
                    json_path = SAMPLES_DIR / f"{clean_name}.json"
                    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                    if audio_unmodified:
                        return (
                            f"Sample '{clean_name}' updated (text only)",
                            get_sample_choices(),
                        )

                    import os

                    audio_data, sr = sf.read(audio)
                    sf.write(str(SAMPLES_DIR / f"{clean_name}.wav"), audio_data, sr)

                    cleared = []
                    for model_size in ["0.6B", "1.7B"]:
                        cache_path = get_prompt_cache_path(clean_name, model_size)
                        if cache_path.exists():
                            os.remove(cache_path)
                            cleared.append(model_size)
                    lux_cleared = 0
                    if get_luxtts_prompt_cache_files:
                        for lux_cache in get_luxtts_prompt_cache_files(clean_name):
                            lux_cache.unlink(missing_ok=True)
                            lux_cleared += 1
                    if tts_manager:
                        tts_manager.clear_prompt_cache_for_sample(clean_name)

                    cache_details = []
                    if cleared:
                        cache_details.append(", ".join(cleared))
                    if lux_cleared:
                        cache_details.append(f"LuxTTS {lux_cleared}")
                    cache_msg = (
                        f", cache cleared ({'; '.join(cache_details)})"
                        if cache_details
                        else ""
                    )
                    return (
                        f"Sample '{clean_name}' saved (audio + text{cache_msg})",
                        get_sample_choices(),
                    )

                except Exception as e:
                    return f"Error saving sample: {str(e)}", gr.update()

            return gr.update(), gr.update()

        # ===== Batch transcribe (dataset mode only) =====

        def batch_transcribe_handler(
            folder, replace_existing, language, transcribe_model, progress=gr.Progress()
        ):
            """Batch transcribe all audio files in a dataset folder."""
            if not folder or folder in ("(No folders)", "(Select Dataset)"):
                return "Please select a dataset folder first."

            try:
                base_dir = DATASETS_DIR / folder
                if not base_dir.exists():
                    return f"Folder not found: {folder}"

                audio_files = sorted(
                    list(base_dir.glob("*.wav")) + list(base_dir.glob("*.mp3"))
                )
                if not audio_files:
                    return f"No audio files found in {folder}"

                files_to_process = [
                    f
                    for f in audio_files
                    if not f.with_suffix(".txt").exists() or replace_existing
                ]
                if not files_to_process:
                    return (
                        f"All {len(audio_files)} files already have transcripts. "
                        "Check 'Replace existing' to re-transcribe."
                    )

                status_log = [
                    f"Batch transcribing folder: {folder}",
                    f"Found {len(audio_files)} audio files ({len(files_to_process)} to process)",
                    "",
                ]

                # Load model once
                options = {}
                if transcribe_model == "VibeVoice ASR":
                    progress(0.05, desc="Loading VibeVoice ASR model...")
                    try:
                        model = asr_manager.get_vibevoice_asr()
                        status_log.append("Loaded VibeVoice ASR model")
                    except Exception as e:
                        return f"VibeVoice ASR not available: {str(e)}"
                else:
                    if not asr_manager.whisper_available:
                        return "Whisper not available. Use VibeVoice ASR instead."
                    progress(0.05, desc="Loading Whisper model...")
                    try:
                        model = asr_manager.get_whisper()
                        status_log.append("Loaded Whisper model")
                    except ImportError as e:
                        return f"Error: {str(e)}"
                    if language and language != "Auto-detect":
                        lang_code = {
                            "English": "en",
                            "Chinese": "zh",
                            "Japanese": "ja",
                            "Korean": "ko",
                            "German": "de",
                            "French": "fr",
                            "Russian": "ru",
                            "Portuguese": "pt",
                            "Spanish": "es",
                            "Italian": "it",
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
                        status_log.append(
                            f"Skipped: {audio_file.name} (transcript exists)"
                        )
                        skipped_count += 1
                        continue

                    progress_val = 0.1 + (0.9 * i / len(audio_files))
                    progress(
                        progress_val,
                        desc=f"Transcribing {i + 1}/{len(audio_files)}: {audio_file.name[:30]}...",
                    )

                    try:
                        result = (
                            model.transcribe(str(audio_file), **options)
                            if transcribe_model != "VibeVoice ASR"
                            else model.transcribe(str(audio_file))
                        )

                        text = result["text"].strip()
                        if transcribe_model == "VibeVoice ASR":
                            text = re.sub(r"\[.*?\]\s*:", "", text)
                            text = re.sub(r"\[.*?\]", "", text)
                            text = " ".join(text.split())

                        txt_file.write_text(text, encoding="utf-8")
                        status_log.append(f"{audio_file.name} -> {len(text)} chars")
                        transcribed_count += 1

                    except Exception as e:
                        status_log.append(f"Error: {audio_file.name} - {str(e)}")
                        error_count += 1

                status_log.extend(
                    [
                        "=" * 60,
                        "",
                        "Summary:",
                        f"   Transcribed: {transcribed_count}",
                        f"   Skipped: {skipped_count}",
                        f"   Errors: {error_count}",
                        f"   Total: {len(audio_files)}",
                    ]
                )

                progress(1.0, desc="Batch transcription complete!")
                if play_completion_beep:
                    play_completion_beep()

                return "\n".join(status_log)

            except Exception as e:
                return f"Error during batch transcription: {str(e)}"

        # ====================================================
        # Wire up events
        # ====================================================

        # --- Mode switching ---
        components["prep_data_type"].change(
            on_mode_change,
            inputs=[components["prep_data_type"]],
            outputs=[
                components["samples_col"],
                components["datasets_col"],
                components["clear_btn"],
                components["clear_cache_btn"],
                components["refresh_preview_btn"],
                components["batch_col"],
                components["prep_file_input"],
                components["prep_audio_editor"],  # visibility
                components["prep_audio_editor"],  # value (clear)
                components["transcription_output"],
                components["prep_status"],
                components["existing_sample_info"],
                components["editor_heading"],
            ],
        )

        # --- Sample lister events ---
        components["sample_lister"].change(
            on_sample_selection_change,
            inputs=[components["sample_lister"]],
            outputs=[
                components["prep_audio_editor"],
                components["prep_file_input"],
                components["prep_audio_editor"],
                components["transcription_output"],
                components["existing_sample_info"],
            ],
        )

        components["sample_lister"].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#prep-audio-editor .play-pause-button'); if (btn) btn.click(); }, 150); }",
        )

        # --- Dataset lister events ---
        components["finetune_folder_dropdown"].change(
            lambda folder: get_dataset_files(folder),
            inputs=[components["finetune_folder_dropdown"]],
            outputs=[components["dataset_lister"]],
        )

        components["refresh_folder_btn"].click(
            lambda: gr.update(
                choices=["(Select Dataset)"] + get_dataset_folders(),
                value="(Select Dataset)",
            ),
            outputs=[components["finetune_folder_dropdown"]],
        )

        components["dataset_lister"].change(
            on_dataset_selection_change,
            inputs=[
                components["finetune_folder_dropdown"],
                components["dataset_lister"],
            ],
            outputs=[
                components["prep_audio_editor"],
                components["prep_audio_editor"],
                components["transcription_output"],
                components["existing_sample_info"],
            ],
        )

        components["dataset_lister"].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#prep-audio-editor .play-pause-button'); if (btn) btn.click(); }, 150); }",
        )

        # --- Refresh (mode-aware) ---
        def refresh_handler(data_type, folder):
            if is_dataset_mode(data_type):
                return gr.update(), get_dataset_files(folder)
            return get_sample_choices(), gr.update()

        components["refresh_preview_btn"].click(
            refresh_handler,
            inputs=[
                components["prep_data_type"],
                components["finetune_folder_dropdown"],
            ],
            outputs=[components["sample_lister"], components["dataset_lister"]],
        )

        # --- Delete (mode-aware JS context) ---
        sample_delete_js = show_confirmation_modal_js(
            title="Delete Sample(s)?",
            message="This will permanently delete the selected sample(s), metadata, and cached files. This action cannot be undone.",
            confirm_button_text="Delete",
            context="sample_",
        )
        dataset_delete_js = show_confirmation_modal_js(
            title="Delete Dataset Item(s)?",
            message="This will permanently delete the selected audio file(s) and transcript(s). This action cannot be undone.",
            confirm_button_text="Delete",
            context="finetune_",
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

        components["delete_btn"].click(
            fn=None, inputs=[components["prep_data_type"]], js=delete_js
        )

        confirm_trigger.change(
            delete_handler,
            inputs=[
                confirm_trigger,
                components["prep_data_type"],
                components["sample_lister"],
                components["dataset_lister"],
                components["finetune_folder_dropdown"],
            ],
            outputs=[
                components["prep_status"],
                components["sample_lister"],
                components["dataset_lister"],
            ],
        )

        # --- Clear cache (samples only) ---
        components["clear_cache_btn"].click(
            clear_sample_cache_handler,
            inputs=[components["sample_lister"]],
            outputs=[components["prep_status"], components["existing_sample_info"]],
        )

        # --- Clear button (samples only) ---
        components["clear_btn"].click(
            lambda: (gr.update(visible=True), gr.update(visible=False), None, ""),
            outputs=[
                components["prep_file_input"],
                components["prep_audio_editor"],
                components["prep_audio_editor"],
                components["prep_status"],
            ],
        )

        # --- File input (samples mode - load new audio/video) ---
        components["prep_file_input"].change(
            on_prep_audio_load_handler,
            inputs=[components["prep_file_input"]],
            outputs=[components["prep_audio_editor"], components["prep_status"]],
        ).then(
            lambda audio: (
                gr.update(visible=audio is not None),
                gr.update(visible=audio is None),
            ),
            inputs=[components["prep_audio_editor"]],
            outputs=[components["prep_audio_editor"], components["prep_file_input"]],
        )

        # --- Audio processing (shared) ---
        components["normalize_btn"].click(
            normalize_audio,
            inputs=[components["prep_audio_editor"]],
            outputs=[components["prep_audio_editor"], components["prep_status"]],
        )

        components["mono_btn"].click(
            convert_to_mono,
            inputs=[components["prep_audio_editor"]],
            outputs=[components["prep_audio_editor"], components["prep_status"]],
        )

        components["clean_btn"].click(
            clean_audio,
            inputs=[components["prep_audio_editor"]],
            outputs=[components["prep_audio_editor"], components["prep_status"]],
        )

        # --- Transcribe (shared) ---
        components["transcribe_btn"].click(
            transcribe_audio_handler,
            inputs=[
                components["prep_audio_editor"],
                components["whisper_language"],
                components["transcribe_model"],
            ],
            outputs=[components["transcription_output"], components["prep_status"]],
        )

        # --- Save button (mode-aware) ---
        # JS opens name modal in samples mode, does nothing in datasets mode
        # fn saves directly in datasets mode, does nothing in samples mode
        modal_js = show_input_modal_js(
            title="Save Voice Sample",
            message="Enter a name for this voice sample:",
            placeholder="e.g., MyVoice, Female-Accent, John-Doe",
            context="save_sample_",
        )
        save_js = f"""
        (dataType, audio, transcription, folder, datasetLister, sampleLister) => {{
            if (dataType === 'Samples') {{
                let name = '';
                if (sampleLister && sampleLister.selected && sampleLister.selected.length === 1) {{
                    name = sampleLister.selected[0].replace(/\\.wav$/i, '');
                }}
                const openModal = {modal_js};
                openModal(name);
            }}
            return [dataType, audio, transcription, folder, datasetLister, sampleLister];
        }}
        """

        components["save_btn"].click(
            fn=save_btn_handler,
            inputs=[
                components["prep_data_type"],
                components["prep_audio_editor"],
                components["transcription_output"],
                components["finetune_folder_dropdown"],
                components["dataset_lister"],
                components["sample_lister"],
            ],
            outputs=[components["prep_status"]],
            js=save_js,
        )

        # Save sample input handler (after modal, samples mode only)
        input_trigger.change(
            handle_save_sample_input,
            inputs=[
                input_trigger,
                components["prep_audio_editor"],
                components["transcription_output"],
            ],
            outputs=[components["prep_status"], components["sample_lister"]],
        )

        # --- Batch transcribe (dataset mode only) ---
        components["batch_transcribe_btn"].click(
            batch_transcribe_handler,
            inputs=[
                components["finetune_folder_dropdown"],
                components["batch_replace_existing"],
                components["whisper_language"],
                components["transcribe_model"],
            ],
            outputs=[components["prep_status"]],
        )

        # --- Save preferences ---
        components["transcribe_model"].change(
            lambda x: (
                save_preference("transcribe_model", x),
                gr.update(visible=(x == "Whisper")),
            )[1],
            inputs=[components["transcribe_model"]],
            outputs=[components["whisper_language"]],
        )

        components["whisper_language"].change(
            lambda x: save_preference("whisper_language", x),
            inputs=[components["whisper_language"]],
            outputs=[],
        )


# Export for tab registry
get_tool_class = lambda: PrepSamplesTool


if __name__ == "__main__":
    """Standalone testing of Prep Samples tool."""
    from modules.core_components.tools import run_tool_standalone

    run_tool_standalone(PrepSamplesTool, port=7865, title="Prep Samples - Standalone")
