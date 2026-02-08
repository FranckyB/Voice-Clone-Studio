"""
Model utilities for AI model management.

Shared utilities for model loading, device management, and VRAM optimization.
"""

import torch
from pathlib import Path


def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_dtype(device=None):
    """Get appropriate dtype based on device."""
    if device is None:
        device = get_device()

    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def get_attention_implementation(user_preference="auto"):
    """
    Get list of attention implementations to try, in order.

    Args:
        user_preference: User's attention preference from config:
            - "auto": Try best options in order
            - "flash_attention_2": Use Flash Attention 2
            - "sdpa": Use Scaled Dot-Product Attention
            - "eager": Use eager attention

    Returns:
        List of attention mechanism strings to try
    """
    if user_preference == "flash_attention_2":
        return ["flash_attention_2", "sdpa", "eager"]
    elif user_preference == "sdpa":
        return ["sdpa", "flash_attention_2", "eager"]
    elif user_preference == "eager":
        return ["eager"]
    else:  # "auto"
        return ["flash_attention_2", "sdpa", "eager"]


def check_model_available_locally(model_name):
    """
    Check if model is available in local models directory.

    Args:
        model_name: Model name/path (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    Returns:
        Path to local model or None if not found
    """
    models_dir = Path(__file__).parent.parent.parent / "models"

    # Try exact model name
    model_path = models_dir / model_name.split("/")[-1]
    if model_path.exists() and (
        list(model_path.glob("*.safetensors"))
        or list(model_path.glob("*.onnx"))
        or list(model_path.glob("*.pt"))
    ):
        return model_path

    return None


def download_model_from_huggingface(model_id, models_dir=None, local_folder_name=None, progress=None):
    """Download model from HuggingFace using git clone (not cache).

    Uses git-lfs to download directly to models/ folder without using HF cache.
    Users can also manually clone with:
    git clone https://huggingface.co/{model_id} models/{folder_name}

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        models_dir: Path to models directory (defaults to project models folder)
        local_folder_name: Custom local folder name (default: extract from model_id)
        progress: Optional Gradio progress callback

    Returns:
        Tuple: (success: bool, message: str, local_path: str or None)
    """
    import subprocess
    import threading

    try:
        # Validate inputs
        if not model_id or "/" not in model_id:
            return False, f"Invalid model ID: {model_id}. Use format 'Author/ModelName'", None

        # Determine local folder name
        if not local_folder_name:
            local_folder_name = model_id.split("/")[-1]

        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent.parent / "models"
        else:
            models_dir = Path(models_dir)

        models_dir.mkdir(exist_ok=True)
        local_path = models_dir / local_folder_name

        # Check if already downloaded (look for model files)
        if list(local_path.glob("*.safetensors")) or list(local_path.glob("*.onnx")) or list(local_path.glob("*.pt")):
            return True, f"Model already exists at {local_path}", str(local_path)

        # Check if git-lfs is installed
        try:
            subprocess.run(["git", "lfs", "version"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            error_msg = (
                "git-lfs is not installed or not in PATH. Install from: https://git-lfs.com\n"
                "Or manually download from HuggingFace and place in: models/" + local_folder_name
            )
            print(error_msg, flush=True)
            return False, error_msg, None

        # Clone repository with git-lfs
        hf_url = f"https://huggingface.co/{model_id}"

        try:
            print(f"\nStarting download: {model_id}", flush=True)
            print(f"URL: {hf_url}", flush=True)
            print(f"Destination: {local_path}\n", flush=True)

            # Track download state
            download_complete = {"done": False, "returncode": None}

            def run_download():
                """Run git clone without capturing output so it shows in console."""
                try:
                    result = subprocess.run(
                        ["git", "clone", hf_url, str(local_path)],
                        timeout=3600
                    )
                    download_complete["returncode"] = result.returncode
                except Exception as e:
                    print(f"Download error: {e}", flush=True)
                    download_complete["returncode"] = -1
                finally:
                    download_complete["done"] = True

            # Start download thread
            download_thread = threading.Thread(target=run_download, daemon=True)
            download_thread.start()

            # Wait for download to complete (progress shown in console)
            download_thread.join()

            if download_complete["returncode"] != 0:
                return False, "Download failed. Check console for details.", None

            # Verify model files exist
            if not (list(local_path.glob("*.safetensors")) or list(local_path.glob("*.onnx")) or list(local_path.glob("*.pt"))):
                return False, "Model files not found - download may be incomplete.", None

            print(f"\nSuccessfully downloaded to {local_path}\n", flush=True)
            return True, f"Successfully downloaded to {local_path}", str(local_path)

        except subprocess.TimeoutExpired:
            if local_path.exists():
                import shutil
                shutil.rmtree(local_path, ignore_errors=True)
            return False, "Download timed out after 1 hour. Check your internet connection and try again.", None
        except Exception as e:
            if local_path.exists():
                import shutil
                shutil.rmtree(local_path, ignore_errors=True)
            return False, f"Download error: {str(e)}", None

    except Exception as e:
        return False, f"Unexpected error: {str(e)}", None


def empty_cuda_cache():
    """Empty CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_gpu_memory(label=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        label_str = f" ({label})" if label else ""
        print(f"GPU Memory{label_str}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def get_trained_models(models_dir=None):
    """
    Find trained model checkpoints in the models directory.

    Args:
        models_dir: Path to models directory (defaults to project models folder)

    Returns:
        List of dicts with display_name, path, and speaker_name
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent.parent / "models"

    models = []
    if models_dir.exists():
        for folder in models_dir.iterdir():
            if folder.is_dir():
                for checkpoint in folder.glob("checkpoint-*"):
                    if checkpoint.is_dir():
                        models.append({
                            'display_name': f"{folder.name} - {checkpoint.name}",
                            'path': str(checkpoint),
                            'speaker_name': folder.name
                        })
    return models


def get_trained_model_names(models_dir=None):
    """Get list of existing trained model folder names.

    Args:
        models_dir: Path to models directory (defaults to project models folder)

    Returns:
        List of folder name strings
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent.parent / "models"

    if not models_dir.exists():
        return []

    return [folder.name for folder in models_dir.iterdir() if folder.is_dir()]


def train_model(folder, speaker_name, ref_audio_filename, batch_size,
                learning_rate, num_epochs, save_interval,
                user_config, datasets_dir, project_root,
                play_completion_beep=None, progress=None):
    """Complete training workflow: validate, prepare data, and train model.

    Args:
        folder: Dataset subfolder name
        speaker_name: Name for the trained model/speaker
        ref_audio_filename: Reference audio file from dataset
        batch_size: Training batch size
        learning_rate: Training learning rate
        num_epochs: Number of training epochs
        save_interval: Save checkpoint every N epochs
        user_config: User configuration dict
        datasets_dir: Path to datasets directory
        project_root: Path to project root
        play_completion_beep: Optional callback for audio notification
        progress: Optional Gradio progress callback
    """
    import subprocess
    import sys
    import json
    import os
    from modules.core_components.audio_utils import check_audio_format

    if progress is None:
        def progress(*a, **kw):
            pass

    # ============== STEP 1: Validation ==============
    progress(0.0, desc="Step 1/3: Validating dataset...")

    if not folder or folder == "(No folders)" or folder == "(Select Dataset)":
        return "Error: Please select a dataset folder"

    if not speaker_name or not speaker_name.strip():
        return "Error: Please enter a speaker name"

    if not ref_audio_filename:
        return "Error: Please select a reference audio file"

    if save_interval is None:
        save_interval = 5

    # Create output directory
    trained_models_folder = user_config.get("trained_models_folder", "models")
    output_dir = project_root / trained_models_folder / speaker_name.strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = datasets_dir / folder
    if not base_dir.exists():
        return f"Error: Folder not found: {folder}"

    ref_audio_path = base_dir / ref_audio_filename
    if not ref_audio_path.exists():
        return f"Error: Reference audio not found: {ref_audio_filename}"

    # Only get audio files
    audio_files = [f for f in (list(base_dir.glob("*.wav")) + list(base_dir.glob("*.mp3")))
                   if not f.name.endswith('.txt') and not f.name.endswith('.jsonl')]
    if not audio_files:
        return "Error: No audio files found in folder"

    issues = []
    valid_files = []
    converted_count = 0
    total = len(audio_files)

    status_log = []
    status_log.append("=" * 60)
    status_log.append("STEP 1/3: DATASET VALIDATION")
    status_log.append("=" * 60)

    for i, audio_path in enumerate(audio_files):
        progress(0.0 + (0.2 * (i + 1) / total), desc=f"Validating {audio_path.name}...")

        txt_path = audio_path.with_suffix(".txt")

        if not txt_path.exists():
            issues.append(f"[X] {audio_path.name}: Missing transcript")
            continue

        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
            if not transcript:
                issues.append(f"[X] {audio_path.name}: Empty transcript")
                continue
        except Exception:
            issues.append(f"[X] {audio_path.name}: Cannot read transcript")
            continue

        is_correct, info = check_audio_format(str(audio_path))
        if not is_correct:
            if not info:
                issues.append(f"[X] {audio_path.name}: Cannot read audio file")
                continue

            progress(0.0 + (0.2 * (i + 1) / total), desc=f"Converting {audio_path.name}...")
            temp_output = audio_path.parent / f"temp_{audio_path.name}"
            cmd = [
                'ffmpeg', '-y', '-i', str(audio_path),
                '-ar', '24000', '-ac', '1', '-sample_fmt', 's16',
                '-acodec', 'pcm_s16le', str(temp_output)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and temp_output.exists():
                    audio_path.unlink()
                    temp_output.rename(audio_path)
                    converted_count += 1
                else:
                    issues.append(f"[X] {audio_path.name}: Conversion failed - {result.stderr[:100]}")
                    continue
            except FileNotFoundError:
                issues.append(f"[X] {audio_path.name}: ffmpeg not found")
                continue
            except Exception as e:
                issues.append(f"[X] {audio_path.name}: Conversion error - {str(e)[:100]}")
                continue

        valid_files.append(audio_path.name)

    if not valid_files:
        return "Error: No valid training samples found\n" + "\n".join(issues[:10])

    status_log.append(f"Found {len(valid_files)} valid training samples")
    if converted_count > 0:
        status_log.append(f"Auto-converted {converted_count} files to 24kHz 16-bit mono")
    if issues:
        status_log.append(f"{len(issues)} files skipped:")
        for issue in issues[:5]:
            status_log.append(f"   {issue}")
        if len(issues) > 5:
            status_log.append(f"   ... and {len(issues) - 5} more")

    # Ensure reference audio is correct format
    progress(0.2, desc="Preparing reference audio...")
    is_correct, info = check_audio_format(str(ref_audio_path))
    if not is_correct:
        temp_output = ref_audio_path.parent / f"temp_{ref_audio_path.name}"
        cmd = [
            'ffmpeg', '-y', '-i', str(ref_audio_path),
            '-ar', '24000', '-ac', '1', '-sample_fmt', 's16',
            '-acodec', 'pcm_s16le', str(temp_output)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and temp_output.exists():
            ref_audio_path.unlink()
            temp_output.rename(ref_audio_path)
        else:
            return f"Error: Failed to convert reference audio: {result.stderr[:200]}"

    # Generate train_raw.jsonl
    progress(0.25, desc="Generating train_raw.jsonl...")
    train_raw_path = base_dir / "train_raw.jsonl"
    jsonl_entries = []

    for filename in valid_files:
        audio_path_entry = base_dir / filename
        txt_path = audio_path_entry.with_suffix(".txt")
        transcript = txt_path.read_text(encoding="utf-8").strip()

        entry = {
            "audio": str(audio_path_entry.absolute()),
            "text": transcript,
            "ref_audio": str(ref_audio_path.absolute())
        }
        jsonl_entries.append(entry)

    try:
        with open(train_raw_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        status_log.append(f"Generated train_raw.jsonl with {len(jsonl_entries)} entries")
    except Exception as e:
        return f"Error: Failed to write train_raw.jsonl: {str(e)}"

    # ============== STEP 2: Prepare Data (extract audio codes) ==============
    status_log.append("")
    status_log.append("=" * 60)
    status_log.append("STEP 2/3: EXTRACTING AUDIO CODES")
    status_log.append("=" * 60)
    progress(0.3, desc="Step 2/3: Extracting audio codes...")

    train_with_codes_path = base_dir / "train_with_codes.jsonl"
    modules_dir = project_root / "modules"
    prepare_script = modules_dir / "qwen_finetune" / "prepare_data.py"

    if not prepare_script.exists():
        status_log.append("[X] Qwen3-TTS finetuning scripts not found!")
        status_log.append("   Please ensure Qwen3-TTS repository is cloned.")
        return "\n".join(status_log)

    prepare_cmd = [
        sys.executable,
        "-u",
        str(prepare_script.absolute()),
        "--device", "cuda:0",
        "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "--input_jsonl", str(train_raw_path),
        "--output_jsonl", str(train_with_codes_path)
    ]

    status_log.append(f"Running: {' '.join(prepare_cmd)}")
    status_log.append("")

    try:
        result = subprocess.Popen(
            prepare_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(base_dir)
        )

        for line in result.stdout:
            line = line.strip()
            if line:
                status_log.append(f"  {line}")

        result.wait()

        if result.returncode != 0:
            status_log.append(f"[X] prepare_data.py failed with exit code {result.returncode}")
            return "\n".join(status_log)

        if not train_with_codes_path.exists():
            status_log.append("[X] train_with_codes.jsonl was not generated")
            return "\n".join(status_log)

        status_log.append("")
        status_log.append("[OK] Audio codes extracted successfully")

    except Exception as e:
        status_log.append(f"[X] Error running prepare_data.py: {str(e)}")
        return "\n".join(status_log)

    # ============== STEP 3: Fine-tune ==============
    status_log.append("")
    status_log.append("=" * 60)
    status_log.append("STEP 3/3: TRAINING MODEL")
    status_log.append("=" * 60)
    progress(0.5, desc="Step 3/3: Training model (this may take a while)...")

    sft_script = modules_dir / "qwen_finetune" / "sft_12hz.py"

    if not sft_script.exists():
        status_log.append("[X] sft_12hz.py not found!")
        return "\n".join(status_log)

    base_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    status_log.append(f"Locating base model: {base_model_id}")
    try:
        from huggingface_hub import snapshot_download
        offline_mode = user_config.get("offline_mode", False)
        base_model_path = snapshot_download(
            repo_id=base_model_id,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.npz"],
            local_files_only=offline_mode
        )
        status_log.append(f"[OK] Using cached model at: {base_model_path}")
    except Exception as e:
        status_log.append(f"[X] Failed to locate/download base model: {str(e)}")
        return "\n".join(status_log)

    attn_impl = user_config.get("attention_mechanism", "auto")

    sft_cmd = [
        sys.executable,
        "-u",
        str(sft_script.absolute()),
        "--init_model_path", base_model_path,
        "--output_model_path", str(output_dir),
        "--train_jsonl", str(train_with_codes_path),
        "--batch_size", str(int(batch_size)),
        "--lr", str(learning_rate),
        "--num_epochs", str(int(num_epochs)),
        "--save_interval", str(int(save_interval)),
        "--speaker_name", speaker_name.strip().lower(),
        "--attn_implementation", attn_impl
    ]

    status_log.append("")
    status_log.append("Training configuration:")
    status_log.append(f"  Base model: {base_model_id}")
    status_log.append(f"  Attention implementation: {attn_impl}")
    status_log.append(f"  Batch size: {int(batch_size)}")
    status_log.append(f"  Learning rate: {learning_rate}")
    status_log.append(f"  Epochs: {int(num_epochs)}")
    status_log.append(f"  Save interval: Every {int(save_interval)} epoch(s)" if save_interval > 0 else "  Save interval: Every epoch")
    status_log.append(f"  Speaker name: {speaker_name.strip()}")
    status_log.append(f"  Output: {output_dir}")
    status_log.append("")
    status_log.append("Starting training...")
    status_log.append(f"Running: {' '.join([str(arg) for arg in sft_cmd])}")
    status_log.append("")

    try:
        env = os.environ.copy()
        env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        env['TOKENIZERS_PARALLELISM'] = 'false'

        result = subprocess.Popen(
            sft_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        for line in result.stdout:
            line = line.strip()
            if line:
                status_log.append(f"  {line}")

                if "Epoch" in line and "Step" in line:
                    try:
                        epoch_num = int(line.split("Epoch")[1].split("|")[0].strip())
                        progress_val = 0.5 + (0.5 * (epoch_num + 1) / int(num_epochs))
                        progress(progress_val, desc=f"Training: {line[:60]}...")
                    except Exception:
                        pass

        result.wait()

        if result.returncode != 0:
            status_log.append("")
            status_log.append(f"[X] Training failed with exit code {result.returncode}")
            return "\n".join(status_log)

        status_log.append("")
        status_log.append("=" * 60)
        status_log.append("TRAINING COMPLETED SUCCESSFULLY!")
        status_log.append("=" * 60)
        status_log.append(f"Model saved to: {output_dir}")
        status_log.append(f"Speaker name: {speaker_name.strip()}")
        status_log.append("")
        status_log.append("To use your trained model:")
        status_log.append("  1. Go to Voice Presets tab")
        status_log.append("  2. Select 'Trained Models' radio button")
        status_log.append(f"  3. Click refresh and select '{speaker_name.strip()}'")

        progress(1.0, desc="Training complete!")
        if play_completion_beep:
            play_completion_beep()

    except Exception as e:
        status_log.append(f"[X] Error during training: {str(e)}")
        return "\n".join(status_log)

    return "\n".join(status_log)
