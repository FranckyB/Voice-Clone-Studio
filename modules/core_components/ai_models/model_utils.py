"""
Model utilities for AI model management.

Shared utilities for model loading, device management, and VRAM optimization.
"""

import torch
from pathlib import Path

# --- Training process management ---
_active_training_process = None
_training_stop_requested = False


def stop_training():
    """Request the active training subprocess to stop.

    Terminates the subprocess and sets a flag so the training loop
    breaks cleanly on the next iteration.
    """
    global _active_training_process, _training_stop_requested
    _training_stop_requested = True
    if _active_training_process is not None:
        try:
            _active_training_process.terminate()
        except Exception:
            pass


def is_training_active():
    """Return True if a training subprocess is currently running."""
    if _active_training_process is None:
        return False
    return _active_training_process.poll() is None


def get_device(gpu_index=0):
    """Get the best available device (CUDA > MPS > CPU).

    Args:
        gpu_index: CUDA GPU index to use (default 0). Ignored for MPS/CPU.
    """
    if torch.cuda.is_available():
        gpu_index = int(gpu_index) if gpu_index is not None else 0
        if gpu_index >= torch.cuda.device_count():
            gpu_index = 0
        return f"cuda:{gpu_index}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_available_gpus():
    """Return list of available CUDA GPU names for UI dropdowns.

    Returns:
        List of tuples: [(index, name), ...] or empty list if no CUDA.
    """
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        gpus.append((i, name))
    return gpus


def get_dtype(device=None):
    """Get appropriate dtype based on device.

    CUDA uses bfloat16 for best quality.
    MPS uses float32 (float16 causes torch.multinomial failures during
    sampling due to NaN/subnormal values from softmax precision loss;
    unified memory makes float32 low-cost on Apple Silicon).
    CPU uses float32.
    """
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


# Brand mapping: model name prefixes to brand folder names.
# Used for organized storage under models/<brand>/<model_folder>/
BRAND_MAP = {
    "Qwen3": "qwen3",
    "VibeVoice": "vibevoice",
    "LuxTTS": "luxtts",
}


def get_model_brand(model_id):
    """Derive brand folder name from a model ID.

    Checks the model name (part after '/') against BRAND_MAP prefixes.
    Falls back to the HuggingFace org/author name lowercased.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    Returns:
        Brand folder name string (e.g., "qwen3")
    """
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id

    for prefix, brand in BRAND_MAP.items():
        if model_name.startswith(prefix):
            return brand

    # Fallback: use the org/author name lowercased
    if "/" in model_id:
        return model_id.split("/")[0].lower()
    return ""


def _has_model_files(path):
    """Check if a directory contains recognized model files."""
    return path.exists() and (
        list(path.glob("*.safetensors"))
        or list(path.glob("*.onnx"))
        or list(path.glob("*.pt"))
    )


def check_model_available_locally(model_name):
    """
    Check if model is available in local models directory.

    Searches in brand subfolder first (e.g., models/qwen3/ModelName/),
    then falls back to flat layout (models/ModelName/) for backward
    compatibility.

    Args:
        model_name: Model name/path (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    Returns:
        Path to local model or None if not found
    """
    models_dir = Path(__file__).parent.parent.parent.parent / "models"
    folder_name = model_name.split("/")[-1]

    # 1. Try brand subfolder: models/<brand>/<folder_name>/
    brand = get_model_brand(model_name)
    if brand:
        brand_path = models_dir / brand / folder_name
        if _has_model_files(brand_path):
            return brand_path

    # 2. Fallback: flat layout models/<folder_name>/ (backward compat)
    flat_path = models_dir / folder_name
    if _has_model_files(flat_path):
        return flat_path

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

        # Organize into brand subfolder (e.g., models/qwen3/Qwen3-TTS-...)
        brand = get_model_brand(model_id)
        if brand:
            brand_dir = models_dir / brand
            brand_dir.mkdir(parents=True, exist_ok=True)
            local_path = brand_dir / local_folder_name
        else:
            models_dir.mkdir(exist_ok=True)
            local_path = models_dir / local_folder_name

        # Check if already downloaded (look for model files)
        # Check brand subfolder first, then flat layout for backward compat
        if _has_model_files(local_path):
            return True, f"Model already exists at {local_path}", str(local_path)

        # Also check flat layout (backward compat: models/<folder_name>/)
        flat_path = models_dir / local_folder_name
        if flat_path != local_path and _has_model_files(flat_path):
            return True, f"Model already exists at {flat_path}", str(flat_path)

        # Check if git-lfs is installed
        try:
            subprocess.run(["git", "lfs", "version"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            error_msg = (
                "git-lfs is not installed or not in PATH. Install from: https://git-lfs.com\n"
                "Or manually download from HuggingFace and place in: " + str(local_path.relative_to(local_path.parents[2]) if len(local_path.parts) > 2 else local_path)
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


def empty_device_cache():
    """Empty GPU cache (CUDA or MPS) if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# Keep old name as alias for compatibility
empty_cuda_cache = empty_device_cache


# ============================================================================
# Pre-model-load hooks
# External processes (e.g., llama.cpp server) register shutdown callbacks here
# so they get stopped before any AI model loads to free VRAM.
# ============================================================================

_pre_load_hooks = []


def register_pre_load_hook(hook):
    """Register a callback to run before any AI model is loaded.

    Used by external processes (e.g., llama.cpp) that need to be shut
    down to free VRAM before loading GPU-resident models.
    """
    if hook not in _pre_load_hooks:
        _pre_load_hooks.append(hook)


def run_pre_load_hooks():
    """Run all registered pre-load hooks (e.g., stop llama.cpp server)."""
    for hook in _pre_load_hooks:
        try:
            hook()
        except Exception:
            pass


def set_seed(seed):
    """Set random seed across all available devices for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_gpu_memory(label=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        label_str = f" ({label})" if label else ""
        print(f"GPU Memory{label_str}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't expose detailed memory stats, just note it's active
        label_str = f" ({label})" if label else ""
        print(f"GPU Memory{label_str}: MPS device active (detailed stats not available)")


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
                        # Qwen checkpoints contain model.safetensors
                        if not (checkpoint / "model.safetensors").exists():
                            continue
                        # Extract epoch number for sorting
                        epoch_num = 0
                        parts = checkpoint.name.split("-")
                        for i, part in enumerate(parts):
                            if part == "epoch" and i + 1 < len(parts):
                                try:
                                    epoch_num = int(parts[i + 1])
                                except ValueError:
                                    pass
                        models.append({
                            'display_name': f"{folder.name} - {checkpoint.name}",
                            'path': str(checkpoint),
                            'speaker_name': folder.name,
                            '_epoch': epoch_num
                        })
    # Sort by speaker name ascending, then epoch descending (highest first)
    models.sort(key=lambda m: (m['speaker_name'].lower(), -m['_epoch']))
    for m in models:
        del m['_epoch']
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


def get_trained_vibevoice_models(models_dir=None):
    """Find trained VibeVoice LoRA checkpoints in the models directory.

    VibeVoice LoRA checkpoints have a lora/ subdirectory containing
    adapter_config.json. This searches both the top-level model folder
    (final save) and any checkpoint-epoch-* subdirectories (interval saves).

    Args:
        models_dir: Path to models directory (defaults to project models folder)

    Returns:
        List of dicts with display_name, path, and speaker_name
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent.parent / "models"

    models = []
    def _has_adapter_files(directory):
        """Check if a directory contains LoRA adapter files (directly or in lora/ subdir)."""
        for candidate in [directory / "lora", directory]:
            if (candidate / "adapter_model.safetensors").exists():
                return True
            if (candidate / "adapter_model.bin").exists():
                return True
        return False

    if models_dir.exists():
        for folder in models_dir.iterdir():
            if folder.is_dir():
                # Check top-level (supports both folder/lora/files and folder/files layouts)
                if _has_adapter_files(folder):
                    models.append({
                        'display_name': folder.name,
                        'path': str(folder),
                        'speaker_name': folder.name,
                        '_epoch': 999999,
                    })

                # Check checkpoint-epoch-* subdirs (interval saves)
                for checkpoint in folder.glob("checkpoint-epoch-*"):
                    if checkpoint.is_dir() and _has_adapter_files(checkpoint):
                        epoch_num = 0
                        parts = checkpoint.name.split("-")
                        for i, part in enumerate(parts):
                            if part == "epoch" and i + 1 < len(parts):
                                try:
                                    epoch_num = int(parts[i + 1])
                                except ValueError:
                                    pass
                        models.append({
                            'display_name': f"{folder.name} - {checkpoint.name}",
                            'path': str(checkpoint),
                            'speaker_name': folder.name,
                            '_epoch': epoch_num,
                        })

    # Sort by speaker name ascending, then epoch descending (final model on top)
    models.sort(key=lambda m: (m['speaker_name'].lower(), -m['_epoch']))
    for m in models:
        del m['_epoch']
    return models


def train_model(folder, speaker_name, ref_audio_filename, batch_size,
                learning_rate, num_epochs, save_interval,
                user_config, datasets_dir, project_root,
                play_completion_beep=None, progress=None):
    """Legacy Qwen finetuning wrapper.

    Qwen finetuning is disabled in this DramaBox build.
    """
    return (
        "Error: Qwen3 finetuning is disabled in this build.\n"
        "Use Model Type = DramaBox in Train Model."
    )

def train_vibevoice_model(folder, speaker_name, batch_size, learning_rate,
                          num_epochs, save_interval, ddpm_batch_mul,
                          diffusion_loss_weight, ce_loss_weight,
                          voice_prompt_drop_rate, train_diffusion_head,
                          gradient_accumulation_steps, warmup_steps,
                          ema_decay, base_model_size,
                          user_config, datasets_dir, project_root,
                          play_completion_beep=None, progress=None):
    """Legacy VibeVoice training wrapper.

    VibeVoice finetuning is disabled in this DramaBox build.
    """
    return (
        "Error: VibeVoice finetuning is disabled in this build.\n"
        "Use Model Type = DramaBox in Train Model."
    )


def train_dramabox_model(folder, speaker_name, batch_size, learning_rate,
                         num_epochs, save_interval,
                         gradient_accumulation_steps, gradient_checkpointing, num_workers, warmup_steps,
                         lora_rank, lora_alpha, lora_dropout,
                         lr_scheduler, base_model, ref_ratio,
                         text_dropout, seed, resume_lora,
                         user_config=None, datasets_dir=None, project_root=None,
                         play_completion_beep=None, progress=None):
    """Run DramaBox finetuning via its training script."""
    global _active_training_process, _training_stop_requested
    import os
    import subprocess
    import sys

    _training_stop_requested = False

    if progress is None:
        def progress(*a, **kw):
            pass

    if not folder or folder in ("(No folders)", "(Select Dataset)"):
        return "Error: Please select a dataset folder"

    if not speaker_name or not speaker_name.strip():
        return "Error: Please enter a model name"

    dataset_dir = datasets_dir / folder
    if not dataset_dir.exists():
        return f"Error: Dataset folder not found: {folder}"

    configured_dramabox = user_config.get("dramabox_folder", "").strip()
    if configured_dramabox:
        dramabox_root = Path(configured_dramabox)
    else:
        dramabox_root = project_root / "modules" / "dramabox"
        if not dramabox_root.exists():
            dramabox_root = project_root.parent / "DramaBox"
        if not dramabox_root.exists():
            dramabox_root = project_root / "DramaBox"

    train_script = dramabox_root / "src" / "train.py"
    if not train_script.exists():
        return (
            "Error: DramaBox training script not found.\n"
            f"Expected: {train_script}\n"
            "Set 'dramabox_folder' in config.json if DramaBox is elsewhere."
        )

    import json as _json
    import importlib.util as _ilu

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(dramabox_root),
        str(dramabox_root / "src"),
        str(dramabox_root / "ltx2"),
        env.get("PYTHONPATH", ""),
    ]).strip(os.pathsep)

    # ------------------------------------------------------------------
    # Step 1 — Always regenerate train_dramabox.jsonl to keep durations accurate
    # ------------------------------------------------------------------
    dramabox_manifest = dataset_dir / "train_dramabox.jsonl"
    if True:
        progress(0.02, desc="Generating train_dramabox.jsonl...")
        import soundfile as _sf
        entries = []
        for wav in sorted(dataset_dir.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if txt.exists():
                text = txt.read_text(encoding="utf-8").strip()
                if text:
                    try:
                        _info = _sf.info(str(wav))
                        _dur = _info.frames / _info.samplerate
                    except Exception:
                        _dur = 0.0
                    entries.append({"audio_filepath": str(wav), "text": text, "duration": round(_dur, 3)})
        if not entries:
            return (
                "Error: No audio+transcript pairs found in dataset folder.\n"
                "Use Prep Samples > Datasets to add audio clips and transcribe them first."
            )
        with open(dramabox_manifest, "w", encoding="utf-8") as f:
            for e in entries:
                _json.dump(e, f, ensure_ascii=False)
                f.write("\n")

    # ------------------------------------------------------------------
    # Step 2 — Run preprocess.py if audio_latents not yet encoded
    # ------------------------------------------------------------------
    audio_latents_dir = dataset_dir / "audio_latents"
    has_latents = audio_latents_dir.exists() and any(audio_latents_dir.glob("*.pt"))

    progress(0.05, desc="Resolving DramaBox model paths...")
    dl_path = dramabox_root / "src" / "model_downloader.py"
    _spec = _ilu.spec_from_file_location("_dramabox_dl_train", str(dl_path))
    _dl_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_dl_mod)
    model_paths = _dl_mod.get_all_paths()

    if not has_latents:
        preprocess_script = dramabox_root / "src" / "preprocess.py"
        if not preprocess_script.exists():
            return (
                "Error: DramaBox preprocess.py not found.\n"
                f"Expected: {preprocess_script}"
            )

        preprocess_cmd = [
            sys.executable, "-u", str(preprocess_script),
            "--dataset-type", "manifest",
            "--index", str(dramabox_manifest),
            "--audio-dir", str(dataset_dir),
            "--output-dir", str(dataset_dir),
            "--checkpoint", str(model_paths["transformer"]),
            "--audio-only-ckpt", str(model_paths["audio_components"]),
            "--gemma-root", str(model_paths["gemma_root"]),
            "--skip-existing",
            "--min-duration", "1.0",
        ]

        progress(0.10, desc="Preprocessing dataset (encoding audio + text)...")
        preprocess_result = subprocess.run(
            preprocess_cmd,
            cwd=str(dramabox_root),
            env=env,
            capture_output=True,
            text=True,
        )

        if preprocess_result.returncode != 0:
            err = (preprocess_result.stderr or preprocess_result.stdout or "").strip()
            return (
                f"Error: Preprocessing failed (exit {preprocess_result.returncode})\n"
                f"{err[:3000]}"
            )

        if not any(audio_latents_dir.glob("*.pt")):
            preprocess_log = (preprocess_result.stderr or preprocess_result.stdout or "").strip()
            return (
                "Error: Preprocessing completed but no audio_latents were created.\n"
                "Check that your audio files are valid WAV format (minimum 1 second each).\n\n"
                f"Preprocessor output:\n{preprocess_log[:3000]}"
            )

    # ------------------------------------------------------------------
    # Step 3 — Auto-generate speaker_index.txt from audio_latents
    # ------------------------------------------------------------------
    # Always regenerate speaker_index.txt from the actual .pt files to avoid
    # stale entries (e.g. old integer IDs after a stem-based re-preprocessing).
    speaker_index = dataset_dir / "speaker_index.txt"
    pt_files = sorted(audio_latents_dir.glob("*.pt"))
    lines = []
    for pt in pt_files:
        lines.append(f"{pt.stem}~{folder}~en~0~0~_~\n")
    if not lines:
        return "Error: Failed to generate speaker_index.txt — no valid audio_latents found."
    speaker_index.write_text("".join(lines), encoding="utf-8")

    trained_models_folder = user_config.get("trained_models_folder", "loras")
    output_dir = project_root / trained_models_folder / speaker_name.strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = max(100, int(num_epochs or 100))

    cmd = [
        sys.executable,
        "-u",
        str(train_script),
        "--data-dir", str(dataset_dir),
        "--speaker-index", str(speaker_index),
        "--output-dir", str(output_dir),
        "--checkpoint", str(model_paths["transformer"]),
        "--full-checkpoint", str(model_paths["audio_components"]),
        "--steps", str(steps),
        "--lr", str(float(learning_rate or 3e-5)),
        "--batch-size", str(max(1, int(batch_size or 1))),
        "--save-every", str(max(50, int(save_interval or 500))),
        "--grad-accum", str(max(1, int(gradient_accumulation_steps or 4))),
        "--gradient-checkpointing", str(int(gradient_checkpointing) if gradient_checkpointing is not None else 1),
        "--num-workers", str(max(0, int(num_workers or 2))),
        "--warmup-steps", str(max(0, int(warmup_steps or 0))),
        "--lora-rank", str(max(1, int(lora_rank or 128))),
        "--lora-alpha", str(max(1, int(lora_alpha or 128))),
        "--lora-dropout", str(float(lora_dropout or 0.0)),
        "--lr-scheduler", str(lr_scheduler or "cosine"),
        "--base-model", str(base_model or "dev"),
        "--ref-ratio", str(float(ref_ratio or 0.3)),
        "--text-dropout", str(float(text_dropout or 0.0)),
        "--seed", str(int(seed) if seed is not None else 42),
        "--log-every", "10",
    ]

    if resume_lora and str(resume_lora).strip():
        cmd.extend(["--resume-lora", str(resume_lora).strip()])

    progress(0.0, desc="Starting DramaBox training...")
    status_log = []
    result = subprocess.Popen(
        cmd,
        cwd=str(dramabox_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    _active_training_process = result

    total_steps = int(steps)
    current_progress = 0.0
    last_step_desc = "Starting DramaBox training..."
    for line in result.stdout:
        if _training_stop_requested:
            break
        line = line.strip()
        if not line:
            continue
        status_log.append(line)
        # Parse: "...Step 12/1000 | loss=0.4321 | ..."
        if "Step " in line and "/" in line and "loss=" in line:
            try:
                step_part = line.split("Step ")[1].split(" ")[0]
                current_step = int(step_part.split("/")[0])
                current_progress = min(current_step / total_steps, 0.99)
                loss_val = line.split("loss=")[1].split(" ")[0].rstrip("|").strip()
                eta_str = ""
                if "ETA " in line:
                    eta_raw = line.split("ETA ")[1].split(" ")[0]
                    eta_str = " | ETA " + eta_raw if eta_raw.endswith("min") else " | ETA " + eta_raw + "min"
                last_step_desc = f"Step {current_step}/{total_steps} | loss={loss_val}{eta_str}"
                progress(current_progress, desc=last_step_desc)
            except Exception:
                pass
        elif "New best:" in line:
            # Show step progress + new-best on one line (Gradio strips newlines in progress desc)
            msg = line
            parts = line.split(" ", 3)
            if len(parts) == 4 and parts[2] in ("INFO", "WARNING", "ERROR", "DEBUG"):
                msg = parts[3]
            progress(current_progress, desc=f"{last_step_desc} | {msg}")
        else:
            # Show all other log lines (model loading, dataset build, etc.)
            # Strip the logging timestamp prefix if present: "2026-01-01 00:00:00,000 INFO msg"
            msg = line
            parts = line.split(" ", 3)
            if len(parts) == 4 and parts[2] in ("INFO", "WARNING", "ERROR", "DEBUG"):
                msg = parts[3]
            if len(msg) > 120:
                msg = msg[:117] + "..."
            progress(current_progress, desc=msg)

    if _training_stop_requested:
        try:
            result.kill()
            result.wait(timeout=5)
        except Exception:
            pass
        _active_training_process = None
        return "Training stopped by user.\n" + "\n".join(status_log[-20:])

    result.wait()
    _active_training_process = None

    if result.returncode != 0:
        err = "\n".join(status_log[-50:])
        return f"Error: DramaBox training failed (exit {result.returncode})\n{err}"

    progress(1.0, desc="DramaBox training complete")
    if play_completion_beep:
        play_completion_beep()

    # Auto-convert the best/final LoRA to LTX-compatible format
    # Match speaker-named checkpoints: {slug}_dramabox_*.safetensors and {slug}_dramabox_best_*.safetensors
    converted_paths = []
    for lora_file in sorted(output_dir.glob("*_dramabox_*.safetensors")):
        if lora_file.name.endswith("_ltx.safetensors"):
            continue
        ltx_path = convert_dramabox_lora_to_ltx(lora_file)
        if ltx_path:
            converted_paths.append(ltx_path.name)

    msg = f"DramaBox training finished.\nOutput: {output_dir}"
    if converted_paths:
        msg += f"\nConverted for LTX inference: {', '.join(converted_paths)}"
    return msg


def convert_dramabox_lora_to_ltx(input_path, output_path=None):
    """Convert a PEFT-format DramaBox LoRA to LTX-compatible safetensors format.

    PEFT saves keys as:
        base_model.model.<layer>.lora_A.default.weight
    LTX fuse_loras expects:
        <layer>.lora_A.weight

    Args:
        input_path: Path to the PEFT .safetensors LoRA file
        output_path: Optional output path. Defaults to input with _ltx suffix.

    Returns:
        Path of the converted file, or None on failure.
    """
    import logging
    from safetensors import safe_open
    from safetensors.torch import save_file

    input_path = Path(input_path)
    if not input_path.exists():
        logging.warning(f"LoRA converter: input not found: {input_path}")
        return None

    if output_path is None:
        stem = input_path.stem
        # Replace _dramabox with _ltx in the filename to keep naming consistent
        # e.g. sleepy_dramabox_00100 → sleepy_ltx_00100
        # e.g. sleepy_dramabox_best_00050 → sleepy_ltx_best_00050
        # e.g. sleepy_dramabox → sleepy_ltx
        if "_dramabox" in stem:
            new_stem = stem.replace("_dramabox", "_ltx", 1)
        else:
            new_stem = stem + "_ltx"
        ltx_dir = input_path.parent / "LTX"
        ltx_dir.mkdir(exist_ok=True)
        output_path = ltx_dir / (new_stem + ".safetensors")
    output_path = Path(output_path)

    try:
        tensors = {}
        with safe_open(str(input_path), framework="pt") as f:
            for key in f.keys():
                # Strip PEFT prefix: base_model.model.<layer>... → <layer>...
                clean = key
                if clean.startswith("base_model.model."):
                    clean = clean[len("base_model.model."):]
                # Remove .default. infix: lora_A.default.weight → lora_A.weight
                clean = clean.replace(".lora_A.default.weight", ".lora_A.weight")
                clean = clean.replace(".lora_B.default.weight", ".lora_B.weight")
                tensors[clean] = f.get_tensor(key)

        metadata = {"reference_downscale_factor": "1"}
        save_file(tensors, str(output_path), metadata=metadata)
        logging.info(f"LoRA converter: saved {len(tensors)} keys to {output_path.name}")
        return output_path

    except Exception as e:
        logging.warning(f"LoRA converter: failed for {input_path.name}: {e}")
        return None
