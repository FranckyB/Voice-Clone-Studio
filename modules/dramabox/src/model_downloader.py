#!/usr/bin/env python3
"""
Download Dramabox models from HuggingFace.

Models are cached locally after first download.
Gemma text encoder is fetched separately from Google's repo.
"""
import logging
import os
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

DRAMABOX_REPO = "ResembleAI/Dramabox"
GEMMA_REPO = "unsloth/gemma-3-12b-it-bnb-4bit"

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_app_models_root():
    """Resolve the app's configured models folder from config.json.

    Always uses the project config — never overridden by HF_HOME.
    This is where the Settings > Download Model button places files.
    """
    models_folder = "models"
    config_path = PROJECT_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            models_folder = config.get("models_folder", models_folder)
        except Exception:
            pass
    return PROJECT_ROOT / models_folder


def _resolve_cache_root():
    """Resolve the HF cache root for DramaBox downloads.

    Preference order:
    1. An explicit HF_HOME environment override
    2. The app's configured models folder from config.json
    3. The default project models folder
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    return _resolve_app_models_root()

# Flat download directory — always inside the app's models folder, ignores HF_HOME
FLAT_DRAMABOX_DIR = str(_resolve_app_models_root() / "dramabox")

# HF cache directory — may be overridden by HF_HOME
DEFAULT_CACHE = str(_resolve_cache_root() / "dramabox")


def _read_offline_mode():
    """Read offline_mode from config.json. Returns False if unreadable."""
    config_path = PROJECT_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f).get("offline_mode", False)
        except Exception:
            pass
    return False

# Model files in the HF repo (flat structure)
MODEL_FILES = {
    "transformer": "dramabox-dit-v1.safetensors",
    "audio_components": "dramabox-audio-components.safetensors",
    "silence_latent": "assets/silence_latent_frame.pt",
}

# Assets bundled with the DramaBox code (no download needed)
BUNDLED_ASSETS = {
    "silence_latent": Path(__file__).parent.parent / "assets" / "silence_latent_frame.pt",
}


def get_model_path(name: str, cache_dir: str = None) -> str:
    """Download a model file from HF and return local path.

    Checks for a flat copy in {models_dir}/dramabox/ first (placed there by
    the Settings > Download Model button).  Falls back to hf_hub_download via
    the HF cache when the flat copy is absent.

    Args:
        name: One of 'transformer', 'audio_components', 'silence_latent'
        cache_dir: Local cache directory (default: ~/.cache/dramabox)

    Returns:
        Local file path
    """
    cache_dir = cache_dir or DEFAULT_CACHE

    if name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_FILES.keys())}")

    # Check bundled assets first (no download ever needed)
    if name in BUNDLED_ASSETS and BUNDLED_ASSETS[name].exists():
        logger.info(f"Using bundled asset for {name}: {BUNDLED_ASSETS[name]}")
        return str(BUNDLED_ASSETS[name])

    repo_path = MODEL_FILES[name]
    filename_only = Path(repo_path).name

    # Check flat layout first: {app_models}/dramabox/<filename>
    flat_path = Path(FLAT_DRAMABOX_DIR) / filename_only
    if flat_path.exists():
        logger.info(f"Found {name} at {flat_path}")
        return str(flat_path)

    if _read_offline_mode():
        raise FileNotFoundError(
            f"Offline mode is enabled and {filename_only} was not found in {Path(FLAT_DRAMABOX_DIR)}. "
            f"Disable offline mode or use Settings > Download Model to download it first."
        )

    logger.info(f"Fetching {name} from {DRAMABOX_REPO}/{repo_path}...")

    local_path = hf_hub_download(
        repo_id=DRAMABOX_REPO,
        filename=repo_path,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info(f"  -> {local_path}")
    return local_path


def get_gemma_path(cache_dir: str = None) -> str:
    """Download Gemma 3 12B IT (pre-quantized bnb-4bit via unsloth) and return
    the snapshot directory. Using the pre-quantized variant skips runtime
    bitsandbytes quantization and ~halves the Gemma load time.

    Checks for a flat copy in {models_dir}/<repo_name>/ first (placed there by
    the Settings > Download Model button).  Falls back to snapshot_download via
    the HF cache when the flat copy is absent.
    """
    cache_dir = cache_dir or DEFAULT_CACHE

    # Check flat layout first: {app_models}/<repo_name>/
    repo_name = GEMMA_REPO.split("/")[-1]
    flat_path = Path(FLAT_DRAMABOX_DIR).parent / repo_name
    if flat_path.exists() and (flat_path / "config.json").exists():
        logger.info(f"Found Gemma at {flat_path}")
        return str(flat_path)

    if _read_offline_mode():
        raise FileNotFoundError(
            f"Offline mode is enabled and Gemma was not found in {flat_path}. "
            f"Disable offline mode or use Settings > Download Model to download it first."
        )

    logger.info(f"Fetching Gemma from {GEMMA_REPO}...")
    local_dir = snapshot_download(
        repo_id=GEMMA_REPO,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info(f"  -> {local_dir}")
    return local_dir


def get_all_paths(cache_dir: str = None) -> dict:
    """Download all required models and return paths dict.

    Returns:
        {
            'transformer': '/path/to/transformer.safetensors',
            'audio_components': '/path/to/audio-components.safetensors',
            'silence_latent': '/path/to/silence_latent_frame.pt',
            'gemma_root': '/path/to/unsloth/gemma-3-12b-it-bnb-4bit/',
        }
    """
    cache_dir = cache_dir or DEFAULT_CACHE
    paths = {}

    for name in MODEL_FILES:
        paths[name] = get_model_path(name, cache_dir)

    paths["gemma_root"] = get_gemma_path(cache_dir)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = get_all_paths()
    print("\nAll models downloaded:")
    for k, v in paths.items():
        size = os.path.getsize(v) / 1e9 if os.path.isfile(v) else "dir"
        print(f"  {k}: {v} ({size:.2f}GB)" if isinstance(size, float) else f"  {k}: {v} (directory)")
