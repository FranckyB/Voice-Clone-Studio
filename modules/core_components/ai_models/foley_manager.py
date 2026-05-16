"""
Foley/Sound Effects Model Manager

Centralized management for MMAudio sound effects generation models.
Supports text-to-audio and video-to-audio synthesis.
"""

import gc
import sys
import torch
from pathlib import Path

from .model_utils import get_device, get_dtype, empty_device_cache, run_pre_load_hooks


# MMAudio model configurations
# Maps display names to internal model identifiers
MMAUDIO_MODELS = {
    "Medium (44kHz)": {
        "model_name": "medium_44k",
        "weight_file": "mmaudio_medium_44k.pth",
        "mode": "44k",
    },
    "Large v2 (44kHz)": {
        "model_name": "large_44k_v2",
        "weight_file": "mmaudio_large_44k_v2.pth",
        "mode": "44k",
    },
}

# Shared components needed for all 44kHz models
MMAUDIO_SHARED_WEIGHTS = {
    "vae": "v1-44.pth",
    "synchformer": "synchformer_state_dict.pth",
}


def _ensure_mmaudio_path():
    """Add MMAudio repo to sys.path if not already present."""
    mmaudio = Path(__file__).parent.parent.parent / "mmaudio"
    repo_str = str(mmaudio)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


class FoleyManager:
    """Manages MMAudio models with lazy loading and VRAM optimization."""

    def __init__(self, user_config=None, models_dir=None):
        """
        Initialize Foley Manager.

        Args:
            user_config: User configuration dict
            models_dir: Base path to models directory
        """
        self.user_config = user_config or {}
        self.models_dir = models_dir or Path("models")

        # MMAudio weight directories
        self._weights_dir = self.models_dir / "mmaudio" / "weights"
        self._ext_weights_dir = self.models_dir / "mmaudio" / "ext_weights"

        # Cached model components
        self._net = None
        self._feature_utils = None
        self._current_model_name = None
        self._current_mode = None
        self._current_display_name = None

    def _get_model_config(self, display_name):
        """
        Get model configuration for a display name.
        Handles both built-in models and user-provided custom .pth files.

        Returns:
            dict with model_name, weight_path, mode
        """
        # Check built-in models first
        if display_name in MMAUDIO_MODELS:
            cfg = MMAUDIO_MODELS[display_name]
            return {
                "model_name": cfg["model_name"],
                "weight_path": self._weights_dir / cfg["weight_file"],
                "mode": cfg["mode"],
            }

        # Custom model — user placed .pth or .safetensors file in weights dir
        # Auto-detect architecture from state dict keys
        custom_path = self._weights_dir / display_name
        if custom_path.exists() and custom_path.suffix in (".pth", ".safetensors"):
            arch = self._detect_architecture(custom_path)
            return {
                "model_name": arch,
                "weight_path": custom_path,
                "mode": "44k" if "44k" in arch else "16k",
            }

        raise ValueError(f"Unknown model: {display_name}")

    def _detect_architecture(self, weight_path):
        """
        Auto-detect MMAudio architecture from a checkpoint's state dict.
        Inspects block counts and key patterns to determine model variant.

        Architecture signatures (joint_blocks / fused_blocks):
            small_16k/44k: 4 / 8  (num_heads=7,  hidden=448)
            medium_44k:    4 / 8  (num_heads=14, hidden=896)
            large_44k:     7 / 14 (clip_input_proj.1, v2=False)
            large_44k_v2:  7 / 14 (clip_input_proj.2, v2=True)

        Returns:
            Architecture name string
        """
        import torch

        weight_path = Path(weight_path)
        if weight_path.suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(weight_path), framework="pt", device="cpu") as f:
                keys = set(f.keys())
        else:
            sd = torch.load(weight_path, map_location="cpu", weights_only=True)
            keys = set(sd.keys())
            del sd

        # Count joint_blocks to determine depth
        jb_indices = set()
        for k in keys:
            if k.startswith("joint_blocks."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    jb_indices.add(int(parts[1]))

        num_joint = len(jb_indices)

        # large vs large_v2: v2 uses clip_input_proj.2, v1 uses .1
        has_clip_proj_2 = any(k.startswith("clip_input_proj.2.") for k in keys)

        if num_joint >= 7:
            # Large family
            detected = "large_44k_v2" if has_clip_proj_2 else "large_44k"
        elif num_joint >= 4:
            # Medium or small — differentiate by hidden dim
            # medium has num_heads=14 (hidden=896), small has num_heads=7 (hidden=448)
            # Check qkv weight shape: hidden_dim * 3 for qkv
            qkv_key = "joint_blocks.0.latent_block.attn.qkv.weight"
            if qkv_key in keys:
                if weight_path.suffix == ".safetensors":
                    with safe_open(str(weight_path), framework="pt", device="cpu") as f:
                        qkv_shape = f.get_tensor(qkv_key).shape
                else:
                    sd = torch.load(weight_path, map_location="cpu", weights_only=True)
                    qkv_shape = sd[qkv_key].shape
                    del sd
                # qkv weight is [3*hidden, hidden] — medium=896, small=448
                hidden = qkv_shape[1]
                detected = "medium_44k" if hidden > 500 else "small_44k"
            else:
                detected = "medium_44k"
        else:
            detected = "large_44k_v2"

        print(f"Auto-detected architecture: {detected} (joint_blocks={num_joint})")
        return detected

    def get_available_models(self):
        """
        Get list of available model display names.
        Includes built-in models + any user-added .pth files in the weights dir.

        Returns:
            List of display name strings
        """
        choices = list(MMAUDIO_MODELS.keys())

        # Scan for user-added custom models (.pth and .safetensors)
        if self._weights_dir.exists():
            known_files = {cfg["weight_file"] for cfg in MMAUDIO_MODELS.values()}
            for f in sorted(self._weights_dir.iterdir()):
                if f.suffix in (".pth", ".safetensors") and f.name not in known_files:
                    choices.append(f.name)

        return choices

    def _download_if_needed(self, weight_path):
        """Download a weight file if it doesn't exist locally."""
        if weight_path.exists():
            return

        _ensure_mmaudio_path()
        from mmaudio.utils.download_utils import download_model_if_needed

        # MMAudio's download system uses relative paths from CWD
        # We need to override by creating the path structure it expects
        weight_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a temporary Path that mimics the expected relative structure
        # The download function matches by filename
        download_model_if_needed(weight_path)

    def load_model(self, display_name="Large v2 (44kHz)", progress_callback=None):
        """
        Load an MMAudio model. Unloads previous model if different.

        Args:
            display_name: Model display name from get_available_models()
            progress_callback: Optional callable(fraction, desc) for progress updates

        Returns:
            True if model is ready
        """
        cfg = self._get_model_config(display_name)

        # Already loaded?
        if (self._net is not None and
                self._current_model_name == cfg["model_name"] and
                self._current_mode == cfg["mode"] and
                self._current_display_name == display_name):
            return True

        # Unload previous model
        if self._net is not None:
            print(f"Switching MMAudio model to {display_name} - unloading previous...")
            self.unload_all()

        # Stop external servers (e.g., llama.cpp) to free VRAM before loading
        run_pre_load_hooks()

        _ensure_mmaudio_path()

        if progress_callback:
            progress_callback(0.1, "Downloading weights if needed...")

        # Ensure weight directories exist
        self._weights_dir.mkdir(parents=True, exist_ok=True)
        self._ext_weights_dir.mkdir(parents=True, exist_ok=True)

        # Download required files
        self._download_if_needed(cfg["weight_path"])

        vae_path = self._ext_weights_dir / MMAUDIO_SHARED_WEIGHTS["vae"]
        synchformer_path = self._ext_weights_dir / MMAUDIO_SHARED_WEIGHTS["synchformer"]
        self._download_if_needed(vae_path)
        self._download_if_needed(synchformer_path)

        if progress_callback:
            progress_callback(0.3, "Loading MMAudio network...")

        device = get_device()
        dtype = get_dtype(device)

        # Load the flow prediction network
        from mmaudio.model.networks import get_my_mmaudio
        net = get_my_mmaudio(cfg["model_name"]).to(device, dtype).eval()

        weight_path = cfg["weight_path"]
        if str(weight_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(str(weight_path), device=str(device))
        else:
            state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        net.load_weights(state_dict)
        print(f"Loaded MMAudio weights from {weight_path.name}")

        if progress_callback:
            progress_callback(0.6, "Loading feature utilities (CLIP, VAE, vocoder)...")

        # Load feature utilities — suppress verbose logging from open_clip,
        # synchformer (MotionFormer), and BigVGAN during initialization
        import logging as _logging
        _root_logger = _logging.getLogger()
        _prev_root_level = _root_logger.level
        _root_logger.setLevel(_logging.ERROR)

        from mmaudio.model.utils.features_utils import FeaturesUtils
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=vae_path,
            synchformer_ckpt=synchformer_path,
            enable_conditions=True,
            mode=cfg["mode"],
            bigvgan_vocoder_ckpt=None,  # 44kHz vocoder auto-downloads from HF
            need_vae_encoder=False
        )
        feature_utils = feature_utils.to(device, dtype).eval()

        _root_logger.setLevel(_prev_root_level)

        if progress_callback:
            progress_callback(0.9, "Model ready!")

        self._net = net
        self._feature_utils = feature_utils
        self._current_model_name = cfg["model_name"]
        self._current_mode = cfg["mode"]
        self._current_display_name = display_name

        return True

    @torch.inference_mode()
    def generate_text_to_audio(self, prompt, negative_prompt="", duration=8.0,
                               seed=42, num_steps=25, cfg_strength=4.5,
                               progress_callback=None):
        """
        Generate audio from a text prompt.

        Args:
            prompt: Text description of the sound to generate
            negative_prompt: What to avoid in the generation
            duration: Audio duration in seconds (max ~30s)
            seed: Random seed for reproducibility
            num_steps: Number of ODE solver steps (more = better quality, slower)
            cfg_strength: Classifier-free guidance strength
            progress_callback: Optional callable(fraction, desc)

        Returns:
            Tuple of (sample_rate, audio_numpy_array) for Gradio
        """
        if self._net is None or self._feature_utils is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        _ensure_mmaudio_path()
        from mmaudio.model.flow_matching import FlowMatching
        from mmaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
        from mmaudio.eval_utils import generate

        device = get_device()

        # Get sequence config
        seq_cfg = CONFIG_44K if self._current_mode == "44k" else CONFIG_16K
        seq_cfg.duration = duration
        self._net.update_seq_lengths(
            seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len
        )

        if progress_callback:
            progress_callback(0.2, "Generating audio...")

        # Setup RNG and flow matching
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        # Generate (text-to-audio: no video frames)
        audios = generate(
            clip_video=None,
            sync_video=None,
            text=[prompt],
            negative_text=[negative_prompt] if negative_prompt else None,
            feature_utils=self._feature_utils,
            net=self._net,
            fm=fm,
            rng=rng,
            cfg_strength=cfg_strength,
        )

        audio = audios.float().cpu()[0]  # Shape: [1, num_samples]

        if progress_callback:
            progress_callback(1.0, "Done!")

        # Return as (sample_rate, numpy_array) for Gradio audio component
        return seq_cfg.sampling_rate, audio.numpy()

    @torch.inference_mode()
    def generate_video_to_audio(self, video_path, prompt="", negative_prompt="",
                                duration=8.0, seed=42, num_steps=25,
                                cfg_strength=4.5, progress_callback=None):
        """
        Generate audio synchronized to a video clip.

        Args:
            video_path: Path to input video file
            prompt: Optional text description to guide generation
            negative_prompt: What to avoid
            duration: Audio duration in seconds
            seed: Random seed
            num_steps: ODE solver steps
            cfg_strength: Guidance strength
            progress_callback: Optional callable(fraction, desc)

        Returns:
            Tuple of (sample_rate, audio_numpy_array) for Gradio
        """
        if self._net is None or self._feature_utils is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        _ensure_mmaudio_path()
        from mmaudio.model.flow_matching import FlowMatching
        from mmaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
        from mmaudio.eval_utils import generate, load_video

        device = get_device()

        if progress_callback:
            progress_callback(0.1, "Preparing video...")

        # Convert video to 25 FPS for optimal Synchformer input
        # (MMAudio's sync model expects 25 FPS; low-FPS sources get frame duplication
        # without this, which is suboptimal vs proper interpolation)
        video_25fps = self._convert_video_to_25fps(Path(video_path))

        # Load and preprocess video
        video_info = load_video(video_25fps, duration)
        clip_frames = video_info.clip_frames.unsqueeze(0)
        sync_frames = video_info.sync_frames.unsqueeze(0)
        duration = video_info.duration_sec  # May be truncated to video length

        # Get sequence config
        seq_cfg = CONFIG_44K if self._current_mode == "44k" else CONFIG_16K
        seq_cfg.duration = duration
        self._net.update_seq_lengths(
            seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len
        )

        if progress_callback:
            progress_callback(0.3, "Generating synchronized audio...")

        # Setup RNG and flow matching
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        # Generate with video conditioning
        audios = generate(
            clip_video=clip_frames,
            sync_video=sync_frames,
            text=[prompt] if prompt else [" "],
            negative_text=[negative_prompt] if negative_prompt else None,
            feature_utils=self._feature_utils,
            net=self._net,
            fm=fm,
            rng=rng,
            cfg_strength=cfg_strength,
        )

        audio = audios.float().cpu()[0]

        if progress_callback:
            progress_callback(1.0, "Done!")

        return seq_cfg.sampling_rate, audio.numpy()

    def _convert_video_to_25fps(self, video_path):
        """
        Convert a video to 25 FPS using ffmpeg for optimal Synchformer input.
        Uses frame interpolation to maintain smooth motion rather than frame duplication.
        Caches the result based on filename — returns original if already 25 FPS.

        Returns:
            Path to the 25 FPS video (may be same as input if already 25 FPS)
        """
        import subprocess
        import json as _json

        video_path = Path(video_path)

        # Probe the source FPS
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate",
                 "-of", "json", str(video_path)],
                capture_output=True, text=True, timeout=10
            )
            info = _json.loads(probe.stdout)
            fps_str = info["streams"][0]["r_frame_rate"]
            num, den = fps_str.split("/")
            source_fps = float(num) / float(den)
        except Exception:
            source_fps = 0

        # Skip conversion if already ~25 FPS
        if abs(source_fps - 25.0) < 1.0:
            return video_path

        # Output to temp dir with deterministic name
        temp_dir = self.models_dir.parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        output_path = temp_dir / f"sfx_25fps_{video_path.stem}.mp4"

        # Use cached version if it exists
        if output_path.exists():
            return output_path

        print(f"Converting video from {source_fps:.1f} FPS to 25 FPS...")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path),
                 "-r", "25", "-vsync", "cfr",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-an",  # Drop audio — we're generating new audio
                 "-loglevel", "error",
                 str(output_path)],
                check=True, timeout=120
            )
            print(f"Video converted to 25 FPS: {output_path.name}")
            return output_path
        except Exception as e:
            print(f"FPS conversion failed ({e}), using original video")
            return video_path

    def unload_all(self):
        """Unload all MMAudio models and free VRAM."""
        was_loaded = self._net is not None or self._feature_utils is not None

        if self._net is not None:
            del self._net
            self._net = None
        if self._feature_utils is not None:
            del self._feature_utils
            self._feature_utils = None

        self._current_model_name = None
        self._current_mode = None

        if was_loaded:
            gc.collect()
            empty_device_cache()
            print("MMAudio models unloaded")


# Singleton pattern
_foley_manager = None

def get_foley_manager(user_config=None, models_dir=None):
    """Get or create the singleton FoleyManager instance."""
    global _foley_manager
    if _foley_manager is None:
        _foley_manager = FoleyManager(user_config, models_dir)
    return _foley_manager
