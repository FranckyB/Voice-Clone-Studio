"""
TTS Model Manager

Centralized management for all TTS models (Qwen3, VibeVoice, etc.)
"""

import gc
import hashlib
import importlib.util
import inspect
import logging
import os
import re
import sys
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

from .model_utils import (
    get_device, get_dtype, get_attention_implementation,
    check_model_available_locally, empty_device_cache, log_gpu_memory, set_seed,
    run_pre_load_hooks
)

# Suppress noisy info/warning messages from upstream libraries:
# - qwen_tts config init: "speaker_encoder_config is None...", "talker_config is None...", etc.
# - transformers generation: "Setting pad_token_id to eos_token_id..."
# - transformers modeling: "Flash Attention 2 without specifying a torch dtype..."
# - transformers tensor_parallel: "TP rules were not applied...", "layers were not sharded..."
for _logger_name in [
    "qwen_tts",
    "transformers.modeling_utils",
    "transformers.generation.utils",
    "transformers.integrations.tensor_parallel",
]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


class TTSManager:
    """Manages all TTS models with lazy loading and VRAM optimization."""

    def __init__(self, user_config: Dict = None, samples_dir: Path = None):
        """
        Initialize TTS Manager.

        Args:
            user_config: User configuration dict (with attention_mechanism, low_cpu_mem_usage, offline_mode)
            samples_dir: Path to samples directory for prompt caching
        """
        self.user_config = user_config or {}
        self.samples_dir = samples_dir or Path("samples")

        # Model cache
        self._qwen3_base_model = None
        self._qwen3_base_size = None
        self._qwen3_voice_design_model = None
        self._qwen3_custom_voice_model = None
        self._qwen3_custom_voice_size = None
        self._vibevoice_tts_model = None
        self._vibevoice_tts_size = None
        self._luxtts_model = None

        # Trained model cache
        self._trained_model = None
        self._trained_model_path = None
        self._trained_model_is_faster = False

        # VibeVoice Streaming model cache
        self._vibevoice_streaming_model = None
        self._vibevoice_streaming_processor = None
        self._vibevoice_streaming_voice_cache = {}

        # VibeVoice Trained (LoRA) model cache
        self._trained_vibevoice_model = None
        self._trained_vibevoice_path = None
        self._trained_vv_base_connectors = None    # original connector state dicts
        self._trained_vv_lora_connectors = None    # trained connector state dicts
        self._trained_vv_base_pred_head = None     # original prediction head state dict (CPU)
        self._trained_vv_lora_pred_head = None     # trained prediction head state dict (CPU)
        self._trained_vv_pred_head_is_peft = False  # True if diffusion head loaded as LoRA

        # Chatterbox models
        self._chatterbox_tts_model = None
        self._chatterbox_vc_model = None
        self._chatterbox_mtl_model = None

        # Fish Speech models
        self._fish_speech_model = None
        self._fish_speech_codec = None
        self._fish_speech_decode_one_token = None
        self._fish_speech_checkpoint_dir = None

        # Prompt cache
        self._voice_prompt_cache = {}
        self._luxtts_prompt_cache = {}
        self._last_loaded_model = None

        # CUDA graphs acceleration state
        self._faster_qwen3_available = None  # lazy-checked

        # DramaBox integration state
        self._dramabox_root = None
        self._dramabox_downloader_mod = None
        self._dramabox_inference_mod = None
        self._dramabox_supported_flags = None
        # DramaBox warm-server cache (TTSServer instance, keyed by checkpoint paths)
        self._dramabox_server = None
        self._dramabox_server_key = None

    def _use_cuda_graphs(self):
        """Check if CUDA graphs acceleration should be used for Qwen3 models."""
        # Must be enabled in config
        if not self.user_config.get("cuda_graphs", True):
            return False
        # Must have CUDA
        if not torch.cuda.is_available():
            return False
        # Multi-GPU: FasterQwen3TTS currently only supports cuda:0
        tts_gpu = int(self.user_config.get("tts_gpu", 0))
        if tts_gpu != 0:
            return False
        # Check package availability (cache result)
        if self._faster_qwen3_available is None:
            try:
                from faster_qwen3_tts import FasterQwen3TTS  # noqa: F401
                self._faster_qwen3_available = True
            except ImportError:
                self._faster_qwen3_available = False
        return self._faster_qwen3_available

    def _load_faster_model(self, model_name):
        """Load a model using FasterQwen3TTS with CUDA graph acceleration."""
        from faster_qwen3_tts import FasterQwen3TTS

        offline_mode = self.user_config.get("offline_mode", False)
        local_path = check_model_available_locally(model_name)

        if local_path:
            print(f"Found local model: {local_path}")
            model_to_load = str(local_path)
        elif offline_mode:
            raise RuntimeError(
                f"Offline mode enabled but model not available locally: {model_name}\n"
                f"To use offline mode, download the model first or disable offline mode in Settings."
            )
        else:
            model_to_load = model_name

        model = FasterQwen3TTS.from_pretrained(
            model_to_load,
            device="cuda",
            dtype=get_dtype(),
            attn_implementation="eager",
        )
        print("[OK] Model loaded with CUDA graphs acceleration")
        return model

    def _check_and_unload_if_different(self, model_id):
        """If switching to a different model, unload all. Stops external servers on first/new load."""
        if self._last_loaded_model is not None and self._last_loaded_model != model_id:
            print(f"Switching from {self._last_loaded_model} to {model_id} - unloading all TTS models...")
            self.unload_all()
            run_pre_load_hooks()
        elif self._last_loaded_model is None:
            # First model load — stop external servers (e.g., llama.cpp) to free VRAM
            run_pre_load_hooks()
        self._last_loaded_model = model_id

    def _load_model_with_attention(self, model_class, model_name: str, **kwargs):
        """
        Load a HuggingFace model with best available attention mechanism.

        Returns:
            Tuple: (loaded_model, attention_mechanism_used)
        """
        offline_mode = self.user_config.get("offline_mode", False)

        # Check local availability
        local_path = check_model_available_locally(model_name)
        if local_path:
            print(f"Found local model: {local_path}")
            model_to_load = str(local_path)
        elif offline_mode:
            raise RuntimeError(
                f"❌ Offline mode enabled but model not available locally: {model_name}\n"
                f"To use offline mode, download the model first or disable offline mode in Settings."
            )
        else:
            model_to_load = model_name

        mechanisms = get_attention_implementation(
            self.user_config.get("attention_mechanism", "auto")
        )

        last_error = None
        for attn in mechanisms:
            try:
                model = model_class.from_pretrained(
                    model_to_load,
                    attn_implementation=attn,
                    trust_remote_code=True,
                    **kwargs
                )
                print(f"[OK] Model loaded with {attn}")
                return model, attn
            except Exception as e:
                error_msg = str(e).lower()
                last_error = e

                is_attn_error = any(
                    keyword in error_msg
                    for keyword in ["flash", "attention", "sdpa", "not supported"]
                )

                if is_attn_error:
                    print(f"  {attn} not available, trying next option...")
                    continue
                else:
                    raise e

        raise RuntimeError(f"Failed to load model: {str(last_error)}")

    def get_qwen3_base(self, size="1.7B"):
        """Load Qwen3 Base TTS model."""
        model_id = f"qwen3_base_{size}"
        self._check_and_unload_if_different(model_id)

        if self._qwen3_base_model is None:
            model_name = f"Qwen/Qwen3-TTS-12Hz-{size}-Base"
            print(f"Loading {model_name}...")

            if self._use_cuda_graphs():
                self._qwen3_base_model = self._load_faster_model(model_name)
            else:
                from qwen_tts import Qwen3TTSModel
                self._qwen3_base_model, _ = self._load_model_with_attention(
                    Qwen3TTSModel,
                    model_name,
                    device_map=get_device(self.user_config.get("tts_gpu", 0)),
                    dtype=get_dtype(),
                    low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
                )
            self._qwen3_base_size = size
            print(f"Qwen3 Base TTS ({size}) loaded!")

        return self._qwen3_base_model

    def get_qwen3_voice_design(self):
        """Load Qwen3 VoiceDesign model (1.7B only)."""
        self._check_and_unload_if_different("qwen3_voice_design")

        if self._qwen3_voice_design_model is None:
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            print("Loading Qwen3 VoiceDesign model (1.7B)...")

            if self._use_cuda_graphs():
                self._qwen3_voice_design_model = self._load_faster_model(model_name)
            else:
                from qwen_tts import Qwen3TTSModel
                self._qwen3_voice_design_model, _ = self._load_model_with_attention(
                    Qwen3TTSModel,
                    model_name,
                    device_map=get_device(self.user_config.get("tts_gpu", 0)),
                    dtype=get_dtype(),
                    low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
                )
            print("VoiceDesign model loaded!")

        return self._qwen3_voice_design_model

    def get_qwen3_custom_voice(self, size="1.7B"):
        """Load Qwen3 CustomVoice model."""
        model_id = f"qwen3_custom_voice_{size}"
        self._check_and_unload_if_different(model_id)

        if self._qwen3_custom_voice_model is None:
            model_name = f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"
            print(f"Loading {model_name}...")

            if self._use_cuda_graphs():
                self._qwen3_custom_voice_model = self._load_faster_model(model_name)
            else:
                from qwen_tts import Qwen3TTSModel
                self._qwen3_custom_voice_model, _ = self._load_model_with_attention(
                    Qwen3TTSModel,
                    model_name,
                    device_map=get_device(self.user_config.get("tts_gpu", 0)),
                    dtype=get_dtype(),
                    low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
                )
            self._qwen3_custom_voice_size = size
            print(f"CustomVoice model ({size}) loaded!")

        return self._qwen3_custom_voice_model

    def get_vibevoice_tts(self, size: str = "1.5B"):
        """Load VibeVoice TTS model."""
        model_id = f"vibevoice_tts_{size}"
        self._check_and_unload_if_different(model_id)

        if self._vibevoice_tts_model is None:
            print(f"Loading VibeVoice TTS ({size})...")
            try:
                from modules.vibevoice_tts.modular.modeling_vibevoice_inference import (
                    VibeVoiceForConditionalGenerationInference
                )
                import warnings

                # Map size to model path
                if size == "Large (4-bit)":
                    model_path = "FranckyB/VibeVoice-Large-4bit"
                    try:
                        import bitsandbytes
                    except ImportError:
                        raise ImportError(
                            "bitsandbytes required for 4-bit models. Install with: pip install bitsandbytes"
                        )
                else:
                    model_path = f"FranckyB/VibeVoice-{size}"

                import logging
                logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)

                    self._vibevoice_tts_model, _ = self._load_model_with_attention(
                        VibeVoiceForConditionalGenerationInference,
                        model_path,
                        dtype=get_dtype(),
                        device_map=get_device(self.user_config.get("tts_gpu", 0)),
                        low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
                    )

                self._vibevoice_tts_size = size
                print(f"VibeVoice TTS ({size}) loaded!")

            except ImportError as e:
                print(f"❌ VibeVoice TTS not available: {e}")
                raise
            except Exception as e:
                print(f"❌ Error loading VibeVoice TTS: {e}")
                raise

        return self._vibevoice_tts_model

    def get_vibevoice_streaming(self):
        """Load VibeVoice Streaming 0.5B model and processor."""
        self._check_and_unload_if_different("vibevoice_streaming")

        if self._vibevoice_streaming_model is None:
            print("Loading VibeVoice Streaming 0.5B...")
            try:
                from modules.vibevoice_tts.modular.modeling_vibevoice_streaming_inference import (
                    VibeVoiceStreamingForConditionalGenerationInference
                )
                from modules.vibevoice_tts.processor.vibevoice_streaming_processor import (
                    VibeVoiceStreamingProcessor
                )
                import warnings

                model_path = "microsoft/VibeVoice-Realtime-0.5B"
                device = get_device(self.user_config.get("tts_gpu", 0))
                dtype = get_dtype(device)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)

                    # Load model
                    offline_mode = self.user_config.get("offline_mode", False)
                    local_path = check_model_available_locally(model_path)
                    load_path = str(local_path) if local_path else model_path

                    if local_path:
                        print(f"Loading from local: {local_path}")
                    else:
                        print(f"Downloading model from HuggingFace: {model_path} (~2 GB)...")

                    self._vibevoice_streaming_model = (
                        VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                            load_path,
                            dtype=dtype,
                            local_files_only=offline_mode,
                        ).to(device)
                    )

                    # Load processor
                    self._vibevoice_streaming_processor = (
                        VibeVoiceStreamingProcessor.from_pretrained(
                            load_path,
                            local_files_only=offline_mode,
                        )
                    )

                print("VibeVoice Streaming 0.5B loaded!")

            except ImportError as e:
                print(f"VibeVoice Streaming not available: {e}")
                raise
            except Exception as e:
                print(f"Error loading VibeVoice Streaming: {e}")
                raise

        return self._vibevoice_streaming_model, self._vibevoice_streaming_processor

    def _get_streaming_voice_prompt(self, voice_name):
        """Load or cache a streaming voice prompt (.pt file).

        Voice prompts are pre-computed KV cache embeddings stored  in the
        microsoft/VibeVoice GitHub repo.  On first use the .pt file is
        downloaded and cached under models/vibevoice/voices/.
        """
        if voice_name in self._vibevoice_streaming_voice_cache:
            return self._vibevoice_streaming_voice_cache[voice_name]

        from modules.core_components.constants import VIBEVOICE_STREAMING_VOICE_FILES

        repo_filename = VIBEVOICE_STREAMING_VOICE_FILES.get(voice_name)
        if not repo_filename:
            raise RuntimeError(
                f"Unknown streaming voice '{voice_name}'. "
                f"Available: {list(VIBEVOICE_STREAMING_VOICE_FILES.keys())}"
            )

        # Local cache directory: models/vibevoice/voices/
        models_dir = Path(__file__).parent.parent.parent.parent / "models"
        voices_dir = models_dir / "vibevoice" / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        local_pt = voices_dir / f"{repo_filename}.pt"

        # Download from GitHub if not cached locally
        if not local_pt.exists():
            offline_mode = self.user_config.get("offline_mode", False)
            if offline_mode:
                raise RuntimeError(
                    f"Voice prompt '{voice_name}' not found locally and offline mode is enabled. "
                    f"Disable offline mode to download it automatically."
                )
            import urllib.request
            url = (
                f"https://github.com/microsoft/VibeVoice/raw/main/"
                f"demo/voices/streaming_model/{repo_filename}.pt"
            )
            print(f"Downloading voice prompt: {voice_name} from GitHub...")
            print(f"  URL: {url}")
            try:
                urllib.request.urlretrieve(url, str(local_pt))
                size_mb = local_pt.stat().st_size / (1024 * 1024)
                print(f"  Saved to {local_pt} ({size_mb:.1f} MB)")
            except Exception as dl_err:
                # Clean up partial download
                if local_pt.exists():
                    local_pt.unlink()
                raise RuntimeError(
                    f"Failed to download voice prompt for '{voice_name}' from {url}: {dl_err}"
                )

        device = get_device(self.user_config.get("tts_gpu", 0))
        # weights_only=False required: .pt files contain BaseModelOutputWithPast
        # objects (pre-computed KV caches), not plain tensors.  Source is the
        # official microsoft/VibeVoice repo.
        cached_prompt = torch.load(str(local_pt), map_location=device, weights_only=False)
        self._vibevoice_streaming_voice_cache[voice_name] = cached_prompt
        return cached_prompt

    def get_luxtts(self):
        """Load LuxTTS model (lazy import to avoid slowing app startup)."""
        self._check_and_unload_if_different("luxtts")

        if self._luxtts_model is None:
            print("Loading LuxTTS model...")
            try:
                import warnings
                import logging

                # Suppress k2 import warning — PyTorch fallback works fine
                k2_logger = logging.getLogger()
                prev_level = k2_logger.level
                k2_logger.setLevel(logging.ERROR)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*k2.*")
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
                    from zipvoice.luxvoice import LuxTTS

                    device = get_device(self.user_config.get("tts_gpu", 0))
                    if device.startswith("cuda"):
                        self._luxtts_model = LuxTTS("YatharthS/LuxTTS", device="cuda")
                    elif device == "mps":
                        self._luxtts_model = LuxTTS("YatharthS/LuxTTS", device="mps")
                    else:
                        threads = int(self.user_config.get("luxtts_cpu_threads", 2))
                        self._luxtts_model = LuxTTS(
                            "YatharthS/LuxTTS", device="cpu", threads=max(1, threads)
                        )

                k2_logger.setLevel(prev_level)

                print("LuxTTS loaded!")

            except ImportError as e:
                raise ImportError(
                    f"LuxTTS not available: {e}\n"
                    "Install with: pip install zipvoice@git+https://github.com/ysharma3501/LuxTTS.git"
                )
            except Exception as e:
                print(f"Error loading LuxTTS: {e}")
                raise

        return self._luxtts_model

    def get_fish_speech(self):
        """Load Fish Speech S2 Pro model and DAC codec.

        Downloads the model from HuggingFace on first use. Subsequent calls
        return cached instances. Respects offline_mode config.

        Returns:
            Tuple of (model, decode_one_token, codec)
        """
        if self._fish_speech_model is not None:
            return self._fish_speech_model, self._fish_speech_decode_one_token, self._fish_speech_codec

        self._check_and_unload_if_different("fish_speech")

        import sys
        from pathlib import Path as _Path

        # Add vendored fish_speech to sys.path so 'from fish_speech.xxx' resolves
        fish_speech_vendor_dir = str(_Path(__file__).parent.parent.parent / "fish_speech")
        if fish_speech_vendor_dir not in sys.path:
            sys.path.insert(0, fish_speech_vendor_dir)

        from fish_speech.models.text2semantic.inference import (
            init_model, load_codec_model
        )

        device = get_device()
        dtype = get_dtype()

        # Download or locate model via huggingface_hub
        offline_mode = self.user_config.get("offline_mode", False)
        repo_id = "fishaudio/s2-pro"

        local_path = check_model_available_locally(repo_id)
        if local_path:
            checkpoint_dir = str(local_path)
            print(f"Found local Fish Speech model: {checkpoint_dir}")
        elif offline_mode:
            raise RuntimeError(
                f"Offline mode enabled but Fish Speech model not available locally: {repo_id}\n"
                "Download the model first or disable offline mode in Settings."
            )
        else:
            from huggingface_hub import snapshot_download
            print(f"Downloading Fish Speech S2 Pro model (~8 GB): {repo_id}")
            checkpoint_dir = snapshot_download(repo_id=repo_id)
            print(f"Fish Speech model downloaded to: {checkpoint_dir}")

        self._fish_speech_checkpoint_dir = checkpoint_dir

        # Load the main transformer model
        print("[FISH SPEECH] Loading S2 Pro model (Triton/Inductor caching active)...")
        model, decode_one_token = init_model(
            checkpoint_path=checkpoint_dir,
            device=device,
            precision=dtype,
            compile=True,  # Enable Triton/Inductor compilation
        )
        print(f"Fish Speech model loaded on {device}")

        # Load the DAC codec for audio encoding/decoding
        codec_path = _Path(checkpoint_dir) / "codec.pth"
        if not codec_path.exists():
            # Try alternate name
            codec_path = _Path(checkpoint_dir) / "codec" / "model.pth"
        if not codec_path.exists():
            # Search for any codec-related file
            for candidate in ["codec.pth", "codec.pt", "dac.pth"]:
                p = _Path(checkpoint_dir) / candidate
                if p.exists():
                    codec_path = p
                    break

        print(f"Loading Fish Speech DAC codec from {codec_path}...")
        codec = load_codec_model(
            codec_checkpoint_path=str(codec_path),
            device=device,
            precision=dtype,
        )
        print(f"Fish Speech DAC codec loaded (sample rate: {codec.sample_rate}Hz)")

        self._fish_speech_model = model
        self._fish_speech_decode_one_token = decode_one_token
        self._fish_speech_codec = codec

        log_gpu_memory("After Fish Speech load")
        return model, decode_one_token, codec

    def generate_voice_clone_fish_speech(self, text, voice_sample_path,
                                         ref_text="", seed=-1,
                                         temperature=0.9, top_p=0.9, top_k=30,
                                         repetition_penalty=1.05,
                                         max_new_tokens=0, chunk_length=300,
                                         split_by_paragraph=False,
                                         progress_callback=None):
        """Generate speech using Fish Speech S2 Pro.

        Args:
            text: Text to synthesize
            voice_sample_path: Path to reference audio for voice cloning
            ref_text: Transcript of the reference audio
            seed: Random seed (-1 for random)
            temperature: Sampling temperature (0.1-2.0)
            top_p: Top-p nucleus sampling (0.1-1.0)
            top_k: Top-k sampling (1-100)
            repetition_penalty: Repetition penalty (1.0-2.0)
            max_new_tokens: Max tokens to generate (0=auto)
            chunk_length: Bytes per batch for generate_long
            progress_callback: Optional Gradio progress callback

        Returns:
            Tuple of (audio_numpy_array, sample_rate)
        """
        import random
        import numpy as np

        if seed < 0:
            seed = random.randint(0, 2147483647)
        set_seed(seed)

        # Optimization: Calculate max_new_tokens based on text length to avoid VRAM leaks
        model, decode_one_token, codec = self.get_fish_speech()
        device = get_device()

        import sys
        from pathlib import Path as _Path
        fish_speech_vendor_dir = str(_Path(__file__).parent.parent.parent / "fish_speech")
        if fish_speech_vendor_dir not in sys.path:
            sys.path.insert(0, fish_speech_vendor_dir)

        from fish_speech.models.text2semantic.inference import (
            encode_audio, generate_long, decode_to_audio
        )

        if progress_callback:
            progress_callback(0.2, desc="Encoding reference audio...")

        # Encode reference audio to VQ codes
        prompt_tokens = encode_audio(str(voice_sample_path), codec, device)

        # Use startup cache check result (set in voice_clone_studio.py)
        has_cache = os.environ.get("FISH_SPEECH_CACHE_READY", "0") == "1"

        def _print_cache_status():
            """Print cache status message once, right before first generation call."""
            if _print_cache_status._done:
                return
            _print_cache_status._done = True
            if has_cache:
                if progress_callback:
                    progress_callback(0.3, desc="Using cached kernels...")
                print("[FISH SPEECH] Using cached kernels...")
            else:
                if progress_callback:
                    progress_callback(0.3, desc="Compiling GPU kernels (one-time process, may take several minutes)...")
                print("")
                print("[FISH SPEECH] " + "=" * 60)
                print("[FISH SPEECH] Building kernel cache for the first time.")
                print("[FISH SPEECH] This may take several minutes - please be patient.")
                print("[FISH SPEECH] Future generations will be much faster.")
                print("[FISH SPEECH] " + "=" * 60)
                print("")
        _print_cache_status._done = False

        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        if split_by_paragraph and len(paragraphs) > 1:
            print(f"[FISH SPEECH] Splitting text into {len(paragraphs)} paragraphs (0.5s pause)...")
            audio_segments = []
            for idx, para in enumerate(paragraphs):
                para_max = max_new_tokens
                # Optimization: Calculate max_new_tokens based on paragraph length
                if para_max <= 0:
                    # Optimized heuristic: 1 token roughly maps to 1 character (multiplied by 1.5 to be safe) + 100 padding
                    para_max = max(int(len(para) * 1.5) + 100, 128)
                    print(f"[FISH SPEECH] Paragraph {idx + 1}: Optimized max_tokens = {para_max}")

                if progress_callback:
                    progress_callback(0.4 + (0.4 * (idx / len(paragraphs))), desc=f"Generating paragraph {idx + 1}/{len(paragraphs)}...")

                _print_cache_status()
                para_codes = []
                for response in generate_long(
                    model=model,
                    device=device,
                    decode_one_token=decode_one_token,
                    text=para,
                    num_samples=1,
                    max_new_tokens=para_max,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    compile=True,
                    iterative_prompt=True,
                    chunk_length=chunk_length,
                    prompt_text=[ref_text] if ref_text else None,
                    prompt_tokens=[prompt_tokens] if ref_text else None,
                ):
                    if response.action == "sample":
                        para_codes.append(response.codes)
                    elif response.action == "next":
                        break

                if not para_codes:
                    continue

                merged_codes = torch.cat(para_codes, dim=1)
                audio_tensor = decode_to_audio(merged_codes.to(device), codec)
                audio_np = audio_tensor.cpu().float().numpy()
                audio_segments.append(audio_np)

                if idx < len(paragraphs) - 1:
                    # Add 0.5 second of silence
                    silence = np.zeros(int(codec.sample_rate * 0.5), dtype=np.float32)
                    audio_segments.append(silence)

            if not audio_segments:
                raise RuntimeError("Fish Speech generation produced no output codes.")

            if progress_callback:
                progress_callback(0.8, desc="Decoding audio...")

            final_audio = np.concatenate(audio_segments)
            return final_audio, codec.sample_rate

        else:
            if progress_callback:
                progress_callback(0.4, desc="Generating speech...")

            # Calculate global max_new_tokens if not splitting
            if max_new_tokens <= 0:
                max_new_tokens = max(int(len(text) * 1.5) + 100, 128)
                print(f"[FISH SPEECH] Optimized max_new_tokens calculated: {max_new_tokens}")

            _print_cache_status()
            all_codes = []
            for response in generate_long(
                model=model,
                device=device,
                decode_one_token=decode_one_token,
                text=text,
                num_samples=1,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                compile=True,  # Use compiled kernels
                iterative_prompt=True,
                chunk_length=chunk_length,
                prompt_text=[ref_text] if ref_text else None,
                prompt_tokens=[prompt_tokens] if ref_text else None,
            ):
                if response.action == "sample":
                    all_codes.append(response.codes)
                elif response.action == "next":
                    break

            if not all_codes:
                raise RuntimeError("Fish Speech generation produced no output codes.")

            if progress_callback:
                progress_callback(0.8, desc="Decoding audio...")

            # Merge codes and decode to audio waveform
            merged_codes = torch.cat(all_codes, dim=1)
            audio_tensor = decode_to_audio(merged_codes.to(device), codec)
            audio_np = audio_tensor.cpu().float().numpy()
            sr = codec.sample_rate

        return audio_np, sr

    # ============================================================
    # DRAMABOX METHODS
    # ============================================================

    def _resolve_dramabox_root(self):
        """Return the vendored DramaBox root, always from modules/dramabox/."""
        project_root = Path(__file__).resolve().parents[3]
        vendored = project_root / "modules" / "dramabox"
        if vendored.exists():
            return vendored
        raise RuntimeError(
            f"Vendored DramaBox not found at {vendored}. "
            "Ensure modules/dramabox/ exists in the project."
        )

    def _ensure_dramabox_paths(self, dramabox_root):
        """Add DramaBox subdirectories to sys.path if not already present."""
        src_dir = dramabox_root / "src"
        ltx_dir = dramabox_root / "ltx2"
        for path in (str(dramabox_root), str(src_dir), str(ltx_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)

    def _load_module_from_path(self, module_name, module_path):
        """Dynamically load a Python module from a file path."""
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module: {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _ensure_dramabox_modules(self):
        """Resolve and load DramaBox modules (model_downloader, inference)."""
        dramabox_root = self._resolve_dramabox_root()

        if self._dramabox_root != dramabox_root:
            self._dramabox_root = dramabox_root
            self._dramabox_downloader_mod = None
            self._dramabox_inference_mod = None
            self._dramabox_supported_flags = None

        self._ensure_dramabox_paths(dramabox_root)

        src_dir = dramabox_root / "src"
        downloader_path = src_dir / "model_downloader.py"
        inference_path = src_dir / "inference.py"

        if not downloader_path.exists():
            raise RuntimeError(f"DramaBox model_downloader not found: {downloader_path}")
        if not inference_path.exists():
            raise RuntimeError(f"DramaBox inference script not found: {inference_path}")

        if self._dramabox_downloader_mod is None:
            self._dramabox_downloader_mod = self._load_module_from_path("dramabox_model_downloader", downloader_path)
        if self._dramabox_inference_mod is None:
            self._dramabox_inference_mod = self._load_module_from_path("dramabox_inference", inference_path)

        return dramabox_root

    def _get_supported_dramabox_flags(self):
        """Extract supported CLI flags from DramaBox src/inference.py."""
        if self._dramabox_supported_flags is not None:
            return self._dramabox_supported_flags

        flags = set()
        if self._dramabox_root is None:
            self._dramabox_supported_flags = flags
            return flags

        inference_path = self._dramabox_root / "src" / "inference.py"
        try:
            text = inference_path.read_text(encoding="utf-8")
            flags.update(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', text))
        except OSError:
            pass

        self._dramabox_supported_flags = flags
        return flags

    def _detect_lora_rank(self, lora_path):
        """Best-effort LoRA rank detection from a safetensors adapter file."""
        if not lora_path:
            return None

        try:
            from safetensors import safe_open
        except Exception:
            return None

        try:
            with safe_open(str(lora_path), framework="pt") as f:
                keys = list(f.keys())

                # Prefer A matrix rank: [r, in]
                for key in keys:
                    if key.endswith(".lora_A.weight"):
                        tensor = f.get_tensor(key)
                        if tensor.ndim >= 2 and tensor.shape[0] > 0:
                            return int(tensor.shape[0])

                # Fallback to B matrix rank: [out, r]
                for key in keys:
                    if key.endswith(".lora_B.weight"):
                        tensor = f.get_tensor(key)
                        if tensor.ndim >= 2 and tensor.shape[1] > 0:
                            return int(tensor.shape[1])
        except Exception:
            return None

        return None

    def generate_dramabox_to_file(self, prompt, output_path, voice_sample=None, seed=42, lora_path=None, cpu_offload=False, dramabox_params=None):
        """Generate audio using DramaBox TTS and write to output_path.

        When cpu_offload=False (default) a TTSServer instance is created once
        and kept warm, eliminating the ~19 s model-reload overhead — fast like
        the reference app.py.

        When cpu_offload=True the legacy argv path is used (lower peak VRAM but
        slow, reloads models on every call) and any existing warm server is torn
        down to free VRAM.
        """
        self._check_and_unload_if_different("dramabox_tts")

        if not prompt or not str(prompt).strip():
            raise RuntimeError("Prompt is empty.")

        dramabox_root = self._ensure_dramabox_modules()
        params = dramabox_params or {}

        # ------------------------------------------------------------------
        # Warm-server path (cpu_offload=False)
        # ------------------------------------------------------------------
        if not cpu_offload:
            # If server is already warm, skip get_all_paths() entirely — it hits
            # HuggingFace every call even when files are cached, printing noisy
            # progress output. The paths are already baked into _dramabox_server_key.
            if self._dramabox_server is not None:
                paths = None  # not needed — server is already loaded
            else:
                paths = self._dramabox_downloader_mod.get_all_paths()

            server_key = (
                str(paths["transformer"]),
                str(paths["audio_components"]),
                str(paths["gemma_root"]),
            ) if paths else self._dramabox_server_key

            if self._dramabox_server is not None and self._dramabox_server_key != server_key:
                logging.info("DramaBox: checkpoint changed — unloading old warm server")
                del self._dramabox_server
                self._dramabox_server = None
                self._dramabox_server_key = None
                empty_device_cache()

            if self._dramabox_server is None:
                logging.info("DramaBox: creating warm TTSServer (models load once)...")
                from inference_server import TTSServer
                self._dramabox_server = TTSServer(
                    checkpoint=str(paths["transformer"]),
                    full_checkpoint=str(paths["audio_components"]),
                    gemma_root=str(paths["gemma_root"]),
                    device="cuda",
                    dtype="bf16",
                    compile_model=False,  # torch.compile fails on Windows (missing omp.h)
                    bnb_4bit=True,
                )
                self._dramabox_server_key = server_key

            # Resolve LoRA rank
            lora_rank = params.get("lora_rank")
            if lora_rank is None and lora_path:
                lora_rank = self._detect_lora_rank(lora_path) or 128

            # Map rescale_scale: slider value -1 means "auto"
            raw_rescale = params.get("rescale_scale")
            if raw_rescale is not None and float(raw_rescale) < 0:
                rescale_val = "auto"
            elif raw_rescale is not None:
                rescale_val = float(raw_rescale)
            else:
                rescale_val = "auto"

            # steps=0 means "use default"
            raw_steps = params.get("steps")
            steps_val = int(raw_steps) if raw_steps and int(raw_steps) > 0 else 30

            self._dramabox_server.generate_to_file(
                prompt=str(prompt).strip(),
                output=str(output_path),
                voice_ref=str(voice_sample) if voice_sample else None,
                seed=int(seed),
                lora_path=str(lora_path) if lora_path else None,
                lora_rank=int(lora_rank) if lora_rank else 128,
                watermark=not bool(params.get("no_watermark", False)),
                cfg_scale=float(params["cfg_scale"]) if params.get("cfg_scale") is not None else 2.5,
                stg_scale=float(params["stg_scale"]) if params.get("stg_scale") is not None else 1.5,
                rescale_scale=rescale_val,
                duration_multiplier=float(params["duration_multiplier"]) if params.get("duration_multiplier") is not None else 1.1,
                gen_duration=float(params["gen_duration"]) if params.get("gen_duration") is not None else 0.0,
                ref_duration=float(params["ref_duration"]) if params.get("ref_duration") is not None else 10.0,
                steps=steps_val,
                sampler=params.get("sampler") or "euler",
                negative_prompt=params.get("negative_prompt") or None,
                speed=float(params["speed"]) if params.get("speed") is not None else 1.0,
                stg_block=int(params["stg_block"]) if params.get("stg_block") is not None else 29,
            )

            out_file = Path(output_path)
            if not out_file.exists():
                raise RuntimeError("DramaBox did not produce an output audio file.")
            return str(out_file)

        # ------------------------------------------------------------------
        # CPU-offload / legacy argv path (cpu_offload=True)
        # ------------------------------------------------------------------

        # Unload the warm server if the user switched back to cpu_offload mode
        if self._dramabox_server is not None:
            logging.info("DramaBox: switching to cpu-offload mode — unloading warm server")
            del self._dramabox_server
            self._dramabox_server = None
            self._dramabox_server_key = None
            empty_device_cache()

        paths = self._dramabox_downloader_mod.get_all_paths()
        argv = [
            "--prompt", str(prompt).strip(),
            "--output", str(output_path),
            "--seed", str(int(seed)),
            "--checkpoint", str(paths["transformer"]),
            "--full-checkpoint", str(paths["audio_components"]),
            "--gemma-root", str(paths["gemma_root"]),
        ]

        if voice_sample:
            argv.extend(["--voice-sample", str(voice_sample)])
        else:
            argv.append("--no-ref")

        supported_flags = self._get_supported_dramabox_flags()

        if "--cpu-offload" in supported_flags:
            argv.append("--cpu-offload")

        if lora_path:
            argv.extend(["--lora", str(lora_path)])

        if params.get("ref_duration") is not None:
            argv.extend(["--ref-duration", str(float(params.get("ref_duration")))])
        if params.get("gen_duration") is not None:
            argv.extend(["--gen-duration", str(float(params.get("gen_duration")))])
        if params.get("pad_start") is not None:
            argv.extend(["--pad-start", str(float(params.get("pad_start")))])
        if params.get("speed") is not None:
            argv.extend(["--speed", str(float(params.get("speed")))])
        if params.get("duration_multiplier") is not None:
            argv.extend(["--duration-multiplier", str(float(params.get("duration_multiplier")))])

        if params.get("sampler") in {"euler", "heun"}:
            argv.extend(["--sampler", params.get("sampler")])

        if params.get("cfg_scale") is not None:
            argv.extend(["--cfg-scale", str(float(params.get("cfg_scale")))])
        if params.get("stg_scale") is not None:
            argv.extend(["--stg-scale", str(float(params.get("stg_scale")))])
        if params.get("stg_block") is not None:
            argv.extend(["--stg-block", str(int(params.get("stg_block")))])
        if params.get("rescale_scale") is not None:
            argv.extend(["--rescale-scale", str(float(params.get("rescale_scale")))])
        if params.get("modality_scale") is not None:
            argv.extend(["--modality-scale", str(float(params.get("modality_scale")))])
        if params.get("cfg_clamp") is not None:
            argv.extend(["--cfg-clamp", str(float(params.get("cfg_clamp")))])
        if params.get("steps") is not None:
            argv.extend(["--steps", str(int(params.get("steps")))])
        if params.get("fps") is not None:
            argv.extend(["--fps", str(float(params.get("fps")))])
        if params.get("negative_prompt"):
            argv.extend(["--negative-prompt", str(params.get("negative_prompt"))])

        lora_rank = params.get("lora_rank")
        if lora_rank is None and lora_path:
            lora_rank = self._detect_lora_rank(lora_path)

        if lora_rank is not None and "--lora-rank" in supported_flags:
            argv.extend(["--lora-rank", str(int(lora_rank))])
        if params.get("id_guidance_scale") is not None:
            argv.extend(["--id-guidance-scale", str(float(params.get("id_guidance_scale")))])

        if params.get("no_watermark"):
            argv.append("--no-watermark")

        if params.get("bnb_4bit") is False:
            argv.append("--no-bnb-4bit")

        old_cwd = os.getcwd()
        try:
            os.chdir(str(dramabox_root))
            self._invoke_dramabox_inference(argv)
        finally:
            os.chdir(old_cwd)

        out_file = Path(output_path)
        if not out_file.exists():
            raise RuntimeError("DramaBox did not produce an output audio file.")

        return str(out_file)

    def _invoke_dramabox_inference(self, argv):
        """Run DramaBox inference across both legacy and modern entry-point signatures."""
        mod = self._dramabox_inference_mod
        if mod is None:
            raise RuntimeError("DramaBox inference module is not loaded.")

        if hasattr(mod, "main"):
            main_fn = mod.main
            try:
                sig = inspect.signature(main_fn)
                accepts_args = len(sig.parameters) > 0
            except Exception:
                accepts_args = True

            if accepts_args:
                return main_fn(argv)

            old_argv = sys.argv[:]
            try:
                sys.argv = ["inference.py"] + list(argv)
                return main_fn()
            finally:
                sys.argv = old_argv

        if hasattr(mod, "parse_args") and hasattr(mod, "run_inference"):
            parsed = mod.parse_args(argv)
            return mod.run_inference(parsed)

        raise RuntimeError("DramaBox inference module has no supported entry point.")

    def unload_all(self):
        """Unload all TTS models to free VRAM."""
        freed = []

        if self._qwen3_base_model is not None:
            del self._qwen3_base_model
            self._qwen3_base_model = None
            freed.append("Qwen3 Base")

        if self._qwen3_voice_design_model is not None:
            del self._qwen3_voice_design_model
            self._qwen3_voice_design_model = None
            freed.append("Qwen3 VoiceDesign")

        if self._qwen3_custom_voice_model is not None:
            del self._qwen3_custom_voice_model
            self._qwen3_custom_voice_model = None
            freed.append("Qwen3 CustomVoice")

        if self._vibevoice_tts_model is not None:
            del self._vibevoice_tts_model
            self._vibevoice_tts_model = None
            freed.append("VibeVoice TTS")

        if self._luxtts_model is not None:
            del self._luxtts_model
            self._luxtts_model = None
            freed.append("LuxTTS")

        if self._chatterbox_tts_model is not None:
            del self._chatterbox_tts_model
            self._chatterbox_tts_model = None
            freed.append("Chatterbox TTS")

        if self._chatterbox_vc_model is not None:
            del self._chatterbox_vc_model
            self._chatterbox_vc_model = None
            freed.append("Chatterbox VC")

        if self._chatterbox_mtl_model is not None:
            del self._chatterbox_mtl_model
            self._chatterbox_mtl_model = None
            freed.append("Chatterbox Multilingual")

        if self._fish_speech_model is not None:
            del self._fish_speech_model
            self._fish_speech_model = None
            freed.append("Fish Speech Model")

        if self._fish_speech_codec is not None:
            del self._fish_speech_codec
            self._fish_speech_codec = None
            freed.append("Fish Speech Codec")

        if self._fish_speech_decode_one_token is not None:
            self._fish_speech_decode_one_token = None

        self._fish_speech_checkpoint_dir = None

        if self._trained_model is not None:
            del self._trained_model
            self._trained_model = None
            self._trained_model_path = None
            self._trained_model_is_faster = False
            freed.append("Trained Model")

        if self._vibevoice_streaming_model is not None:
            del self._vibevoice_streaming_model
            self._vibevoice_streaming_model = None
            self._vibevoice_streaming_processor = None
            self._vibevoice_streaming_voice_cache = {}
            freed.append("VibeVoice Streaming")

        if self._trained_vibevoice_model is not None:
            del self._trained_vibevoice_model
            self._trained_vibevoice_model = None
            self._trained_vibevoice_path = None
            self._trained_vv_base_connectors = None
            self._trained_vv_lora_connectors = None
            self._trained_vv_base_pred_head = None
            self._trained_vv_lora_pred_head = None
            self._trained_vv_pred_head_is_peft = False
            freed.append("VibeVoice Trained")

        if self._dramabox_downloader_mod is not None or self._dramabox_inference_mod is not None:
            self._dramabox_downloader_mod = None
            self._dramabox_inference_mod = None
            self._dramabox_root = None
            self._dramabox_supported_flags = None
            freed.append("DramaBox")

        if self._dramabox_server is not None:
            del self._dramabox_server
            self._dramabox_server = None
            self._dramabox_server_key = None
            if "DramaBox" not in freed:
                freed.append("DramaBox (warm server)")

        if freed:
            gc.collect()
            empty_device_cache()
            print(f"Unloaded TTS models: {', '.join(freed)}")

        return bool(freed)

    # ============================================================
    # GENERATION METHODS
    # ============================================================

    def generate_voice_design(self, text: str, language: str, instruct: str, seed: int = -1,
                              do_sample: bool = True, temperature: float = 0.9, top_k: int = 50,
                              top_p: float = 1.0, repetition_penalty: float = 1.05,
                              max_new_tokens: int = 2048) -> Tuple[str, int]:
        """
        Generate audio using voice design with natural language instructions.

        Args:
            text: Text to generate
            language: Language for TTS
            instruct: Natural language voice design instructions
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        # Load model
        model = self.get_qwen3_voice_design()

        # Generate
        wavs, sr = model.generate_voice_design(
            text=text.strip(),
            language=language if language != "Auto" else "Auto",
            instruct=instruct.strip(),
            do_sample=do_sample,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            max_new_tokens=int(max_new_tokens)
        )

        # Convert to numpy if needed
        audio_data = wavs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        return audio_data, sr

    def generate_custom_voice(self, text: str, language: str, speaker: str, instruct: str = None,
                              model_size: str = "1.7B", seed: int = -1,
                              do_sample: bool = True, temperature: float = 0.9, top_k: int = 50,
                              top_p: float = 1.0, repetition_penalty: float = 1.05,
                              max_new_tokens: int = 2048) -> Tuple[str, int]:
        """
        Generate audio using CustomVoice model with premium speakers.

        Args:
            text: Text to generate
            language: Language for TTS
            speaker: Speaker name
            instruct: Optional style instructions
            model_size: Model size (1.7B, 0.6B, etc.)
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        # Load model
        model = self.get_qwen3_custom_voice(model_size)

        # Build kwargs
        kwargs = {
            "text": text.strip(),
            "language": language if language != "Auto" else "Auto",
            "speaker": speaker,
            "do_sample": do_sample,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens)
        }
        if instruct and instruct.strip():
            kwargs["instruct"] = instruct.strip()

        wavs, sr = model.generate_custom_voice(**kwargs)

        # Convert to numpy if needed
        audio_data = wavs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        return audio_data, sr

    def generate_with_trained_model(self, text, language, speaker_name,
                                    checkpoint_path, instruct=None, seed=-1,
                                    do_sample=True, temperature=0.9, top_k=50,
                                    top_p=1.0, repetition_penalty=1.05,
                                    max_new_tokens=2048, user_config=None,
                                    icl_mode=False, voice_sample_path=None, ref_text=None):
        """
        Generate audio using a trained custom voice model checkpoint.

        Supports two modes:
        - Speaker embedding mode (default): Uses the baked-in speaker embedding
        - ICL mode: Uses In-Context Learning with a reference audio sample for
          much more natural and expressive voice cloning

        Args:
            text: Text to generate
            language: Language for TTS
            speaker_name: Speaker name the model was trained with
            checkpoint_path: Path to trained model checkpoint
            instruct: Optional style instructions (only used in speaker embedding mode)
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate
            user_config: User configuration dict
            icl_mode: If True, use In-Context Learning with voice sample
            voice_sample_path: Path to reference audio (required for ICL mode)
            ref_text: Transcript of the reference audio (required for ICL mode)

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        if user_config is None:
            user_config = {}

        # Determine device and dtype
        device = get_device(user_config.get("tts_gpu", 0))
        dtype = get_dtype(device)

        # Reuse cached trained model if same checkpoint
        checkpoint_str = str(checkpoint_path)
        if self._trained_model is not None and self._trained_model_path == checkpoint_str:
            model = self._trained_model
            is_faster = self._trained_model_is_faster
            print(f"Reusing cached trained model: {Path(checkpoint_str).name}")
        else:
            # Unload previous trained model if switching checkpoints
            if self._trained_model is not None:
                print("Switching trained model checkpoint, unloading previous...")
                del self._trained_model
                self._trained_model = None
                self._trained_model_path = None
                empty_device_cache()

            # Try CUDA graphs acceleration first, fall back to standard loading
            use_faster = self._use_cuda_graphs()
            model = None

            if use_faster:
                try:
                    from faster_qwen3_tts import FasterQwen3TTS
                    model = FasterQwen3TTS.from_pretrained(
                        checkpoint_path,
                        device="cuda",
                        dtype=dtype,
                        attn_implementation="eager",
                    )
                    print("[OK] Trained model loaded with CUDA graphs acceleration")
                except Exception as e:
                    print(f"  CUDA graphs failed for trained model, falling back to standard: {e}")
                    model = None

            if model is None:
                from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
                # Load with attention fallback
                mechanisms = get_attention_implementation(
                    user_config.get("attention_mechanism", "auto")
                )
                for attn in mechanisms:
                    try:
                        model = Qwen3TTSModel.from_pretrained(
                            checkpoint_path,
                            device_map=device,
                            torch_dtype=dtype,
                            attn_implementation=attn,
                            low_cpu_mem_usage=user_config.get("low_cpu_mem_usage", False)
                        )
                        print(f"Trained model loaded with {attn}")
                        break
                    except Exception as e:
                        error_msg = str(e).lower()
                        is_attn_error = any(
                            kw in error_msg for kw in ["flash", "attention", "sdpa", "not supported"]
                        )
                        if is_attn_error:
                            print(f"  {attn} not available for trained model, trying next...")
                            continue
                        raise

            if model is None:
                raise RuntimeError("Failed to load trained model with any attention mechanism")

            is_faster = hasattr(model, 'talker_graph')

            # Cache the loaded model
            self._trained_model = model
            self._trained_model_path = checkpoint_str
            self._trained_model_is_faster = is_faster

        # FasterQwen3TTS wraps the Qwen3TTSModel: model.model is the inner Qwen3TTSModel
        # For standard Qwen3TTSModel: model itself is the Qwen3TTSModel
        qwen_model = model.model if is_faster else model

        if icl_mode and voice_sample_path and ref_text:
            # ICL mode: use generate_voice_clone with reference audio
            # Ensure model type allows voice cloning (patch legacy "custom_voice" models)
            if hasattr(qwen_model, 'model') and hasattr(qwen_model.model, 'tts_model_type'):
                if qwen_model.model.tts_model_type != "base":
                    print(f"Patching tts_model_type from '{qwen_model.model.tts_model_type}' to 'base' for ICL inference")
                    qwen_model.model.tts_model_type = "base"

            # Training drops speaker_encoder weights from checkpoints to save space.
            # ICL needs it to compute speaker embeddings from reference audio.
            # Borrow speaker_encoder from the base model of matching size.
            if qwen_model.model.speaker_encoder is None:
                import json as json_mod
                config_path = Path(checkpoint_path) / "config.json"
                with open(config_path, "r", encoding="utf-8") as f:
                    ckpt_config = json_mod.load(f)
                base_size = "1.7B" if ckpt_config.get("tts_model_size") == "1b7" else "0.6B"
                base_name = f"Qwen/Qwen3-TTS-12Hz-{base_size}-Base"
                print(f"Speaker encoder missing - borrowing from {base_size} base model...")

                # Try local first, then HuggingFace
                local_path = check_model_available_locally(base_name)
                base_to_load = str(local_path) if local_path else base_name

                from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as _Qwen3TTSModel
                base_model = _Qwen3TTSModel.from_pretrained(
                    base_to_load,
                    device_map=device,
                    torch_dtype=dtype,
                )
                qwen_model.model.speaker_encoder = base_model.model.speaker_encoder
                qwen_model.model.speaker_encoder_sample_rate = base_model.model.speaker_encoder_sample_rate
                del base_model
                empty_device_cache()
                print("Speaker encoder transplanted successfully")

            # Build generation kwargs
            gen_kwargs = {
                'max_new_tokens': int(max_new_tokens),
            }
            if do_sample:
                gen_kwargs['do_sample'] = True
                gen_kwargs['temperature'] = float(temperature)
                if top_k > 0:
                    gen_kwargs['top_k'] = int(top_k)
                if top_p < 1.0:
                    gen_kwargs['top_p'] = float(top_p)
                if repetition_penalty != 1.0:
                    gen_kwargs['repetition_penalty'] = float(repetition_penalty)

            if is_faster:
                # FasterQwen3TTS: pass ref_audio/ref_text directly
                # xvec_only=False enables full ICL (reference audio in context)
                # instead of just speaker embedding, matching standard Qwen3 quality
                wavs, sr = model.generate_voice_clone(
                    text=text.strip(),
                    language=language if language != "Auto" else "Auto",
                    ref_audio=str(voice_sample_path),
                    ref_text=ref_text,
                    xvec_only=False,
                    **gen_kwargs
                )
            else:
                # Standard: create voice clone prompt first
                prompt_items = model.create_voice_clone_prompt(
                    ref_audio=str(voice_sample_path),
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                )
                wavs, sr = model.generate_voice_clone(
                    text=text.strip(),
                    language=language if language != "Auto" else "Auto",
                    voice_clone_prompt=prompt_items,
                    **gen_kwargs
                )
        else:
            # Speaker embedding mode: use generate_custom_voice
            # Ensure model type allows custom voice (patch "base" models)
            if hasattr(qwen_model, 'model') and hasattr(qwen_model.model, 'tts_model_type'):
                if qwen_model.model.tts_model_type != "custom_voice":
                    print(f"Patching tts_model_type from '{qwen_model.model.tts_model_type}' to 'custom_voice' for speaker embedding inference")
                    qwen_model.model.tts_model_type = "custom_voice"

            kwargs = {
                "text": text.strip(),
                "language": language if language != "Auto" else "Auto",
                "speaker": speaker_name,
                "do_sample": do_sample,
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "max_new_tokens": int(max_new_tokens)
            }
            if instruct and instruct.strip():
                kwargs["instruct"] = instruct.strip()

            wavs, sr = model.generate_custom_voice(**kwargs)

        # Convert to numpy if needed
        audio_data = wavs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        return audio_data, sr

    def generate_vibevoice_streaming(self, text, voice_name, cfg_scale=1.5,
                                     ddpm_steps=20, seed=-1):
        """
        Generate audio using VibeVoice Streaming 0.5B with baked-in voices.

        Args:
            text: Text to generate
            voice_name: Name of pre-built voice (Carter, Davis, Emma, etc.)
            cfg_scale: Classifier-free guidance scale
            ddpm_steps: Number of DDPM denoising steps
            seed: Random seed (-1 for random)

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        import copy

        if seed < 0:
            seed = random.randint(0, 2147483647)
        set_seed(seed)

        print(f"VibeVoice Speakers: voice={voice_name}, seed={seed}, cfg={cfg_scale}, steps={ddpm_steps}")
        print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        print("Loading model...")
        model, processor = self.get_vibevoice_streaming()

        print(f"Loading voice prompt: {voice_name}...")
        cached_prompt = self._get_streaming_voice_prompt(voice_name)

        print("Processing input...")
        # Process text with cached voice prompt
        inputs = processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move inputs to model device
        device = next(model.parameters()).device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        # Generate
        import time
        print("Generating audio...")
        gen_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                tokenizer=processor.tokenizer,
                cfg_scale=cfg_scale,
                ddpm_steps=ddpm_steps,
                all_prefilled_outputs=copy.deepcopy(cached_prompt),
            )
        gen_time = time.time() - gen_start
        print(f"Generation complete in {gen_time:.1f}s")

        # Extract audio from generation output
        print("Decoding audio...")
        sr = 24000
        audio_data = None
        if outputs.speech_outputs:
            audio_data = outputs.speech_outputs[0]  # batch index 0

        if audio_data is None:
            raise RuntimeError("No speech output was generated")

        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.float().cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        # Flatten to 1D mono for soundfile
        audio_data = audio_data.squeeze()
        if audio_data.ndim > 1:
            audio_data = audio_data[0]

        return audio_data, sr

    def _apply_lora_scale(self, model, scale):
        """Apply a scaling factor to all trained components.

        Handles three kinds of trained weights:
        1. PEFT LoRA layers (language model, optionally diffusion head):
           Uses set_scale to multiply the adapter contribution.
        2. Full-weight diffusion head (diffusion_head_full.bin):
           Linearly interpolates between original and trained state dicts.
        3. Connectors (acoustic_connector, semantic_connector):
           Linearly interpolates between original and trained state dicts.

        At scale 0.0 the model behaves as the unmodified base model.
        At scale 1.0 all trained weights are fully applied.

        Args:
            model: The VibeVoice model with trained weights loaded
            scale: Float multiplier (0.0 = base model, 1.0 = full trained)
        """
        # --- 1. PEFT LoRA layers (language model + optionally prediction head) ---
        try:
            from peft.tuners.lora.layer import LoraLayer
        except ImportError:
            LoraLayer = None

        lora_count = 0
        if LoraLayer is not None:
            for component in [model.model.language_model, model.model.prediction_head]:
                if component is None:
                    continue
                for module in component.modules():
                    if isinstance(module, LoraLayer):
                        for adapter_name in module.scaling:
                            module.set_scale(adapter_name, scale)
                        lora_count += 1

        # --- 2. Full-weight diffusion head interpolation ---
        base_pred = self._trained_vv_base_pred_head
        lora_pred = self._trained_vv_lora_pred_head
        if base_pred and lora_pred and not self._trained_vv_pred_head_is_peft:
            device = next(model.model.prediction_head.parameters()).device
            interpolated = {}
            for key in base_pred:
                if key in lora_pred:
                    interpolated[key] = (
                        base_pred[key] * (1.0 - scale) + lora_pred[key] * scale
                    ).to(device)
                else:
                    interpolated[key] = base_pred[key].to(device)
            model.model.prediction_head.load_state_dict(interpolated)

        # --- 3. Connector interpolation ---
        base_connectors = self._trained_vv_base_connectors
        lora_connectors = self._trained_vv_lora_connectors
        if base_connectors and lora_connectors:
            for cname in base_connectors:
                connector = getattr(model.model, cname, None)
                if connector is None or cname not in lora_connectors:
                    continue
                base_sd = base_connectors[cname]
                lora_sd = lora_connectors[cname]
                interpolated = {}
                for key in base_sd:
                    if key in lora_sd:
                        interpolated[key] = base_sd[key] * (1.0 - scale) + lora_sd[key] * scale
                    else:
                        interpolated[key] = base_sd[key]
                connector.load_state_dict(interpolated)

        print(f"LoRA scale applied: {scale:.2f} "
              f"({lora_count} LoRA layers, "
              f"pred_head={'interpolated' if (base_pred and lora_pred and not self._trained_vv_pred_head_is_peft) else 'peft' if self._trained_vv_pred_head_is_peft else 'unchanged'}, "
              f"connectors={'interpolated' if (base_connectors and lora_connectors) else 'unchanged'})")

    def generate_with_trained_vibevoice(self, text, language, checkpoint_path,
                                        seed=-1, do_sample=False,
                                        temperature=1.0, top_k=50,
                                        top_p=1.0, repetition_penalty=1.0,
                                        cfg_scale=1.3, num_steps=10,
                                        max_new_tokens=2048, user_config=None,
                                        voice_sample_path=None,
                                        lora_scale=1.0,
                                        progress_callback=None):
        """
        Generate audio using a trained VibeVoice LoRA checkpoint.

        Loads the base VibeVoice 1.5B model, applies LoRA adapters,
        and generates speech. Optionally uses a voice sample for additional
        voice conditioning on top of the LoRA weights.

        Args:
            text: Text to generate
            language: Language for TTS (used in prompt formatting)
            checkpoint_path: Path to trained model folder (containing lora/ subdir)
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            cfg_scale: Classifier-free guidance scale
            num_steps: Number of DDPM denoising steps
            max_new_tokens: Maximum tokens to generate
            user_config: User configuration dict
            voice_sample_path: Optional path to voice sample WAV for additional conditioning
            lora_scale: LoRA adapter strength multiplier (0.0 = base model, 1.0 = full LoRA)

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random

        if seed < 0:
            seed = random.randint(0, 2147483647)
        set_seed(seed)

        if user_config is None:
            user_config = {}

        device = get_device(user_config.get("tts_gpu", 0))
        dtype = get_dtype(device)
        checkpoint_str = str(checkpoint_path)

        # Reuse cached model if same checkpoint
        if self._trained_vibevoice_model is not None and self._trained_vibevoice_path == checkpoint_str:
            model = self._trained_vibevoice_model
            model.set_ddpm_inference_steps(num_steps=int(num_steps))
            print(f"Reusing cached VibeVoice LoRA model: {Path(checkpoint_str).name}")
        else:
            # Unload previous
            if self._trained_vibevoice_model is not None:
                print("Switching VibeVoice LoRA checkpoint, unloading previous...")
                del self._trained_vibevoice_model
                self._trained_vibevoice_model = None
                self._trained_vibevoice_path = None
                empty_device_cache()

            # Unload other models to free VRAM
            self._check_and_unload_if_different("trained_vibevoice")

            print(f"Loading VibeVoice base model + LoRA from {Path(checkpoint_str).name}...")
            if progress_callback:
                progress_callback(0.2, desc="Loading VibeVoice base model...")

            from modules.vibevoice_tts.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference
            )
            from modules.vibevoice_tts.modular.lora_loading import load_lora_assets

            # Determine which base model to load from checkpoint metadata.
            # vcs_metadata.json lives inside the lora/ folder so it ships with shared models.
            # Fall back to 1.5B for checkpoints trained before this feature.
            base_model_size = "1.5B"
            checkpoint_path = Path(checkpoint_str)
            # Check lora/ subdir first (primary location), then checkpoint root
            for candidate_dir in [checkpoint_path / "lora", checkpoint_path]:
                meta_candidate = candidate_dir / "vcs_metadata.json"
                if meta_candidate.exists():
                    try:
                        import json as _json
                        meta = _json.loads(meta_candidate.read_text(encoding="utf-8"))
                        base_model_size = meta.get("base_model_size", "1.5B")
                    except Exception:
                        pass
                    break

            # Load base model — try FranckyB variant first (same weights,
            # often already cached from voice cloning)
            offline_mode = user_config.get("offline_mode", False)
            load_path = None
            if base_model_size == "7B":
                candidates = ["FranckyB/VibeVoice-Large", "vibevoice/VibeVoice-7B"]
            else:
                candidates = ["FranckyB/VibeVoice-1.5B", "vibevoice/VibeVoice-1.5B"]
            for candidate_id in candidates:
                local_path = check_model_available_locally(candidate_id)
                if local_path:
                    load_path = str(local_path)
                    break
            if not load_path:
                # Fall back to HF hub resolution
                load_path = candidates[0]

            # Pick attention implementation (flash_attention_2 for CUDA, sdpa otherwise)
            attn_candidates = get_attention_implementation(
                user_config.get("attention_mechanism", "auto")
            )
            attn_impl = attn_candidates[0] if isinstance(attn_candidates, list) else attn_candidates

            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                try:
                    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        load_path,
                        torch_dtype=dtype,
                        local_files_only=offline_mode,
                        attn_implementation=attn_impl,
                    ).to(device)
                except Exception as e:
                    if attn_impl == "flash_attention_2":
                        print(f"Flash Attention 2 failed ({e}), falling back to SDPA")
                        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            load_path,
                            torch_dtype=dtype,
                            local_files_only=offline_mode,
                            attn_implementation="sdpa",
                        ).to(device)
                    else:
                        raise

            # Configure noise scheduler (SDE solver, matches upstream)
            model.model.noise_scheduler = model.model.noise_scheduler.from_config(
                model.model.noise_scheduler.config,
                algorithm_type='sde-dpmsolver++',
                beta_schedule='squaredcos_cap_v2',
            )
            model.set_ddpm_inference_steps(num_steps=int(num_steps))

            # Snapshot original weights before LoRA replaces them.
            # Connectors stay on-device (small), prediction head goes to CPU
            # (can be 235 MB) to avoid doubling VRAM usage.
            import copy
            self._trained_vv_base_connectors = {}
            for cname in ('acoustic_connector', 'semantic_connector'):
                connector = getattr(model.model, cname, None)
                if connector is not None:
                    self._trained_vv_base_connectors[cname] = copy.deepcopy(connector.state_dict())

            # Snapshot prediction head on CPU before LoRA loading
            self._trained_vv_base_pred_head = {
                k: v.clone().cpu() for k, v in model.model.prediction_head.state_dict().items()
            }

            # Apply LoRA adapters (supports both model/lora/files and model/files layouts)
            if progress_callback:
                progress_callback(0.35, desc="Applying LoRA adapters...")
            report = load_lora_assets(model, checkpoint_str)
            print(f"LoRA loaded: {report}")

            # Snapshot trained connector weights so we can interpolate later
            self._trained_vv_lora_connectors = {}
            for cname in ('acoustic_connector', 'semantic_connector'):
                connector = getattr(model.model, cname, None)
                if connector is not None:
                    self._trained_vv_lora_connectors[cname] = copy.deepcopy(connector.state_dict())

            # Diffusion head: if loaded as full weights (not LoRA), snapshot
            # the trained weights for interpolation.
            self._trained_vv_pred_head_is_peft = report.diffusion_head_lora
            if report.diffusion_head_full:
                self._trained_vv_lora_pred_head = {
                    k: v.clone().cpu()
                    for k, v in model.model.prediction_head.state_dict().items()
                }
                print(f"Prediction head full weights cached for interpolation "
                      f"({len(self._trained_vv_lora_pred_head)} tensors)")
            else:
                self._trained_vv_lora_pred_head = None

            # Cache
            self._trained_vibevoice_model = model
            self._trained_vibevoice_path = checkpoint_str

        # Load processor — try FranckyB variant first (same tokenizer)
        if progress_callback:
            progress_callback(0.5, desc="Loading processor...")
        from modules.vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor
        offline_mode = user_config.get("offline_mode", False)

        processor_path = None
        for candidate_id in ["FranckyB/VibeVoice-1.5B", "vibevoice/VibeVoice-1.5B"]:
            local_path = check_model_available_locally(candidate_id)
            if local_path:
                processor_path = str(local_path)
                break
        if not processor_path:
            processor_path = "FranckyB/VibeVoice-1.5B"

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The tokenizer class you load from this checkpoint")
            try:
                processor = VibeVoiceProcessor.from_pretrained(
                    processor_path,
                    local_files_only=offline_mode,
                )
            except Exception:
                # Last resort: try the other repo ID
                fallback_id = "vibevoice/VibeVoice-1.5B"
                fb_local = check_model_available_locally(fallback_id)
                if fb_local:
                    processor = VibeVoiceProcessor.from_pretrained(str(fb_local))
                else:
                    processor = VibeVoiceProcessor.from_pretrained(fallback_id)

        # Apply LoRA scaling if not default (1.0)
        if lora_scale != 1.0:
            self._apply_lora_scale(model, float(lora_scale))
            print(f"LoRA strength: {lora_scale:.2f}")
        else:
            # Reset to default scale in case it was changed previously
            self._apply_lora_scale(model, 1.0)

        # Build generation config (same pattern as voice clone)
        gen_config = {'do_sample': do_sample}
        if do_sample:
            gen_config['temperature'] = float(temperature)
            if top_k > 0:
                gen_config['top_k'] = int(top_k)
            if top_p < 1.0:
                gen_config['top_p'] = float(top_p)
            if repetition_penalty != 1.0:
                gen_config['repetition_penalty'] = float(repetition_penalty)

        # Format input text with Speaker prefix required by processor
        # Speaker 0 = no voice conditioning, Speaker 1 = with voice sample
        formatted_text = f"[{language}] {text.strip()}" if language and language != "Auto" else text.strip()
        use_voice_sample = voice_sample_path is not None
        speaker_id = "1" if use_voice_sample else "0"
        formatted_text = f"Speaker {speaker_id}: {formatted_text}"

        if use_voice_sample:
            print(f"Using voice sample for conditioning: {Path(voice_sample_path).name}")

        if progress_callback:
            progress_callback(0.6, desc="Generating audio...")

        with torch.no_grad():
            inputs = processor(
                text=[formatted_text],
                voice_samples=[[voice_sample_path]] if use_voice_sample else None,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move to device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # is_prefill=True enables speech prefill (voice conditioning)
            # Must be True when voice sample is provided, False otherwise
            wavs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=float(cfg_scale),
                tokenizer=processor.tokenizer,
                generation_config=gen_config,
                verbose=False,
                is_prefill=use_voice_sample,
                progress_callback=progress_callback,
                progress_start=0.6,
                progress_end=0.95,
            )

        # VibeVoiceGenerationOutput: .sequences (token IDs), .speech_outputs (list of audio tensors)
        if not wavs.speech_outputs or wavs.speech_outputs[0] is None:
            raise RuntimeError("Model generated tokens but produced no speech audio. Try increasing max_new_tokens or adjusting CFG scale.")

        audio_data = wavs.speech_outputs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.float().cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        import numpy as np
        # Squeeze extra dims (e.g. (1, T) or (1, 1, T) → (T,)) — matches upstream save_audio
        audio_data = np.squeeze(audio_data)
        if audio_data.dtype not in (np.float32, np.float64, np.int16, np.int32):
            audio_data = audio_data.astype(np.float32)

        sr = 24000
        return audio_data, sr

    def generate_voice_clone_qwen(self, text, language, prompt_items, seed=-1,
                                  do_sample=True, temperature=0.9, top_k=50,
                                  top_p=1.0, repetition_penalty=1.05,
                                  max_new_tokens=2048, model_size="1.7B",
                                  ref_audio=None, ref_text=None):
        """
        Generate audio using Qwen3 voice cloning.

        When CUDA graphs are enabled, uses ref_audio/ref_text directly.
        Otherwise, uses pre-computed voice_clone_prompt (prompt_items).

        Args:
            text: Text to generate
            language: Language for TTS
            prompt_items: Pre-computed voice clone prompt (standard path)
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate
            model_size: Model size (1.7B or 0.6B)
            ref_audio: Path to reference audio (CUDA graphs path)
            ref_text: Transcript of reference audio (CUDA graphs path)

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        # Load BASE model (not CustomVoice - Base supports voice cloning)
        model = self.get_qwen3_base(model_size)

        # Check if this is a FasterQwen3TTS model (CUDA graphs)
        is_faster = hasattr(model, 'talker_graph')

        if is_faster and ref_audio:
            # FasterQwen3TTS: pass ref_audio/ref_text directly
            # xvec_only=False enables full ICL for proper voice matching
            wavs, sr = model.generate_voice_clone(
                text=text.strip(),
                language=language if language != "Auto" else "Auto",
                ref_audio=ref_audio,
                ref_text=ref_text or "",
                xvec_only=False,
                do_sample=do_sample,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                max_new_tokens=int(max_new_tokens),
            )
        else:
            # Standard Qwen3TTSModel: use pre-computed voice_clone_prompt
            gen_kwargs = {
                'max_new_tokens': int(max_new_tokens),
            }
            if do_sample:
                gen_kwargs['do_sample'] = True
                gen_kwargs['temperature'] = float(temperature)
                if top_k > 0:
                    gen_kwargs['top_k'] = int(top_k)
                if top_p < 1.0:
                    gen_kwargs['top_p'] = float(top_p)
                if repetition_penalty != 1.0:
                    gen_kwargs['repetition_penalty'] = float(repetition_penalty)

            wavs, sr = model.generate_voice_clone(
                text=text.strip(),
                language=language if language != "Auto" else "Auto",
                voice_clone_prompt=prompt_items,
                **gen_kwargs
            )

        # Convert to numpy if needed
        audio_data = wavs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        return audio_data, sr

    def generate_voice_clone_vibevoice(self, text, voice_sample_path, seed=-1,
                                       do_sample=False, temperature=1.0, top_k=50,
                                       top_p=1.0, repetition_penalty=1.0,
                                       cfg_scale=1.3, num_steps=10,
                                       paragraph_per_chunk=False,
                                       model_size="Large", user_config=None,
                                       progress_callback=None):
        """
        Generate audio using VibeVoice voice cloning.

        Args:
            text: Text to generate
            voice_sample_path: Path to voice sample WAV file
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            cfg_scale: Classifier-free guidance scale
            num_steps: DDPM inference steps
            paragraph_per_chunk: Split text by paragraphs and generate each
                separately. Prevents quality degradation on long text.
            model_size: Model size (Large, 1.5B, or Large (4-bit))
            user_config: User configuration dict

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        import warnings
        import logging

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        # Load model
        if progress_callback:
            progress_callback(0.1, desc=f"Loading VibeVoice - {model_size}...")
        model = self.get_vibevoice_tts(model_size)

        from modules.vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor

        # Map model_size to valid HuggingFace repo path
        if model_size == "Large (4-bit)":
            model_path = "FranckyB/VibeVoice-Large-4bit"
        else:
            model_path = f"FranckyB/VibeVoice-{model_size}"

        # Suppress tokenizer mismatch warning
        prev_level = logging.getLogger("transformers.tokenization_utils_base").level
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if user_config is None:
                user_config = {}
            offline_mode = user_config.get("offline_mode", False)
            processor = VibeVoiceProcessor.from_pretrained(model_path, local_files_only=offline_mode)

        logging.getLogger("transformers.tokenization_utils_base").setLevel(prev_level)

        # Build generation config
        gen_config = {'do_sample': do_sample}
        if do_sample:
            gen_config['temperature'] = float(temperature)
            if top_k > 0:
                gen_config['top_k'] = int(top_k)
            if top_p < 1.0:
                gen_config['top_p'] = float(top_p)
            if repetition_penalty != 1.0:
                gen_config['repetition_penalty'] = float(repetition_penalty)

        sr = 24000  # VibeVoice uses 24kHz
        device = get_device(self.user_config.get("tts_gpu", 0))

        # Normalize newlines: collapse \r\n and multiple blank lines into a single
        # space so paragraph breaks don't produce lines without a "Speaker N:"
        # prefix, which VibeVoice's parser silently drops.
        import re
        text = re.sub(r'\r\n|\r', '\n', text)   # normalise CR/CRLF

        # If paragraph chunking is enabled, split by paragraph boundaries first,
        # then generate each paragraph separately and concatenate.  Each chunk
        # gets its own inference call so the generation state resets, preventing
        # quality degradation (screaming/rushing) on long text.
        if paragraph_per_chunk:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            if len(paragraphs) > 1:
                import numpy as np
                print(f"VibeVoice chunking: {len(paragraphs)} paragraphs")
                audio_segments = []

                for idx, para in enumerate(paragraphs):
                    if progress_callback:
                        pct = 0.2 + (idx / len(paragraphs)) * 0.7
                        progress_callback(pct, desc=f"Generating paragraph {idx + 1}/{len(paragraphs)}...")

                    chunk_script = f"Speaker 1: {para}"

                    chunk_inputs = processor(
                        text=[chunk_script],
                        voice_samples=[[voice_sample_path]],
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )

                    for k, v in chunk_inputs.items():
                        if torch.is_tensor(v):
                            chunk_inputs[k] = v.to(device)

                    model.set_ddpm_inference_steps(num_steps=int(num_steps))

                    # Map chunk progress within its slice of the overall bar
                    _chunk_start = 0.2 + 0.7 * idx / len(paragraphs)
                    _chunk_end = 0.2 + 0.7 * (idx + 1) / len(paragraphs)
                    outputs = model.generate(
                        **chunk_inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=processor.tokenizer,
                        generation_config=gen_config,
                        verbose=False,
                        progress_callback=progress_callback,
                        progress_start=_chunk_start,
                        progress_end=_chunk_end,
                    )

                    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                        audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
                        audio_segments.append(audio_tensor.squeeze().numpy())
                        print(f"  Paragraph {idx + 1}/{len(paragraphs)} done ({len(para.split())} words)")
                    else:
                        print(f"  Paragraph {idx + 1}/{len(paragraphs)} produced no audio, skipping")

                if not audio_segments:
                    raise RuntimeError("VibeVoice failed to generate audio for any paragraph")

                audio_data = np.concatenate(audio_segments)
                return audio_data, sr

            # Single paragraph — fall through to standard generation
            text = paragraphs[0] if paragraphs else text

        # Collapse remaining newlines to space for single-pass generation
        text = re.sub(r'\n+', ' ', text).strip()

        # Standard single-pass generation (no chunking)
        if progress_callback:
            progress_callback(0.2, desc="Preparing VibeVoice generation...")
        formatted_script = f"Speaker 1: {text}"

        # Process inputs
        inputs = processor(
            text=[formatted_script],
            voice_samples=[[voice_sample_path]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        # Set inference steps
        model.set_ddpm_inference_steps(num_steps=int(num_steps))

        # Generate
        if progress_callback:
            progress_callback(0.3, desc="Generating audio with VibeVoice...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config=gen_config,
            verbose=False,
            progress_callback=progress_callback,
            progress_start=0.3,
            progress_end=0.95,
        )

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            # Convert bfloat16 to float32 for soundfile compatibility
            audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
            audio_data = audio_tensor.squeeze().numpy()
            sr = 24000  # VibeVoice uses 24kHz
        else:
            raise RuntimeError("VibeVoice failed to generate audio")

        return audio_data, sr

    # Voice prompt caching
    def get_prompt_cache_path(self, sample_name: str, model_size: str = "1.7B") -> Path:
        """Get path to cached voice prompt."""
        return self.samples_dir / f"{sample_name}_{model_size}.pt"

    def compute_sample_hash(self, wav_path: str, ref_text: str) -> str:
        """Compute hash of sample to detect changes."""
        hasher = hashlib.md5()
        with open(wav_path, 'rb') as f:
            hasher.update(f.read())
        hasher.update(ref_text.encode('utf-8'))
        return hasher.hexdigest()

    def save_voice_prompt(self, sample_name: str, prompt_items, sample_hash: str, model_size: str = "1.7B") -> bool:
        """Save voice prompt to cache."""
        cache_path = self.get_prompt_cache_path(sample_name, model_size)
        try:
            # Move tensors to CPU
            if isinstance(prompt_items, dict):
                cpu_prompt = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in prompt_items.items()
                }
            elif isinstance(prompt_items, (list, tuple)):
                cpu_prompt = [
                    item.cpu() if isinstance(item, torch.Tensor) else item
                    for item in prompt_items
                ]
            else:
                cpu_prompt = prompt_items.cpu() if isinstance(prompt_items, torch.Tensor) else prompt_items

            cache_data = {
                'prompt': cpu_prompt,
                'hash': sample_hash,
                'version': '1.0'
            }
            torch.save(cache_data, cache_path)
            print(f"Saved voice prompt cache: {cache_path}")
            return True
        except Exception as e:
            print(f"Failed to save voice prompt cache: {e}")
            return False

    def load_voice_prompt(self, sample_name: str, expected_hash: str, model_size: str = "1.7B") -> Optional[dict]:
        """Load voice prompt from cache if valid."""
        cache_key = f"{sample_name}_{model_size}"

        # Check memory cache first
        if cache_key in self._voice_prompt_cache:
            cached = self._voice_prompt_cache[cache_key]
            if cached['hash'] == expected_hash:
                return cached['prompt']

        # Check disk cache
        cache_path = self.get_prompt_cache_path(sample_name, model_size)
        if not cache_path.exists():
            return None

        try:
            cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)

            if cache_data.get('hash') != expected_hash:
                return None

            # Move to device
            cached_prompt = cache_data['prompt']
            device = get_device(self.user_config.get("tts_gpu", 0))

            if isinstance(cached_prompt, dict):
                prompt_items = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in cached_prompt.items()
                }
            elif isinstance(cached_prompt, (list, tuple)):
                prompt_items = [
                    item.to(device) if isinstance(item, torch.Tensor) else item
                    for item in cached_prompt
                ]
            else:
                prompt_items = cached_prompt.to(device) if isinstance(cached_prompt, torch.Tensor) else cached_prompt

            # Store in memory cache
            self._voice_prompt_cache[cache_key] = {
                'prompt': prompt_items,
                'hash': expected_hash
            }

            return prompt_items

        except Exception as e:
            print(f"Failed to load voice prompt cache: {e}")
            return None

    # ============================================================
    # LUXTTS PROMPT CACHING
    # ============================================================

    def compute_audio_hash(self, wav_path):
        """Compute a hash of the raw audio file bytes (used for LuxTTS prompt caching)."""
        hasher = hashlib.md5()
        with open(wav_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def get_luxtts_prompt_cache_path(self, sample_name):
        """Get the path to the cached LuxTTS encoded prompt file."""
        return self.samples_dir / f"{sample_name}_luxtts.pt"

    def save_luxtts_prompt(self, sample_name, encoded_prompt, audio_hash, rms=0.01, ref_duration=30):
        """Save LuxTTS encoded prompt to disk (CPU tensors only)."""
        cache_path = self.get_luxtts_prompt_cache_path(sample_name)

        try:
            if isinstance(encoded_prompt, dict):
                cpu_prompt = {}
                for key, value in encoded_prompt.items():
                    cpu_prompt[key] = value.cpu() if isinstance(value, torch.Tensor) else value
            elif isinstance(encoded_prompt, (list, tuple)):
                cpu_prompt = [
                    item.cpu() if isinstance(item, torch.Tensor) else item
                    for item in encoded_prompt
                ]
            else:
                cpu_prompt = encoded_prompt.cpu() if isinstance(encoded_prompt, torch.Tensor) else encoded_prompt

            cache_data = {
                "prompt": cpu_prompt,
                "audio_hash": audio_hash,
                "params": {
                    "rms": round(float(rms), 6),
                    "ref_duration": int(ref_duration),
                },
                "version": "luxtts-1.0",
            }
            torch.save(cache_data, cache_path)
            return True
        except Exception as e:
            print(f"Failed to save LuxTTS prompt: {e}")
            return False

    def load_luxtts_prompt(self, sample_name, expected_audio_hash, rms=0.01, ref_duration=30):
        """Load LuxTTS encoded prompt from disk/memory if valid."""
        cache_key = sample_name

        # Check memory cache
        if cache_key in self._luxtts_prompt_cache:
            cached = self._luxtts_prompt_cache[cache_key]
            if cached.get("audio_hash") == expected_audio_hash:
                return cached["prompt"]

        # Check disk cache
        cache_path = self.get_luxtts_prompt_cache_path(sample_name)
        if not cache_path.exists():
            return None

        try:
            device = get_device(self.user_config.get("tts_gpu", 0))
            cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)

            if cache_data.get("audio_hash") != expected_audio_hash:
                return None

            params = cache_data.get("params") or {}
            if round(float(params.get("rms", -1)), 6) != round(float(rms), 6) or int(
                params.get("ref_duration", -1)
            ) != int(ref_duration):
                return None

            cached_prompt = cache_data.get("prompt")
            if isinstance(cached_prompt, dict):
                prompt = {}
                for key, value in cached_prompt.items():
                    prompt[key] = value.to(device) if isinstance(value, torch.Tensor) else value
            elif isinstance(cached_prompt, (list, tuple)):
                prompt = [
                    item.to(device) if isinstance(item, torch.Tensor) else item
                    for item in cached_prompt
                ]
            else:
                prompt = cached_prompt.to(device) if isinstance(cached_prompt, torch.Tensor) else cached_prompt

            self._luxtts_prompt_cache[cache_key] = {
                "prompt": prompt,
                "audio_hash": expected_audio_hash,
            }

            return prompt

        except Exception as e:
            print(f"Failed to load LuxTTS prompt cache: {e}")
            return None

    def _encode_luxtts_prompt_direct(self, wav_path, ref_text, rms=0.01, ref_duration=30):
        """Encode LuxTTS prompt directly using known transcript text (bypasses Whisper).

        Replicates zipvoice's process_audio() but substitutes the known transcript
        instead of running Whisper transcription.
        """
        import librosa
        from zipvoice.utils.infer import rms_norm

        lux_model = self.get_luxtts()

        # Load audio at 24kHz (same as process_audio)
        prompt_wav, sr = librosa.load(str(wav_path), sr=24000, duration=int(ref_duration))
        prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0)
        prompt_wav, prompt_rms = rms_norm(prompt_wav, float(rms))

        # Extract features
        prompt_features = lux_model.feature_extractor.extract(
            prompt_wav, sampling_rate=24000
        ).to(lux_model.device)
        prompt_features = prompt_features.unsqueeze(0) * 0.1  # feat_scale=0.1

        prompt_features_lens = torch.tensor([prompt_features.size(1)], device=lux_model.device)

        # Tokenize the known transcript directly (no Whisper needed)
        prompt_tokens = lux_model.tokenizer.texts_to_token_ids([ref_text])

        return {
            "prompt_tokens": prompt_tokens,
            "prompt_features_lens": prompt_features_lens,
            "prompt_features": prompt_features,
            "prompt_rms": prompt_rms,
        }

    def get_or_create_luxtts_prompt(self, sample_name, wav_path, rms=0.01, ref_duration=30, ref_text=None, progress_callback=None):
        """Get cached LuxTTS encoded prompt or create a new one."""
        audio_hash = self.compute_audio_hash(wav_path)

        cached = self.load_luxtts_prompt(
            sample_name,
            expected_audio_hash=audio_hash,
            rms=rms,
            ref_duration=ref_duration,
        )
        if cached is not None:
            if progress_callback:
                progress_callback(0.35, desc="Using cached LuxTTS voice prompt...")
            return cached, True

        if progress_callback:
            progress_callback(0.2, desc="Encoding LuxTTS voice prompt (first time)...")

        # Use direct encoding with known text (bypasses Whisper entirely)
        if ref_text:
            encoded_prompt = self._encode_luxtts_prompt_direct(
                wav_path, ref_text, rms=rms, ref_duration=ref_duration
            )
        else:
            raise ValueError(
                f"No transcript found for sample '{sample_name}'. "
                "Please transcribe this sample first in the Prep Audio tab "
                "(using Whisper or VibeVoice ASR), then try again."
            )

        if progress_callback:
            progress_callback(0.35, desc="Caching LuxTTS voice prompt...")

        self.save_luxtts_prompt(
            sample_name, encoded_prompt, audio_hash, rms=rms, ref_duration=ref_duration
        )

        self._luxtts_prompt_cache[sample_name] = {
            "prompt": encoded_prompt,
            "audio_hash": audio_hash,
        }

        return encoded_prompt, False

    def generate_voice_clone_luxtts(
        self, text, voice_sample_path, sample_name,
        num_steps=4, t_shift=0.5, speed=1.0,
        return_smooth=False, rms=0.01, ref_duration=30,
        guidance_scale=3.0, seed=-1, ref_text=None, progress_callback=None
    ):
        """Generate audio using LuxTTS voice cloning.

        Args:
            text: Text to generate
            voice_sample_path: Path to voice sample WAV file
            sample_name: Name of the sample (for caching)
            num_steps: Sampling steps (3-4 recommended)
            t_shift: Sampling parameter (higher = better quality but more pronunciation errors)
            speed: Speed multiplier (lower=slower)
            return_smooth: Smoother output (may reduce metallic artifacts)
            rms: Loudness (0.01 recommended)
            ref_duration: How many seconds of reference audio to use (30 default, increase to 1000 if artifacts)
            guidance_scale: Classifier-free guidance scale (3.0 default)
            seed: Random seed (-1 for random)
            ref_text: Known transcript of the voice sample (bypasses Whisper if provided)
            progress_callback: Optional Gradio progress callback

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        import numpy as np

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        set_seed(seed)

        # Get or create encoded prompt (with caching)
        encoded_prompt, was_cached = self.get_or_create_luxtts_prompt(
            sample_name=sample_name,
            wav_path=voice_sample_path,
            rms=rms,
            ref_duration=ref_duration,
            ref_text=ref_text,
            progress_callback=progress_callback,
        )

        cache_status = "cached" if was_cached else "newly processed"
        if progress_callback:
            progress_callback(0.6, desc=f"Generating audio ({cache_status} prompt)...")

        # Load model and generate
        lux_model = self.get_luxtts()
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
            wav_tensor = lux_model.generate_speech(
                text.strip(),
                encoded_prompt,
                num_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                t_shift=float(t_shift),
                speed=float(speed),
                return_smooth=bool(return_smooth),
            )

        # Convert to numpy
        if isinstance(wav_tensor, torch.Tensor):
            audio_data = wav_tensor.detach().cpu().to(torch.float32).numpy().squeeze()
        else:
            audio_data = np.array(wav_tensor).squeeze()

        return audio_data, 48000, was_cached

    # ============================================================
    # CHATTERBOX METHODS
    # ============================================================

    def get_chatterbox_tts(self):
        """Load Chatterbox TTS model (English)."""
        self._check_and_unload_if_different("chatterbox_tts")

        if self._chatterbox_tts_model is None:
            print("Loading Chatterbox TTS model...")
            try:
                from modules.chatterbox import ChatterboxTTS

                device = get_device(self.user_config.get("tts_gpu", 0))
                local_path = check_model_available_locally("ResembleAI/chatterbox")
                if local_path:
                    print(f"Found local Chatterbox model: {local_path}")
                    self._chatterbox_tts_model = ChatterboxTTS.from_local(local_path, device)
                elif self.user_config.get("offline_mode", False):
                    raise RuntimeError(
                        "Offline mode enabled but Chatterbox model not available locally.\n"
                        "Download it first via Settings or disable offline mode."
                    )
                else:
                    self._chatterbox_tts_model = ChatterboxTTS.from_pretrained(device)
                print("Chatterbox TTS loaded!")

            except ImportError as e:
                raise ImportError(f"Chatterbox not available: {e}")
            except Exception as e:
                print(f"Error loading Chatterbox TTS: {e}")
                raise

        return self._chatterbox_tts_model

    def get_chatterbox_multilingual(self):
        """Load Chatterbox Multilingual TTS model (23 languages)."""
        self._check_and_unload_if_different("chatterbox_mtl")

        if self._chatterbox_mtl_model is None:
            print("Loading Chatterbox Multilingual TTS model...")
            try:
                from modules.chatterbox import ChatterboxMultilingualTTS

                device = get_device(self.user_config.get("tts_gpu", 0))
                local_path = check_model_available_locally("ResembleAI/chatterbox")
                if local_path:
                    print(f"Found local Chatterbox model: {local_path}")
                    self._chatterbox_mtl_model = ChatterboxMultilingualTTS.from_local(local_path, device)
                elif self.user_config.get("offline_mode", False):
                    raise RuntimeError(
                        "Offline mode enabled but Chatterbox model not available locally.\n"
                        "Download it first via Settings or disable offline mode."
                    )
                else:
                    self._chatterbox_mtl_model = ChatterboxMultilingualTTS.from_pretrained(device)
                print("Chatterbox Multilingual TTS loaded!")

            except ImportError as e:
                raise ImportError(f"Chatterbox Multilingual not available: {e}")
            except Exception as e:
                print(f"Error loading Chatterbox Multilingual TTS: {e}")
                raise

        return self._chatterbox_mtl_model

    def get_chatterbox_vc(self):
        """Load Chatterbox Voice Conversion model."""
        self._check_and_unload_if_different("chatterbox_vc")

        if self._chatterbox_vc_model is None:
            print("Loading Chatterbox Voice Conversion model...")
            try:
                from modules.chatterbox import ChatterboxVC

                device = get_device(self.user_config.get("tts_gpu", 0))
                local_path = check_model_available_locally("ResembleAI/chatterbox")
                if local_path:
                    print(f"Found local Chatterbox model: {local_path}")
                    self._chatterbox_vc_model = ChatterboxVC.from_local(local_path, device)
                elif self.user_config.get("offline_mode", False):
                    raise RuntimeError(
                        "Offline mode enabled but Chatterbox model not available locally.\n"
                        "Download it first via Settings or disable offline mode."
                    )
                else:
                    self._chatterbox_vc_model = ChatterboxVC.from_pretrained(device)
                print("Chatterbox VC loaded!")

            except ImportError as e:
                raise ImportError(f"Chatterbox VC not available: {e}")
            except Exception as e:
                print(f"Error loading Chatterbox VC: {e}")
                raise

        return self._chatterbox_vc_model

    def generate_voice_clone_chatterbox(self, text, voice_sample_path, seed=-1,
                                        exaggeration=0.5, cfg_weight=0.5,
                                        temperature=0.8, repetition_penalty=1.2,
                                        top_p=1.0):
        """Generate audio using Chatterbox TTS (English voice cloning).

        Args:
            text: Text to speak
            voice_sample_path: Path to reference voice WAV
            seed: Random seed (-1 for random)
            exaggeration: Emotion intensity (0-2)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty
            top_p: Top-p sampling

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        import numpy as np

        if seed < 0:
            seed = random.randint(0, 2147483647)
        set_seed(seed)

        model = self.get_chatterbox_tts()

        wav_tensor = model.generate(
            text=text.strip(),
            audio_prompt_path=str(voice_sample_path),
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            top_p=float(top_p),
        )

        # Output is [1, N] float tensor at 24kHz
        audio_data = wav_tensor.squeeze(0).detach().cpu().numpy()

        return audio_data, 24000

    def generate_voice_clone_chatterbox_multilingual(self, text, language_code,
                                                     voice_sample_path, seed=-1,
                                                     exaggeration=0.5, cfg_weight=0.5,
                                                     temperature=0.8, repetition_penalty=2.0,
                                                     top_p=1.0):
        """Generate audio using Chatterbox Multilingual TTS (23 languages).

        Args:
            text: Text to speak
            language_code: 2-letter ISO language code (e.g. "en", "fr")
            voice_sample_path: Path to reference voice WAV
            seed: Random seed (-1 for random)
            exaggeration: Emotion intensity (0-2)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty (default 2.0 for multilingual)
            top_p: Top-p sampling

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        import numpy as np

        if seed < 0:
            seed = random.randint(0, 2147483647)
        set_seed(seed)

        model = self.get_chatterbox_multilingual()

        wav_tensor = model.generate(
            text=text.strip(),
            language_id=language_code,
            audio_prompt_path=str(voice_sample_path),
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            top_p=float(top_p),
        )

        # Output is [1, N] float tensor at 24kHz
        audio_data = wav_tensor.squeeze(0).detach().cpu().numpy()

        return audio_data, 24000

    def generate_voice_convert_chatterbox(self, source_audio_path, target_voice_path):
        """Convert voice in source audio to match target voice.

        Args:
            source_audio_path: Path to source audio WAV to convert
            target_voice_path: Path to target voice reference WAV

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        model = self.get_chatterbox_vc()

        wav_tensor = model.generate(
            audio=str(source_audio_path),
            target_voice_path=str(target_voice_path),
        )

        # Output is [1, N] float tensor at 24kHz
        audio_data = wav_tensor.squeeze(0).detach().cpu().numpy()

        return audio_data, 24000

    # ============================================================
    # UNIFIED VOICE CLONE DISPATCH
    # ============================================================

    @staticmethod
    def parse_model_selection(model_selection):
        """Parse a UI model selection string into (engine, model_size).

        Args:
            model_selection: UI string like 'Qwen3 - Small', 'VibeVoice - Large (4-bit)', etc.

        Returns:
            Tuple of (engine, model_size) e.g. ('qwen', '0.6B')
        """
        if "DramaBox" in model_selection:
            return "dramabox", "Default"
        elif "Fish Speech" in model_selection:
            return "fish_speech", "Pro"
        elif "LuxTTS" in model_selection:
            return "luxtts", "Default"
        elif "VibeVoice" in model_selection:
            if "Small" in model_selection:
                return "vibevoice", "1.5B"
            elif "4-bit" in model_selection:
                return "vibevoice", "Large (4-bit)"
            else:
                return "vibevoice", "Large"
        elif "Chatterbox" in model_selection:
            if "Multilingual" in model_selection:
                return "chatterbox", "Multilingual"
            else:
                return "chatterbox", "Default"
        else:
            if "Small" in model_selection:
                return "qwen", "0.6B"
            else:
                return "qwen", "1.7B"

    def generate_voice_clone_dispatch(self, text, engine, model_size,
                                      sample_wav_path, sample_name, sample_ref_text,
                                      language="Auto", seed=-1,
                                      qwen_params=None, vv_params=None,
                                      lux_params=None, cb_params=None,
                                      prompt_items=None, user_config=None,
                                      progress_callback=None, **kwargs):
        """Unified voice clone generation that dispatches to the correct engine.

        This is the single entry point for all voice clone generation
        (single clip or per-paragraph in a split loop).

        Args:
            text: Text to generate
            engine: Engine id ('qwen', 'vibevoice', 'luxtts', 'chatterbox')
            model_size: Model size string
            sample_wav_path: Path to voice sample WAV
            sample_name: Name of the sample (for caching)
            sample_ref_text: Transcript of the sample
            language: Language selection
            seed: Random seed
            qwen_params: Dict of Qwen-specific params
            vv_params: Dict of VibeVoice-specific params
            lux_params: Dict of LuxTTS-specific params
            cb_params: Dict of Chatterbox-specific params
            prompt_items: Pre-computed Qwen voice clone prompt (optional)
            user_config: User config dict (for VibeVoice)
            progress_callback: Optional Gradio progress callback

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        qp = qwen_params or {}
        vp = vv_params or {}
        lp = lux_params or {}
        cp = cb_params or {}

        if engine == "qwen":
            return self.generate_voice_clone_qwen(
                text=text, language=language, prompt_items=prompt_items, seed=seed,
                do_sample=qp.get('do_sample', True),
                temperature=qp.get('temperature', 0.9),
                top_k=qp.get('top_k', 50),
                top_p=qp.get('top_p', 1.0),
                repetition_penalty=qp.get('repetition_penalty', 1.05),
                max_new_tokens=qp.get('max_new_tokens', 2048),
                model_size=model_size,
                ref_audio=sample_wav_path,
                ref_text=sample_ref_text,
            )
        elif engine == "vibevoice":
            return self.generate_voice_clone_vibevoice(
                text=text, voice_sample_path=sample_wav_path, seed=seed,
                do_sample=vp.get('do_sample', False),
                temperature=vp.get('temperature', 1.0),
                top_k=vp.get('top_k', 50),
                top_p=vp.get('top_p', 1.0),
                repetition_penalty=vp.get('repetition_penalty', 1.0),
                cfg_scale=vp.get('cfg_scale', 1.3),
                num_steps=vp.get('num_steps', 10),
                paragraph_per_chunk=bool(vp.get('paragraph_per_chunk', False)),
                model_size=model_size,
                user_config=user_config or {},
                progress_callback=progress_callback,
            )
        elif engine == "luxtts":
            audio_data, sr, _ = self.generate_voice_clone_luxtts(
                text=text, voice_sample_path=sample_wav_path,
                sample_name=sample_name,
                num_steps=int(lp.get('num_steps', 4)),
                t_shift=float(lp.get('t_shift', 0.5)),
                speed=float(lp.get('speed', 1.0)),
                return_smooth=bool(lp.get('return_smooth', False)),
                rms=float(lp.get('rms', 0.01)),
                ref_duration=int(lp.get('ref_duration', 30)),
                guidance_scale=float(lp.get('guidance_scale', 3.0)),
                seed=seed,
                ref_text=sample_ref_text,
                progress_callback=progress_callback,
            )
            return audio_data, sr
        elif engine == "chatterbox":
            cb_language = cp.get('language', 'English')
            if model_size == "Multilingual":
                from modules.core_components.constants import CHATTERBOX_LANG_TO_CODE
                lang_code = CHATTERBOX_LANG_TO_CODE.get(cb_language, "en")
                return self.generate_voice_clone_chatterbox_multilingual(
                    text=text, language_code=lang_code,
                    voice_sample_path=sample_wav_path, seed=seed,
                    exaggeration=float(cp.get('exaggeration', 0.5)),
                    cfg_weight=float(cp.get('cfg_weight', 0.5)),
                    temperature=float(cp.get('temperature', 0.8)),
                    repetition_penalty=float(cp.get('repetition_penalty', 1.2)),
                    top_p=float(cp.get('top_p', 1.0)),
                )
            else:
                return self.generate_voice_clone_chatterbox(
                    text=text, voice_sample_path=sample_wav_path, seed=seed,
                    exaggeration=float(cp.get('exaggeration', 0.5)),
                    cfg_weight=float(cp.get('cfg_weight', 0.5)),
                    temperature=float(cp.get('temperature', 0.8)),
                    repetition_penalty=float(cp.get('repetition_penalty', 1.2)),
                    top_p=float(cp.get('top_p', 1.0)),
                )
        elif engine == "fish_speech":
            fp = kwargs.get('fs_params', {})
            return self.generate_voice_clone_fish_speech(
                text=text, voice_sample_path=sample_wav_path,
                ref_text=sample_ref_text, seed=seed,
                temperature=float(fp.get('temperature', 0.9)),
                top_p=float(fp.get('top_p', 0.9)),
                top_k=int(fp.get('top_k', 30)),
                repetition_penalty=float(fp.get('repetition_penalty', 1.05)),
                max_new_tokens=int(fp.get('max_new_tokens', 2048)),
                chunk_length=int(fp.get('chunk_length', 1000)),
                split_by_paragraph=bool(fp.get('split_by_paragraph', False)),
                progress_callback=progress_callback,
            )
        else:
            raise ValueError(f"Unknown engine: {engine}")


# Global singleton instance
_tts_manager = None


def get_tts_manager(user_config: Dict = None, samples_dir: Path = None) -> TTSManager:
    """Get or create the global TTS manager."""
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = TTSManager(user_config, samples_dir)
    return _tts_manager
