"""
TTS Model Manager

Centralized management for all TTS models (Qwen3, VibeVoice, etc.)
"""

import torch
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional

from .model_utils import (
    get_device,
    get_dtype,
    get_attention_implementation,
    check_model_available_locally,
    empty_cuda_cache,
    log_gpu_memory,
)


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

        # Prompt cache
        self._voice_prompt_cache = {}
        self._luxtts_prompt_cache = {}
        self._last_loaded_model = None

    def _get_luxtts_class(self):
        """Import LuxTTS with a fallback module path."""
        try:
            from zipvoice.luxtts import LuxTTS

            return LuxTTS
        except ImportError:
            try:
                from zipvoice.luxvoice import LuxTTS

                return LuxTTS
            except ImportError as exc:
                raise ImportError(
                    "LuxTTS is not installed. Install it from the LuxTTS repository and retry."
                ) from exc

    def _check_and_unload_if_different(self, model_id: str):
        """If switching to a different model, unload all."""
        if self._last_loaded_model is not None and self._last_loaded_model != model_id:
            print(
                f"📦 Switching from {self._last_loaded_model} to {model_id} - unloading all TTS models..."
            )
            self.unload_all()
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
                    **kwargs,
                )
                print(f"✓ Model loaded with {attn}")
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

    def get_qwen3_base(self, size: str = "1.7B"):
        """Load Qwen3 Base TTS model."""
        model_id = f"qwen3_base_{size}"
        self._check_and_unload_if_different(model_id)

        if self._qwen3_base_model is None:
            from qwen_tts import Qwen3TTSModel

            model_name = f"Qwen/Qwen3-TTS-12Hz-{size}-Base"
            print(f"Loading {model_name}...")

            self._qwen3_base_model, _ = self._load_model_with_attention(
                Qwen3TTSModel,
                model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False),
            )
            self._qwen3_base_size = size
            print(f"Qwen3 Base TTS ({size}) loaded!")

        return self._qwen3_base_model

    def get_qwen3_voice_design(self):
        """Load Qwen3 VoiceDesign model (1.7B only)."""
        self._check_and_unload_if_different("qwen3_voice_design")

        if self._qwen3_voice_design_model is None:
            from qwen_tts import Qwen3TTSModel

            print("Loading Qwen3 VoiceDesign model (1.7B)...")

            self._qwen3_voice_design_model, _ = self._load_model_with_attention(
                Qwen3TTSModel,
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map="cuda:0",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False),
            )
            print("VoiceDesign model loaded!")

        return self._qwen3_voice_design_model

    def get_qwen3_custom_voice(self, size: str = "1.7B"):
        """Load Qwen3 CustomVoice model."""
        model_id = f"qwen3_custom_voice_{size}"
        self._check_and_unload_if_different(model_id)

        if self._qwen3_custom_voice_model is None:
            from qwen_tts import Qwen3TTSModel

            model_name = f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"
            print(f"Loading {model_name}...")

            self._qwen3_custom_voice_model, _ = self._load_model_with_attention(
                Qwen3TTSModel,
                model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False),
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
                    VibeVoiceForConditionalGenerationInference,
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

                logging.getLogger("transformers.tokenization_utils_base").setLevel(
                    logging.ERROR
                )

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)

                    self._vibevoice_tts_model, _ = self._load_model_with_attention(
                        VibeVoiceForConditionalGenerationInference,
                        model_path,
                        dtype=torch.bfloat16,
                        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                        low_cpu_mem_usage=self.user_config.get(
                            "low_cpu_mem_usage", False
                        ),
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
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message=".*torch.cuda.amp.autocast.*",
                    )
                    LuxTTS = self._get_luxtts_class()

                    device_pref = self.user_config.get("luxtts_device", "auto")
                    if device_pref == "auto":
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    else:
                        device = device_pref

                    if device == "cuda" and not torch.cuda.is_available():
                        device = "cpu"

                    if device == "cpu":
                        threads = int(self.user_config.get("luxtts_threads", 2))
                        self._luxtts_model = LuxTTS(
                            "YatharthS/LuxTTS", device="cpu", threads=max(1, threads)
                        )
                    else:
                        self._luxtts_model = LuxTTS("YatharthS/LuxTTS", device="cuda")

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
            freed.append("LuxTTS")

        self._luxtts_prompt_cache.clear()
        if freed:
            empty_cuda_cache()
            print(f"🗑️ Unloaded TTS models: {', '.join(freed)}")

        return bool(freed)

    # ============================================================
    # GENERATION METHODS
    # ============================================================

    def generate_voice_design(
        self,
        text: str,
        language: str,
        instruct: str,
        seed: int = -1,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 2048,
    ) -> Tuple[str, int]:
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

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load model
        model = self.get_qwen3_voice_design()

        # Generate
        wavs, sr = model.generate_voice_design(
            text=text.strip(),
            language=language if language != "Auto" else "Auto",
            instruct=instruct.strip(),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

        # Convert to numpy if needed
        audio_data = wavs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        return audio_data, sr

    def generate_custom_voice(
        self,
        text: str,
        language: str,
        speaker: str,
        instruct: str = None,
        model_size: str = "1.7B",
        seed: int = -1,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 2048,
    ) -> Tuple[str, int]:
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

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load model
        model = self.get_qwen3_custom_voice(model_size)

        # Build kwargs
        kwargs = {
            "text": text.strip(),
            "language": language if language != "Auto" else "Auto",
            "speaker": speaker,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
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

    def generate_with_trained_model(
        self,
        text: str,
        language: str,
        speaker_name: str,
        checkpoint_path: str,
        instruct: str = None,
        seed: int = -1,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 2048,
        user_config: dict = None,
    ) -> Tuple[str, int]:
        """
        Generate audio using a trained custom voice model checkpoint.

        Args:
            text: Text to generate
            language: Language for TTS
            speaker_name: Speaker name the model was trained with
            checkpoint_path: Path to trained model checkpoint
            instruct: Optional style instructions
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate
            user_config: User configuration dict (for attention mechanism, etc.)

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Model management - unload if different model
        model_id = f"trained_{checkpoint_path}"
        # Note: check_and_unload_if_different should be called by the caller if needed

        # Load the trained model checkpoint
        # This would need check_and_unload_if_different and load_model_with_attention
        # which should also be moved here eventually
        if user_config is None:
            user_config = {}

        # For now, load directly - proper model management to be added
        model = Qwen3TTSModel.from_pretrained(
            checkpoint_path,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=user_config.get("low_cpu_mem_usage", False),
        )

        # Build kwargs
        kwargs = {
            "text": text.strip(),
            "language": language if language != "Auto" else "Auto",
            "speaker": speaker_name,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
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

    def generate_voice_clone_qwen(
        self,
        text: str,
        language: str,
        prompt_items,
        seed: int = -1,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 2048,
        model_size: str = "1.7B",
    ) -> Tuple[str, int]:
        """
        Generate audio using Qwen3 voice cloning with cached prompt.

        Args:
            text: Text to generate
            language: Language for TTS
            prompt_items: Pre-computed voice clone prompt
            seed: Random seed (-1 for random)
            do_sample: Enable sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate
            model_size: Model size (1.7B or 0.6B)

        Returns:
            Tuple: (audio_array, sample_rate)
        """
        import random

        # Set seed for reproducibility
        if seed < 0:
            seed = random.randint(0, 2147483647)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load BASE model (not CustomVoice - Base supports voice cloning)
        model = self.get_qwen3_base(model_size)

        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            if top_k > 0:
                gen_kwargs["top_k"] = int(top_k)
            if top_p < 1.0:
                gen_kwargs["top_p"] = top_p
            if repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = repetition_penalty

        # Generate using the cached prompt
        wavs, sr = model.generate_voice_clone(
            text=text.strip(),
            language=language if language != "Auto" else "Auto",
            voice_clone_prompt=prompt_items,
            **gen_kwargs,
        )

        # Convert to numpy if needed
        audio_data = wavs[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        return audio_data, sr

    def generate_voice_clone_vibevoice(
        self,
        text: str,
        voice_sample_path: str,
        seed: int = -1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        cfg_scale: float = 3.0,
        num_steps: int = 20,
        model_size: str = "Large",
        user_config: dict = None,
    ) -> Tuple[str, int]:
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

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load model
        model = self.get_vibevoice_tts(model_size)

        from modules.vibevoice_tts.processor.vibevoice_processor import (
            VibeVoiceProcessor,
        )

        # Map model_size to valid HuggingFace repo path
        if model_size == "Large (4-bit)":
            model_path = "FranckyB/VibeVoice-Large-4bit"
        else:
            model_path = f"FranckyB/VibeVoice-{model_size}"

        # Suppress tokenizer mismatch warning
        prev_level = logging.getLogger("transformers.tokenization_utils_base").level
        logging.getLogger("transformers.tokenization_utils_base").setLevel(
            logging.ERROR
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if user_config is None:
                user_config = {}
            offline_mode = user_config.get("offline_mode", False)
            processor = VibeVoiceProcessor.from_pretrained(
                model_path, local_files_only=offline_mode
            )

        logging.getLogger("transformers.tokenization_utils_base").setLevel(prev_level)

        # Format script for VibeVoice (single speaker)
        formatted_script = f"Speaker 1: {text.strip()}"

        # Process inputs
        inputs = processor(
            text=[formatted_script],
            voice_samples=[[voice_sample_path]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        # Set inference steps
        model.set_ddpm_inference_steps(num_steps=int(num_steps))

        # Prepare generation config
        gen_config = {"do_sample": do_sample}
        if do_sample:
            gen_config["temperature"] = temperature
            if top_k > 0:
                gen_config["top_k"] = int(top_k)
            if top_p < 1.0:
                gen_config["top_p"] = top_p
            if repetition_penalty != 1.0:
                gen_config["repetition_penalty"] = repetition_penalty

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config=gen_config,
            verbose=False,
        )

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            # Convert bfloat16 to float32 for soundfile compatibility
            audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
            audio_data = audio_tensor.squeeze().numpy()
            sr = 24000  # VibeVoice uses 24kHz
        else:
            raise RuntimeError("VibeVoice failed to generate audio")

        return audio_data, sr

    def clear_prompt_cache_for_sample(self, sample_name):
        """Clear in-memory prompt cache for a sample."""
        qwen_keys = [
            k
            for k in self._voice_prompt_cache.keys()
            if k.startswith(f"{sample_name}_")
        ]
        for key in qwen_keys:
            del self._voice_prompt_cache[key]

        lux_keys = [
            k
            for k in self._luxtts_prompt_cache.keys()
            if k.startswith(f"{sample_name}_")
        ]
        for key in lux_keys:
            del self._luxtts_prompt_cache[key]

    # Voice prompt caching
    def get_prompt_cache_path(self, sample_name: str, model_size: str = "1.7B") -> Path:
        """Get path to cached voice prompt."""
        return self.samples_dir / f"{sample_name}_{model_size}.pt"

    def compute_sample_hash(self, wav_path: str, ref_text: str) -> str:
        """Compute hash of sample to detect changes."""
        hasher = hashlib.md5()
        with open(wav_path, "rb") as f:
            hasher.update(f.read())
        hasher.update(ref_text.encode("utf-8"))
        return hasher.hexdigest()

    def save_voice_prompt(
        self, sample_name: str, prompt_items, sample_hash: str, model_size: str = "1.7B"
    ) -> bool:
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
                cpu_prompt = (
                    prompt_items.cpu()
                    if isinstance(prompt_items, torch.Tensor)
                    else prompt_items
                )

            cache_data = {"prompt": cpu_prompt, "hash": sample_hash, "version": "1.0"}
            torch.save(cache_data, cache_path)
            print(f"Saved voice prompt: {cache_path}")
            return True
        except Exception as e:
            print(f"Failed to save voice prompt: {e}")
            return False

    def load_voice_prompt(
        self, sample_name: str, expected_hash: str, model_size: str = "1.7B"
    ) -> Optional[dict]:
        """Load voice prompt from cache if valid."""
        cache_key = f"{sample_name}_{model_size}"

        # Check memory cache first
        if cache_key in self._voice_prompt_cache:
            cached = self._voice_prompt_cache[cache_key]
            if cached["hash"] == expected_hash:
                return cached["prompt"]

        # Check disk cache
        cache_path = self.get_prompt_cache_path(sample_name, model_size)
        if not cache_path.exists():
            return None

        try:
            cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)

            if cache_data.get("hash") != expected_hash:
                return None

            # Move to device
            cached_prompt = cache_data["prompt"]
            device = get_device()

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
                prompt_items = (
                    cached_prompt.to(device)
                    if isinstance(cached_prompt, torch.Tensor)
                    else cached_prompt
                )

            # Store in memory cache
            self._voice_prompt_cache[cache_key] = {
                "prompt": prompt_items,
                "hash": expected_hash,
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

    def save_luxtts_prompt(
        self, sample_name, encoded_prompt, audio_hash, rms=0.01, ref_duration=30
    ):
        """Save LuxTTS encoded prompt to disk (CPU tensors only)."""
        cache_path = self.get_luxtts_prompt_cache_path(sample_name)

        try:
            if isinstance(encoded_prompt, dict):
                cpu_prompt = {}
                for key, value in encoded_prompt.items():
                    cpu_prompt[key] = (
                        value.cpu() if isinstance(value, torch.Tensor) else value
                    )
            elif isinstance(encoded_prompt, (list, tuple)):
                cpu_prompt = [
                    item.cpu() if isinstance(item, torch.Tensor) else item
                    for item in encoded_prompt
                ]
            else:
                cpu_prompt = (
                    encoded_prompt.cpu()
                    if isinstance(encoded_prompt, torch.Tensor)
                    else encoded_prompt
                )

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

    def load_luxtts_prompt(
        self, sample_name, expected_audio_hash, rms=0.01, ref_duration=30
    ):
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
            device = get_device()
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
                    prompt[key] = (
                        value.to(device) if isinstance(value, torch.Tensor) else value
                    )
            elif isinstance(cached_prompt, (list, tuple)):
                prompt = [
                    item.to(device) if isinstance(item, torch.Tensor) else item
                    for item in cached_prompt
                ]
            else:
                prompt = (
                    cached_prompt.to(device)
                    if isinstance(cached_prompt, torch.Tensor)
                    else cached_prompt
                )

            self._luxtts_prompt_cache[cache_key] = {
                "prompt": prompt,
                "audio_hash": expected_audio_hash,
            }

            return prompt

        except Exception as e:
            print(f"Failed to load LuxTTS prompt cache: {e}")
            return None

    def get_or_create_luxtts_prompt(
        self,
        sample_name,
        wav_path,
        rms=0.01,
        ref_duration=30,
        progress_callback=None,
    ):
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

        lux_model = self.get_luxtts()
        encoded_prompt = lux_model.encode_prompt(
            wav_path, duration=int(ref_duration), rms=float(rms)
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
        self,
        text,
        voice_sample_path,
        sample_name,
        num_steps=4,
        t_shift=0.5,
        speed=1.0,
        return_smooth=False,
        rms=0.01,
        ref_duration=30,
        guidance_scale=3.0,
        seed=-1,
        progress_callback=None,
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

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Get or create encoded prompt (with caching)
        encoded_prompt, was_cached = self.get_or_create_luxtts_prompt(
            sample_name=sample_name,
            wav_path=voice_sample_path,
            rms=rms,
            ref_duration=ref_duration,
            progress_callback=progress_callback,
        )

        cache_status = "cached" if was_cached else "newly processed"
        if progress_callback:
            progress_callback(0.6, desc=f"Generating audio ({cache_status} prompt)...")

        # Load model and generate
        lux_model = self.get_luxtts()
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*"
            )
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


# Global singleton instance
_tts_manager = None


def get_tts_manager(user_config: Dict = None, samples_dir: Path = None) -> TTSManager:
    """Get or create the global TTS manager."""
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = TTSManager(user_config, samples_dir)
    return _tts_manager
