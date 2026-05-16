#!/usr/bin/env python3
"""
Warm TTS server — loads models once, accepts requests via stdin or function call.

The key insight: inference.py spends 11s on Gemma + 8s on model load every call.
This server loads everything once and keeps it warm.

We import and call the same code paths as inference.py but cache the heavy objects.
"""
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import torch
import soundfile as sf

# Setup paths
APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR / "ltx2"))
sys.path.insert(0, str(APP_DIR / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from audio_conditioning import AudioConditionByReferenceLatent
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.loader import DummyRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.text_projection import create_caption_projection
from ltx_core.model.transformer.attention import AttentionFunction
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.tools import AudioLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_pipelines.utils.blocks import AudioConditioner, AudioDecoder, PromptEncoder
from ltx_pipelines.utils.media_io import decode_audio_from_file
from ltx_pipelines.utils.denoisers import GuidedDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop, heun_denoising_loop
from safetensors import safe_open


DEFAULT_NEG = "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear, unnatural, monotone"


def estimate_duration(prompt, multiplier=1.1, speed=1.0):
    """Defer to the richer CLI estimator (sentence-aware + non-verbal action
    budget) so warm-server outputs match the lengths of the per-call CLI runs."""
    from inference import estimate_speech_duration
    base = estimate_speech_duration(prompt, speed)
    return max(3.0, round(base * multiplier, 1))


def auto_rescale_for_cfg(cfg: float) -> float:
    """CFG-aware std-rescale schedule that prevents output clipping at high cfg.

    The CFG formula `pred = cond + (cfg-1)*(cond - uncond)` makes pred.std()
    grow roughly linearly with cfg, which the audio VAE+vocoder render as
    progressively louder waveforms. By cfg≈3 the output starts hard-clipping
    at 0 dBFS — and clipped information is unrecoverable in post.

    Empirical sweep on the blues prompt with the back-porch-boogie ref
    (rescale_scale needed for ≥1 dB peak headroom):
        cfg=2.5 → 0.2 ;  cfg=3 → 0.6 ; cfg=4 → 0.8 ; cfg=5–8 → 0.8 ; cfg=10 → 1.0

    Piecewise-linear fit through those points; returns 0 below cfg=2 (no CFG
    even applied at cfg=1), plateaus at 0.8 between cfg=4 and cfg=8 to
    preserve the "extra punch" of high-CFG generations, and ramps to 1.0 by
    cfg=10.
    """
    if cfg <= 2.0:
        return 0.0
    if cfg <= 3.0:
        return 0.6 * (cfg - 2.0)               # 0 → 0.6
    if cfg <= 4.0:
        return 0.6 + 0.2 * (cfg - 3.0)         # 0.6 → 0.8
    if cfg <= 8.0:
        return 0.8                              # plateau
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))   # 0.8 → 1.0 at cfg=10


class TTSServer:
    def __init__(self, checkpoint=None, full_checkpoint=None, gemma_root=None,
                 device="cuda", dtype="bf16", compile_model=True, bnb_4bit=True):
        MODELS = APP_DIR / "models"
        self.checkpoint = checkpoint or str(MODELS / "ltx-2.3-22b-dev-audio-only-v13-merged.safetensors")
        self.full_checkpoint = full_checkpoint or os.environ.get(
            "LTX_FULL_CHECKPOINT", "/mnt/persistent0/manmay/models/ltx23/ltx-2.3-22b-dev.safetensors")
        if gemma_root is None and not os.environ.get("GEMMA_DIR"):
            from model_downloader import get_gemma_path
            gemma_root = get_gemma_path()
        self.gemma_root = gemma_root or os.environ["GEMMA_DIR"]
        self.device = torch.device(device)
        self.dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        self.compile_model = compile_model
        self.bnb_4bit = bnb_4bit
        self.patchifier = AudioPatchifier(patch_size=1)

        # Cached models
        self._prompt_encoder = None
        self._velocity_model = None
        self._audio_conditioner = None
        self._audio_decoder = None

        # LoRA tracking — None means base model is loaded
        self._loaded_lora_path = None

        logging.info(f"TTSServer loading on {device}...")
        t0 = time.time()
        self._load_all()
        logging.info(f"All models loaded in {time.time()-t0:.1f}s — ready for requests")

    def _load_all(self):
        # 1. Prompt encoder (Gemma + embeddings processor kept warm)
        t0 = time.time()
        self._prompt_encoder = PromptEncoder(
            checkpoint_path=self.full_checkpoint,
            gemma_root=self.gemma_root,
            dtype=self.dtype, device=self.device,
            warm=True,
            use_bnb_4bit=self.bnb_4bit,
            audio_only=True,
        )
        logging.info(f"  PromptEncoder (warm): {time.time()-t0:.1f}s")

        # 2. Audio conditioner (VAE encoder kept warm)
        t0 = time.time()
        self._audio_conditioner = AudioConditioner(
            checkpoint_path=self.full_checkpoint,
            dtype=self.dtype, device=self.device,
            warm=True,
        )
        logging.info(f"  AudioConditioner (warm): {time.time()-t0:.1f}s")

        # 3. Transformer
        t0 = time.time()
        with safe_open(self.checkpoint, framework="pt") as f:
            config = json.loads(f.metadata()["config"])

        t = config.get("transformer", {})

        class AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
            @classmethod
            def from_config(cls, cfg):
                t = cfg.get("transformer", {})
                cp = None
                if not t.get("caption_proj_before_connector", False):
                    with torch.device("meta"):
                        cp = create_caption_projection(t, audio=True)
                return LTXModel(
                    model_type=LTXModelType.AudioOnly,
                    audio_num_attention_heads=t.get("audio_num_attention_heads", 32),
                    audio_attention_head_dim=t.get("audio_attention_head_dim", 64),
                    audio_in_channels=t.get("audio_in_channels", 128),
                    audio_out_channels=t.get("audio_out_channels", 128),
                    num_layers=t.get("num_layers", 48),
                    audio_cross_attention_dim=t.get("audio_cross_attention_dim", 2048),
                    norm_eps=t.get("norm_eps", 1e-6),
                    attention_type=AttentionFunction(t.get("attention_type", "default")),
                    positional_embedding_theta=10000.0,
                    audio_positional_embedding_max_pos=[20.0],
                    timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
                    use_middle_indices_grid=t.get("use_middle_indices_grid", True),
                    rope_type=LTXRopeType(t.get("rope_type", "interleaved")),
                    double_precision_rope=t.get("frequencies_precision", False) == "float64",
                    apply_gated_attention=t.get("apply_gated_attention", False),
                    audio_caption_projection=cp,
                    cross_attention_adaln=t.get("cross_attention_adaln", False),
                )

        audio_sd_ops = SDOps("AO").with_matching(prefix="model.diffusion_model.").with_replacement(
            "model.diffusion_model.", "")
        builder = Builder(
            model_path=self.checkpoint,
            model_class_configurator=AudioOnlyConfigurator,
            model_sd_ops=audio_sd_ops,
            registry=DummyRegistry(),
        )
        # Builder already places loaded tensors on the target device.
        self._velocity_model = builder.build(device=self.device, dtype=self.dtype).eval()
        n_params = sum(p.numel() for p in self._velocity_model.parameters()) / 1e9
        vram_gb = sum(p.numel() * p.element_size() for p in self._velocity_model.parameters()) / 1e9
        logging.info(f"  Transformer: {time.time()-t0:.1f}s ({n_params:.1f}B params, {vram_gb:.1f}GB VRAM, {self.dtype})")

        # torch.compile for faster denoising
        if self.compile_model:
            t0 = time.time()
            logging.info("  Compiling transformer with torch.compile (default mode)...")
            self._velocity_model = torch.compile(self._velocity_model, mode="default", dynamic=True)
            logging.info(f"  Compiled: {time.time()-t0:.1f}s (first call triggers actual compilation)")

        self._loaded_lora_path = None

        # 4. Audio decoder (VAE decoder + vocoder kept warm)
        t0 = time.time()
        self._audio_decoder = AudioDecoder(
            checkpoint_path=self.full_checkpoint,
            dtype=self.dtype, device=self.device,
            warm=True,
        )
        logging.info(f"  AudioDecoder (warm): {time.time()-t0:.1f}s")

    def _reload_for_lora(self, lora_path, lora_rank=128):
        """Reload the transformer when the active LoRA adapter changes.

        Gemma, AudioConditioner and AudioDecoder stay warm — only the
        3.3 B transformer is rebuilt (~8 s vs ~30 s for a full reload).
        """
        normalized = str(lora_path) if lora_path and os.path.exists(str(lora_path)) else None
        if normalized == self._loaded_lora_path:
            return

        label = f"LoRA: {normalized}" if normalized else "base model (no LoRA)"
        logging.info(f"DramaBox: reloading transformer for {label}...")
        t0 = time.time()

        # Free current transformer
        if self._velocity_model is not None:
            del self._velocity_model
            self._velocity_model = None
            torch.cuda.empty_cache()

        # Rebuild from checkpoint
        with safe_open(self.checkpoint, framework="pt") as f:
            config = json.loads(f.metadata()["config"])
        t = config.get("transformer", {})

        class _AOC(ModelConfigurator[LTXModel]):
            @classmethod
            def from_config(cls, cfg):
                _t = cfg.get("transformer", {})
                cp = None
                if not _t.get("caption_proj_before_connector", False):
                    with torch.device("meta"):
                        cp = create_caption_projection(_t, audio=True)
                return LTXModel(
                    model_type=LTXModelType.AudioOnly,
                    audio_num_attention_heads=_t.get("audio_num_attention_heads", 32),
                    audio_attention_head_dim=_t.get("audio_attention_head_dim", 64),
                    audio_in_channels=_t.get("audio_in_channels", 128),
                    audio_out_channels=_t.get("audio_out_channels", 128),
                    num_layers=_t.get("num_layers", 48),
                    audio_cross_attention_dim=_t.get("audio_cross_attention_dim", 2048),
                    norm_eps=_t.get("norm_eps", 1e-6),
                    attention_type=AttentionFunction(_t.get("attention_type", "default")),
                    positional_embedding_theta=10000.0,
                    audio_positional_embedding_max_pos=[20.0],
                    timestep_scale_multiplier=_t.get("timestep_scale_multiplier", 1000),
                    use_middle_indices_grid=_t.get("use_middle_indices_grid", True),
                    rope_type=LTXRopeType(_t.get("rope_type", "interleaved")),
                    double_precision_rope=_t.get("frequencies_precision", False) == "float64",
                    apply_gated_attention=_t.get("apply_gated_attention", False),
                    audio_caption_projection=cp,
                    cross_attention_adaln=_t.get("cross_attention_adaln", False),
                )

        sd_ops = SDOps("AO").with_matching(prefix="model.diffusion_model.").with_replacement(
            "model.diffusion_model.", "")
        builder = Builder(
            model_path=self.checkpoint,
            model_class_configurator=_AOC,
            model_sd_ops=sd_ops,
            registry=DummyRegistry(),
        )
        model = builder.build(device=self.device, dtype=self.dtype).eval()

        # Apply LoRA if provided
        if normalized:
            from peft import LoraConfig, get_peft_model
            from safetensors.torch import load_file as st_load
            lora_sd = st_load(normalized)
            is_peft = any("base_model.model." in k for k in lora_sd)
            is_id = any("diffusion_model." in k for k in lora_sd)
            lora_cfg = LoraConfig(
                r=lora_rank, lora_alpha=lora_rank, lora_dropout=0.0, bias="none",
                target_modules=[
                    "audio_attn1.to_k", "audio_attn1.to_q",
                    "audio_attn1.to_v", "audio_attn1.to_out.0",
                    "audio_attn2.to_k", "audio_attn2.to_q",
                    "audio_attn2.to_v", "audio_attn2.to_out.0",
                    "audio_ff.net.0.proj", "audio_ff.net.2",
                ],
            )
            model = get_peft_model(model, lora_cfg)
            if is_peft:
                # PEFT format: base_model.model.<layer>.lora_A.weight
                mapped = {}
                for k, v in lora_sd.items():
                    nk = k.replace(".lora_A.weight", ".lora_A.default.weight") \
                          .replace(".lora_B.weight", ".lora_B.default.weight")
                    mapped[nk] = v
                model.load_state_dict(mapped, strict=False)
            elif is_id:
                # Original IC-LoRA format: diffusion_model.<layer>.lora_A.weight
                mapped = {}
                for k, v in lora_sd.items():
                    if not any(x in k for x in ("audio_attn1", "audio_attn2", "audio_ff")):
                        continue
                    nk = k.replace("diffusion_model.", "base_model.model.") \
                          .replace(".lora_A.weight", ".lora_A.default.weight") \
                          .replace(".lora_B.weight", ".lora_B.default.weight")
                    mapped[nk] = v
                model.load_state_dict(mapped, strict=False)
            else:
                # Clean LTX format: <layer>.lora_A.weight (no prefix, no .default.)
                mapped = {}
                for k, v in lora_sd.items():
                    nk = "base_model.model." + k
                    nk = nk.replace(".lora_A.weight", ".lora_A.default.weight") \
                           .replace(".lora_B.weight", ".lora_B.default.weight")
                    mapped[nk] = v
                model.load_state_dict(mapped, strict=False)
            model = model.merge_and_unload()
            logging.info(f"  LoRA merged")

        self._velocity_model = model
        self._loaded_lora_path = normalized

        # Re-apply torch.compile if it was active on the previous transformer
        if self.compile_model:
            logging.info("  Re-compiling reloaded transformer...")
            self._velocity_model = torch.compile(self._velocity_model, mode="default", dynamic=True)

        logging.info(f"  Transformer reload: {time.time()-t0:.1f}s")

    @torch.inference_mode()
    def generate(self, prompt, voice_ref=None, cfg_scale=2.5, stg_scale=1.5,
                 duration_multiplier=1.1, seed=42, ref_duration=10.0,
                 rescale_scale="auto", gen_duration: float = 0.0,
                 steps=30, sampler="euler", negative_prompt=None, speed=1.0,
                 stg_block=29, lora_path=None, lora_rank=128):
        """Generate audio. Returns (waveform_path, duration_seconds).

        rescale_scale: latent-side CFG std-rescale that prevents clipping at
            high cfg. Set to "auto" (default) for the cfg-aware schedule, a
            float in [0, 1] for a fixed override, or 0 to disable.
        gen_duration: explicit target duration in seconds. 0 (default) → auto
            from prompt + duration_multiplier; >0 overrides everything else.
        """
        t_total = time.time()

        # Hot-swap LoRA if it has changed since the last call
        self._reload_for_lora(lora_path, lora_rank)

        # Duration + target shape — explicit gen_duration wins over the estimator.
        if gen_duration and gen_duration > 0:
            gen_dur = float(gen_duration)
        else:
            gen_dur = estimate_duration(prompt, duration_multiplier, speed)
        fps = 25.0
        n_frames = int(round(gen_dur * fps)) + 1
        n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
        pixel_shape = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
        target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools = AudioLatentTools(patchifier=self.patchifier, target_shape=target_shape)

        # Initial state
        state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype)

        # Voice ref conditioning
        if voice_ref and os.path.exists(voice_ref):
            t0 = time.time()
            voice = decode_audio_from_file(voice_ref, self.device, 0.0, ref_duration)
            w = voice.waveform
            if w.dim() == 2:
                if w.shape[0] == 1:
                    w = w.repeat(2, 1)
                w = w.unsqueeze(0)
            elif w.dim() == 3 and w.shape[1] == 1:
                w = w.repeat(1, 2, 1)
            target_samples = int(ref_duration * voice.sampling_rate)
            if w.shape[-1] < target_samples:
                w = w.repeat(1, 1, (target_samples // w.shape[-1]) + 1)
            w = w[..., :target_samples]
            peak = w.abs().max()
            if peak > 0:
                w = w * (10 ** (-4.0 / 20) / peak)
            voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)
            ref_latent = self._audio_conditioner(lambda enc: vae_encode_audio(voice, enc, None))
            cond = AudioConditionByReferenceLatent(latent=ref_latent.to(self.device, self.dtype), strength=1.0)
            state = cond.apply_to(state, audio_tools)
            logging.info(f"Voice ref: {time.time()-t0:.2f}s")

        # Noise
        gen = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=gen)
        state = noiser(state, noise_scale=1.0)

        # Prompt encode
        t0 = time.time()
        neg = negative_prompt if negative_prompt else DEFAULT_NEG
        prompts = [prompt, neg] if cfg_scale > 1.0 else [prompt]
        ctx = self._prompt_encoder(prompts, streaming_prefetch_count=None)
        a_ctx = ctx[0].audio_encoding
        a_ctx_neg = ctx[1].audio_encoding if cfg_scale > 1.0 else None
        logging.info(f"Prompt: {time.time()-t0:.2f}s")

        # Denoiser
        resc = auto_rescale_for_cfg(cfg_scale) if rescale_scale == "auto" else float(rescale_scale)
        if rescale_scale == "auto":
            logging.info(f"Auto rescale_scale = {resc:.2f} for cfg={cfg_scale}")
        guider = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=cfg_scale, stg_scale=stg_scale,
                stg_blocks=[stg_block] if stg_scale > 0 else [], rescale_scale=resc, modality_scale=1.0,
            ),
            negative_context=a_ctx_neg,
        )
        denoiser = GuidedDenoiser(
            v_context=None, a_context=a_ctx,
            video_guider=None, audio_guider=guider,
        )

        # Sigmas
        sigmas = LTX2Scheduler().execute(steps=steps, latent=state.latent).to(self.device)

        # Denoise
        t0 = time.time()
        x0 = X0Model(self._velocity_model)
        denoise_fn = heun_denoising_loop if sampler == "heun" else euler_denoising_loop
        _, audio_state = denoise_fn(
            sigmas=sigmas, video_state=None, audio_state=state,
            stepper=EulerDiffusionStep(), transformer=x0, denoiser=denoiser,
        )
        logging.info(f"Denoise ({steps} steps, {sampler}): {time.time()-t0:.2f}s")

        # Strip + unpatchify + decode
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        # End-of-clip silence-prior fix.
        # The base LTX-2.3 22B DiT was trained on audio clips ≤ ~20 s and
        # learned a strong "clip-end silence" prior that lands on the next
        # patchifier-aligned latent frame after 20 s — index 513 = 8*64+1.
        # When inference produces longer audio, this prior leaks through as a
        # high-norm latent burst at frame 513 (and adjacent 512), which the
        # audio VAE + vocoder render as a ~30 ms hard silence dip near 20.4 s.
        # Linear interpolation across the two affected frames removes the dip
        # cleanly without any retraining. Only runs when the latent is long
        # enough to actually contain the boundary.
        latent = audio_state.latent
        if latent.shape[2] > 513:
            f0, f1 = 511, 514          # neighbours used for interpolation
            n = f1 - f0                # = 3
            patched = latent.clone()
            for f in (512, 513):
                t = (f - f0) / n
                patched[:, :, f, :] = (1.0 - t) * latent[:, :, f0, :] + t * latent[:, :, f1, :]
            latent = patched

        t0 = time.time()
        decoded = self._audio_decoder(latent)
        logging.info(f"Decode: {time.time()-t0:.2f}s")

        total = time.time() - t_total
        dur = decoded.waveform.shape[-1] / decoded.sampling_rate
        logging.info(f"Total: {total:.2f}s for {dur:.1f}s audio")
        return decoded.waveform, decoded.sampling_rate

    def generate_to_file(self, prompt, output, watermark: bool = True, **kwargs):
        waveform, sr = self.generate(prompt, **kwargs)
        wav_cpu = waveform.cpu().float()
        if watermark:
            try:
                import numpy as np, perth
                if not hasattr(self, "_perth"):
                    self._perth = perth.PerthImplicitWatermarker()
                mono = wav_cpu.mean(dim=0).numpy() if wav_cpu.shape[0] > 1 else wav_cpu[0].numpy()
                mono_wm = self._perth.apply_watermark(mono, sample_rate=sr)
                mono_wm_t = torch.from_numpy(np.asarray(mono_wm, dtype=np.float32)).unsqueeze(0)
                wav_cpu = mono_wm_t if wav_cpu.shape[0] == 1 else mono_wm_t.repeat(wav_cpu.shape[0], 1)
            except Exception as e:
                logging.warning(f"Perth watermark skipped ({e})")
        _wav_data = wav_cpu.numpy()
        if _wav_data.ndim == 2:
            _wav_data = _wav_data.T
        sf.write(output, _wav_data, sr)
        logging.info(f"Saved: {output}")
        return output


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-bnb-4bit", action="store_true",
                   help="Disable bitsandbytes 4-bit path (default: on, since the default "
                        "unsloth Gemma checkpoint is pre-quantized).")
    args = p.parse_args()

    server = TTSServer(device=args.device, dtype=args.dtype, compile_model=not args.no_compile,
                       bnb_4bit=not args.no_bnb_4bit)

    # First call - includes any warmup
    logging.info("=== First request ===")
    server.generate_to_file(
        prompt='A woman speaks clearly, "The weather today will be sunny."',
        output="/tmp/warm_test1.wav",
        voice_ref="/mnt/persistent0/manmay/expressive/female_radio_nikole/female_radio_nikole.wav",
    )

    # Second call - should be much faster (models already warm)
    logging.info("\n=== Second request (warm) ===")
    server.generate_to_file(
        prompt='A man speaks excitedly, "This is amazing, I cannot believe it!"',
        output="/tmp/warm_test2.wav",
        voice_ref="/mnt/persistent0/manmay/expressive/male_arnie/male_arnie.mp3",
    )
