#!/usr/bin/env python3
"""
LTX-2.3 TTS with IC-LoRA voice cloning.

Uses AudioConditionByReferenceLatent to append reference audio tokens to the
end of the target sequence.  Auto-detects distilled vs dev checkpoint and
selects the appropriate denoiser (SimpleDenoiser / GuidedDenoiser) and sigma
schedule.  Leverages the official euler_denoising_loop, AudioLatentTools,
GaussianNoiser, and X0Model wrapper throughout.

Usage (distilled):
    python tts_iclora.py \
        --voice-sample reference.wav \
        --prompt "A woman speaks clearly: The weather today will be sunny." \
        --output tts_output.wav

Usage (dev):
    python tts_iclora.py \
        --voice-sample reference.wav \
        --prompt "A woman speaks clearly: The weather today will be sunny." \
        --checkpoint ltx-2.3-22b-dev-audio-only.safetensors \
        --full-checkpoint ltx-2.3-22b-dev.safetensors \
        --output tts_output.wav
"""

import argparse
import json
import logging
import os
import re
import struct
import sys
import time
from pathlib import Path

import torch
import soundfile as sf

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ltx2"))
# ltx-pipelines already on path via ltx2/

# Also add the local directory so audio_conditioning.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_models_dir():
    config_path = PROJECT_ROOT / "config.json"
    models_folder = "models"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            models_folder = config.get("models_folder", models_folder)
        except Exception:
            pass
    return PROJECT_ROOT / models_folder


MODEL_DIR = str(_resolve_models_dir())
GEMMA_DIR = os.environ.get("GEMMA_DIR", "gemma-3-12b-it-qat-q4_0-unquantized")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_model_type(checkpoint_path: str) -> str:
    """Detect if checkpoint is distilled or dev by checking filename and metadata."""
    path_lower = checkpoint_path.lower()
    if "distilled" in path_lower:
        return "distilled"
    if "dev" in path_lower:
        return "dev"
    # Fallback: try to read safetensors metadata
    try:
        with open(checkpoint_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode())
        metadata = header.get("__metadata__", {})
        version = metadata.get("model_version", "")
        if "distilled" in version.lower():
            return "distilled"
    except Exception:
        pass
    # Default to distilled (most common for audio-only)
    return "distilled"


_LAUGH_VERBS = {
    # base seconds per occurrence; gets scaled by the modifier found nearby.
    # Verb regex covers inflections: laugh/laughs/laughed/laughing.
    r"\blaugh(?:s|ed|ing)?\b": 1.5,
    r"\bcackl(?:e|es|ed|ing)\b": 1.5,
    r"\bchuckl(?:e|es|ed|ing)\b": 1.0,
    r"\bgiggl(?:e|es|ed|ing)\b": 1.0,
    r"\bsnicker(?:s|ed|ing)?\b": 0.8,
    r"\bcru?el laugh\b": 1.5,
}


def _contextual_laugh_duration(text: str) -> float:
    """Context-aware laugh budget.

    For each laugh verb in the prompt, look at the adjective/adverb that
    modifies it and scale the base duration:
      - short modifiers  (briefly, softly, once)     -> 0.4x base
      - long modifiers   (maniacally, heartily, ...) -> 1.2x base
      - default (no mod / neutral)                   -> 1.0x base
    Also reward phonetic repetition inside quotes -- 'Hahahahahaha' buys more
    time than 'Haha' -- at ~0.2s per extra repeated syllable.
    """
    # "softly" / "quietly" describe volume not length, so keep at default 1.0x.
    short_mod = re.compile(
        r"^\s*(?:[a-z]+ly )?(?:briefly|shortly|once|quickly)",
        re.IGNORECASE)
    long_mod = re.compile(
        r"^\s*(?:[a-z]+ly )?(?:maniacally|heartily|uproariously|uncontrollably|"
        r"hysterically|darkly|wickedly|evilly|loudly|long)"
        r"|^\s*between phrases", re.IGNORECASE)

    total = 0.0
    for pat, base_dur in _LAUGH_VERBS.items():
        for m in re.finditer(pat, text, re.IGNORECASE):
            ctx = text[m.end(): m.end() + 40]
            if short_mod.match(ctx):
                total += base_dur * 0.4
            elif long_mod.match(ctx):
                total += base_dur * 1.2
            else:
                total += base_dur

    # Phonetic laugh repetition inside quotes:
    #   'Haha' = 2 syllables (base, no bonus)
    #   'Hahahaha' = 4 syllables (+0.4s)
    #   'Hehehehahahahahahahaha' ~ 10 syllables (+1.6s)
    for q in re.findall(r'"([^"]+)"', text) + re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", text):
        for run in re.findall(r"(?:h[ae]){3,}|(?:h[ae][ \-]?){3,}", q, re.IGNORECASE):
            syls = len(re.findall(r"h[ae]", run, re.IGNORECASE))
            total += 0.2 * max(syls - 2, 0)
    return total


def _estimate_nonverbal_duration(text: str) -> float:
    """Estimate extra duration for non-verbal sounds and actions in the prompt.

    Laugh-verb handling lives in ``_contextual_laugh_duration`` so cackle /
    chuckle / laugh budgets scale with the adjective ("maniacally" vs
    "briefly") and with the repetition length of 'Ha'/'He' tokens inside
    quotes.
    """
    PATTERNS = {
        # Breathing / sighs
        r'\bsighs?\b': 0.8, r'\bshaky breath\b': 1.0, r'\bbreathing deeply\b': 1.0,
        r'\bgasps?\b': 0.5, r'\bburps?\b': 0.5, r'\byawns?\b': 1.0,
        r'\bpants?\b': 0.8, r'\bwheezes?\b': 0.8, r'\bcoughs?\b': 0.8,
        r'\bsniffles?\b': 0.5, r'\bsnorts?\b': 0.3, r'\bgroans?\b': 0.8,
        # Pauses (trimmed; earlier values over-budgeted silence)
        r'\blong pause\b': 1.0, r'\bpauses? briefly\b': 0.3,
        r'\bpauses?\b': 0.5, r'\bsilence\b': 1.0,
        r'\blets? the .{1,20} hang\b': 1.0, r'\blets? .{1,20} sink in\b': 1.0,
        # Physical actions that produce sound
        r'\bslams?\b': 0.5, r'\bclaps?\b': 0.3,
        r'\bdraws? (?:his|her|a) sword\b': 0.5,
        r'\btakes? a (?:drag|swig|sip|drink)\b': 0.5,
        r'\bwhistles?\b': 1.0, r'\bhums?\b': 0.8,
        # Vocal actions (not in quotes but take time)
        r'\bmutters?\b': 1.5, r'\bmumbles?\b': 1.0, r'\bwhispers?\b': 0.0,
        r'\bclears? (?:his|her) throat\b': 0.5, r'\bgulps?\b': 0.5,
        r'\bswallows?\b': 0.5,
        # (laugh / chuckle / cackle / giggle / snicker handled by
        # _contextual_laugh_duration below -- modifier-aware, not flat.)
        # Emotional transitions
        r'\bvoice (?:breaks?|cracks?|trembles?|drops?|rises?)\b': 0.5,
        r'\bsteadies? (?:him|her)self\b': 1.0,
        r'\bcatches? (?:his|her) breath\b': 1.0,
        r'\bcomposes? (?:him|her)self\b': 0.8,
        # Scene transitions that imply time
        r'\bdemeanor shifts?\b': 0.5, r'\bsettles? in\b': 0.5,
        r'\bleans? in\b': 0.3, r'\bwipes? (?:his|her) eyes\b': 0.5,
    }
    extra = 0.0
    for pattern, dur in PATTERNS.items():
        extra += dur * len(re.findall(pattern, text, re.IGNORECASE))
    extra += _contextual_laugh_duration(text)
    return extra


def estimate_speech_duration(text: str, speed: float = 1.0) -> float:
    """Estimate speech duration from spoken content + non-verbal actions.

    Extracts spoken text by priority:
    1. Quoted text ('...' or "...") -- official prompt guide format
    2. Text after colon -- simple "Speaker: dialogue" format
    3. Full text -- fallback

    Also scans the full prompt for non-verbal cues (laughs, pauses, sighs,
    gasps, etc.) and adds estimated duration for each.
    """
    # Try double quotes first (clean, no contraction issues)
    quotes = re.findall(r'"([^"]+)"', text)
    if not quotes:
        # Single quotes: allow apostrophes in contractions (don't, can't, it's)
        # Match ' to ' but apostrophes NOT followed by space/punctuation are kept inside
        quotes = re.findall(r"'((?:[^']|'(?![\s.,!?)\]]))+)'", text)
        # Filter out short fragments (scene directions like "He pauses")
        quotes = [q for q in quotes if len(q.split()) > 3]
    if quotes:
        spoken = " ".join(quotes)
    elif ":" in text:
        spoken = text.split(":", 1)[1].strip()
    else:
        spoken = text

    CHARS_PER_SEC = 14.0
    text_len = len(spoken)

    if text_len < 40:
        chars_per_sec = CHARS_PER_SEC * 0.6
    elif text_len < 80:
        chars_per_sec = CHARS_PER_SEC * 0.8
    else:
        chars_per_sec = CHARS_PER_SEC

    chars_per_sec *= speed
    duration = text_len / chars_per_sec

    sentence_count = spoken.count(".") + spoken.count("!") + spoken.count("?")
    duration += sentence_count * 0.3

    # Add time for non-verbal sounds/actions in the full prompt
    duration += _estimate_nonverbal_duration(text)

    return max(3.0, round(duration + 2.0, 1))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LTX-2.3 TTS with IC-LoRA voice cloning")

    p.add_argument("--voice-sample", default=None, help="Voice reference WAV")
    p.add_argument("--no-ref", action="store_true", help="Skip voice reference conditioning (raw base model)")
    p.add_argument("--prompt", required=True, help="Text/scene description to synthesize")
    p.add_argument("--output", default="tts_output.wav")

    p.add_argument("--ref-duration", type=float, default=10.0, help="Seconds of voice reference to use")
    p.add_argument("--gen-duration", type=float, default=0.0,
                   help="Target output duration in seconds (0 = auto from prompt + multiplier). "
                        "Set explicitly for long-form prompts (e.g. --gen-duration 30 for music). "
                        "Outputs >20.5s automatically engage the end-of-clip silence-prior patch.")
    p.add_argument("--pad-start", type=float, default=0.0,
                   help="Prepend N seconds of silent padding, trimmed after decode (use 0 for clean starts)")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--duration-multiplier", type=float, default=1.0,
                   help="Multiply auto-estimated duration by this factor (e.g. 1.1 for 10%% more breathing room)")

    p.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "ltx-2.3-audio-only.safetensors"))
    p.add_argument("--full-checkpoint", default=os.path.join(MODEL_DIR, "ltx-2.3-22b-distilled.safetensors"))
    p.add_argument("--gemma-root", default=GEMMA_DIR)
    p.add_argument("--bnb-4bit", dest="bnb_4bit", action="store_true", default=True,
                   help="Load Gemma text encoder via the bitsandbytes 4-bit path "
                        "(required for the default unsloth/gemma-3-12b-it-bnb-4bit "
                        "pre-quantized weights). Default: on.")
    p.add_argument("--no-bnb-4bit", dest="bnb_4bit", action="store_false",
                   help="Disable the bitsandbytes path (use only if --gemma-root "
                        "points at an unquantized Gemma checkpoint).")
    p.add_argument("--lora", default=None, help="Path to trained IC-LoRA .safetensors (audio-only)")
    p.add_argument("--lora-rank", type=int, default=128, help="LoRA rank (must match training)")
    p.add_argument("--id-guidance-scale", type=float, default=3.0, help="Identity guidance scale (0=disabled)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-offload", dest="cpu_offload", action="store_true", default=True,
                   help="Keep auxiliary DramaBox stages on CPU to reduce peak VRAM (default: on).")
    p.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false",
                   help="Disable CPU offloading and load auxiliary stages on the main device.")

    # Auto-set based on model type but overridable
    p.add_argument("--no-watermark", action="store_true",
                   help="Skip Perth audio watermarking on the output (default: watermark on).")
    p.add_argument("--sampler", choices=["euler", "heun"], default="euler",
                   help="Denoising loop. 'heun' = jkass_quality 2nd-order predictor-corrector (~2x model calls, cleaner audio).")
    p.add_argument("--cfg-scale", type=float, default=None, help="CFG scale (auto: 1.0 distilled, 7.0 dev)")
    p.add_argument("--stg-scale", type=float, default=None, help="STG scale (auto: 0.0 distilled, 1.0 dev)")
    p.add_argument("--stg-block", type=int, default=29, help="Block index for STG perturbation")
    p.add_argument("--rescale-scale", type=float, default=None,
                   help="Latent CFG std-rescale (default auto: cfg-aware schedule that prevents "
                        "output clipping at high cfg; pass any float in [0,1] to override).")
    p.add_argument("--modality-scale", type=float, default=None, help="Modality (auto: 1.0 distilled, 3.0 dev)")
    p.add_argument("--cfg-clamp", type=float, default=0.0, help="Clamp guided pred std to N * cond std (0=disabled)")
    p.add_argument("--steps", type=int, default=None, help="Override steps (auto: distilled sigmas / 30 dev)")
    p.add_argument("--fps", type=float, default=None, help="FPS (auto: 24.0 distilled, 25.0 dev)")
    p.add_argument(
        "--negative-prompt",
        default=(
            "worst quality, inconsistent motion, blurry, jittery, distorted, "
            "robotic voice, echo, background noise, off-sync audio, repetitive speech"
        ),
        help="Negative prompt for CFG (dev model)",
    )

    return p.parse_args(argv)


@torch.inference_mode()
def run_inference(args):
    t0 = time.time()

    # ---- Imports (deferred to avoid startup cost when checking --help) ----
    from audio_conditioning import AudioConditionByReferenceLatent

    from ltx_core.batch_split import BatchSplitAdapter
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import AudioPatchifier
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.model.model_protocol import ModelConfigurator
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.tools import AudioLatentTools
    from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoPixelShape
    from ltx_pipelines.utils.blocks import AudioConditioner, AudioDecoder, PromptEncoder
    from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
    from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
    from ltx_pipelines.utils.gpu_model import gpu_model
    from ltx_pipelines.utils.media_io import decode_audio_from_file
    from ltx_pipelines.utils.samplers import euler_denoising_loop, heun_denoising_loop

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    offload_device = torch.device("cpu") if args.cpu_offload and torch.cuda.is_available() else device
    patchifier = AudioPatchifier(patch_size=1)

    # ---- Detect model type and set defaults ----
    model_type = detect_model_type(args.full_checkpoint)
    logging.info(f"Detected model type: {model_type}")

    is_distilled = model_type == "distilled"

    if args.cfg_scale is None:
        args.cfg_scale = 1.0 if is_distilled else 7.0
    if args.stg_scale is None:
        args.stg_scale = 0.0 if is_distilled else 1.0
    if args.rescale_scale is None:
        # Auto cfg-aware rescale: imported from inference_server to keep one source of truth.
        from inference_server import auto_rescale_for_cfg
        args.rescale_scale = 0.0 if is_distilled else auto_rescale_for_cfg(args.cfg_scale)
    if args.modality_scale is None:
        args.modality_scale = 1.0 if is_distilled else 3.0
    if args.fps is None:
        args.fps = 24.0 if is_distilled else 25.0

    logging.info(
        f"Params: cfg={args.cfg_scale}, stg={args.stg_scale}, rescale={args.rescale_scale}, "
        f"modality={args.modality_scale}, fps={args.fps}"
    )

    # ---- Auto duration ----
    if args.gen_duration <= 0:
        args.gen_duration = estimate_speech_duration(args.prompt, args.speed)
        if args.duration_multiplier != 1.0:
            args.gen_duration = round(args.gen_duration * args.duration_multiplier, 1)
        logging.info(f"Auto duration: {args.gen_duration}s for {len(args.prompt)} chars"
                     f"{f' (x{args.duration_multiplier})' if args.duration_multiplier != 1.0 else ''}")

    # ---- Compute target shape (include pad_start in duration) ----
    padded_duration = args.gen_duration + args.pad_start
    raw_frames = int(round(padded_duration * args.fps)) + 1
    num_frames = ((raw_frames - 1 + 4) // 8) * 8 + 1
    pixel_shape = VideoPixelShape(batch=1, frames=num_frames, height=64, width=64, fps=args.fps)
    tgt_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
    logging.info(f"Target shape: {tgt_shape} ({args.gen_duration}s, {num_frames} frames)")

    # ---- AudioLatentTools for target ----
    audio_tools = AudioLatentTools(patchifier=patchifier, target_shape=tgt_shape)

    # ---- Create initial state ----
    state = audio_tools.create_initial_state(device, dtype)
    logging.info(
        f"Initial state: latent={state.latent.shape}, positions={state.positions.shape}, "
        f"denoise_mask={state.denoise_mask.shape}"
    )

    if not args.no_ref and args.voice_sample:
        # ---- Encode voice reference ----
        logging.info(f"Loading voice reference: {args.voice_sample}")
        voice = decode_audio_from_file(args.voice_sample, device, 0.0, args.ref_duration)
        if voice is None:
            raise ValueError(f"Could not load audio from {args.voice_sample}")

        w = voice.waveform
        if w.dim() == 2:
            if w.shape[0] == 1:
                w = w.repeat(2, 1)
            w = w.unsqueeze(0)
        elif w.dim() == 3 and w.shape[1] == 1:
            w = w.repeat(1, 2, 1)

        target_samples = int(args.ref_duration * voice.sampling_rate)
        if w.shape[-1] < target_samples:
            w = w.repeat(1, 1, (target_samples // w.shape[-1]) + 1)
        w = w[..., :target_samples]

        # Peak normalize reference
        peak = w.abs().max()
        if peak > 0:
            target_peak = 10 ** (-4.0 / 20)  # -4dB
            w = w * (target_peak / peak)
            logging.info(f"Normalized reference: peak {peak:.4f} -> {target_peak:.4f}")

        voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)

        logging.info("Encoding voice through Audio VAE...")
        ac = AudioConditioner(checkpoint_path=args.full_checkpoint, dtype=dtype, device=offload_device)
        ref_latent = ac(lambda enc: vae_encode_audio(voice, enc, None))
        del ac
        torch.cuda.empty_cache()
        logging.info(f"Reference latent: {ref_latent.shape}")

        # ---- Apply conditioning: append ref tokens to END ----
        conditioning = AudioConditionByReferenceLatent(latent=ref_latent.to(device, dtype), strength=1.0)
        state = conditioning.apply_to(latent_state=state, latent_tools=audio_tools)
        logging.info(
            f"After conditioning: latent={state.latent.shape}, positions={state.positions.shape}, "
            f"attention_mask={'None' if state.attention_mask is None else state.attention_mask.shape}"
        )
    else:
        logging.info("No voice reference — running raw base model")

    # ---- Apply noise ----
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator)
    noised_state = noiser(state, noise_scale=1.0)
    logging.info("Applied Gaussian noise to state")

    # ---- Encode prompt ----
    use_cfg = args.cfg_scale > 1.0
    logging.info("Encoding prompt...")
    pe = PromptEncoder(checkpoint_path=args.full_checkpoint, gemma_root=args.gemma_root, dtype=dtype, device=device,
                       use_bnb_4bit=args.bnb_4bit, warm=True, audio_only=True)
    prompts_to_encode = [args.prompt]
    if use_cfg:
        prompts_to_encode.append(args.negative_prompt)
    ctx = pe(prompts_to_encode, streaming_prefetch_count=None)
    a_ctx = ctx[0].audio_encoding
    a_ctx_neg = ctx[1].audio_encoding if use_cfg else None
    del pe
    torch.cuda.empty_cache()
    logging.info(f"Prompt encoded: a_ctx={a_ctx.shape}" + (f", a_ctx_neg={a_ctx_neg.shape}" if a_ctx_neg is not None else ""))

    # ---- Build audio-only model ----
    logging.info("Building audio-only model...")
    audio_only_sd_ops = SDOps("AO").with_matching(prefix="model.diffusion_model.").with_replacement(
        "model.diffusion_model.", ""
    )

    class AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
        @classmethod
        def from_config(cls, config):
            t = config.get("transformer", {})
            cp = None
            if not t.get("caption_proj_before_connector", False):
                from ltx_core.model.transformer.text_projection import create_caption_projection

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

    builder = Builder(
        model_path=args.checkpoint,
        model_class_configurator=AudioOnlyConfigurator,
        model_sd_ops=audio_only_sd_ops,
        registry=DummyRegistry(),
    )
    # Builder already places loaded tensors on the requested device.
    # Calling .to(device) again fails if pruned branches remain on meta.
    velocity_model = builder.build(device=device, dtype=dtype).eval()

    # ---- Load LoRA weights (if provided) ----
    if args.lora and os.path.exists(args.lora):
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file as st_load

        logging.info(f"Loading LoRA: {args.lora}")
        lora_sd = st_load(args.lora)

        is_peft_format = any("base_model.model." in k for k in lora_sd.keys())
        is_original_idlora = any("diffusion_model." in k for k in lora_sd.keys())

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "audio_attn1.to_k",
                "audio_attn1.to_q",
                "audio_attn1.to_v",
                "audio_attn1.to_out.0",
                "audio_attn2.to_k",
                "audio_attn2.to_q",
                "audio_attn2.to_v",
                "audio_attn2.to_out.0",
                "audio_ff.net.0.proj",
                "audio_ff.net.2",
            ],
        )
        velocity_model = get_peft_model(velocity_model, lora_config)

        if is_peft_format:
            mapped_sd = {}
            for k, v in lora_sd.items():
                new_key = k
                if ".lora_A.weight" in k and ".lora_A.default.weight" not in k:
                    new_key = k.replace(".lora_A.weight", ".lora_A.default.weight")
                if ".lora_B.weight" in k and ".lora_B.default.weight" not in k:
                    new_key = k.replace(".lora_B.weight", ".lora_B.default.weight")
                mapped_sd[new_key] = v
            missing, unexpected = velocity_model.load_state_dict(mapped_sd, strict=False)
            loaded = len(mapped_sd) - len(unexpected)
            logging.info(f"Loaded {loaded} LoRA weights (peft format)")
        elif is_original_idlora:
            audio_keys = {
                k: v
                for k, v in lora_sd.items()
                if "audio_attn1" in k or "audio_attn2" in k or "audio_ff" in k
            }
            mapped_sd = {}
            for k, v in audio_keys.items():
                new_key = k.replace("diffusion_model.", "base_model.model.")
                new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
                new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
                mapped_sd[new_key] = v
            missing, unexpected = velocity_model.load_state_dict(mapped_sd, strict=False)
            loaded = len(mapped_sd) - len(unexpected)
            logging.info(f"Loaded {loaded} LoRA weights (original ID-LoRA)")
        else:
            # Clean LTX format: <layer>.lora_A.weight (no prefix, no .default.)
            mapped_sd = {}
            for k, v in lora_sd.items():
                new_key = "base_model.model." + k
                new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
                new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
                mapped_sd[new_key] = v
            missing, unexpected = velocity_model.load_state_dict(mapped_sd, strict=False)
            loaded = len(mapped_sd) - len(unexpected)
            logging.info(f"Loaded {loaded} LoRA weights (clean LTX format)")

        velocity_model = velocity_model.merge_and_unload()
        logging.info("Merged LoRA into model")

    logging.info(f"Model: {sum(p.numel() for p in velocity_model.parameters()) / 1e9:.1f}B params")

    # ---- Wrap velocity model in X0Model ----
    x0_model = X0Model(velocity_model)

    # ---- Build denoiser and sigmas ----
    stepper = EulerDiffusionStep()

    # ---- Sigma schedule ----
    if is_distilled:
        if args.steps is not None and args.steps > 0:
            sigmas = LTX2Scheduler().execute(steps=args.steps, latent=noised_state.latent).to(device)
            logging.info(f"Distilled with custom {args.steps}-step schedule")
        else:
            sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)
            logging.info(f"Distilled {len(DISTILLED_SIGMA_VALUES) - 1}-step schedule")
    else:
        steps = args.steps if args.steps is not None and args.steps > 0 else 30
        sigmas = LTX2Scheduler().execute(steps=steps, latent=noised_state.latent).to(device)
        logging.info(f"Dev {steps}-step schedule")

    # ---- Denoiser: use GuidedDenoiser if any guidance is active, SimpleDenoiser otherwise ----
    needs_guidance = args.cfg_scale > 1.0 or args.stg_scale > 0.0 or args.modality_scale > 1.0
    if needs_guidance:
        audio_guider = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=args.cfg_scale,
                stg_scale=args.stg_scale,
                stg_blocks=[args.stg_block] if args.stg_scale > 0 else [],
                rescale_scale=args.rescale_scale,
                modality_scale=args.modality_scale,
                cfg_clamp_scale=args.cfg_clamp,
            ),
            negative_context=a_ctx_neg,
        )
        denoiser = GuidedDenoiser(
            v_context=None,
            a_context=a_ctx,
            video_guider=None,
            audio_guider=audio_guider,
        )
        logging.info(f"GuidedDenoiser: cfg={args.cfg_scale}, stg={args.stg_scale}, "
                     f"rescale={args.rescale_scale}, modality={args.modality_scale}")
    else:
        denoiser = SimpleDenoiser(v_context=None, a_context=a_ctx)
        logging.info("SimpleDenoiser (no guidance)")

    logging.info(f"Sigmas: {sigmas.tolist()}")

    # ---- Denoising loop ----
    logging.info(f"Running denoising loop ({len(sigmas) - 1} steps)...")
    with gpu_model(x0_model) as model:
        batched_model = BatchSplitAdapter(model, max_batch_size=1)

        denoise_fn = heun_denoising_loop if args.sampler == "heun" else euler_denoising_loop
        _, audio_state = denoise_fn(
            sigmas=sigmas,
            video_state=None,
            audio_state=noised_state,
            stepper=stepper,
            transformer=batched_model,
            denoiser=denoiser,
        )

    del velocity_model, x0_model
    torch.cuda.empty_cache()

    # ---- Strip ref tokens and unpatchify ----
    logging.info("Stripping conditioning and unpatchifying...")
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)
    logging.info(f"Final latent shape: {audio_state.latent.shape}")

    # ---- End-of-clip silence-prior fix ----
    # Base LTX-2.3 22B was trained on audio clips ≤ ~20 s and learned a strong
    # "clip-end silence" prior at the next patchifier-aligned latent boundary
    # (frame 513 = 8 × 64 + 1). For longer outputs that prior leaks through as
    # a ~30 ms hard silence dip near 20.4 s. Linearly interpolating frames
    # 512–513 between their neighbours (511 and 514) removes the dip cleanly.
    latent_in = audio_state.latent
    if latent_in.shape[2] > 513:
        f0, f1 = 511, 514
        n = f1 - f0
        patched = latent_in.clone()
        for f in (512, 513):
            t = (f - f0) / n
            patched[:, :, f, :] = (1.0 - t) * latent_in[:, :, f0, :] + t * latent_in[:, :, f1, :]
        latent_in = patched

    # ---- Decode audio ----
    logging.info("Decoding audio...")
    ad = AudioDecoder(checkpoint_path=args.full_checkpoint, dtype=dtype, device=offload_device)
    decoded = ad(latent_in)
    del ad
    torch.cuda.empty_cache()

    wav = decoded.waveform
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    sr = decoded.sampling_rate

    # Trim leading pad if --pad-start was used
    if args.pad_start > 0:
        trim_samples = int(args.pad_start * sr)
        wav = wav[..., trim_samples:]
        logging.info(f"Trimmed {args.pad_start}s ({trim_samples} samples) of start padding")

    # Apply Perth (Perceptual Threshold) imperceptible neural watermark — see
    # https://github.com/resemble-ai/perth. Mono waveform required; if stereo,
    # we average to mono for the watermark and broadcast back. Skip on
    # --no-watermark for debugging.
    wav_cpu = wav.float().cpu()
    if not getattr(args, "no_watermark", False):
        try:
            import perth
            import numpy as np
            wm = perth.PerthImplicitWatermarker()
            mono = wav_cpu.mean(dim=0).numpy() if wav_cpu.shape[0] > 1 else wav_cpu[0].numpy()
            mono_wm = wm.apply_watermark(mono, sample_rate=sr)
            mono_wm_t = torch.from_numpy(np.asarray(mono_wm, dtype=np.float32)).unsqueeze(0)
            wav_cpu = mono_wm_t if wav_cpu.shape[0] == 1 else mono_wm_t.repeat(wav_cpu.shape[0], 1)
        except Exception as e:
            logging.warning(f"Perth watermark skipped ({e})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    _wav_data = wav_cpu.numpy()
    if _wav_data.ndim == 2:
        _wav_data = _wav_data.T
    sf.write(args.output, _wav_data, sr)

    elapsed = time.time() - t0
    logging.info(f"Output: {args.output} ({wav.shape[-1] / sr:.1f}s)")
    logging.info(f"Total time: {elapsed:.1f}s")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    run_inference(args)


if __name__ == "__main__":
    main()
