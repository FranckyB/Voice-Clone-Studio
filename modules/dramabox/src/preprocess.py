#!/usr/bin/env python3
"""
Preprocess TTS datasets for LTX-2.3 audio-only LoRA fine-tuning.

Takes paired (audio, transcript) data and produces the format expected by
the LTX trainer:
    .precomputed/
    ├── latents/sample_N.pt         # Dummy video latents (minimal)
    ├── conditions/sample_N.pt      # Text embeddings from Gemma
    └── audio_latents/sample_N.pt   # Audio VAE-encoded latents

Supports multiple dataset formats:
  - gemini_synthetic: index.txt with ~-separated fields (id~speaker~lang~sr~samples~dur~phonemes~text)
  - libriheavy: index_ft.txt with ~-separated fields (id~speaker~lang~samples~dur~phonemes~text)
  - manifest: JSON/JSONL with {"audio_filepath": ..., "text": ...}
  - tsv: TSV file with audio_path<TAB>text columns

Usage:
    python preprocess_tts_data.py \
        --dataset-type gemini_synthetic \
        --index /mnt/large-datasets/gemini_synthetic_dataset/conversational_dataset_pp/index.txt \
        --audio-dir /mnt/large-datasets/gemini_synthetic_dataset/conversational_dataset_pp/wavs \
        --output-dir /mnt/persistent0/manmay/tts_training_data \
        --max-samples 10000 \
        --max-duration 20.0 \
        --min-duration 3.0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torchaudio

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ltx2"))
# ltx-pipelines on path via ltx2/

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMMA_DIR = os.environ.get("GEMMA_DIR", "gemma-3-12b-it-qat-q4_0-unquantized")


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess TTS data for LTX-2.3 fine-tuning")
    p.add_argument("--dataset-type", required=True,
                   choices=["gemini_synthetic", "libriheavy", "manifest", "tsv"],
                   help="Dataset format type")
    p.add_argument("--index", required=True, help="Path to index/manifest file")
    p.add_argument("--audio-dir", default=None,
                   help="Base directory for audio files (if paths in index are relative)")
    p.add_argument("--output-dir", required=True, help="Output directory for preprocessed data")
    p.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "ltx-2.3-22b-distilled.safetensors"))
    p.add_argument("--gemma-root", default=GEMMA_DIR)
    p.add_argument("--max-samples", type=int, default=0, help="Max samples to process (0=all)")
    p.add_argument("--max-duration", type=float, default=20.0, help="Max audio duration in seconds")
    p.add_argument("--min-duration", type=float, default=2.0, help="Min audio duration in seconds")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for text encoding")
    p.add_argument("--skip-existing", action="store_true", help="Skip already processed samples")
    p.add_argument("--audio-only-ckpt", default=None,
                   help="Audio-only checkpoint for VAE encoding (optional, uses full ckpt if not set)")
    p.add_argument("--bnb-4bit", dest="bnb_4bit", action="store_true", default=True,
                   help="Load Gemma with bitsandbytes 4-bit (required for the unsloth pre-quantized weights). Default: on.")
    p.add_argument("--no-bnb-4bit", dest="bnb_4bit", action="store_false",
                   help="Disable bitsandbytes 4-bit loading (use when gemma_root points at a full-precision checkpoint).")
    p.add_argument("--shard", type=int, default=0, help="Shard index (for parallel processing)")
    p.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    p.add_argument("--gpu", type=int, default=None, help="GPU device index to use")
    return p.parse_args()


def parse_gemini_synthetic(index_path: str, audio_dir: str | None) -> list[dict]:
    """Parse gemini_synthetic format: id~speaker~lang~sr~samples~dur~phonemes~text"""
    samples = []
    with open(index_path) as f:
        for line in f:
            parts = line.strip().split("~")
            if len(parts) < 7:
                continue
            file_id = parts[0]
            text = parts[-1]  # Last field is always the text
            sr = int(parts[3])
            n_samples = int(parts[4])
            duration = n_samples / sr

            # Find audio file
            if audio_dir:
                # Try common extensions
                for ext in [".flac", ".wav", ".mp3"]:
                    audio_path = os.path.join(audio_dir, file_id + ext)
                    if os.path.exists(audio_path):
                        break
                else:
                    continue
            else:
                audio_path = file_id

            samples.append({
                "id": file_id,
                "audio_path": audio_path,
                "text": text,
                "duration": duration,
            })
    return samples


def parse_libriheavy(index_path: str, audio_dir: str | None) -> list[dict]:
    """Parse libriheavy format: id~speaker~lang~samples~dur~phonemes~text"""
    samples = []
    with open(index_path) as f:
        for line in f:
            parts = line.strip().split("~")
            if len(parts) < 7:
                continue
            file_id = parts[0]
            text = parts[-1]
            n_samples = int(parts[3])
            duration = int(parts[4]) / 1000.0  # milliseconds to seconds

            if audio_dir:
                for ext in [".flac", ".wav", ".mp3"]:
                    audio_path = os.path.join(audio_dir, file_id + ext)
                    if os.path.exists(audio_path):
                        break
                else:
                    continue
            else:
                audio_path = file_id

            samples.append({
                "id": file_id,
                "audio_path": audio_path,
                "text": text,
                "duration": duration,
            })
    return samples


def parse_manifest(index_path: str, audio_dir: str | None) -> list[dict]:
    """Parse JSON/JSONL manifest with audio_filepath and text fields."""
    samples = []
    with open(index_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            audio_path = entry.get("audio_filepath", entry.get("audio_path", ""))
            text = entry.get("text", entry.get("transcript", ""))
            duration = entry.get("duration", 0.0)

            if audio_dir and not os.path.isabs(audio_path):
                audio_path = os.path.join(audio_dir, audio_path)

            if os.path.exists(audio_path) and text:
                samples.append({
                    "id": Path(audio_path).stem,
                    "audio_path": audio_path,
                    "text": text,
                    "duration": duration,
                })
    return samples


def parse_tsv(index_path: str, audio_dir: str | None) -> list[dict]:
    """Parse TSV file with audio_path<TAB>text."""
    samples = []
    with open(index_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            audio_path, text = parts[0], parts[1]
            if audio_dir and not os.path.isabs(audio_path):
                audio_path = os.path.join(audio_dir, audio_path)
            if os.path.exists(audio_path):
                samples.append({
                    "id": Path(audio_path).stem,
                    "audio_path": audio_path,
                    "text": text,
                    "duration": 0.0,
                })
    return samples


PARSERS = {
    "gemini_synthetic": parse_gemini_synthetic,
    "libriheavy": parse_libriheavy,
    "manifest": parse_manifest,
    "tsv": parse_tsv,
}


def _load_feature_extractor(checkpoint_path, device, dtype):
    """Load feature extractor directly from checkpoint, bypassing SingleGPUModelBuilder.

    SingleGPUModelBuilder returns a meta model if ANY param is missing from the
    checkpoint. The DramaBox audio_components file only has audio_aggregate_embed
    weights — video_aggregate_embed is absent from both provided checkpoints.
    Since the trainer only uses audio_prompt_embeds (not video_prompt_embeds),
    video_aggregate_embed can be default-initialized.
    """
    import safetensors as _st
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import _create_feature_extractor

    f = _st.safe_open(str(checkpoint_path), framework="pt")
    meta = f.metadata() or {}
    transformer_config = json.loads(meta.get("config", "{}")).get("transformer", {})

    # Creates V1 or V2 feature extractor on CPU with default (random) initialization
    fe = _create_feature_extractor(transformer_config)

    # Map checkpoint keys: text_embedding_projection.<key> → <key>
    state = {}
    prefix = "text_embedding_projection."
    for key in f.keys():
        if key.startswith(prefix):
            state[key[len(prefix):]] = f.get_tensor(key).to(dtype)

    missing, unexpected = fe.load_state_dict(state, strict=False)
    if missing:
        logging.info(f"Feature extractor: using default init for missing keys: {missing}")
    if unexpected:
        logging.warning(f"Feature extractor: unexpected keys ignored: {unexpected}")

    return fe.to(dtype).to(device)


@torch.inference_mode()
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.types import Audio
    from ltx_pipelines.utils.blocks import AudioConditioner
    from ltx_pipelines.utils.media_io import decode_audio_from_file
    from ltx_trainer.model_loader import load_text_encoder

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Create output directories
    out = Path(args.output_dir)
    (out / "latents").mkdir(parents=True, exist_ok=True)
    (out / "conditions").mkdir(parents=True, exist_ok=True)
    (out / "audio_latents").mkdir(parents=True, exist_ok=True)

    # Parse dataset
    logging.info(f"Parsing {args.dataset_type} dataset from {args.index}...")
    samples = PARSERS[args.dataset_type](args.index, args.audio_dir)
    logging.info(f"Found {len(samples)} samples")

    # Filter by duration
    before = len(samples)
    samples = [s for s in samples if args.min_duration <= s["duration"] <= args.max_duration]
    logging.info(f"After duration filter [{args.min_duration}s, {args.max_duration}s]: {len(samples)} (dropped {before - len(samples)})")

    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        logging.info(f"Limiting to {len(samples)} samples")

    # Prune orphaned .pt files whose source audio was removed from the dataset
    valid_ids = {s["id"] for s in samples}
    pruned = 0
    for subdir in ("audio_latents", "conditions", "latents"):
        for pt in (out / subdir).glob("*.pt"):
            if pt.stem not in valid_ids:
                pt.unlink()
                pruned += 1
                logging.info(f"Pruned orphaned file: {subdir}/{pt.name}")
    if pruned:
        logging.info(f"Pruned {pruned} orphaned .pt file(s) with no matching audio source.")

    # Shard the data for parallel processing
    if args.num_shards > 1:
        total = len(samples)
        samples = samples[args.shard::args.num_shards]
        logging.info(f"Shard {args.shard}/{args.num_shards}: {len(samples)} samples (of {total} total)")

    # ── Step 1: Encode text with Gemma (Blocks 1+2 only) ──
    # The trainer runs Block 3 (embeddings processor/connectors) during training,
    # so we only precompute Blocks 1+2 here (Gemma LLM + feature extractor).
    logging.info("Loading text encoder (Gemma + feature extractor)...")
    if args.bnb_4bit:
        from ltx_pipelines.utils.blocks import PromptEncoder as _PE
        _pe = _PE.__new__(_PE)
        _pe._dtype = dtype
        _pe._device = torch.device(device)
        text_encoder = _pe._load_bnb_4bit_encoder(args.gemma_root)
    else:
        text_encoder = load_text_encoder(args.gemma_root, device=device, dtype=dtype)

    logging.info("Loading feature extractor...")
    ckpt_for_fe = args.audio_only_ckpt or args.checkpoint
    text_encoder.feature_extractor = _load_feature_extractor(ckpt_for_fe, device, dtype)
    torch.cuda.empty_cache()

    logging.info("Encoding text prompts (Blocks 1+2: Gemma + feature extractor)...")
    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        cond_path = out / "conditions" / f"{sample_id}.pt"
        if args.skip_existing and cond_path.exists():
            continue

        text = sample["text"]
        # Run Blocks 1+2: Gemma LLM → feature extractor
        hidden_states, attention_mask = text_encoder.encode(text)
        video_feats, audio_feats = text_encoder.feature_extractor(
            hidden_states, attention_mask, "left"
        )

        # Trim left-padding: Gemma uses left-padding so real tokens are at the end.
        # Dropping the zero-masked prefix saves ~4-8x disk space and speeds up
        # training. train.py re-pads to the nearest 128-token multiple before use.
        mask_1d = attention_mask.squeeze(0).bool()  # [seq_len]
        real_token_indices = mask_1d.nonzero(as_tuple=False)
        if real_token_indices.numel() > 0:
            first_real = real_token_indices[0].item()
        else:
            first_real = 0
        v_save = video_feats.squeeze(0)[first_real:].cpu()
        a_save = (audio_feats.squeeze(0)[first_real:].cpu()
                  if audio_feats is not None else v_save)
        m_save = mask_1d[first_real:].cpu()

        torch.save({
            "video_prompt_embeds": v_save,
            "audio_prompt_embeds": a_save,
            "prompt_attention_mask": m_save,
        }, cond_path)

        if i % 100 == 0:
            logging.info(f"  Text encoding: {i}/{len(samples)}")

    del text_encoder
    torch.cuda.empty_cache()

    # ── Step 2: Encode audio with Audio VAE ──
    ckpt_for_vae = args.audio_only_ckpt or args.checkpoint
    logging.info(f"Loading audio VAE from {ckpt_for_vae}...")

    ac = AudioConditioner(checkpoint_path=ckpt_for_vae, dtype=dtype, device=device)

    logging.info("Encoding audio samples...")
    for idx, sample in enumerate(samples):
        sample_id = sample["id"]
        audio_path = out / "audio_latents" / f"{sample_id}.pt"
        if args.skip_existing and audio_path.exists():
            continue

        try:
            # Load audio
            voice = decode_audio_from_file(sample["audio_path"], device, 0.0, args.max_duration)
            if voice is None:
                logging.warning(f"  Skipping {sample['id']}: no audio")
                continue

            w = voice.waveform
            if w.dim() == 2:
                if w.shape[0] == 1:
                    w = w.repeat(2, 1)
                w = w.unsqueeze(0)
            elif w.dim() == 3 and w.shape[1] == 1:
                w = w.repeat(1, 2, 1)
            voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)

            # Encode through Audio VAE
            audio_latent = ac(lambda enc: vae_encode_audio(voice, enc, None))

            # Save audio latent
            torch.save({
                "latents": audio_latent.squeeze(0).cpu(),  # [C=8, T, F=16]
                "sample_rate": 16000,
            }, audio_path)

        except Exception as e:
            logging.warning(f"  Skipping {sample['id']}: {e}")
            continue

        if idx % 100 == 0:
            logging.info(f"  Audio encoding: {idx}/{len(samples)}")

    del ac
    torch.cuda.empty_cache()

    # ── Step 3: Create dummy video latents ──
    logging.info("Creating dummy video latents...")
    # Minimal video: 1 frame, 64x64 = 2x2 in latent space
    dummy_video = {
        "latents": torch.zeros(128, 1, 2, 2),
        "num_frames": 1,
        "height": 2,
        "width": 2,
        "fps": 24.0,
    }
    for idx, sample in enumerate(samples):
        sample_id = sample["id"]
        latent_path = out / "latents" / f"{sample_id}.pt"
        if args.skip_existing and latent_path.exists():
            continue
        torch.save(dummy_video, latent_path)

    # ── Summary ──
    n_audio = len(list((out / "audio_latents").glob("*.pt")))
    n_cond = len(list((out / "conditions").glob("*.pt")))
    n_lat = len(list((out / "latents").glob("*.pt")))
    logging.info(f"\nDone! Output: {args.output_dir}")
    logging.info(f"  audio_latents: {n_audio} files")
    logging.info(f"  conditions:    {n_cond} files")
    logging.info(f"  latents:       {n_lat} files")


if __name__ == "__main__":
    main()
