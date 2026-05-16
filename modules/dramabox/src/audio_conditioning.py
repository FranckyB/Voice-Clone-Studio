"""Audio reference conditioning item for IC-LoRA voice cloning."""

import torch

from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import AudioLatentTools
from ltx_core.types import AudioLatentShape, LatentState


class AudioConditionByReferenceLatent(ConditioningItem):
    """Conditions audio generation on a reference audio latent for voice cloning.

    Mirrors VideoConditionByReferenceLatent but for audio:
    - Patchifies reference latent [B, C, T, F] -> [B, ref_T, 128]
    - Computes 1D temporal positions via AudioPatchifier
    - Sets denoise_mask = 1.0 - strength (strength=1.0 -> mask=0 -> frozen)
    - Builds ASYMMETRIC attention mask: target->ref=1 (attend), ref->target=0 (read-only)
    - APPENDS ref tokens to END of latent sequence (IC-LoRA pattern)
    - Uses OVERLAPPING positions (same coordinate space) so RoPE doesn't
      decay target->ref attention. The asymmetric mask provides the structural
      signal that ref tokens are conditioning, not reconstruction targets.

    Args:
        latent: Reference audio latent [B, C, T, F] (pre-VAE-encoded).
        strength: Conditioning strength. 1.0 = full (ref kept clean),
            0.0 = none (ref fully denoised). Default 1.0.
    """

    def __init__(self, latent: torch.Tensor, strength: float = 1.0):
        self.latent = latent
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: AudioLatentTools,
    ) -> LatentState:
        """Append reference audio tokens with positions and attention mask."""
        tokens = latent_tools.patchifier.patchify(self.latent)

        # Compute positions for the reference audio — small offset (0.5s) from
        # target start to avoid exact t=0 overlap (which causes ref content to
        # bleed into target start), while keeping RoPE decay minimal.
        # 0.5s / max_pos(20s) = 0.025 fractional — negligible RoPE decay.
        ref_shape = AudioLatentShape(
            batch=self.latent.shape[0],
            channels=self.latent.shape[1],
            frames=self.latent.shape[2],
            mel_bins=self.latent.shape[3],
        )
        positions = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=ref_shape,
            device=self.latent.device,
        )
        # Small offset to prevent t=0 position collision between target and ref
        positions = positions + 0.5

        # Denoise mask: 0 for frozen (strength=1.0), 1 for fully denoised (strength=0.0)
        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device,
            dtype=torch.float32,
        )

        # Build ASYMMETRIC attention mask manually.
        # Structure:
        #              target (N)    ref (M)
        #            ┌────────────┬──────────┐
        #   target   │    1.0     │   1.0    │  target attends to everything
        #    (N)     │            │          │
        #            ├────────────┼──────────┤
        #    ref     │    0.0     │   1.0    │  ref only attends to itself
        #    (M)     │            │          │
        #            └────────────┴──────────┘
        #
        # This makes reference tokens "read-only conditioning":
        # - Target tokens freely attend to ref (voice cloning signal)
        # - Ref tokens don't attend to noisy target (stays clean/stable)
        batch_size = tokens.shape[0]
        num_target = latent_state.latent.shape[1]
        num_ref = tokens.shape[1]
        total = num_target + num_ref

        # Use float32 for the [0,1] mask — _prepare_self_attention_mask converts
        # to log-space bias in the model's compute dtype before it reaches attention.
        mask = torch.zeros(
            (batch_size, total, total),
            device=self.latent.device,
            dtype=torch.float32,
        )

        # Incorporate existing mask if present, otherwise full attention for target
        if latent_state.attention_mask is not None:
            mask[:, :num_target, :num_target] = latent_state.attention_mask
        else:
            mask[:, :num_target, :num_target] = 1.0

        # Target -> ref: FULL attention (target can read reference voice)
        mask[:, :num_target, num_target:] = 1.0

        # Ref -> target: BLOCKED (ref is read-only, doesn't see noisy target)
        # mask[:, num_target:, :num_target] remains 0.0

        # Ref -> ref: full self-attention within reference
        mask[:, num_target:, num_target:] = 1.0

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=mask,
        )
