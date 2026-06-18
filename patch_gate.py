"""Bag-size-invariant patch gate driven by PatchMILHead attention logits.

Gates patch features by z-scoring the pre-softmax attention logits across
the bag and applying a doubled sigmoid (range [0, 2], identity-equivalent
at init). The z-scoring makes the gate threshold a pure statistical
quantity (sigma above bag mean) rather than a per-bag-size-tuned magnitude,
which is essential for WSI bags where N varies from a few hundred to
many thousands of patches per slide.
"""

import torch
import torch.nn as nn


class PatchGate(nn.Module):
    """Z-scored attention-logit gate.

    Args:
        tau_init: Initial value of the learnable temperature scalar.
    """
    def __init__(self, tau_init: float = 1.0) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

    def forward(
        self,
        patch_feats: torch.Tensor,
        attn_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Gate patch features by z-scored attention logits.

        Args:
            patch_feats:  [B, N, D] patch features.
            attn_logits: [B, N]    pre-softmax attention logits from PatchMILHead.

        Returns:
            gated patch features, shape [B, N, D].
        """
        mu = attn_logits.mean(dim=1, keepdim=True)
        sigma = attn_logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        z = (attn_logits - mu) / sigma
        g = 2.0 * torch.sigmoid(self.tau * z)
        return patch_feats * g.unsqueeze(-1)
