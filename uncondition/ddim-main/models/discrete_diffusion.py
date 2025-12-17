"""Placeholder implementation for discrete diffusion models.

The class implements the same constructor/forward signature expected by
runners so you can drop in a discrete-time or token-based diffusion
architecture while reusing the existing training and sampling loops.
"""

from typing import Any, Optional, Tuple

import torch
from torch import nn


class DiscreteDiffusionModel(nn.Module):
    """Template module for integrating discrete diffusion models.

    Replace the embeddings, transition logic, and ``forward`` method with
    the discrete formulation your project requires (e.g., octree tokens).
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        # Example placeholder components; swap these for your own layers.
        self.token_embedding = nn.Identity()
        self.score_head = nn.Identity()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits/score predictions for discrete states.

        Parameters
        ----------
        x_t: torch.Tensor
            Input discrete state at timestep ``t`` (e.g., token ids or
            probability logits).
        t: torch.Tensor
            Current timestep indices.
        mask: Optional[torch.Tensor]
            Optional mask for padding/invalid tokens (shape broadcastable
            to ``x_t``).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A pair ``(pred_x0, score)`` mirroring the interface used by the
            original continuous diffusion ``Model`` so the runner can cache
            both predictions for PFDiff updates.
        """

        del mask  # silence lint on unused argument in the placeholder

        embedded = self.token_embedding(x_t)
        pred_x0 = embedded  # replace with actual denoising prediction
        score = self.score_head(embedded)
        return pred_x0, score
