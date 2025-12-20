"""Octree token diffusion backbone.

The module mirrors the continuous diffusion model interface so the DDIM
runner can reuse the same training/sampling loops while operating on
BFS-ordered octree tokens.
"""

from typing import Any, Optional, Tuple

import math
import torch
from torch import nn


def _timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings."""
    if timesteps.dim() != 1:
        timesteps = timesteps.view(-1)
    half_dim = embedding_dim // 2
    freq = math.log(10000.0) / max(half_dim - 1, 1)
    exp_term = torch.exp(torch.arange(half_dim, device=timesteps.device) * -freq)
    emb = timesteps.float().unsqueeze(1) * exp_term.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def _positional_embedding(num_tokens: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(num_tokens, device=device, dtype=dtype).unsqueeze(1)
    half_dim = dim // 2
    freq = math.log(10000.0) / max(half_dim - 1, 1)
    exp_term = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -freq)
    emb = positions * exp_term.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class OctreeTokenEmbedding(nn.Module):
    """Embed octree occupancy bits plus token attributes."""

    def __init__(self, embed_dim: int, num_attributes: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.occupancy_proj = nn.Linear(8, embed_dim)
        self.attribute_embedding = nn.Embedding(num_attributes, embed_dim, padding_idx=num_attributes - 1)

    def forward(self, token_ids: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        token_ids = token_ids.long()
        bits = ((token_ids.unsqueeze(-1) >> torch.arange(8, device=token_ids.device)) & 1).float()
        occupancy = self.occupancy_proj(bits)

        attribute_ids = torch.where(token_ids <= 1, token_ids, torch.full_like(token_ids, 2))
        if mask is not None:
            attribute_ids = torch.where(mask, attribute_ids, torch.full_like(attribute_ids, 3))
        attributes = self.attribute_embedding(attribute_ids)
        return occupancy + attributes


class DiscreteDiffusionModel(nn.Module):
    """Octree token diffusion model with transformer backbone."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

        model_cfg = getattr(config, "model", object())
        self.vocab_size = int(getattr(model_cfg, "vocab_size", 256))
        self.hidden_dim = int(getattr(model_cfg, "hidden_dim", 256))
        self.num_layers = int(getattr(model_cfg, "num_layers", 6))
        self.num_heads = int(getattr(model_cfg, "num_heads", 8))
        self.dropout = float(getattr(model_cfg, "dropout", 0.1))

        self.token_embedding = OctreeTokenEmbedding(self.hidden_dim, num_attributes=4)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.pred_x0_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.vocab_size),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.vocab_size),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits/score predictions for octree tokens."""

        if x_t.dim() == 3:
            token_ids = x_t.argmax(dim=-1)
        else:
            token_ids = x_t

        if mask is not None:
            mask = mask.bool()

        token_emb = self.token_embedding(token_ids, mask=mask)
        temb = self.timestep_mlp(_timestep_embedding(t, self.hidden_dim))
        token_emb = token_emb + temb.unsqueeze(1)

        pos_emb = _positional_embedding(token_emb.size(1), self.hidden_dim, token_emb.device, token_emb.dtype)
        token_emb = token_emb + pos_emb.unsqueeze(0)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        hidden = self.backbone(token_emb, src_key_padding_mask=key_padding_mask)
        pred_x0 = self.pred_x0_head(hidden)
        score = self.score_head(hidden)

        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(pred_x0.dtype)
            pred_x0 = pred_x0 * mask_f
            score = score * mask_f

        return pred_x0, score
