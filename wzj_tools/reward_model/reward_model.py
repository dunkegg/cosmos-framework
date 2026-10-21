from __future__ import annotations

import torch
import torch.nn as nn


class MeanPoolRewardModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 48,
        hidden_dim: int = 256,
        drop_t0: bool = True,
    ):
        super().__init__()

        self.drop_t0 = drop_t0

        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,C,T,H,W]

        if self.drop_t0:
            z = z[:, :, 1:]

        # Spatial pooling.
        z = z.mean(dim=(-1, -2))
        # [B,C,T]

        # Temporal pooling.
        z = z.mean(dim=-1)
        # [B,C]

        reward = self.head(z)

        return reward.squeeze(-1)


class TemporalRewardModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 48,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        drop_t0: bool = True,
    ):
        super().__init__()

        self.drop_t0 = drop_t0

        self.input_norm = nn.LayerNorm(latent_dim)

        self.proj = nn.Linear(
            latent_dim,
            hidden_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,C,T,H,W]

        if self.drop_t0:
            z = z[:, :, 1:]
        # [B,48,15,H,W]

        z = z.mean(dim=(-1, -2))
        # [B,48,15]

        z = z.transpose(1, 2)
        # [B,15,48]

        z = self.input_norm(z)
        z = self.proj(z)
        # [B,15,256]

        z = self.temporal_encoder(z)
        # [B,15,256]

        attn = self.attn_pool(z)
        # [B,15,1]

        attn = torch.softmax(
            attn,
            dim=1,
        )

        pooled = (
            z * attn
        ).sum(dim=1)
        # [B,256]

        reward = self.head(pooled)

        return reward.squeeze(-1)


def build_reward_model(
    model_type: str,
    latent_dim: int = 48,
    hidden_dim: int = 256,
):
    if model_type == "mean":
        return MeanPoolRewardModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

    if model_type == "temporal":
        return TemporalRewardModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

    raise ValueError(
        f"Unknown model_type: {model_type}"
    )