from __future__ import annotations

import torch
from torch import nn


class DirectMLP(nn.Module):
    """Pairwise disease ranker over pooled seed and PrimeKG disease embeddings."""

    def __init__(
        self,
        query_dim: int = 128,
        disease_dim: int = 64,
        hidden_dim: int = 256,
        bottleneck_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.query_projection = nn.Linear(query_dim, disease_dim)
        pair_dim = disease_dim * 4
        self.scorer = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )

    def pair_features(self, query_emb: torch.Tensor, disease_emb: torch.Tensor) -> torch.Tensor:
        query_projected = self.query_projection(query_emb)
        query_projected, disease_emb = torch.broadcast_tensors(query_projected, disease_emb)
        return torch.cat(
            (
                query_projected,
                disease_emb,
                torch.abs(query_projected - disease_emb),
                query_projected * disease_emb,
            ),
            dim=-1,
        )

    def forward(self, query_emb: torch.Tensor, disease_emb: torch.Tensor) -> torch.Tensor:
        return self.scorer(self.pair_features(query_emb, disease_emb)).squeeze(-1)
