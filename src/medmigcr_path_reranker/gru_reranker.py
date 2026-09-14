from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GRUPathRerankerConfig:
    num_relations: int
    num_node_types: int
    graph_dim: int
    relation_dim: int = 32
    direction_dim: int = 8
    node_type_dim: int = 8
    node_projection_dim: int = 32
    hidden_dim: int = 64
    dropout: float = 0.1
    use_score_features: bool = False


class GRUPathReranker(nn.Module):
    def __init__(self, config: GRUPathRerankerConfig, node_embeddings: torch.Tensor):
        super().__init__()
        self.config = config
        self.node_embeddings = nn.Embedding.from_pretrained(node_embeddings.float(), freeze=True)
        self.relation_embedding = nn.Embedding(config.num_relations + 1, config.relation_dim, padding_idx=0)
        self.direction_embedding = nn.Embedding(3, config.direction_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(config.num_node_types + 1, config.node_type_dim, padding_idx=0)
        self.node_projection = nn.Linear(config.graph_dim, config.node_projection_dim)

        token_dim = (
            config.relation_dim
            + config.direction_dim
            + config.node_type_dim
            + config.node_projection_dim
        )
        self.input_dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(
            input_size=token_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        extra_dim = 2 if config.use_score_features else 0
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim + extra_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def encode_tokens(
        self,
        relation_ids: torch.Tensor,
        direction_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        dst_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = self.relation_embedding(relation_ids)
        direction_emb = self.direction_embedding(direction_ids)
        node_type_emb = self.node_type_embedding(node_type_ids)
        node_emb = self.node_projection(self.node_embeddings(dst_node_ids))
        tokens = torch.cat([relation_emb, direction_emb, node_type_emb, node_emb], dim=-1)
        return self.input_dropout(tokens)

    def score_hidden(
        self,
        hidden: torch.Tensor,
        lengths: torch.Tensor,
        additive_scores: torch.Tensor | None = None,
        max_len: int = 1,
    ) -> torch.Tensor:
        path_repr = hidden[-1]
        if self.config.use_score_features:
            if additive_scores is None:
                raise ValueError("additive_scores is required when use_score_features=True")
            len_feature = lengths.float().unsqueeze(1) / max(1, max_len)
            score_feature = additive_scores.float().unsqueeze(1)
            path_repr = torch.cat([path_repr, score_feature, len_feature], dim=1)
        return self.head(path_repr).squeeze(1)

    def forward_incremental(
        self,
        relation_ids: torch.Tensor,
        direction_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        dst_node_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        additive_scores: torch.Tensor | None = None,
        max_len: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encode_tokens(
            relation_ids,
            direction_ids,
            node_type_ids,
            dst_node_ids,
        ).unsqueeze(1)
        if hidden is None:
            hidden = torch.zeros(
                (1, tokens.shape[0], self.config.hidden_dim),
                device=tokens.device,
                dtype=tokens.dtype,
            )
        _outputs, new_hidden = self.gru(tokens, hidden)
        if lengths is None:
            lengths = torch.ones(tokens.shape[0], dtype=torch.long, device=tokens.device)
        scores = self.score_hidden(new_hidden, lengths, additive_scores, max_len)
        return scores, new_hidden

    def forward(
        self,
        relation_ids: torch.Tensor,
        direction_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        dst_node_ids: torch.Tensor,
        lengths: torch.Tensor,
        additive_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.encode_tokens(relation_ids, direction_ids, node_type_ids, dst_node_ids)

        lengths_cpu = lengths.detach().cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            tokens,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        _outputs, hidden = self.gru(packed)
        return self.score_hidden(
            hidden,
            lengths,
            additive_scores,
            max_len=relation_ids.shape[1],
        )

