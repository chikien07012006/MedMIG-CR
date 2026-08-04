from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ClinicalMIND


@dataclass
class ContrastiveLossOutput:
    total: torch.Tensor
    target: torch.Tensor
    evidence: torch.Tensor
    diversity: torch.Tensor


class ProjectedClinicalMIND(nn.Module):
    """ClinicalMIND plus a trainable head into PrimeKG graph embedding space."""

    def __init__(
        self,
        *,
        num_symptoms: int,
        mind_dim: int,
        graph_dim: int,
        num_interests: int,
        max_seq_len: int,
        num_routing_iters: int,
        symptom_padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.mind = ClinicalMIND(
            num_symptoms=num_symptoms,
            num_diseases=1,
            dim=mind_dim,
            num_interests=num_interests,
            max_seq_len=max_seq_len,
            num_routing_iters=num_routing_iters,
            symptom_padding_idx=symptom_padding_idx,
            disease_padding_idx=0,
        )
        self.projection = nn.Linear(mind_dim, graph_dim)

    def forward(self, symptom_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, _, active_mask = self.mind.encode_symptoms(symptom_ids)
        projected = F.normalize(self.projection(latent), dim=-1)
        return projected, latent, active_mask


def capsule_node_scores(
    projected_interests: torch.Tensor,
    node_embeddings: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Score query interests against candidate graph nodes with logsumexp over K."""
    node_embeddings = F.normalize(node_embeddings, dim=-1)
    per_capsule = torch.einsum("bkd,nd->bkn", projected_interests, node_embeddings) / temperature
    return torch.logsumexp(per_capsule, dim=1)


def multi_positive_nce(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Multi-positive InfoNCE over a fixed candidate universe."""
    has_positive = positive_mask.any(dim=1)
    if not torch.any(has_positive):
        return scores.new_tensor(0.0)
    scores = scores[has_positive]
    positive_mask = positive_mask[has_positive]
    neg_large = torch.finfo(scores.dtype).min / 4
    numerator = torch.logsumexp(torch.where(positive_mask, scores, neg_large), dim=1)
    denominator = torch.logsumexp(scores, dim=1)
    return -(numerator - denominator).mean()


def diversity_loss(projected_interests: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Penalize duplicate interest directions without forcing opposite directions."""
    batch_size, num_interests, _ = projected_interests.shape
    if num_interests <= 1:
        return projected_interests.new_tensor(0.0)
    z = F.normalize(projected_interests, dim=-1)
    sim = torch.matmul(z, z.transpose(1, 2)).pow(2)
    eye = torch.eye(num_interests, dtype=torch.bool, device=z.device).unsqueeze(0)
    pair_mask = ~eye.expand(batch_size, -1, -1)
    if active_mask is not None:
        pair_mask = pair_mask & active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
    denom = pair_mask.sum().clamp(min=1)
    return sim.masked_select(pair_mask).sum() / denom


def contrastive_loss(
    *,
    projected_interests: torch.Tensor,
    target_embeddings: torch.Tensor,
    evidence_embeddings: torch.Tensor,
    target_positive_mask: torch.Tensor,
    evidence_positive_mask: torch.Tensor,
    active_interest_mask: torch.Tensor,
    temperature: float,
    evidence_weight: float,
    diversity_weight: float,
) -> ContrastiveLossOutput:
    target_scores = capsule_node_scores(projected_interests, target_embeddings, temperature=temperature)
    target = multi_positive_nce(target_scores, target_positive_mask)

    evidence_scores = capsule_node_scores(projected_interests, evidence_embeddings, temperature=temperature)
    evidence = multi_positive_nce(evidence_scores, evidence_positive_mask)

    diversity = diversity_loss(projected_interests, active_interest_mask)
    total = target + evidence_weight * evidence + diversity_weight * diversity
    return ContrastiveLossOutput(total=total, target=target, evidence=evidence, diversity=diversity)


def average_pairwise_cosine(
    projected_interests: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> Tuple[float, Dict[str, float]]:
    num_interests = projected_interests.shape[1]
    if num_interests <= 1:
        return 0.0, {}
    z = F.normalize(projected_interests.detach(), dim=-1)
    sim = torch.matmul(z, z.transpose(1, 2))
    values = []
    per_pair: Dict[str, float] = {}
    for i in range(num_interests):
        for j in range(i + 1, num_interests):
            pair_values = sim[:, i, j]
            if active_mask is not None:
                mask = active_mask[:, i] & active_mask[:, j]
                pair_values = pair_values[mask]
            if pair_values.numel() == 0:
                continue
            pair_mean = float(pair_values.mean().cpu().item())
            per_pair[f"{i}-{j}"] = pair_mean
            values.append(pair_mean)
    mean_value = float(sum(values) / len(values)) if values else 0.0
    return mean_value, per_pair
