"""
Clinical MIND: multi-interest encoding from symptom sequences (MedMIG-CR).

Maps a clinical query (sequence of symptom entity IDs) to K latent interest vectors
Z ∈ R^{B × K × D} via capsule dynamic routing (B2I), adapted from the original MIND
item-sequence encoder. No label-aware attention — training scores use max-pooling
over interests vs disease embeddings (see train_mind_medmigcr.py).

Designed as a retrieval anchor for later PrimeKG / LogosKG integration.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(caps: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Squash capsule vectors to bounded magnitude while preserving direction.

    Original MIND: ||v|| in (0, 1) after squash. Semantically: each interest vector
    lives on a "probability simplex" of direction with bounded activation magnitude,
    stabilizing routing updates.

    caps: (B, K, D)
    """
    n = torch.norm(caps, p=2, dim=-1, keepdim=True)  # (B, K, 1)
    n_sq = n * n
    scale = n_sq / ((1.0 + n_sq) * n.clamp_min(eps))
    return scale * caps


def make_active_interest_mask(padding_mask: torch.Tensor, max_interests: int) -> torch.Tensor:
    """Compute active interest indices per query based on valid symptoms."""
    num_valid = padding_mask.sum(dim=1).clamp(min=1)
    kq = torch.clamp(torch.floor(torch.log2(num_valid.float())), min=1, max=max_interests).long()
    arange = torch.arange(max_interests, device=padding_mask.device).unsqueeze(0)
    return arange < kq.unsqueeze(1)


def average_active_interest_cosine_similarity(interests: torch.Tensor, active_interest_mask: torch.Tensor) -> torch.Tensor:
    """Average cosine similarity across active interests only."""
    z = F.normalize(interests, dim=-1)
    sim = torch.matmul(z, z.transpose(1, 2))
    active_pair = active_interest_mask.unsqueeze(2) & active_interest_mask.unsqueeze(1)
    eye = torch.eye(interests.size(1), device=interests.device, dtype=torch.bool).unsqueeze(0)
    active_pair = active_pair & ~eye
    valid_pairs = active_pair.sum(dim=(1, 2)).clamp(min=1)
    sim = sim * active_pair.to(sim.dtype)
    return (sim.sum(dim=(1, 2)) / valid_pairs).mean()


class DynamicRoutingB2I(nn.Module):
    """
    Behavior-to-Interest (B2I) dynamic routing over a *symptom* sequence.

    Notation (per MIND paper):
    - L: sequence length (symptoms), K: number of interest capsules, D: dimension.
    - b_{jk}: routing logit from symptom slot j toward interest capsule k (here stored
      as B[k, j] then softmax over j for each k).
    - After softmax, W[b, k, j] is how much interest k "listens" to symptom j.
    - Agreement update: add inner product <squashed s_k, transformed symptom_j>
      to routing logits (coupling stronger when capsule agrees with the symptom embedding).

    Biomedical interpretation:
    - Each of K capsules can specialize to a latent hypothesis axis (e.g. inflammatory
      vs metabolic overlap is learned from data, not hand-labeled).
    - Routing distributes evidence across HPO symptoms into competing explanations.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_interests: int,
        max_seq_len: int,
        num_routing_iters: int,
    ) -> None:
        super().__init__()
        self.D = dim
        self.K = num_interests
        self.L = max_seq_len
        self.R = num_routing_iters

        # Shared linear transform on symptom embeddings before routing (same role as S in MIND).
        self.S = nn.Parameter(torch.empty(dim, dim))
        nn.init.normal_(self.S, mean=0.0, std=0.02 * (2.0 / dim) ** 0.5)

        # Initial routing logits (K, L); broadcast per batch then masked softmax over L.
        self.routing_logits_init = nn.Parameter(torch.empty(num_interests, max_seq_len))
        nn.init.normal_(self.routing_logits_init, mean=0.0, std=0.1)

        # Post-routing refinement (same structure as original MIND MLP on capsules).
        hidden = max(dim * 4, dim)
        self.capsule_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        symptom_emb: torch.Tensor,
        padding_mask: torch.Tensor,
        active_interest_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        symptom_emb: (B, L, D) — already embedded symptoms
        padding_mask: (B, L) bool, True = valid token, False = pad
        active_interest_mask: (B, K) bool, True for interests that should route.
                              Only used for tracking; routing happens on all K interests.

        Returns:
            Z: (B, K, D) multi-interest vectors (after squash + MLP).
        """
        B, L, D = symptom_emb.shape
        if L != self.L:
            raise ValueError(f"Expected sequence length L={self.L}, got {L}")

        # Transform symptoms into routing space.
        h = torch.matmul(symptom_emb, self.S)  # (B, L, D)

        # Start from learnable logits (K, L) → (B, K, L)
        b_logits = self.routing_logits_init.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Mask padded positions before softmax over L (dim=2): pad → large negative.
        mask_kl = padding_mask.unsqueeze(1).expand(B, self.K, L)  # (B, K, L)
        neg_large = torch.finfo(b_logits.dtype).min / 4

        for r in range(self.R):
            b_masked = torch.where(mask_kl, b_logits, torch.full_like(b_logits, neg_large))
            w = F.softmax(b_masked, dim=2)  # (B, K, L), distribution over symptoms per interest

            caps = torch.matmul(w, h)  # (B, K, D)
            caps = squash(caps)

            if r < self.R - 1:
                # Agreement: for each (k, j), add <caps_k, h_j> to update routing toward agreeing symptoms.
                agreement = torch.matmul(caps, h.transpose(1, 2))  # (B, K, L)
                b_logits = b_logits + agreement

        z = self.capsule_mlp(caps)
        return z


class ClinicalMIND(nn.Module):
    """
    Symptom sequence → K clinical interest vectors.

    - Symptom tokens: integer indices (0 = PAD, ignored in routing mask).
    - Disease tokens: separate vocabulary for supervised training (optional forward).
    """

    def __init__(
        self,
        *,
        num_symptoms: int,
        num_diseases: int,
        dim: int,
        num_interests: int,
        max_seq_len: int,
        num_routing_iters: int = 3,
        symptom_padding_idx: int = 0,
        disease_padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.D = dim
        self.K = num_interests
        self.L = max_seq_len
        self.symptom_padding_idx = symptom_padding_idx
        self.disease_padding_idx = disease_padding_idx

        self.symptom_emb = nn.Embedding(num_symptoms, dim, padding_idx=symptom_padding_idx)
        self.disease_emb = nn.Embedding(num_diseases, dim, padding_idx=disease_padding_idx)

        nn.init.normal_(self.symptom_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.disease_emb.weight, mean=0.0, std=0.02)
        if symptom_padding_idx is not None:
            with torch.no_grad():
                self.symptom_emb.weight[symptom_padding_idx].zero_()
        if disease_padding_idx is not None:
            with torch.no_grad():
                self.disease_emb.weight[disease_padding_idx].zero_()

        self.routing = DynamicRoutingB2I(
            dim=dim,
            num_interests=num_interests,
            max_seq_len=max_seq_len,
            num_routing_iters=num_routing_iters,
        )

    def encode_symptoms(
        self, symptom_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        symptom_ids: (B, L) long

        Returns:
            Z: (B, K, D) multi-interest vectors
            padding_mask: (B, L) bool
            active_interest_mask: (B, K) bool
        """
        padding_mask = symptom_ids != self.symptom_padding_idx
        active_interest_mask = make_active_interest_mask(padding_mask, self.K)
        x = self.symptom_emb(symptom_ids)
        z = self.routing(x, padding_mask, active_interest_mask)
        return z, padding_mask, active_interest_mask

    def forward(self, symptom_ids: torch.Tensor) -> torch.Tensor:
        """Return Z (B, K, D) only."""
        z, _, _ = self.encode_symptoms(symptom_ids)
        return z


def max_interest_disease_score(
    interests: torch.Tensor,
    disease_vec: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    No label-aware attention: each disease vector scores against all K interests;
    we take the max (which interest best aligns with this disease under dot product).

    interests: (B, K, D)
    disease_vec: (B, D) or (B, N, D)
    Returns: (B,) or (B, N) scores
    """
    if disease_vec.dim() == 2:
        s = (interests * disease_vec.unsqueeze(1)).sum(dim=-1)  # (B, K)
        return s.max(dim=-1).values / temperature
    if disease_vec.dim() == 3:
        # (B, K, D) vs (B, 1, N, D) -> einsum
        logits = torch.einsum("bkd,bnd->bkn", interests, disease_vec)
        return logits.max(dim=1).values / temperature
    raise ValueError("disease_vec must be (B, D) or (B, N, D)")


def training_bce_loss(
    model: ClinicalMIND,
    symptom_ids: torch.Tensor,
    disease_pos: torch.Tensor,
    disease_neg: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    BCE loss: positive diseases should match at least one interest; negatives should not.

    symptom_ids: (B, L)
    disease_pos: (B,)
    disease_neg: (B, N_neg)
    """
    z, _, _ = model.encode_symptoms(symptom_ids)
    d_pos = model.disease_emb(disease_pos)
    d_neg = model.disease_emb(disease_neg)

    pos_logits = max_interest_disease_score(z, d_pos, temperature=temperature)
    neg_logits = max_interest_disease_score(z, d_neg, temperature=temperature)

    pos_targets = torch.ones_like(pos_logits)
    neg_targets = torch.zeros_like(neg_logits.view(-1))

    loss_pos = F.binary_cross_entropy_with_logits(pos_logits, pos_targets)
    loss_neg = F.binary_cross_entropy_with_logits(neg_logits.view(-1), neg_targets)
    return loss_pos + loss_neg
