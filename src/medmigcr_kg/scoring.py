from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def cosine_similarity(
    embeddings,
    interest,
    eps: float = 1e-8,
):
    if torch is not None and hasattr(embeddings, "dtype") and hasattr(interest, "dtype") and isinstance(embeddings, torch.Tensor):
        interest_vec = interest
        if interest_vec.dim() == 1:
            interest_vec = interest_vec.unsqueeze(0)
        emb_norm = embeddings.norm(dim=1, keepdim=True).clamp(min=eps)
        interest_norm = interest_vec.norm(dim=1, keepdim=True).clamp(min=eps)
        normalized_emb = embeddings / emb_norm
        normalized_interest = interest_vec / interest_norm
        return torch.matmul(normalized_emb, normalized_interest.t()).squeeze(1)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    interest = np.asarray(interest, dtype=np.float32)
    if interest.ndim == 1:
        interest = interest.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=eps)
    interest_norm = np.linalg.norm(interest, axis=1, keepdims=True).clip(min=eps)
    normalized_emb = embeddings / norms
    normalized_interest = interest / interest_norm
    return np.dot(normalized_emb, normalized_interest.T).reshape(-1)


def score_neighbors(
    neighbor_embeddings,
    interest_vector,
    degrees,
    alpha: float = 1.0,
    beta: float = 0.4,
):
    if torch is not None and isinstance(neighbor_embeddings, torch.Tensor):
        sim = cosine_similarity(neighbor_embeddings, interest_vector)
        if not isinstance(degrees, torch.Tensor):
            degrees = torch.from_numpy(np.asarray(degrees, dtype=np.float32)).to(neighbor_embeddings.device)
        penalty = beta * torch.log(degrees.float().clamp(min=0) + 1.0)
        return alpha * sim - penalty

    neighbor_embeddings = np.asarray(neighbor_embeddings, dtype=np.float32)
    interest_vector = np.asarray(interest_vector, dtype=np.float32)
    degrees = np.asarray(degrees, dtype=np.float32)
    sim = cosine_similarity(neighbor_embeddings, interest_vector)
    penalty = beta * np.log(degrees + 1.0)
    return alpha * sim - penalty
