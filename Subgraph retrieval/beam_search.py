from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

import scoring
from graph_store import GraphStore


@dataclass(frozen=True)
class BeamItem:
    current_node: int
    path: Tuple[int, ...]
    score: float


class SemanticBeamSearch:
    def __init__(
        self,
        graph_store: GraphStore,
        interest_vector,
        alpha: float = 1.0,
        beta: float = 0.4,
        projection: Optional[dict] = None,
    ):
        self.graph_store = graph_store
        self.alpha = alpha
        self.beta = beta
        self.projection = self._prepare_projection(projection)
        self.interest_vector = self._prepare_interest_vector(interest_vector)

    def _prepare_projection(self, projection: Optional[dict]) -> Optional[dict]:
        if projection is None:
            return None
        
        weight = projection.get("weight")
        bias = projection.get("bias")
        
        if self.graph_store.use_torch:
            import torch
            if isinstance(weight, np.ndarray):
                weight = torch.from_numpy(weight).float().to(self.graph_store.device)
            if isinstance(bias, np.ndarray):
                bias = torch.from_numpy(bias).float().to(self.graph_store.device)
        else:
            weight = np.asarray(weight, dtype=np.float32)
            bias = np.asarray(bias, dtype=np.float32)
            
        return {"weight": weight, "bias": bias}

    def _prepare_interest_vector(self, interest_vector) -> object:
        if self.graph_store.use_torch:
            import torch
            if isinstance(interest_vector, np.ndarray):
                interest_vector = torch.from_numpy(interest_vector).float().to(self.graph_store.device)
            else:
                interest_vector = interest_vector.to(self.graph_store.device)
            
            if self.projection:
                # Apply linear projection: v_aligned = interest_vector @ weight + bias
                # interest_vector shape: (D,) or (1, D)
                # weight shape: (D, D)
                # bias shape: (D,)
                interest_vector = torch.matmul(interest_vector, self.projection["weight"]) + self.projection["bias"]
            
            return interest_vector
        
        interest_vector = np.asarray(interest_vector, dtype=np.float32)
        if self.projection:
            interest_vector = np.dot(interest_vector, self.projection["weight"]) + self.projection["bias"]
            
        return interest_vector

    def search(
        self,
        seed_node_ids: Sequence[int],
        max_hops: int = 5,
        beam_width: int = 32,
        topk_paths: int = 100,
        avoid_cycles: bool = True,
    ) -> List[BeamItem]:
        beam: List[BeamItem] = [BeamItem(int(node_id), (int(node_id),), 0.0) for node_id in seed_node_ids]
        visited_paths: Set[Tuple[int, ...]] = set(item.path for item in beam)
        global_paths: List[BeamItem] = []

        for _hop in range(max_hops):
            candidates: List[BeamItem] = []
            for item in beam:
                out_neighbors = self.graph_store.get_neighbors(item.current_node, direction="out")
                in_neighbors = self.graph_store.get_neighbors(item.current_node, direction="in")
                neighbor_ids = np.unique(np.concatenate([out_neighbors, in_neighbors]))
                if neighbor_ids.size == 0:
                    continue
                embeddings = self.graph_store.get_embeddings(neighbor_ids, as_tensor=self.graph_store.use_torch)
                degree_values = self.graph_store.out_degree[neighbor_ids]
                scores = scoring.score_neighbors(embeddings, self.interest_vector, degree_values, self.alpha, self.beta)
                if self.graph_store.use_torch:
                    scores = scores.detach().cpu().numpy()
                scores = np.asarray(scores, dtype=np.float32)

                for neighbor_id, expansion_score in zip(neighbor_ids, scores):
                    if avoid_cycles and neighbor_id in item.path:
                        continue
                    path = item.path + (int(neighbor_id),)
                    if path in visited_paths:
                        continue
                    visited_paths.add(path)
                    cumulative_score = item.score + float(expansion_score)
                    candidates.append(BeamItem(int(neighbor_id), path, cumulative_score))

            if not candidates:
                break

            candidates.sort(key=lambda item: item.score, reverse=True)
            beam = candidates[:beam_width]
            global_paths.extend(beam)

        global_paths.sort(key=lambda item: item.score, reverse=True)
        if topk_paths is not None:
            return global_paths[:topk_paths]
        return global_paths
