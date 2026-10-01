from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import scoring
from .graph_store import GraphStore
from .target_distance import TargetDistanceGuide


@dataclass(frozen=True)
class BeamItem:
    current_node: int
    path: Tuple[int, ...]
    score: float
    # The original semantic score is retained for diagnostics and optional GRU inputs.
    additive_score: float = 0.0
    relation_path: Tuple[int, ...] = ()
    direction_path: Tuple[int, ...] = ()
    gru_hidden_state: Any = None


@dataclass(frozen=True)
class BeamScore:
    score: float
    gru_hidden_state: Any = None


class SemanticBeamSearch:
    def __init__(
        self,
        graph_store: GraphStore,
        interest_vector,
        alpha: float = 1.0,
        beta: float = 0.4,
        projection: Optional[dict] = None,
        path_scorer: Optional[Callable[[Sequence[BeamItem]], Sequence[float | BeamScore]]] = None,
        distance_guide: Optional[TargetDistanceGuide] = None,
    ):
        self.graph_store = graph_store
        self.distance_guide = distance_guide
        self.alpha = alpha
        self.beta = beta
        self.projection = self._prepare_projection(projection)
        self.interest_vector = self._prepare_interest_vector(interest_vector)
        self.path_scorer = path_scorer

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
        target_node_ids: Optional[Set[int]] = None,
    ) -> List[BeamItem]:
        beam: List[BeamItem] = [BeamItem(int(node_id), (int(node_id),), 0.0) for node_id in seed_node_ids]
        visited_paths: Set[Tuple[int, ...]] = set(item.path for item in beam)
        global_paths: List[BeamItem] = []
        target_paths: dict[int, BeamItem] = {}

        for hop in range(max_hops):
            remaining_hops = max_hops - hop - 1
            candidates: List[BeamItem] = []
            for item in beam:
                out_neighbors = self.graph_store.get_neighbors(item.current_node, direction="out")
                in_neighbors = self.graph_store.get_neighbors(item.current_node, direction="in")
                neighbor_ids = np.unique(np.concatenate([out_neighbors, in_neighbors]))
                if self.distance_guide is not None:
                    neighbor_ids = neighbor_ids[self.distance_guide.admissible(neighbor_ids, remaining_hops)]
                if neighbor_ids.size == 0:
                    continue
                embeddings = self.graph_store.get_embeddings(neighbor_ids, as_tensor=self.graph_store.use_torch)
                degree_values = self.graph_store.out_degree[neighbor_ids] + self.graph_store.in_degree[neighbor_ids]
                scores = scoring.score_neighbors(embeddings, self.interest_vector, degree_values, self.alpha, self.beta)
                if self.graph_store.use_torch:
                    scores = scores.detach().cpu().numpy()
                scores = np.asarray(scores, dtype=np.float32)
                if self.distance_guide is not None:
                    heuristics = self.distance_guide.penalty(neighbor_ids, max_hops)
                else:
                    heuristics = np.zeros(len(neighbor_ids), dtype=np.float32)

                for neighbor_id, expansion_score, heuristic in zip(neighbor_ids, scores, heuristics):
                    if avoid_cycles and neighbor_id in item.path:
                        continue
                    path = item.path + (int(neighbor_id),)
                    if path in visited_paths:
                        continue
                    visited_paths.add(path)
                    step_rel_id, step_direction = self.graph_store.get_canonical_relation_step(
                        item.current_node,
                        int(neighbor_id),
                    )
                    additive_score = item.additive_score + float(expansion_score)
                    candidate = BeamItem(
                        int(neighbor_id),
                        path,
                        additive_score - float(heuristic),
                        additive_score,
                        item.relation_path + (step_rel_id,),
                        item.direction_path + (step_direction,),
                        item.gru_hidden_state,
                    )
                    candidates.append(candidate)

            if not candidates:
                break

            if self.path_scorer is not None:
                guided_scores = list(self.path_scorer(candidates))
                if len(guided_scores) != len(candidates):
                    raise ValueError("path_scorer must return exactly one score per candidate path.")
                scored_candidates = []
                for item, guided_score in zip(candidates, guided_scores):
                    if isinstance(guided_score, BeamScore):
                        score = guided_score.score
                        gru_hidden_state = guided_score.gru_hidden_state
                    else:
                        score = float(guided_score)
                        gru_hidden_state = item.gru_hidden_state
                    scored_candidates.append(
                        BeamItem(
                            item.current_node,
                            item.path,
                            float(score),
                            item.additive_score,
                            item.relation_path,
                            item.direction_path,
                            gru_hidden_state,
                        )
                    )
                candidates = scored_candidates

            if target_node_ids is not None:
                for candidate in candidates:
                    if candidate.current_node not in target_node_ids:
                        continue
                    existing = target_paths.get(candidate.current_node)
                    if existing is None or candidate.score > existing.score:
                        target_paths[candidate.current_node] = candidate

            candidates.sort(key=lambda item: item.score, reverse=True)
            beam = candidates[:beam_width]
            global_paths.extend(beam)

        global_paths.sort(key=lambda item: item.score, reverse=True)
        if topk_paths is not None:
            global_paths = global_paths[:topk_paths]
        if target_paths:
            by_path = {item.path: item for item in global_paths}
            for item in target_paths.values():
                existing = by_path.get(item.path)
                if existing is None or item.score > existing.score:
                    by_path[item.path] = item
            return sorted(by_path.values(), key=lambda item: item.score, reverse=True)
        return global_paths
