from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .beam_search import BeamItem, SemanticBeamSearch
from .graph_store import GraphStore
from .subgraph_builder import build_subgraph_from_paths


@dataclass
class RetrievalResult:
    paths: List[BeamItem]
    candidate_scores: List[Tuple[str, float]]
    unique_nodes: List[int]
    latency_seconds: float
    path_diversity: float
    hub_ratio: float


class RetrievalEngine:
    def __init__(
        self,
        graph_store: GraphStore,
        interest_vectors: Optional[np.ndarray] = None,
        projection: Optional[dict] = None,
    ):
        self.graph_store = graph_store
        self.interest_vectors = interest_vectors
        self.projection = projection

    @staticmethod
    def load_projection(path: Path) -> dict:
        data = np.load(path)
        return {"weight": data["weight"], "bias": data["bias"]}

    @staticmethod
    def load_interest_vectors(path: Path) -> np.ndarray:
        return np.load(path)

    def derive_interest_vectors_from_seeds(self, seed_ids: Sequence[int], k: int = 1) -> np.ndarray:
        embeddings = self.graph_store.get_embeddings(seed_ids, as_tensor=False)
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        mean_vector = np.mean(embeddings, axis=0)
        if k <= 1 or embeddings.shape[0] < 2:
            return mean_vector.reshape(1, -1)
        centered = embeddings - mean_vector
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0] if vh.shape[0] > 0 else np.zeros_like(mean_vector)
        vectors = [mean_vector + (i - (k - 1) / 2.0) * direction for i in range(k)]
        return np.vstack(vectors)

    def retrieve(
        self,
        seed_node_ids: Sequence[int],
        max_hops: int = 5,
        beam_width: int = 32,
        topk_paths: int = 100,
        alpha: float = 1.0,
        beta: float = 0.4,
        interest_vectors: Optional[np.ndarray] = None,
        interest_count: int = 1,
        max_paths_per_interest: int = 3,
        projection: Optional[dict] = None,
    ) -> RetrievalResult:
        if interest_vectors is None:
            if self.interest_vectors is not None:
                interest_vectors = self.interest_vectors
            else:
                interest_vectors = self.derive_interest_vectors_from_seeds(seed_node_ids, interest_count)

        if projection is None:
            projection = self.projection

        if interest_vectors.ndim == 1:
            interest_vectors = interest_vectors.reshape(1, -1)

        all_paths_per_interest: List[List[BeamItem]] = []
        start_time = time.perf_counter()

        for interest_index, interest_vector in enumerate(interest_vectors):
            search = SemanticBeamSearch(
                self.graph_store,
                interest_vector,
                alpha=alpha,
                beta=beta,
                projection=projection
            )
            best_paths = search.search(
                seed_node_ids,
                max_hops=max_hops,
                beam_width=beam_width,
                topk_paths=max_paths_per_interest
            )
            all_paths_per_interest.append(best_paths)

        latency_seconds = time.perf_counter() - start_time
        unique_paths = self._deduplicate_paths_across_interests(all_paths_per_interest)
        unique_nodes = self._get_unique_nodes(unique_paths)
        candidate_scores = self._rank_candidates(unique_paths)
        path_diversity = self._compute_path_diversity(unique_paths)
        hub_ratio = self._compute_hub_ratio(unique_paths)

        return RetrievalResult(
            paths=unique_paths,
            candidate_scores=candidate_scores,
            unique_nodes=unique_nodes,
            latency_seconds=latency_seconds,
            path_diversity=path_diversity,
            hub_ratio=hub_ratio,
        )

    def _deduplicate_paths_across_interests(self, paths_per_interest: List[List[BeamItem]]) -> List[BeamItem]:
        path_map: Dict[Tuple[int, ...], BeamItem] = {}
        for interest_paths in paths_per_interest:
            for path in interest_paths:
                existing = path_map.get(path.path)
                if existing is None or path.score > existing.score:
                    path_map[path.path] = path
        return sorted(path_map.values(), key=lambda item: item.score, reverse=True)

    def _get_unique_nodes(self, paths: List[BeamItem]) -> List[int]:
        seen = set()
        for item in paths:
            for node_id in item.path:
                seen.add(node_id)
        return sorted(seen)

    def _rank_candidates(self, paths: List[BeamItem], top_k: int = 50) -> List[Tuple[str, float]]:
        endpoint_scores: Dict[int, float] = {}
        for item in paths:
            node_id = item.current_node
            endpoint_scores[node_id] = max(endpoint_scores.get(node_id, float("-inf")), item.score)
        ranked = sorted(endpoint_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [(self.graph_store.lookup_node_name(node_id) or str(node_id), score) for node_id, score in ranked]

    def _compute_path_diversity(self, paths: List[BeamItem]) -> float:
        if not paths:
            return 0.0
        endpoints = [item.current_node for item in paths]
        return len(set(endpoints)) / len(endpoints)

    def _compute_hub_ratio(self, paths: List[BeamItem], threshold_quantile: float = 0.90) -> float:
        degrees = self.graph_store.out_degree
        if len(degrees) == 0:
            return 0.0
        threshold = float(np.quantile(degrees.astype(np.float32), threshold_quantile))
        visited = set()
        hub_count = 0
        for item in paths:
            for node_id in item.path:
                if node_id in visited:
                    continue
                visited.add(node_id)
                if degrees[node_id] >= threshold:
                    hub_count += 1
        return float(hub_count) / max(1, len(visited))

    def save_subgraph(self, result: RetrievalResult, out_path: Path) -> None:
        subgraph = build_subgraph_from_paths(result.paths, self.graph_store)
        records = [
            {
                "node_id": node_id,
                "node_name": self.graph_store.lookup_node_name(node_id),
            }
            for node_id in subgraph["nodes"]
        ]
        with out_path.open("w", encoding="utf-8") as handle:
            import json

            json.dump(
                {
                    "nodes": records,
                    "edges": [
                        {
                            "source": self.graph_store.lookup_node_name(src),
                            "target": self.graph_store.lookup_node_name(dst),
                            "relations": self.graph_store.get_edge_relations(src, dst),
                        }
                        for src, dst in sorted(subgraph["edges"])
                    ],
                    "paths": [
                        {
                            "score": item.score,
                            "path": [self.graph_store.lookup_node_name(node_id) for node_id in item.path],
                        }
                        for item in result.paths
                    ],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def load_query_seeds(query_id: str, query_csv_path: Path) -> List[str]:
        with query_csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("query_id") == query_id:
                    symptom_ids = row.get("symptom_entity_ids", "")
                    return [token.strip() for token in symptom_ids.split(";") if token.strip()]
        return []
