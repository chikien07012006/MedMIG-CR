from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from .graph_store import GraphStore

UNREACHABLE = np.iinfo(np.int32).max


@dataclass(frozen=True)
class TargetDistanceGuide:
    distance: np.ndarray
    pruning: bool = True
    weight: float = 0.0

    def admissible(self, node_ids: np.ndarray, remaining_hops: int) -> np.ndarray:
        if not self.pruning:
            return np.ones(len(node_ids), dtype=bool)
        return self.distance[node_ids] <= remaining_hops

    def penalty(self, node_ids: np.ndarray, max_hops: int) -> np.ndarray:
        if self.weight == 0.0:
            return np.zeros(len(node_ids), dtype=np.float32)
        capped = np.minimum(self.distance[node_ids], max_hops + 1)
        return (self.weight * capped).astype(np.float32)


def undirected_adjacency(graph_store: GraphStore) -> sp.csr_matrix:
    num_nodes = graph_store.num_nodes
    adjacency = sp.csr_matrix(
        (np.ones(len(graph_store.indices), dtype=np.int8), graph_store.indices, graph_store.indptr),
        shape=(num_nodes, num_nodes),
    )
    return adjacency.maximum(adjacency.T).tocsr()


def distance_to_targets(graph_store: GraphStore, target_node_ids: Iterable[int]) -> np.ndarray:
    targets = np.asarray(sorted(set(int(node_id) for node_id in target_node_ids)), dtype=np.int64)
    if targets.size == 0:
        raise ValueError("At least one target node is required")
    hops = shortest_path(undirected_adjacency(graph_store), directed=False, unweighted=True, indices=targets).min(axis=0)
    distance = np.full(graph_store.num_nodes, UNREACHABLE, dtype=np.int32)
    finite = np.isfinite(hops)
    distance[finite] = hops[finite].astype(np.int32)
    return distance


def build_distance_guide(
    graph_store: GraphStore,
    target_node_ids: Iterable[int] | None,
    pruning: bool,
    weight: float,
) -> TargetDistanceGuide | None:
    if not pruning and weight == 0.0:
        return None
    if not target_node_ids:
        raise ValueError("Target-distance guidance requires a target universe (pass --condition_map)")
    return TargetDistanceGuide(distance_to_targets(graph_store, target_node_ids), pruning=pruning, weight=weight)
