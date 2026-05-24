"""Graph + interaction statistics for sanity checks and reporting."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetStats:
    num_queries: int
    num_diseases: int
    num_entities_total: int
    num_relations: int
    num_triplets: int
    num_interactions: int
    sparsity: float
    avg_query_size: float
    avg_node_degree: float
    hop_distribution: Dict[int, int]


def avg_query_size(query_df: pd.DataFrame) -> float:
    if query_df.empty:
        return 0.0
    sizes = query_df["symptom_entity_ids"].astype(str).apply(lambda s: 0 if not s else len([x for x in s.split(";") if x]))
    return float(sizes.mean())


def avg_degree(num_entities: int, id_triplets: Sequence[Tuple[int, int, int]]) -> float:
    if num_entities == 0:
        return 0.0
    deg = np.zeros((num_entities,), dtype=np.int64)
    for h, _, t in id_triplets:
        deg[h] += 1
        deg[t] += 1
    return float(deg.mean())


def hop_histogram(
    start_nodes: Sequence[int],
    adjacency_list: List[List[Tuple[int, int]]],
    max_depth: int = 3,
) -> Dict[int, int]:
    """
    For each start node, BFS up to max_depth and count how many nodes are reached at each hop.
    Returns aggregated histogram hop->count.
    """
    hist = Counter()
    for s in start_nodes:
        seen = {s}
        frontier = {s}
        for d in range(1, max_depth + 1):
            nxt = set()
            for u in frontier:
                for _, v in adjacency_list[u]:
                    if v not in seen:
                        nxt.add(v)
            hist[d] += len(nxt)
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
    return dict(hist)


def compute_stats(
    query_df: pd.DataFrame,
    disease_entity_keys: Sequence[str],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    id_triplets: Sequence[Tuple[int, int, int]],
    interactions_all: pd.DataFrame,
    interaction_matrix_shape: Tuple[int, int],
    adjacency_list: List[List[Tuple[int, int]]],
    query_entity_ids: Sequence[int],
    max_hop_stat_depth: int = 3,
) -> DatasetStats:
    qn, dn = interaction_matrix_shape
    num_pos = int((interactions_all["label"] == 1).sum())
    sparsity = 1.0 - (num_pos / float(max(1, qn * dn)))
    hop_dist = hop_histogram(query_entity_ids, adjacency_list, max_depth=max_hop_stat_depth)

    return DatasetStats(
        num_queries=int(len(query_df)),
        num_diseases=int(len(disease_entity_keys)),
        num_entities_total=int(len(entity2id)),
        num_relations=int(len(relation2id)),
        num_triplets=int(len(id_triplets)),
        num_interactions=int(len(interactions_all)),
        sparsity=float(sparsity),
        avg_query_size=float(avg_query_size(query_df)),
        avg_node_degree=float(avg_degree(len(entity2id), id_triplets)),
        hop_distribution=hop_dist,
    )

