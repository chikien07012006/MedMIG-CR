"""Export recommendation-compatible KG artifacts (KGAT / RippleNet / KGCN style)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class IdMaps:
    entity2id: Dict[str, int]
    relation2id: Dict[str, int]


def build_id_maps(entities: Iterable[str], relations: Iterable[str]) -> IdMaps:
    entity2id = {e: i for i, e in enumerate(sorted(set(entities)))}
    relation2id = {r: i for i, r in enumerate(sorted(set(relations)))}
    return IdMaps(entity2id=entity2id, relation2id=relation2id)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_triplets_txt(triples: Sequence[Tuple[int, int, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def to_id_triplets(
    triples: Sequence[Tuple[str, str, str]],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for h, r, t in triples:
        out.append((entity2id[h], relation2id[r], entity2id[t]))
    return out


def build_adjacency_list(
    id_triplets: Sequence[Tuple[int, int, int]],
    num_entities: int,
) -> List[List[Tuple[int, int]]]:
    """Adj list per head: list of (relation_id, tail_id)."""
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_entities)]
    for h, r, t in id_triplets:
        adj[h].append((r, t))
    return adj


def build_neighbor_sampler(
    adjacency_list: List[List[Tuple[int, int]]],
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """
    Minimal neighbor sampler cache: entity_id -> ndarray of shape (deg, 2) [relation_id, tail_id]
    KGCN-style samplers can subsample per batch from this.
    """
    rng = np.random.default_rng(seed)
    sampler: Dict[int, np.ndarray] = {}
    for eid, neigh in enumerate(adjacency_list):
        if not neigh:
            sampler[eid] = np.zeros((0, 2), dtype=np.int64)
        else:
            arr = np.asarray(neigh, dtype=np.int64)
            # shuffle once for deterministic stochasticity across epochs if you roll indices
            rng.shuffle(arr, axis=0)
            sampler[eid] = arr
    return sampler


def pyg_edge_index(id_triplets: Sequence[Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (edge_index [2,E], edge_type [E]) for PyTorch Geometric."""
    if not id_triplets:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    h = np.fromiter((x[0] for x in id_triplets), dtype=np.int64, count=len(id_triplets))
    t = np.fromiter((x[2] for x in id_triplets), dtype=np.int64, count=len(id_triplets))
    r = np.fromiter((x[1] for x in id_triplets), dtype=np.int64, count=len(id_triplets))
    edge_index = np.stack([h, t], axis=0)
    return edge_index, r


def multihop_neighbors(
    start_nodes: Sequence[int],
    adjacency_list: List[List[Tuple[int, int]]],
    depth: int,
) -> Dict[int, List[Set[int]]]:
    """
    RippleNet-style multi-hop cache.
    Returns: start_node -> list of hop sets (hop1 entities, hop2 entities, ...)
    """
    cache: Dict[int, List[Set[int]]] = {}
    for s in start_nodes:
        hops: List[Set[int]] = []
        frontier: Set[int] = {s}
        visited: Set[int] = {s}
        for _ in range(depth):
            nxt: Set[int] = set()
            for u in frontier:
                for _, v in adjacency_list[u]:
                    if v not in visited:
                        nxt.add(v)
            hops.append(nxt)
            visited |= nxt
            frontier = nxt
        cache[s] = hops
    return cache

