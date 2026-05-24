from __future__ import annotations

import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def count_relations_from_triplets(triplets_path: Path) -> Counter:
    c = Counter()
    with open(triplets_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            r = int(parts[1])
            c[r] += 1
    return c


def hop_histogram_sample(
    start_nodes: List[int],
    adjacency_list: List[List[Tuple[int, int]]],
    max_depth: int,
    max_starts: int,
    seed: int,
) -> Dict[int, int]:
    rng = random.Random(seed)
    if len(start_nodes) > max_starts:
        start_nodes = rng.sample(start_nodes, max_starts)
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


def main() -> None:
    out = Path("processed_medmigcr_dataset")
    entity2id = load_pickle(out / "entity2id.pkl")
    relation2id = load_pickle(out / "relation2id.pkl")
    id2rel = {v: k for k, v in relation2id.items()}

    # relation counts (proxy for rel_kept)
    rel_counts_id = count_relations_from_triplets(out / "triplets.txt")
    rel_counts_name = {id2rel[rid]: int(cnt) for rid, cnt in rel_counts_id.items()}
    with open(out / "relation_kept_counts.json", "w", encoding="utf-8") as f:
        json.dump(rel_counts_name, f, indent=2)

    query_df = pd.read_csv(out / "query_nodes.csv")
    interactions_all = pd.concat(
        [
            pd.read_csv(out / "interactions_train.csv"),
            pd.read_csv(out / "interactions_valid.csv"),
            pd.read_csv(out / "interactions_test.csv"),
        ],
        axis=0,
        ignore_index=True,
    )

    # disease universe from interactions files (entity keys)
    disease_keys = sorted(interactions_all["disease_id"].astype(str).unique().tolist())

    # compute interaction sparsity from positives
    num_queries = int(query_df.shape[0])
    num_diseases = int(len(disease_keys))
    num_pos = int((interactions_all["label"] == 1).sum())
    sparsity = 1.0 - (num_pos / float(max(1, num_queries * num_diseases)))

    # query size
    sizes = (
        query_df["symptom_entity_ids"]
        .astype(str)
        .apply(lambda s: 0 if not s else len([x for x in s.split(";") if x]))
        .to_numpy()
    )
    avg_query_size = float(sizes.mean()) if len(sizes) else 0.0

    # avg node degree from edge_index/edge_type (fast)
    edge_index = np.load(out / "edge_index.npy")
    num_entities = int(len(entity2id))
    if edge_index.size:
        num_entities = max(num_entities, int(edge_index.max()) + 1)
    deg = np.zeros((num_entities,), dtype=np.int64)
    if edge_index.size:
        deg += np.bincount(edge_index[0], minlength=num_entities)
        deg += np.bincount(edge_index[1], minlength=num_entities)
    avg_node_degree = float(deg.mean()) if num_entities else 0.0

    # hop histogram on a sample of query nodes (expensive on all queries)
    adjacency_list = load_pickle(out / "adjacency_list.pkl")
    query_entity_ids = [
        entity2id[f"query|synthetic|{qid}"]
        for qid in query_df["query_id"].astype(str).tolist()
        if f"query|synthetic|{qid}" in entity2id
    ]
    hop_dist = hop_histogram_sample(
        start_nodes=query_entity_ids,
        adjacency_list=adjacency_list,
        max_depth=3,
        max_starts=2000,
        seed=42,
    )

    stats = {
        "num_queries": int(num_queries),
        "num_diseases": int(num_diseases),
        "num_entities_total": int(num_entities),
        "num_relations": int(len(relation2id)),
        "num_triplets": int(sum(rel_counts_id.values())),
        "num_interactions": int(len(interactions_all)),
        "interaction_sparsity": float(sparsity),
        "avg_query_size": float(avg_query_size),
        "avg_node_degree": float(avg_node_degree),
        "hop_distribution_sampled": hop_dist,
        "hop_distribution_sample_size": int(min(2000, len(query_entity_ids))),
        "hop_distribution_max_depth": 3,
    }
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Wrote stats.json and relation_kept_counts.json")


if __name__ == "__main__":
    main()

