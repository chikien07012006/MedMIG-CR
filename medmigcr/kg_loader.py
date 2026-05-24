"""Chunked loading and filtering of PrimeKG triples."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd

from medmigcr.config import PipelineConfig
from medmigcr.entities import entity_key, head_key, tail_key


def iter_kg_chunks(cfg: PipelineConfig, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(cfg.kg_path, chunksize=chunksize, low_memory=False):
        yield chunk


def build_disease_phenotype_map(cfg: PipelineConfig) -> Dict[str, Set[str]]:
    """disease entity_key -> set of phenotype entity_keys from disease_phenotype_positive."""
    dis_to_phen: Dict[str, Set[str]] = defaultdict(set)
    for chunk in iter_kg_chunks(cfg):
        m = chunk[chunk["relation"] == "disease_phenotype_positive"]
        for row in m.itertuples(index=False):
            xt, yt = str(row.x_type), str(row.y_type)
            if xt == "disease" and yt == "effect/phenotype":
                d = entity_key(xt, str(row.x_source), row.x_id)
                p = entity_key(yt, str(row.y_source), row.y_id)
                dis_to_phen[d].add(p)
            elif xt == "effect/phenotype" and yt == "disease":
                d = entity_key(yt, str(row.y_source), row.y_id)
                p = entity_key(xt, str(row.x_source), row.x_id)
                dis_to_phen[d].add(p)
    return {k: v for k, v in dis_to_phen.items() if len(v) >= cfg.min_symptoms_per_query}


def relation_allowed(
    relation: str,
    cfg: PipelineConfig,
    rel_counts_sample: Optional[Dict[str, int]] = None,
) -> bool:
    if relation in cfg.exclude_relations:
        return False
    cap = cfg.max_edges_per_relation.get(relation, 0)
    if cap and rel_counts_sample is not None:
        return rel_counts_sample.get(relation, 0) < cap
    return True


def collect_filtered_triples(
    cfg: PipelineConfig,
    rng: random.Random,
) -> Tuple[List[Tuple[str, str, str]], Set[str], Dict[str, int]]:
    """
    Returns list of (h, r, t) with string entity keys, entity set, and per-relation kept counts.
    Applies exclude_relations and per-relation caps via reservoir-style subsampling when cap set.
    """
    allowed = cfg.allowed_node_types
    triples: List[Tuple[str, str, str]] = []
    entities: Set[str] = set()
    rel_kept: Dict[str, int] = defaultdict(int)
    rel_seen: Dict[str, int] = defaultdict(int)

    # Reservoir per relation when capped
    reservoirs: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    # Faster than iterrows for tens of millions of rows
    for chunk in iter_kg_chunks(cfg):
        m = chunk[chunk["x_type"].isin(allowed) & chunk["y_type"].isin(allowed)]
        for row in m.itertuples(index=False):
            r = str(row.relation)
            if r in cfg.exclude_relations:
                continue
            h = entity_key(str(row.x_type), str(row.x_source), row.x_id)
            t = entity_key(str(row.y_type), str(row.y_source), row.y_id)
            cap = cfg.max_edges_per_relation.get(r, 0)
            rel_seen[r] += 1
            if cap <= 0:
                triples.append((h, r, t))
                entities.add(h)
                entities.add(t)
                rel_kept[r] += 1
                continue
            # reservoir sample of size cap
            res = reservoirs[r]
            if len(res) < cap:
                res.append((h, r, t))
            else:
                j = rng.randint(1, rel_seen[r])
                if j <= cap:
                    res[j - 1] = (h, r, t)

    for r, res in reservoirs.items():
        for h, rr, t in res:
            triples.append((h, rr, t))
            entities.add(h)
            entities.add(t)
            rel_kept[r] += 1

    return triples, entities, dict(rel_kept)
