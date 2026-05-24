"""Build train/valid/test recommendation interactions from synthetic queries."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from medmigcr.config import PipelineConfig


@dataclass(frozen=True)
class InteractionSplits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def _symptom_set_from_row(symptom_entity_ids: str) -> Set[str]:
    if not isinstance(symptom_entity_ids, str) or not symptom_entity_ids:
        return set()
    return set([s for s in symptom_entity_ids.split(";") if s])


def build_query_symptoms(query_df: pd.DataFrame) -> Dict[str, Set[str]]:
    return {
        str(r.query_id): _symptom_set_from_row(r.symptom_entity_ids)
        for r in query_df.itertuples(index=False)
    }


def _build_inverted_index(dis_to_phen: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """phenotype -> diseases that have it."""
    inv: Dict[str, Set[str]] = {}
    for d, phen in dis_to_phen.items():
        for p in phen:
            inv.setdefault(p, set()).add(d)
    return inv


def _hard_negative_candidates(
    symptoms: Set[str],
    positives: Set[str],
    phen_to_dis: Dict[str, Set[str]],
) -> List[str]:
    cand: Set[str] = set()
    for s in symptoms:
        cand |= phen_to_dis.get(s, set())
    cand -= positives
    return list(cand)


def build_interactions(
    query_df: pd.DataFrame,
    query_positives: Dict[str, Set[str]],
    dis_to_phen: Dict[str, Set[str]],
    cfg: PipelineConfig,
    rng: random.Random,
) -> InteractionSplits:
    """
    Output interactions DataFrames with columns: query_id, disease_id, label
    disease_id is the *entity_key* for disease nodes (same key space as KG).
    """
    query_symptoms = build_query_symptoms(query_df)
    all_diseases = sorted(dis_to_phen.keys())
    phen_to_dis = _build_inverted_index(dis_to_phen)

    rows: List[Tuple[str, str, int]] = []
    for qid in query_df["query_id"].astype(str).tolist():
        positives = set(query_positives.get(qid, set()))
        if not positives:
            continue
        for d in positives:
            rows.append((qid, d, 1))

        symptoms = query_symptoms.get(qid, set())
        hard_cand = _hard_negative_candidates(symptoms, positives, phen_to_dis)
        rng.shuffle(hard_cand)
        hard = hard_cand[: cfg.num_hard_negatives]

        # random negatives from the whole disease universe excluding positives/hard
        forbidden = positives | set(hard)
        rand_neg: List[str] = []
        if cfg.num_random_negatives > 0:
            # sample with retries (disease universe is large)
            for _ in range(cfg.num_random_negatives * 10):
                if len(rand_neg) >= cfg.num_random_negatives:
                    break
                d = rng.choice(all_diseases)
                if d in forbidden:
                    continue
                rand_neg.append(d)
                forbidden.add(d)

        for d in hard:
            rows.append((qid, d, 0))
        for d in rand_neg:
            rows.append((qid, d, 0))

    df = pd.DataFrame(rows, columns=["query_id", "disease_id", "label"])

    # Split by query_id to avoid leakage
    qids = df["query_id"].drop_duplicates().tolist()
    rng.shuffle(qids)
    n = len(qids)
    n_train = int(n * cfg.train_ratio)
    n_valid = int(n * cfg.valid_ratio)
    train_q = set(qids[:n_train])
    valid_q = set(qids[n_train : n_train + n_valid])
    test_q = set(qids[n_train + n_valid :])

    train = df[df["query_id"].isin(train_q)].reset_index(drop=True)
    valid = df[df["query_id"].isin(valid_q)].reset_index(drop=True)
    test = df[df["query_id"].isin(test_q)].reset_index(drop=True)

    return InteractionSplits(train=train, valid=valid, test=test)


def interaction_matrix(
    interactions: pd.DataFrame,
    query_ids: Sequence[str],
    disease_ids: Sequence[str],
) -> "scipy.sparse.csr_matrix":
    """Binary matrix (|Q| x |D|) from label==1 interactions."""
    from scipy import sparse

    q2i = {q: i for i, q in enumerate(query_ids)}
    d2i = {d: i for i, d in enumerate(disease_ids)}
    pos = interactions[interactions["label"] == 1]
    rows = pos["query_id"].map(q2i).to_numpy()
    cols = pos["disease_id"].map(d2i).to_numpy()
    data = np.ones(len(pos), dtype=np.int8)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(query_ids), len(disease_ids)))

