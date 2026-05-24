"""Synthetic patient queries from disease–phenotype associations."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from medmigcr.config import PipelineConfig
from medmigcr.entities import entity_key


def _phen_to_disease_index(dis_to_phen: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    idx: Dict[str, Set[str]] = {}
    for d, phen in dis_to_phen.items():
        for p in phen:
            idx.setdefault(p, set()).add(d)
    return idx


def _true_positive_diseases(symptoms: Set[str], phen_to_dis: Dict[str, Set[str]]) -> Set[str]:
    """
    Diseases that contain ALL symptoms (set containment), computed via set intersections
    instead of scanning all diseases per query.
    """
    if not symptoms:
        return set()
    it = iter(symptoms)
    first = next(it)
    cand = set(phen_to_dis.get(first, set()))
    for s in it:
        cand &= phen_to_dis.get(s, set())
        if not cand:
            break
    return cand


def build_queries(
    dis_to_phen: Dict[str, Set[str]],
    cfg: PipelineConfig,
    rng: random.Random,
    phen_names: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """
    Returns:
      query_df with columns query_id, symptom_entity_ids, query_text
      query_id -> set of positive disease entity keys
      phenotype_name lookup optional — we embed names in query_text via dis_to_phen only keys;
      for names we need id->name map from KG pass — passed in as phen_names dict.
    """
    diseases = [d for d, sy in dis_to_phen.items() if len(sy) >= cfg.min_symptoms_per_query]
    rng.shuffle(diseases)

    if phen_names is None:
        phen_names = {}

    phen_to_dis = _phen_to_disease_index(dis_to_phen)

    rows: List[dict] = []
    query_positives: Dict[str, Set[str]] = {}

    q_counter = 0
    for d in diseases:
        syms = list(dis_to_phen[d])
        n_gen = cfg.queries_per_disease
        for _ in range(n_gen):
            if cfg.max_queries_total is not None and q_counter >= cfg.max_queries_total:
                break
            k = rng.randint(cfg.min_symptoms_per_query, min(cfg.max_symptoms_per_query, len(syms)))
            chosen = rng.sample(syms, k)
            sym_set = set(chosen)

            # Optional: add coherent second disease by sharing 1 symptom (multi-disease queries)
            if rng.random() < 0.25 and len(diseases) > 1:
                other = rng.choice(diseases)
                if other != d:
                    inter = dis_to_phen[d] & dis_to_phen[other]
                    if inter:
                        extra = rng.choice(list(inter))
                        if extra not in sym_set and len(sym_set) < cfg.max_symptoms_per_query:
                            sym_set.add(extra)
                            chosen = list(sym_set)

            positives = _true_positive_diseases(sym_set, phen_to_dis)
            if len(positives) == 0:
                continue

            qid = f"Q{q_counter:07d}"
            q_counter += 1
            sym_list = list(sym_set)
            rng.shuffle(sym_list)
            query_text = _format_query_text(sym_list, phen_names)

            rows.append(
                {
                    "query_id": qid,
                    "symptom_entity_ids": ";".join(sym_list),
                    "query_text": query_text,
                }
            )
            query_positives[qid] = positives

        if cfg.max_queries_total is not None and q_counter >= cfg.max_queries_total:
            break

    df = pd.DataFrame(rows)
    return df, query_positives


def _format_query_text(sym_list: List[str], phen_names: Dict[str, str]) -> str:
    parts = []
    for s in sym_list:
        if s in phen_names:
            parts.append(phen_names[s])
        else:
            # HPO|name style: last segment after last | often empty; use full key readable
            parts.append(s.replace("|", " "))
    return ", ".join(parts)


def attach_phenotype_names(cfg: PipelineConfig) -> Dict[str, str]:
    """Map phenotype entity_key -> display name from disease_phenotype_positive rows."""
    import pandas as pd

    names: Dict[str, str] = {}
    for chunk in pd.read_csv(cfg.kg_path, chunksize=500_000, low_memory=False):
        m = chunk[chunk["relation"] == "disease_phenotype_positive"]
        for _, row in m.iterrows():
            xt = str(row["x_type"])
            if xt == "disease":
                p_key = entity_key(str(row["y_type"]), str(row["y_source"]), row["y_id"])
                nm = row["y_name"]
            else:
                p_key = entity_key(str(row["x_type"]), str(row["x_source"]), row["x_id"])
                nm = row["x_name"]
            if p_key not in names and isinstance(nm, str):
                names[p_key] = nm
    return names
