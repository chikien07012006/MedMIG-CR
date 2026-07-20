from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "associated",
    "do",
    "does",
    "either",
    "feel",
    "felt",
    "have",
    "is",
    "measured",
    "of",
    "or",
    "somewhere",
    "the",
    "to",
    "with",
    "you",
    "your",
}


def normalize(text: Any) -> str:
    raw = "" if text is None else str(text)
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    raw = raw.lower()
    raw = raw.replace("(disease)", " ")
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    tokens = [tok for tok in raw.split() if tok and tok not in STOPWORDS]
    return " ".join(tokens)


def token_set(text: str) -> set[str]:
    return set(text.split())


def match_score(query: str, candidate: str) -> float:
    q = normalize(query)
    c = normalize(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    seq = SequenceMatcher(None, q, c).ratio()
    q_tokens = token_set(q)
    c_tokens = token_set(c)
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
    containment = 0.0
    if len(q) >= 4 and q in c:
        containment = 0.92
    elif len(c) >= 4 and c in q:
        containment = 0.88
    return max(seq, overlap, containment)


def load_nodes(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ranked_candidates(query_labels: Iterable[str], nodes: List[Dict[str, str]], top_n: int) -> List[Dict[str, Any]]:
    scored: Dict[str, Dict[str, Any]] = {}
    for label in query_labels:
        if not label:
            continue
        for node in nodes:
            score = match_score(label, node.get("name", ""))
            key = node["node_key"]
            existing = scored.get(key)
            if existing is None or score > existing["score"]:
                scored[key] = {
                    "node_key": key,
                    "node_name": node.get("name", ""),
                    "node_type": node.get("node_type", ""),
                    "score": round(float(score), 6),
                    "matched_label": label,
                }
    return sorted(scored.values(), key=lambda item: item["score"], reverse=True)[:top_n]


def evidence_label(evidence: Dict[str, Any], value: str | None = None) -> str:
    question = evidence.get("question_en") or evidence.get("question_fr") or evidence.get("name", "")
    if value is None:
        return str(question)
    meanings = evidence.get("value_meaning") or {}
    value_info = meanings.get(str(value), {})
    if isinstance(value_info, dict):
        value_label = value_info.get("en") or value_info.get("fr") or str(value)
    else:
        value_label = str(value_info or value)
    return f"{question} {value_label}"


def evidence_entries(evidences: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for evidence_key, evidence in evidences.items():
        evidence_name = str(evidence.get("name") or evidence_key)
        entries[evidence_name] = {
            "ddxplus_key": evidence_name,
            "ddxplus_label": evidence_label(evidence),
            "data_type": evidence.get("data_type"),
            "is_antecedent": bool(evidence.get("is_antecedent", False)),
        }
        for value in evidence.get("possible-values") or []:
            value_key = f"{evidence_name}_@_{value}"
            entries[value_key] = {
                "ddxplus_key": value_key,
                "base_evidence": evidence_name,
                "value": str(value),
                "ddxplus_label": evidence_label(evidence, str(value)),
                "data_type": evidence.get("data_type"),
                "is_antecedent": bool(evidence.get("is_antecedent", False)),
            }
    return entries


def build_condition_mapping(
    conditions: Dict[str, Any],
    disease_nodes: List[Dict[str, str]],
    top_n: int,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for condition_key, condition in conditions.items():
        labels = [
            condition.get("cond-name-eng", ""),
            condition.get("condition_name", ""),
            condition.get("cond-name-fr", ""),
        ]
        candidates = ranked_candidates(labels, disease_nodes, top_n=top_n)
        selected = [candidates[0]["node_key"]] if candidates and candidates[0]["score"] >= threshold else []
        status = "auto" if selected else "needs_review"
        output_key = str(condition.get("cond-name-eng") or condition.get("condition_name") or condition_key)
        mapping[output_key] = {
            "condition_name": condition.get("condition_name"),
            "cond_name_eng": condition.get("cond-name-eng"),
            "cond_name_fr": condition.get("cond-name-fr"),
            "icd10_id": condition.get("icd10-id"),
            "selected_primekg_nodes": selected,
            "status": status,
            "candidates": candidates,
        }
        if condition_key != output_key:
            mapping.setdefault(
                condition_key,
                {
                    "alias_of": output_key,
                    "selected_primekg_nodes": selected,
                    "status": status,
                },
            )
    return mapping


def build_evidence_mapping(
    evidences: Dict[str, Any],
    phenotype_nodes: List[Dict[str, str]],
    top_n: int,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for evidence_key, entry in evidence_entries(evidences).items():
        candidates = ranked_candidates([entry["ddxplus_label"], evidence_key], phenotype_nodes, top_n=top_n)
        selected = [candidates[0]["node_key"]] if candidates and candidates[0]["score"] >= threshold else []
        mapping[evidence_key] = {
            **entry,
            "selected_primekg_nodes": selected,
            "status": "auto" if selected else "needs_review",
            "candidates": candidates,
        }
    return mapping


def count_auto(mapping: Dict[str, Dict[str, Any]]) -> int:
    return sum(1 for item in mapping.values() if item.get("status") == "auto")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DDXPlus to PrimeKG candidate mappings.")
    parser.add_argument("--conditions_json", type=Path, default=Path("Benchmark data/DDXPlus/release_conditions.json"))
    parser.add_argument("--evidences_json", type=Path, default=Path("Benchmark data/DDXPlus/release_evidences.json"))
    parser.add_argument("--primekg_index_dir", type=Path, default=Path("data/processed/primekg"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/mappings/ddxplus"))
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--condition_threshold", type=float, default=0.82)
    parser.add_argument("--evidence_threshold", type=float, default=0.78)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    disease_nodes = load_nodes(args.primekg_index_dir / "disease_nodes.csv")
    phenotype_nodes = load_nodes(args.primekg_index_dir / "phenotype_nodes.csv")
    with args.conditions_json.open("r", encoding="utf-8-sig") as handle:
        conditions = json.load(handle)
    with args.evidences_json.open("r", encoding="utf-8-sig") as handle:
        evidences = json.load(handle)

    condition_mapping = build_condition_mapping(
        conditions,
        disease_nodes,
        top_n=args.top_n,
        threshold=args.condition_threshold,
    )
    evidence_mapping = build_evidence_mapping(
        evidences,
        phenotype_nodes,
        top_n=args.top_n,
        threshold=args.evidence_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "condition_to_primekg.json").open("w", encoding="utf-8") as handle:
        json.dump(condition_mapping, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "evidence_to_primekg.json").open("w", encoding="utf-8") as handle:
        json.dump(evidence_mapping, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "mapping_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_conditions": len(condition_mapping),
                "num_condition_auto": count_auto(condition_mapping),
                "num_evidences": len(evidence_mapping),
                "num_evidence_auto": count_auto(evidence_mapping),
                "condition_threshold": args.condition_threshold,
                "evidence_threshold": args.evidence_threshold,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Wrote DDXPlus mappings to {args.output_dir}")


if __name__ == "__main__":
    main()
