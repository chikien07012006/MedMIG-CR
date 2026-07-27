from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


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

# DDXPlus uses abbreviations and clinical labels that are not lexical matches
# for the canonical MONDO/HPO names shipped with PrimeKG.
CONDITION_ALIASES = {
    "Spontaneous rib fracture": "bone fracture",
    "GERD": "gastroesophageal reflux disease",
    "HIV (initial infection)": "HIV infectious disease",
    "Whooping cough": "pertussis",
    "PSVT": "Paroxysmal supraventricular tachycardia",
    "Larygospasm": "Laryngospasm",
    "Acute dystonic reactions": "dystonic disorder",
    "SLE": "systemic lupus erythematosus (disease)",
    "Unstable angina": "Angina pectoris",
    "Stable angina": "Angina pectoris",
    "Panic attack": "panic disorder",
    "Acute COPD exacerbation / infection": "chronic obstructive pulmonary disease",
    "Possible NSTEMI / STEMI": "acute myocardial infarction",
    "URTI": "upper respiratory tract disease",
    "Acute rhinosinusitis": "sinusitis",
}

# High-frequency clinical paraphrases. These are deliberately conservative:
# every target is an exact PrimeKG node name and can be inspected in output.
EVIDENCE_ALIASES = {
    "shortness of breath": ("Dyspnea",),
    "difficulty breathing": ("Dyspnea",),
    "nasal congestion": ("Rhinorrhea",),
    "runny nose": ("Rhinorrhea",),
    "increased sweating": ("Hyperhidrosis",),
    "smoke cigarettes": ("Triggered by smoking",),
    "addiction to alcohol": ("Alcoholism",),
    "drink alcohol excessively": ("Alcoholism",),
    "swelling in one or more areas": ("Edema",),
    "lightheaded and dizzy": ("Presyncope",),
    "about to faint": ("Presyncope",),
    "unable to do your usual activities": ("Fatigue",),
    "stuck in your bed": ("Fatigue",),
    "increased with physical exertion": ("Exercise intolerance",),
    "common allergies": ("Allergy",),
    "lesions peel off": ("Desquamation of skin soon after birth",),
    "lesions redness or problems on your skin": ("Localized skin lesion",),
    "high blood pressure": ("essential hypertension",),
    "use a bronchodilator": ("asthma",),
    "have asthma": ("asthma",),
}

NEGATIVE_VALUES = {"n", "no", "none", "false", "0"}


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
    shared = len(q_tokens & c_tokens)
    overlap = shared / max(1, len(q_tokens | c_tokens))
    candidate_coverage = shared / max(1, len(c_tokens))
    if len(c_tokens) == 1 and len(q_tokens) > 3:
        candidate_coverage *= 0.82
    containment = 0.0
    if len(q) >= 4 and q in c:
        containment = 0.92
    elif len(c) >= 4 and c in q:
        containment = 0.88
    return max(seq, overlap, containment, 0.9 * candidate_coverage)


def load_nodes(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        nodes = list(csv.DictReader(handle))
    for node in nodes:
        normalized = normalize(node.get("name", ""))
        node["_normalized_name"] = normalized
        node["_tokens"] = token_set(normalized)
    return nodes


def build_token_index(nodes: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = {}
    for node in nodes:
        for token in node.get("_tokens", set()):
            index.setdefault(token, []).append(node)
    return index


def ranked_candidates(
    query_labels: Iterable[str],
    nodes: List[Dict[str, str]],
    top_n: int,
    token_index: Dict[str, List[Dict[str, str]]] | None = None,
) -> List[Dict[str, Any]]:
    scored: Dict[str, Dict[str, Any]] = {}
    token_index = token_index or build_token_index(nodes)
    for label in query_labels:
        if not label:
            continue
        label_tokens = token_set(normalize(label))
        candidates_by_key = {
            node["node_key"]: node
            for token in label_tokens
            for node in token_index.get(token, [])
        }
        for node in candidates_by_key.values():
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


def exact_name_index(nodes: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = {}
    for node in nodes:
        index.setdefault(normalize(node.get("name", "")), []).append(node)
    return index


def exact_candidates(names: Iterable[str], nodes_by_name: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen = set()
    for name in names:
        for node in nodes_by_name.get(normalize(name), []):
            if node["node_key"] in seen:
                continue
            seen.add(node["node_key"])
            output.append(
                {
                    "node_key": node["node_key"],
                    "node_name": node.get("name", ""),
                    "node_type": node.get("node_type", ""),
                    "score": 1.0,
                    "matched_label": name,
                }
            )
    return output


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


def evidence_value_label(evidence: Dict[str, Any], value: str | None) -> str:
    if value is None:
        return ""
    value_info = (evidence.get("value_meaning") or {}).get(str(value), {})
    if isinstance(value_info, dict):
        return str(value_info.get("en") or value_info.get("fr") or value)
    return str(value_info or value)


def is_negative_evidence(entry: Dict[str, Any]) -> bool:
    value_label = normalize(entry.get("value_label", ""))
    return bool(value_label and value_label in NEGATIVE_VALUES)


def alias_names(label: str) -> List[str]:
    normalized = normalize(label)
    output: List[str] = []
    for phrase, names in EVIDENCE_ALIASES.items():
        if normalize(phrase) in normalized:
            output.extend(names)
    return output


def evidence_entries(evidences: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for evidence_key, evidence in evidences.items():
        evidence_name = str(evidence.get("name") or evidence_key)
        entries[evidence_name] = {
            "ddxplus_key": evidence_name,
            "ddxplus_label": evidence_label(evidence),
            "data_type": evidence.get("data_type"),
            "is_antecedent": bool(evidence.get("is_antecedent", False)),
            "value_label": "",
        }
        for value in evidence.get("possible-values") or []:
            value_key = f"{evidence_name}_@_{value}"
            entries[value_key] = {
                "ddxplus_key": value_key,
                "base_evidence": evidence_name,
                "value": str(value),
                "value_label": evidence_value_label(evidence, str(value)),
                "ddxplus_label": evidence_label(evidence, str(value)),
                "data_type": evidence.get("data_type"),
                "is_antecedent": bool(evidence.get("is_antecedent", False)),
            }
    return entries


def build_condition_mapping(
    conditions: Dict[str, Any],
    target_nodes: List[Dict[str, str]],
    top_n: int,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    nodes_by_name = exact_name_index(target_nodes)
    disease_nodes = [node for node in target_nodes if node.get("node_type") == "disease"]
    token_index = build_token_index(disease_nodes)
    for condition_key, condition in conditions.items():
        labels = [
            condition.get("cond-name-eng", ""),
            condition.get("condition_name", ""),
            condition.get("cond-name-fr", ""),
        ]
        output_key = str(condition.get("cond-name-eng") or condition.get("condition_name") or condition_key)
        alias = CONDITION_ALIASES.get(output_key)
        candidates = exact_candidates([alias] if alias else [], nodes_by_name)
        method = "clinical_alias" if candidates else "lexical"
        if not candidates:
            candidates = ranked_candidates(labels, disease_nodes, top_n=top_n, token_index=token_index)
        selected = [candidates[0]["node_key"]] if candidates and candidates[0]["score"] >= threshold else []
        status = "auto" if selected else "needs_review"
        mapping[output_key] = {
            "condition_name": condition.get("condition_name"),
            "cond_name_eng": condition.get("cond-name-eng"),
            "cond_name_fr": condition.get("cond-name-fr"),
            "icd10_id": condition.get("icd10-id"),
            "selected_primekg_nodes": selected,
            "status": status,
            "selection_method": method if selected else "unmapped",
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
    exact_match_nodes: List[Dict[str, str]],
    top_n: int,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    nodes_by_name = exact_name_index(exact_match_nodes)
    token_index = build_token_index(phenotype_nodes)
    for evidence_key, entry in evidence_entries(evidences).items():
        aliases = alias_names(entry["ddxplus_label"])
        candidates = exact_candidates(aliases, nodes_by_name)
        method = "clinical_alias" if candidates else "lexical"
        if not candidates:
            labels = [entry["ddxplus_label"]]
            if entry.get("value_label") and normalize(entry["value_label"]) not in NEGATIVE_VALUES:
                labels.insert(0, entry["value_label"])
            candidates = ranked_candidates(labels, phenotype_nodes, top_n=top_n, token_index=token_index)
        negative = is_negative_evidence(entry)
        selected: List[str] = []
        if not negative and candidates and candidates[0]["score"] >= threshold:
            best_score = float(candidates[0]["score"])
            selected = [
                candidate["node_key"]
                for candidate in candidates
                if float(candidate["score"]) >= threshold and float(candidate["score"]) >= best_score - 0.03
            ][:2]
        mapping[evidence_key] = {
            **entry,
            "selected_primekg_nodes": selected,
            "status": "negative_value" if negative else ("auto" if selected else "needs_review"),
            "selection_method": "negative_filter" if negative else (method if selected else "unmapped"),
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
    parser.add_argument("--output_dir", type=Path, default=Path("data/mappings/ddxplus_v2"))
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--condition_threshold", type=float, default=0.82)
    parser.add_argument("--evidence_threshold", type=float, default=0.72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_nodes = load_nodes(args.primekg_index_dir / "node_metadata.csv")
    target_nodes = [node for node in all_nodes if node.get("node_type") in {"disease", "effect/phenotype"}]
    with args.conditions_json.open("r", encoding="utf-8-sig") as handle:
        conditions = json.load(handle)
    with args.evidences_json.open("r", encoding="utf-8-sig") as handle:
        evidences = json.load(handle)

    condition_mapping = build_condition_mapping(
        conditions,
        target_nodes,
        top_n=args.top_n,
        threshold=args.condition_threshold,
    )
    evidence_mapping = build_evidence_mapping(
        evidences,
        [node for node in target_nodes if node.get("node_type") == "effect/phenotype"],
        target_nodes,
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
                "num_evidence_negative_filtered": sum(
                    1 for item in evidence_mapping.values() if item.get("status") == "negative_value"
                ),
                "num_unique_condition_nodes": len(
                    {
                        node
                        for item in condition_mapping.values()
                        for node in item.get("selected_primekg_nodes", [])
                    }
                ),
                "num_unique_evidence_nodes": len(
                    {
                        node
                        for item in evidence_mapping.values()
                        for node in item.get("selected_primekg_nodes", [])
                    }
                ),
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
