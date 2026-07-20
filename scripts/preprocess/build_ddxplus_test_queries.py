from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def parse_list_cell(cell: str) -> List[str]:
    if not cell:
        return []
    try:
        value = ast.literal_eval(cell)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def selected_nodes(mapping: Dict[str, Any], key: str) -> List[str]:
    entry = mapping.get(key)
    if not entry:
        return []
    if "alias_of" in entry:
        entry = mapping.get(entry["alias_of"], entry)
    return list(entry.get("selected_primekg_nodes") or [])


def build_queries(
    patients_csv: Path,
    evidence_map: Dict[str, Any],
    condition_map: Dict[str, Any],
    output_csv: Path,
    summary_json: Path,
    min_seed_nodes: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    missing_evidences = Counter()
    missing_pathologies = Counter()

    with patients_csv.open("r", newline="", encoding="utf-8-sig") as in_handle, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as out_handle:
        reader = csv.DictReader(in_handle)
        fieldnames = [
            "patient_index",
            "age",
            "sex",
            "pathology",
            "evidence_keys",
            "seed_node_keys",
            "target_node_keys",
            "num_evidences",
            "num_seed_nodes",
        ]
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()

        for patient_index, row in enumerate(reader):
            stats["patients_total"] += 1
            evidence_keys = parse_list_cell(row.get("EVIDENCES", ""))
            seed_nodes: List[str] = []
            for evidence_key in evidence_keys:
                nodes = selected_nodes(evidence_map, evidence_key)
                if nodes:
                    seed_nodes.extend(nodes)
                else:
                    missing_evidences[evidence_key] += 1

            seed_nodes = sorted(set(seed_nodes))
            pathology = str(row.get("PATHOLOGY", ""))
            target_nodes = selected_nodes(condition_map, pathology)

            if not target_nodes:
                stats["skipped_missing_pathology_mapping"] += 1
                missing_pathologies[pathology] += 1
                continue
            if len(seed_nodes) < min_seed_nodes:
                stats["skipped_too_few_seed_nodes"] += 1
                continue

            stats["patients_written"] += 1
            writer.writerow(
                {
                    "patient_index": patient_index,
                    "age": row.get("AGE", ""),
                    "sex": row.get("SEX", ""),
                    "pathology": pathology,
                    "evidence_keys": ";".join(evidence_keys),
                    "seed_node_keys": ";".join(seed_nodes),
                    "target_node_keys": ";".join(sorted(set(target_nodes))),
                    "num_evidences": len(evidence_keys),
                    "num_seed_nodes": len(seed_nodes),
                }
            )

    summary = {
        **dict(stats),
        "min_seed_nodes": min_seed_nodes,
        "missing_evidence_top20": missing_evidences.most_common(20),
        "missing_pathology_top20": missing_pathologies.most_common(20),
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DDXPlus test queries mapped to PrimeKG seed/target nodes.")
    parser.add_argument("--patients_csv", type=Path, default=Path("Benchmark data/DDXPlus/release_test_patients.csv"))
    parser.add_argument("--evidence_map", type=Path, default=Path("data/mappings/ddxplus/evidence_to_primekg.json"))
    parser.add_argument("--condition_map", type=Path, default=Path("data/mappings/ddxplus/condition_to_primekg.json"))
    parser.add_argument("--output_csv", type=Path, default=Path("data/processed/ddxplus/test_queries.csv"))
    parser.add_argument("--summary_json", type=Path, default=Path("data/processed/ddxplus/test_query_summary.json"))
    parser.add_argument("--min_seed_nodes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_queries(
        patients_csv=args.patients_csv,
        evidence_map=load_json(args.evidence_map),
        condition_map=load_json(args.condition_map),
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        min_seed_nodes=args.min_seed_nodes,
    )
    print(f"Wrote DDXPlus test queries to {args.output_csv}")


if __name__ == "__main__":
    main()
