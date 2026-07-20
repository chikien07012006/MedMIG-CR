from __future__ import annotations

import argparse
import ast
import csv
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


METRIC_KS = (5, 10, 20, 50)
CANDIDATE_COLUMNS = ("candidate", "node_key", "disease_node", "prediction", "disease")


def normalize(text: Any) -> str:
    raw = "" if text is None else str(text)
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    return " ".join(raw.lower().replace("_", " ").split())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def selected_nodes_for_condition(condition_name: str, condition_map: Dict[str, Any]) -> List[str]:
    if condition_name in condition_map:
        entry = condition_map[condition_name]
    else:
        norm_index = {normalize(key): key for key in condition_map}
        key = norm_index.get(normalize(condition_name))
        if key is None:
            return []
        entry = condition_map[key]

    if "alias_of" in entry:
        entry = condition_map.get(entry["alias_of"], entry)
    return list(entry.get("selected_primekg_nodes") or [])


def parse_differential(cell: str) -> List[str]:
    if not cell:
        return []
    try:
        value = ast.literal_eval(cell)
    except (SyntaxError, ValueError):
        return []
    out: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and item:
                out.append(str(item[0]))
            elif isinstance(item, str):
                out.append(item)
    return out


def target_nodes_for_patient(
    row: Dict[str, str],
    condition_map: Dict[str, Any],
    target_mode: str,
) -> List[str]:
    if target_mode == "pathology":
        return selected_nodes_for_condition(row.get("PATHOLOGY", ""), condition_map)
    if target_mode == "differential":
        targets: List[str] = []
        for condition in parse_differential(row.get("DIFFERENTIAL_DIAGNOSIS", "")):
            targets.extend(selected_nodes_for_condition(condition, condition_map))
        return sorted(set(targets))
    raise ValueError(f"Unknown target_mode: {target_mode}")


def patient_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def candidate_from_row(row: Dict[str, str]) -> str:
    for column in CANDIDATE_COLUMNS:
        value = row.get(column)
        if value:
            return str(value)
    raise ValueError(f"Prediction row must contain one of these columns: {CANDIDATE_COLUMNS}")


def load_predictions_csv(path: Path) -> Dict[int, List[str]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "patient_index" not in reader.fieldnames:
            raise ValueError("CSV predictions require a patient_index column.")
        for order, row in enumerate(reader):
            patient_index = int(row["patient_index"])
            rank = row.get("rank")
            score = row.get("score")
            grouped[patient_index].append(
                {
                    "candidate": candidate_from_row(row),
                    "rank": int(rank) if rank not in (None, "") else None,
                    "score": float(score) if score not in (None, "") else None,
                    "order": order,
                }
            )

    predictions: Dict[int, List[str]] = {}
    for patient_index, rows in grouped.items():
        rows.sort(
            key=lambda item: (
                item["rank"] if item["rank"] is not None else 10**9,
                -(item["score"] if item["score"] is not None else float("-inf")),
                item["order"],
            )
        )
        predictions[patient_index] = [row["candidate"] for row in rows]
    return predictions


def candidate_from_json_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for column in CANDIDATE_COLUMNS:
            if item.get(column):
                return str(item[column])
    raise ValueError(f"Unsupported candidate item: {item!r}")


def load_predictions_json(path: Path) -> Dict[int, List[str]]:
    data = load_json(path)
    predictions: Dict[int, List[str]] = {}
    if isinstance(data, dict):
        iterable = data.items()
        for key, value in iterable:
            patient_index = int(key)
            candidates = value.get("candidates", value) if isinstance(value, dict) else value
            predictions[patient_index] = [candidate_from_json_item(item) for item in candidates]
        return predictions

    if isinstance(data, list):
        for entry in data:
            patient_index = int(entry["patient_index"])
            candidates = entry.get("candidates")
            if candidates is None:
                candidates = [entry]
            predictions[patient_index] = [candidate_from_json_item(item) for item in candidates]
        return predictions

    raise ValueError("Predictions JSON must be a dict or list.")


def load_predictions(path: Path) -> Dict[int, List[str]]:
    if path.suffix.lower() == ".json":
        return load_predictions_json(path)
    return load_predictions_csv(path)


def canonical_candidate(candidate: str, condition_map: Dict[str, Any]) -> List[str]:
    direct_nodes = selected_nodes_for_condition(candidate, condition_map)
    if direct_nodes:
        return direct_nodes
    return [candidate]


def first_hit_rank(
    ranked_candidates: Sequence[str],
    target_nodes: set[str],
    condition_map: Dict[str, Any],
) -> int | None:
    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate_nodes = canonical_candidate(candidate, condition_map)
        if target_nodes.intersection(candidate_nodes):
            return rank
    return None


def metric_row(
    patient_index: int,
    patient: Dict[str, str],
    ranked_candidates: Sequence[str],
    target_nodes: List[str],
    condition_map: Dict[str, Any],
    ks: Sequence[int],
) -> Dict[str, Any]:
    targets = set(target_nodes)
    rank = first_hit_rank(ranked_candidates, targets, condition_map) if targets else None
    row: Dict[str, Any] = {
        "patient_index": patient_index,
        "pathology": patient.get("PATHOLOGY", ""),
        "num_targets": len(targets),
        "num_predictions": len(ranked_candidates),
        "first_hit_rank": rank or "",
        "mrr": (1.0 / rank) if rank else 0.0,
    }
    for k in ks:
        row[f"recall@{k}"] = 1.0 if rank is not None and rank <= k else 0.0
    return row


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize(rows: List[Dict[str, Any]], skipped: Dict[str, int], ks: Sequence[int]) -> Dict[str, Any]:
    summary = {
        "num_evaluated": len(rows),
        "skipped": skipped,
        "mrr": mean(float(row["mrr"]) for row in rows),
    }
    for k in ks:
        summary[f"recall@{k}"] = mean(float(row[f"recall@{k}"]) for row in rows)
    return summary


def write_by_patient(path: Path, rows: List[Dict[str, Any]], ks: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["patient_index", "pathology", "num_targets", "num_predictions", "first_hit_rank", "mrr"]
    columns.extend(f"recall@{k}" for k in ks)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DDXPlus KG retrieval predictions.")
    parser.add_argument("--patients_csv", type=Path, default=Path("Benchmark data/DDXPlus/release_test_patients.csv"))
    parser.add_argument("--condition_map", type=Path, default=Path("data/mappings/ddxplus/condition_to_primekg.json"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results/ddxplus_retrieval"))
    parser.add_argument("--target_mode", choices=("pathology", "differential"), default="pathology")
    parser.add_argument("--topk", nargs="*", type=int, default=list(METRIC_KS))
    parser.add_argument("--limit_patients", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patients = patient_rows(args.patients_csv)
    if args.limit_patients is not None:
        patients = patients[: args.limit_patients]
    condition_map = load_json(args.condition_map)
    predictions = load_predictions(args.predictions)

    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = defaultdict(int)
    for patient_index, patient in enumerate(patients):
        target_nodes = target_nodes_for_patient(patient, condition_map, args.target_mode)
        if not target_nodes:
            skipped["missing_target_mapping"] += 1
            continue
        ranked_candidates = predictions.get(patient_index)
        if not ranked_candidates:
            skipped["missing_prediction"] += 1
            continue
        rows.append(
            metric_row(
                patient_index=patient_index,
                patient=patient,
                ranked_candidates=ranked_candidates,
                target_nodes=target_nodes,
                condition_map=condition_map,
                ks=args.topk,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, dict(skipped), args.topk)
    summary.update(
        {
            "patients_csv": str(args.patients_csv),
            "predictions": str(args.predictions),
            "condition_map": str(args.condition_map),
            "target_mode": args.target_mode,
            "topk": args.topk,
        }
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_by_patient(args.output_dir / "by_patient.csv", rows, args.topk)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
