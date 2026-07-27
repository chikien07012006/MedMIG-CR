from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired old/new query cohorts using the same patient IDs.")
    parser.add_argument("--old_queries", type=Path, default=Path("data/processed/ddxplus/test_queries.csv"))
    parser.add_argument("--new_queries", type=Path, default=Path("data/processed/ddxplus_v2/test_queries.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/comparison_cohort"))
    parser.add_argument("--limit", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_rows = read_rows(args.old_queries)
    new_rows = read_rows(args.new_queries)
    new_by_id = {int(row["patient_index"]): row for row in new_rows}
    old_by_id = {int(row["patient_index"]): row for row in old_rows}
    patient_ids = [
        int(row["patient_index"])
        for row in old_rows
        if int(row["patient_index"]) in new_by_id
    ][: args.limit]
    paired_old = [old_by_id[patient_id] for patient_id in patient_ids]
    paired_new = [new_by_id[patient_id] for patient_id in patient_ids]
    write_rows(args.output_dir / "old_queries.csv", paired_old, list(paired_old[0].keys()))
    write_rows(args.output_dir / "new_queries.csv", paired_new, list(paired_new[0].keys()))
    print(f"Wrote {len(patient_ids)} paired queries to {args.output_dir}")


if __name__ == "__main__":
    main()
