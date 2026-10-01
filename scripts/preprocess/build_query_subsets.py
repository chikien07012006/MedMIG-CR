"""Draw fixed, pathology-stratified query subsets shared by every method.

Each subset keeps each pathology's share of its source split (largest-remainder
rounding, at least one query per pathology) and preserves the source row order.
The same seed always yields the same patient_index lists.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_dir", type=Path, default=Path("data/processed/ddxplus_v2"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed/ddxplus_v2_subsets"))
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--valid_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def allocate(counts: Counter[str], size: int) -> Dict[str, int]:
    total = sum(counts.values())
    if size >= total:
        return dict(counts)
    quotas = {key: size * count / total for key, count in counts.items()}
    alloc = {key: max(1, min(counts[key], int(quota))) for key, quota in quotas.items()}
    remainder = size - sum(alloc.values())
    order = sorted(counts, key=lambda key: (quotas[key] - int(quotas[key]), key), reverse=True)
    step = 1 if remainder > 0 else -1
    while remainder != 0:
        changed = False
        for key in order if step > 0 else reversed(order):
            if remainder == 0:
                break
            new_value = alloc[key] + step
            if 1 <= new_value <= counts[key]:
                alloc[key] = new_value
                remainder -= step
                changed = True
        if not changed:
            break
    return alloc


def build_subset(source: Path, out_csv: Path, size: int, rng: random.Random) -> Dict[str, object]:
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    by_pathology: Dict[str, List[int]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        by_pathology[row["pathology"]].append(row_idx)
    alloc = allocate(Counter({key: len(idx) for key, idx in by_pathology.items()}), size)

    chosen: List[int] = []
    for pathology in sorted(by_pathology):
        chosen.extend(rng.sample(by_pathology[pathology], alloc[pathology]))
    chosen.sort()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in chosen:
            writer.writerow(rows[row_idx])

    source_dist = {key: len(idx) / len(rows) for key, idx in by_pathology.items()}
    subset_counts = Counter(rows[row_idx]["pathology"] for row_idx in chosen)
    total_variation = 0.5 * sum(abs(subset_counts[key] / len(chosen) - share) for key, share in source_dist.items())
    return {
        "source_csv": str(source),
        "output_csv": str(out_csv),
        "source_rows": len(rows),
        "subset_rows": len(chosen),
        "num_pathologies": len(subset_counts),
        "total_variation_vs_source": round(total_variation, 6),
        "pathology_counts": dict(sorted(subset_counts.items())),
        "patient_index": [int(rows[row_idx]["patient_index"]) for row_idx in chosen],
    }


def main() -> None:
    args = parse_args()
    sizes = {"train": args.train_size, "valid": args.valid_size, "test": args.test_size}
    summary: Dict[str, object] = {"seed": args.seed, "stratify_by": "pathology", "splits": {}}
    for offset, split in enumerate(SPLITS):
        # A separate generator per split keeps each subset stable if another size changes.
        rng = random.Random(args.seed + offset)
        size = sizes[split]
        result = build_subset(
            args.source_dir / f"{split}_queries.csv",
            args.out_dir / f"{split}_{size // 1000}k.csv",
            size,
            rng,
        )
        summary["splits"][split] = result
        print(f"{split}: {result['subset_rows']}/{result['source_rows']} rows, "
              f"{result['num_pathologies']} pathologies, TV={result['total_variation_vs_source']}")

    with (args.out_dir / "subsets_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
