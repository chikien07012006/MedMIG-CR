from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


DEFAULT_NODE_TYPES = ("disease", "effect/phenotype", "drug", "gene/protein", "pathway", "anatomy")


def entity_key(node_type: str, source: str, node_id: str) -> str:
    return f"{str(node_type).strip()}|{str(source).strip()}|{str(node_id).strip()}"


def iter_nodes(row: pd.Series) -> Iterable[Tuple[str, str, str, str]]:
    yield (
        str(row["x_type"]).strip(),
        str(row["x_source"]).strip(),
        str(row["x_id"]).strip(),
        str(row["x_name"]).strip(),
    )
    yield (
        str(row["y_type"]).strip(),
        str(row["y_source"]).strip(),
        str(row["y_id"]).strip(),
        str(row["y_name"]).strip(),
    )


def write_nodes(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_key", "node_type", "source", "node_id", "name"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_index(primekg_csv: Path, output_dir: Path, chunksize: int, allowed_types: set[str]) -> None:
    required = {
        "relation",
        "display_relation",
        "x_id",
        "x_type",
        "x_name",
        "x_source",
        "y_id",
        "y_type",
        "y_name",
        "y_source",
    }
    seen: Dict[str, Dict[str, str]] = {}
    relation_counts: Counter[str] = Counter()
    display_relation_counts: Counter[str] = Counter()

    for chunk in pd.read_csv(primekg_csv, chunksize=chunksize, dtype=str, low_memory=False):
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"PrimeKG CSV is missing required columns: {sorted(missing)}")

        relation_counts.update(chunk["relation"].fillna("").astype(str))
        display_relation_counts.update(chunk["display_relation"].fillna("").astype(str))

        for row in chunk.itertuples(index=False):
            row_dict = row._asdict()
            for prefix in ("x", "y"):
                node_type = str(row_dict[f"{prefix}_type"]).strip()
                if node_type not in allowed_types:
                    continue
                source = str(row_dict[f"{prefix}_source"]).strip()
                node_id = str(row_dict[f"{prefix}_id"]).strip()
                name = str(row_dict[f"{prefix}_name"]).strip()
                if not node_type or not source or not node_id:
                    continue
                key = entity_key(node_type, source, node_id)
                if key not in seen:
                    seen[key] = {
                        "node_key": key,
                        "node_type": node_type,
                        "source": source,
                        "node_id": node_id,
                        "name": name,
                    }

    rows = sorted(seen.values(), key=lambda item: (item["node_type"], item["source"], item["node_id"]))
    write_nodes(output_dir / "node_metadata.csv", rows)
    write_nodes(output_dir / "disease_nodes.csv", (row for row in rows if row["node_type"] == "disease"))
    write_nodes(
        output_dir / "phenotype_nodes.csv",
        (row for row in rows if row["node_type"] == "effect/phenotype"),
    )

    with (output_dir / "index_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "primekg_csv": str(primekg_csv),
                "num_nodes": len(rows),
                "node_type_counts": Counter(row["node_type"] for row in rows),
                "relation_counts": dict(relation_counts),
                "display_relation_counts": dict(display_relation_counts),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PrimeKG node index with stable typed node keys.")
    parser.add_argument("--primekg_csv", type=Path, default=Path("kg_giant.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/primekg"))
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--node_types", nargs="*", default=list(DEFAULT_NODE_TYPES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(
        primekg_csv=args.primekg_csv,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        allowed_types=set(args.node_types),
    )
    print(f"Wrote PrimeKG node index to {args.output_dir}")


if __name__ == "__main__":
    main()
