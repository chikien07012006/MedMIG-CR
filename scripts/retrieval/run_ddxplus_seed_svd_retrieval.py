from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medmigcr_kg.graph_store import GraphStore  # noqa: E402
from medmigcr_kg.retrieval_engine import RetrievalEngine  # noqa: E402


def split_nodes(cell: str) -> List[str]:
    return [token.strip() for token in str(cell or "").split(";") if token.strip()]


def is_disease_node(node_name: str | None) -> bool:
    return bool(node_name and node_name.startswith("disease|"))


def load_queries(path: Path, limit: int | None = None) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit is not None else rows


def resolve_seed_ids(graph_store: GraphStore, seed_node_keys: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for key in seed_node_keys:
        node_id = graph_store.lookup_node_id(key)
        if node_id is not None:
            ids.append(node_id)
    return sorted(set(ids))


def disease_endpoint_scores(result, graph_store: GraphStore, top_k: int) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for item in result.paths:
        node_name = graph_store.lookup_node_name(item.current_node)
        if not is_disease_node(node_name):
            continue
        current = scores.get(node_name)
        if current is None or item.score > current:
            scores[node_name] = float(item.score)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]


def write_predictions(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["patient_index", "candidate", "score", "rank"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, summary: Dict[str, object]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seed-SVD beam-search retrieval on DDXPlus test queries.")
    parser.add_argument("--test_queries_csv", type=Path, default=Path("data/processed/ddxplus/test_queries.csv"))
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument("--output_csv", type=Path, default=Path("results/seed_svd/predictions.csv"))
    parser.add_argument("--summary_json", type=Path, default=Path("results/seed_svd/retrieval_summary.json"))
    parser.add_argument("--interest_count", type=int, default=1)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--beam_width", type=int, default=32)
    parser.add_argument("--paths_per_interest", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--limit_patients", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_store = GraphStore.load(
        graph_npz=args.graph_dir / "graph_csr.npz",
        node_embeddings_npy=args.graph_dir / "node_embeddings.npy",
        out_degree_npy=args.graph_dir / "out_degree.npy",
        in_degree_npy=args.graph_dir / "in_degree.npy",
        mapping_dir=args.graph_dir / "mappings",
        device=args.device,
    )
    engine = RetrievalEngine(graph_store)
    queries = load_queries(args.test_queries_csv, limit=args.limit_patients)

    prediction_rows: List[Dict[str, object]] = []
    skipped_missing_seed = 0
    no_disease_candidates = 0
    for row in queries:
        patient_index = int(row["patient_index"])
        seed_ids = resolve_seed_ids(graph_store, split_nodes(row.get("seed_node_keys", "")))
        if not seed_ids:
            skipped_missing_seed += 1
            continue
        result = engine.retrieve(
            seed_node_ids=seed_ids,
            max_hops=args.max_hops,
            beam_width=args.beam_width,
            topk_paths=args.paths_per_interest,
            alpha=args.alpha,
            beta=args.beta,
            interest_count=args.interest_count,
            max_paths_per_interest=args.paths_per_interest,
        )
        ranked = disease_endpoint_scores(result, graph_store, top_k=args.top_k)
        if not ranked:
            no_disease_candidates += 1
            continue
        for rank, (candidate, score) in enumerate(ranked, start=1):
            prediction_rows.append(
                {
                    "patient_index": patient_index,
                    "candidate": candidate,
                    "score": f"{score:.8f}",
                    "rank": rank,
                }
            )

    write_predictions(args.output_csv, prediction_rows)
    write_summary(
        args.summary_json,
        {
            "test_queries_csv": str(args.test_queries_csv),
            "graph_dir": str(args.graph_dir),
            "output_csv": str(args.output_csv),
            "num_queries_loaded": len(queries),
            "num_prediction_rows": len(prediction_rows),
            "skipped_missing_seed": skipped_missing_seed,
            "no_disease_candidates": no_disease_candidates,
            "interest_count": args.interest_count,
            "max_hops": args.max_hops,
            "beam_width": args.beam_width,
            "paths_per_interest": args.paths_per_interest,
            "top_k": args.top_k,
        },
    )
    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
