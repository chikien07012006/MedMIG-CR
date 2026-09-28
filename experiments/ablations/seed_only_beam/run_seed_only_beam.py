from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medmigcr_kg.graph_store import GraphStore  # noqa: E402
from medmigcr_kg.retrieval_engine import RetrievalEngine  # noqa: E402
from medmigcr_kg.telemetry import log_run_summary  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def split_nodes(cell: str) -> List[str]:
    return [token.strip() for token in str(cell or "").split(";") if token.strip()]


def load_queries(path: Path, limit: int) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(tqdm(islice(reader, limit), total=limit, desc="Loading test queries", unit="query"))


def resolve_seed_ids(graph_store: GraphStore, seed_node_keys: Sequence[str]) -> List[int]:
    node_ids = {
        node_id
        for key in seed_node_keys
        for node_id in [graph_store.lookup_node_id(key)]
        if node_id is not None
    }
    return sorted(node_ids)


def load_target_universe(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        mapping = json.load(handle)
    return {
        node_key
        for entry in mapping.values()
        for node_key in entry.get("selected_primekg_nodes", [])
    }


def collect_target_scores(result, graph_store: GraphStore, target_universe: set[str], top_k: int) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for item in result.paths:
        node_key = graph_store.lookup_node_name(item.current_node)
        if node_key not in target_universe:
            continue
        current_score = scores.get(node_key)
        if current_score is None or item.score > current_score:
            scores[node_key] = float(item.score)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]


def write_predictions(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_index", "candidate", "score", "rank"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run seed-only PrimeKG beam retrieval and evaluate the first DDXPlus test queries."
    )
    parser.add_argument("--test_queries_csv", type=Path, default=Path("data/processed/ddxplus_v2/test_queries.csv"))
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument("--condition_map", type=Path, default=Path("data/mappings/ddxplus_v2/condition_to_primekg.json"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiments/ablations/seed_only_beam/results/test5000"),
    )
    parser.add_argument("--limit_queries", type=int, default=5000)
    parser.add_argument("--max_hops", type=int, default=10)
    parser.add_argument("--beam_width", type=int, default=128)
    parser.add_argument("--paths_per_interest", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--graph_device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_queries_csv = resolve_path(args.test_queries_csv)
    graph_dir = resolve_path(args.graph_dir)
    condition_map = resolve_path(args.condition_map)
    output_dir = resolve_path(args.output_dir)
    output_csv = output_dir / "predictions.csv"
    retrieval_summary_path = output_dir / "retrieval_summary.json"
    evaluation_dir = output_dir / "evaluation"
    started_at = time.perf_counter()

    graph_store = GraphStore.load(
        graph_npz=graph_dir / "graph_csr.npz",
        node_embeddings_npy=graph_dir / "node_embeddings.npy",
        out_degree_npy=graph_dir / "out_degree.npy",
        in_degree_npy=graph_dir / "in_degree.npy",
        mapping_dir=graph_dir / "mappings",
        device=args.graph_device,
    )
    engine = RetrievalEngine(graph_store)
    queries = load_queries(test_queries_csv, args.limit_queries)
    target_universe = load_target_universe(condition_map)
    target_node_ids = {
        node_id
        for node_key in target_universe
        for node_id in [graph_store.lookup_node_id(node_key)]
        if node_id is not None
    }

    prediction_rows: List[Dict[str, object]] = []
    skipped_missing_seed = 0
    no_target_candidates = 0
    retrieval_seconds = 0.0
    for query in tqdm(queries, desc="Seed-only beam retrieval", unit="query"):
        patient_index = int(query["patient_index"])
        seed_keys = split_nodes(query.get("seed_node_keys", ""))
        seed_ids = resolve_seed_ids(graph_store, seed_keys)
        if not seed_ids:
            skipped_missing_seed += 1
            continue

        result = engine.retrieve(
            seed_node_ids=seed_ids,
            interest_count=1,
            max_hops=args.max_hops,
            beam_width=args.beam_width,
            topk_paths=args.paths_per_interest,
            alpha=args.alpha,
            beta=args.beta,
            max_paths_per_interest=args.paths_per_interest,
            target_node_ids=target_node_ids,
        )
        retrieval_seconds += result.latency_seconds
        ranked = collect_target_scores(result, graph_store, target_universe, args.top_k)
        if not ranked:
            no_target_candidates += 1
            continue
        prediction_rows.extend(
            {
                "patient_index": patient_index,
                "candidate": candidate,
                "score": f"{score:.8f}",
                "rank": rank,
            }
            for rank, (candidate, score) in enumerate(ranked, start=1)
        )

    write_predictions(output_csv, prediction_rows)
    log_run_summary(
        "seed_only_beam_retrieval",
        started_at,
        len(queries),
        "queries",
        prediction_rows=len(prediction_rows),
        retrieval_seconds=round(retrieval_seconds, 3),
    )
    retrieval_summary = {
        "experiment": "seed_only_beam",
        "query_encoder": "none",
        "seed_interest_initialization": "mean PrimeKG embedding of resolved seed nodes",
        "test_queries_csv": str(test_queries_csv),
        "graph_dir": str(graph_dir),
        "condition_map": str(condition_map),
        "target_universe_size": len(target_universe),
        "target_aware_candidate_collection": True,
        "output_csv": str(output_csv),
        "num_queries_loaded": len(queries),
        "num_prediction_rows": len(prediction_rows),
        "skipped_missing_seed": skipped_missing_seed,
        "no_target_candidates": no_target_candidates,
        "max_hops": args.max_hops,
        "beam_width": args.beam_width,
        "paths_per_interest": args.paths_per_interest,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "beta": args.beta,
        "retrieval_seconds": round(retrieval_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    retrieval_summary_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_summary_path.write_text(json.dumps(retrieval_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    evaluator = REPO_ROOT / "scripts/evaluation/evaluate_ddxplus_retrieval.py"
    subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--queries_csv",
            str(test_queries_csv),
            "--condition_map",
            str(condition_map),
            "--predictions",
            str(output_csv),
            "--output_dir",
            str(evaluation_dir),
            "--target_mode",
            "pathology",
            "--topk",
            "1",
            "5",
            "10",
            "20",
            "50",
            "--limit_patients",
            str(args.limit_queries),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    print(f"Predictions: {output_csv}")
    print(f"Retrieval summary: {retrieval_summary_path}")
    print(f"Evaluation metrics: {evaluation_dir / 'summary.json'}")


if __name__ == "__main__":
    main()