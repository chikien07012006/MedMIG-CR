from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm  # noqa: E402

from baseline_common import (  # noqa: E402
    DEFAULT_CONDITION_MAP,
    DEFAULT_GRAPH_DIR,
    DEFAULT_SUBSET_DIR,
    Query,
    candidate_coverage,
    load_queries,
    load_target_universe,
    print_metrics,
    run_shared_evaluator,
    write_json,
    write_predictions,
)
from medmigcr_kg.graph_store import GraphStore  # noqa: E402
from medmigcr_kg.parallel import recommended_workers, run_ordered  # noqa: E402
from medmigcr_kg.retrieval_engine import RetrievalEngine  # noqa: E402
from medmigcr_kg.target_distance import build_distance_guide  # noqa: E402

_STATE: Dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries_csv", type=Path, default=DEFAULT_SUBSET_DIR / "test_10k.csv")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--graph_dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--condition_map", type=Path, default=DEFAULT_CONDITION_MAP)
    parser.add_argument("--max_hops", type=int, default=10)
    parser.add_argument("--beam_width", type=int, default=128)
    parser.add_argument("--paths_per_interest", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--distance_pruning", action="store_true",
                        help="Drop neighbours that cannot reach any target within the remaining hops.")
    parser.add_argument("--distance_weight", type=float, default=0.0,
                        help="Subtract weight * hop distance to the nearest target from each step score.")
    parser.add_argument("--num_workers", type=int, default=recommended_workers(),
                        help="CPU processes; results are identical for any value.")
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N queries (smoke tests).")
    parser.add_argument("--no_eval", action="store_true", help="Skip the shared evaluator.")
    return parser.parse_args()


def init_state(args: argparse.Namespace) -> None:
    graph_dir = args.graph_dir
    graph_store = GraphStore.load(
        graph_npz=graph_dir / "graph_csr.npz",
        node_embeddings_npy=graph_dir / "node_embeddings.npy",
        out_degree_npy=graph_dir / "out_degree.npy",
        in_degree_npy=graph_dir / "in_degree.npy",
        mapping_dir=graph_dir / "mappings",
        device="cpu",
    )
    target_universe = set(load_target_universe(args.condition_map))
    _STATE.update(
        args=args,
        graph_store=graph_store,
        engine=RetrievalEngine(graph_store),
        target_universe=target_universe,
        target_node_ids={graph_store.lookup_node_id(key) for key in target_universe} - {None},
    )
    _STATE["distance_guide"] = build_distance_guide(
        _STATE["graph_store"], _STATE["target_node_ids"], args.distance_pruning, args.distance_weight
    )


def process_query(query: Query) -> Dict[str, object]:
    args: argparse.Namespace = _STATE["args"]
    graph_store: GraphStore = _STATE["graph_store"]
    target_universe: set[str] = _STATE["target_universe"]
    seed_ids = sorted({node_id for key in query.seed_keys for node_id in [graph_store.lookup_node_id(key)] if node_id is not None})
    if not seed_ids:
        return {"status": "missing_seed", "rows": [], "latency": 0.0}

    result = _STATE["engine"].retrieve(
        seed_node_ids=seed_ids,
        interest_count=1,  # mean seed embedding
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        topk_paths=args.paths_per_interest,
        alpha=args.alpha,
        beta=args.beta,
        max_paths_per_interest=args.paths_per_interest,
        target_node_ids=_STATE["target_node_ids"],
        distance_guide=_STATE["distance_guide"],
    )
    # A candidate's score is its best path score, as in the main method without reranking.
    best: Dict[str, float] = {}
    for item in result.paths:
        key = graph_store.lookup_node_name(item.current_node)
        if key in target_universe and item.score > best.get(key, float("-inf")):
            best[key] = float(item.score)
    ranked = sorted(best.items(), key=lambda pair: pair[1], reverse=True)[: args.top_k]
    rows = [
        {"patient_index": query.patient_index, "candidate": key, "score": f"{score:.8f}", "rank": rank}
        for rank, (key, score) in enumerate(ranked, start=1)
    ]
    return {"status": "ok" if rows else "no_candidates", "rows": rows, "latency": float(result.latency_seconds)}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    queries = load_queries(args.queries_csv, args.limit)

    rows: List[dict] = []
    counts = {"missing_seed": 0, "no_candidates": 0, "ok": 0}
    search_seconds = 0.0
    results = run_ordered(queries, process_query, init_fn=init_state, init_args=(args,),
                          num_workers=args.num_workers, chunksize=args.chunksize)
    for outcome in tqdm(results, total=len(queries), desc=f"Seed-only beam ({args.num_workers} workers)", unit="query"):
        counts[outcome["status"]] += 1
        search_seconds += outcome["latency"]
        rows.extend(outcome["rows"])

    predictions_csv = args.out_dir / "predictions.csv"
    write_predictions(predictions_csv, rows)
    elapsed = time.perf_counter() - started
    summary = {
        "method": "Seed-only beam search",
        "query_representation": "mean PrimeKG node2vec embedding of resolved seed nodes (one interest)",
        "queries_csv": str(args.queries_csv),
        "graph_dir": str(args.graph_dir),
        "condition_map": str(args.condition_map),
        "candidate_space": "closed (mapped DDXPlus target universe)",
        "candidate_score": "best path score per candidate",
        "max_hops": args.max_hops,
        "beam_width": args.beam_width,
        "paths_per_interest": args.paths_per_interest,
        "alpha": args.alpha,
        "beta": args.beta,
        "top_k": args.top_k,
        "num_queries": len(queries),
        "query_status_counts": counts,
        "num_prediction_rows": len(rows),
        "mean_search_seconds_per_query": search_seconds / max(len(queries), 1),
        "num_workers": args.num_workers,
        "distance_pruning": args.distance_pruning,
        "distance_weight": args.distance_weight,
        "elapsed_seconds": round(elapsed, 3),
        "queries_per_second": round(len(queries) / max(elapsed, 1e-9), 3),
        "candidate_coverage": candidate_coverage(predictions_csv, queries),
    }
    if not args.no_eval:
        metrics = run_shared_evaluator(args.queries_csv, args.condition_map, predictions_csv, args.out_dir / "evaluation", args.limit)
        metrics["candidate_coverage"] = summary["candidate_coverage"]
        summary["metrics"] = metrics
        print_metrics(summary["method"], metrics)
    write_json(args.out_dir / "run_summary.json", summary)


if __name__ == "__main__":
    main()
