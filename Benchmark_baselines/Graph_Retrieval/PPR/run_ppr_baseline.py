from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]

TOP_KS = (5, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run personalized PageRank for DDXPlus-to-PrimeKG disease retrieval."
    )
    parser.add_argument("--train_csv", type=Path, default=Path("data/processed/ddxplus_v2/train_queries.csv"))
    parser.add_argument("--valid_csv", type=Path, default=Path("data/processed/ddxplus_v2/valid_queries.csv"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/processed/ddxplus_v2/test_queries.csv"))
    parser.add_argument("--condition_map", type=Path, default=Path("data/mappings/ddxplus_v2/condition_to_primekg.json"))
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument("--disease_nodes_csv", type=Path, default=None)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("Benchmark_baselines/Graph_Retrieval/PPR/outputs/ppr_v1"),
    )
    parser.add_argument("--restart_probability", type=float, default=0.15)
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_valid_rows", type=int, default=None)
    parser.add_argument("--max_test_rows", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help="Evaluate only validation queries; do not read or score the test split.",
    )
    return parser.parse_args()


def query_rows(path: Path, limit: int | None, description: str) -> Iterator[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = islice(reader, limit) if limit is not None else reader
        yield from tqdm(rows, total=limit, desc=description, unit="query")


def csv_row_count(path: Path, limit: int | None) -> int:
    with path.open("rb") as handle:
        if limit is None:
            return max(sum(1 for _ in handle) - 1, 0)
        return max(sum(1 for _ in islice(handle, limit + 1)) - 1, 0)


def load_graph(graph_dir: Path) -> tuple[sp.csr_matrix, dict[str, int]]:
    graph_npz = graph_dir / "graph_csr.npz"
    mappings_dir = graph_dir / "mappings"
    with np.load(graph_npz) as arrays:
        indptr = arrays["indptr"].astype(np.int64, copy=False)
        indices = arrays["indices"].astype(np.int64, copy=False)
        data = arrays["data"].astype(np.float32, copy=False)
    node_count = len(indptr) - 1
    adjacency = sp.csr_matrix((data, indices, indptr), shape=(node_count, node_count))
    adjacency.data.fill(1.0)
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data.fill(1.0)

    with (mappings_dir / "node2id.json").open("r", encoding="utf-8") as handle:
        node_to_id = json.load(handle)
    if len(node_to_id) != node_count:
        raise ValueError(f"Graph has {node_count} nodes but node2id has {len(node_to_id)} entries")
    return adjacency, node_to_id


def load_disease_candidates(
    graph_dir: Path,
    disease_nodes_csv: Path | None,
    node_to_id: dict[str, int],
) -> tuple[list[str], np.ndarray]:
    path = disease_nodes_csv or graph_dir / "disease_nodes.csv"
    node_keys: list[str] = []
    node_ids: list[int] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in tqdm(csv.DictReader(handle), desc="Loading disease candidates", unit="node"):
            key = row.get("node_key", "")
            if row.get("node_type") != "disease" or not key.startswith("disease|"):
                raise ValueError(f"Non-disease entry in disease candidate file: {key!r}")
            graph_id = node_to_id.get(key)
            if graph_id is None:
                raise ValueError(f"Candidate {key} is not in the PrimeKG graph index")
            node_keys.append(key)
            node_ids.append(int(graph_id))
    if not node_keys or len(node_keys) != len(set(node_keys)):
        raise ValueError("PrimeKG disease candidate file is empty or contains duplicate nodes")
    return node_keys, np.asarray(node_ids, dtype=np.int64)


def mapped_target_count(condition_map_path: Path) -> int:
    with condition_map_path.open("r", encoding="utf-8-sig") as handle:
        mapping = json.load(handle)
    return len({node for entry in mapping.values() for node in entry.get("selected_primekg_nodes", [])})


def validate_candidate_space(candidate_nodes: list[str], mapped_targets: int) -> None:
    if len(candidate_nodes) <= max(50, mapped_targets):
        raise RuntimeError(
            "Candidate space is too small for PrimeKG retrieval; refusing a DDXPlus-class-only candidate set "
            f"({len(candidate_nodes)} candidates; mapped targets={mapped_targets})."
        )
    if any(not key.startswith("disease|") for key in candidate_nodes):
        raise RuntimeError("PPR candidates must be PrimeKG disease nodes")


def personalized_pagerank(
    transition: sp.csr_matrix,
    seed_ids: np.ndarray,
    restart_probability: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int, bool]:
    node_count = transition.shape[0]
    personalization = np.zeros(node_count, dtype=np.float64)
    unique_seeds = np.unique(seed_ids)
    personalization[unique_seeds] = 1.0 / len(unique_seeds)
    rank = personalization.copy()
    dangling = np.asarray(transition.getnnz(axis=1) == 0)

    for iteration in range(1, max_iterations + 1):
        dangling_mass = float(rank[dangling].sum()) if np.any(dangling) else 0.0
        updated = (
            restart_probability * personalization
            + (1.0 - restart_probability)
            * (transition.transpose().dot(rank) + dangling_mass * personalization)
        )
        delta = float(np.abs(updated - rank).sum())
        rank = updated
        if delta <= tolerance:
            return rank, iteration, True
    return rank, max_iterations, False


def evaluate_rankings(
    query_path: Path,
    row_limit: int | None,
    node_to_id: dict[str, int],
    candidate_nodes: list[str],
    candidate_graph_ids: np.ndarray,
    transition: sp.csr_matrix,
    restart_probability: float,
    max_iterations: int,
    tolerance: float,
    top_k: int,
    output_dir: Path | None,
    description: str,
) -> dict[str, object]:
    candidate_to_position = {key: index for index, key in enumerate(candidate_nodes)}
    expected_rows = csv_row_count(query_path, row_limit)
    candidate_count = len(candidate_nodes)
    output_predictions = output_dir / "predictions.csv" if output_dir else None
    output_by_patient = output_dir / "by_patient.csv" if output_dir else None
    predictions_handle = output_predictions.open("w", newline="", encoding="utf-8") if output_predictions else None
    patient_handle = output_by_patient.open("w", newline="", encoding="utf-8") if output_by_patient else None
    prediction_writer = None
    patient_writer = None
    if predictions_handle:
        prediction_writer = csv.DictWriter(
            predictions_handle, fieldnames=["patient_index", "candidate", "score", "rank"]
        )
        prediction_writer.writeheader()
    patient_columns = [
        "patient_index", "num_targets", "num_targets_in_candidates", "target_candidate_coverage",
        "num_predictions", "first_hit_rank", "mrr", "candidate_recall",
        "recall@5", "recall@10", "recall@20", "recall@50",
    ]
    if patient_handle:
        patient_writer = csv.DictWriter(patient_handle, fieldnames=patient_columns)
        patient_writer.writeheader()

    reciprocal_ranks: list[float] = []
    hit_counts = {k: 0 for k in TOP_KS}
    query_candidate_hits = 0
    target_total = 0
    target_covered = 0
    absent_target_nodes = 0
    total_iterations = 0
    converged_queries = 0
    processed = 0
    started_at = time.perf_counter()
    for row in query_rows(query_path, row_limit, description):
        processed += 1
        patient_index = int(row["patient_index"])
        seed_keys = [key for key in row.get("seed_node_keys", "").split(";") if key]
        seed_ids = [node_to_id[key] for key in seed_keys if key in node_to_id]
        if not seed_ids:
            if predictions_handle:
                predictions_handle.close()
            if patient_handle:
                patient_handle.close()
            raise ValueError(f"Query {patient_index} has no valid PrimeKG seed node")

        scores, iterations, converged = personalized_pagerank(
            transition, np.asarray(seed_ids, dtype=np.int64), restart_probability, max_iterations, tolerance
        )
        total_iterations += iterations
        converged_queries += int(converged)
        candidate_scores = scores[candidate_graph_ids]
        candidate_order = np.lexsort((np.arange(candidate_count), -candidate_scores))
        top_positions = candidate_order[: min(top_k, candidate_count)]

        targets = [key for key in row.get("target_node_keys", "").split(";") if key]
        if not targets:
            raise ValueError(f"Query {patient_index} has no target_node_keys")
        covered_target_positions = sorted({candidate_to_position[key] for key in targets if key in candidate_to_position})
        n_targets = len(set(targets))
        n_covered = len(covered_target_positions)
        target_total += n_targets
        target_covered += n_covered
        absent_target_nodes += n_targets - n_covered
        candidate_hit = n_covered > 0
        query_candidate_hits += int(candidate_hit)

        first_hit_rank: int | None = None
        if covered_target_positions:
            target_scores = candidate_scores[covered_target_positions]
            for target_position, target_score in zip(covered_target_positions, target_scores):
                rank = int(np.count_nonzero(candidate_scores > target_score))
                rank += int(np.count_nonzero((candidate_scores == target_score) & (np.arange(candidate_count) < target_position)))
                candidate_rank = rank + 1
                first_hit_rank = candidate_rank if first_hit_rank is None else min(first_hit_rank, candidate_rank)
        reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank else 0.0
        reciprocal_ranks.append(reciprocal_rank)

        row_metrics: dict[str, object] = {
            "patient_index": patient_index,
            "num_targets": n_targets,
            "num_targets_in_candidates": n_covered,
            "target_candidate_coverage": n_covered / max(n_targets, 1),
            "num_predictions": len(top_positions),
            "first_hit_rank": first_hit_rank if first_hit_rank is not None else "",
            "mrr": reciprocal_rank,
            "candidate_recall": int(candidate_hit),
        }
        for k in TOP_KS:
            hit = int(first_hit_rank is not None and first_hit_rank <= k)
            hit_counts[k] += hit
            row_metrics[f"recall@{k}"] = hit
        if patient_writer:
            patient_writer.writerow(row_metrics)
        if prediction_writer:
            for rank, candidate_position in enumerate(top_positions, start=1):
                prediction_writer.writerow({
                    "patient_index": patient_index,
                    "candidate": candidate_nodes[int(candidate_position)],
                    "score": f"{candidate_scores[candidate_position]:.12g}",
                    "rank": rank,
                })

    if predictions_handle:
        predictions_handle.close()
    if patient_handle:
        patient_handle.close()
    if processed != expected_rows:
        raise RuntimeError(f"Processed {processed} queries from {query_path}, expected {expected_rows}")
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    summary: dict[str, object] = {
        "queries_csv": str(query_path),
        "num_evaluated": processed,
        "candidate_space_size": candidate_count,
        "candidate_recall": query_candidate_hits / max(processed, 1),
        "candidate_recall_definition": "Fraction of queries with at least one mapped target in the full PrimeKG disease candidate set; targets are never inserted.",
        "candidate_recall_at_k": None,
        "candidate_recall_at_k_note": "Not applicable: PPR ranks the full candidate set and has no separate pre-ranking candidate generator.",
        "num_queries_with_target_in_candidates": query_candidate_hits,
        "num_targets_total": target_total,
        "num_targets_in_candidates": target_covered,
        "num_target_nodes_absent_from_candidates": absent_target_nodes,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        **{f"recall@{k}": hit_counts[k] / max(processed, 1) for k in TOP_KS},
        "restart_probability": restart_probability,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "mean_iterations": total_iterations / max(processed, 1),
        "converged_queries": converged_queries,
        "elapsed_seconds": round(elapsed, 3),
        "queries_per_second": round(processed / elapsed, 3),
    }
    if output_dir:
        summary["predictions_csv"] = str(output_predictions)
        with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def build_transition(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
    inverse_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    transition = sp.diags(inverse_degree, format="csr") @ adjacency.astype(np.float64)
    return transition.tocsr()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.restart_probability < 1.0:
        raise ValueError("restart_probability must be between 0 and 1")
    if args.max_iterations < 1 or args.tolerance <= 0 or args.top_k < 1:
        raise ValueError("max_iterations, tolerance, and top_k must be positive")

    run_started = time.perf_counter()
    adjacency, node_to_id = load_graph(args.graph_dir)
    candidate_nodes, candidate_graph_ids = load_disease_candidates(
        args.graph_dir, args.disease_nodes_csv, node_to_id
    )
    mapped_targets = mapped_target_count(args.condition_map)
    validate_candidate_space(candidate_nodes, mapped_targets)
    transition = build_transition(adjacency)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_positive_pairs = 0
    candidate_node_set = set(candidate_nodes)
    for row in query_rows(args.train_csv, args.max_train_rows, "Auditing training targets"):
        targets = [key for key in row.get("target_node_keys", "").split(";") if key]
        train_positive_pairs += sum(key in candidate_node_set for key in targets)
    valid_summary = evaluate_rankings(
        args.valid_csv,
        args.max_valid_rows,
        node_to_id,
        candidate_nodes,
        candidate_graph_ids,
        transition,
        args.restart_probability,
        args.max_iterations,
        args.tolerance,
        args.top_k,
        None,
        "PPR validation metrics",
    )
    test_summary = None
    if not args.validation_only:
        test_summary = evaluate_rankings(
            args.test_csv,
            args.max_test_rows,
            node_to_id,
            candidate_nodes,
            candidate_graph_ids,
            transition,
            args.restart_probability,
            args.max_iterations,
            args.tolerance,
            args.top_k,
            args.out_dir,
            "PPR test ranking",
        )
    summary = {
        "method": "Personalized PageRank",
        "graph_dir": str(args.graph_dir),
        "candidate_source": str(args.disease_nodes_csv or (args.graph_dir / "disease_nodes.csv")),
        "candidate_space_size": len(candidate_nodes),
        "graph_nodes": adjacency.shape[0],
        "graph_edges_undirected": int(adjacency.nnz // 2),
        "graph_direction": "undirected (PrimeKG edge union with transpose; aligns with retrieval's incoming/outgoing expansion)",
        "query_personalization": "uniform probability over valid seed_node_keys",
        "restart_probability": args.restart_probability,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
        "num_train_queries": csv_row_count(args.train_csv, args.max_train_rows),
        "num_valid_queries": csv_row_count(args.valid_csv, args.max_valid_rows),
        "num_test_queries": None if args.validation_only else csv_row_count(args.test_csv, args.max_test_rows),
        "num_train_positive_pairs_in_candidate_space": train_positive_pairs,
        "ranking_loss": "none (non-parametric graph baseline)",
        "graph_traversal_model_training": False,
        "train_positive_pairs_in_candidate_space": train_positive_pairs,
        "valid_metrics": valid_summary,
        "test_metrics": test_summary,
        "validation_only": args.validation_only,
        "elapsed_seconds_including_split_metrics": round(time.perf_counter() - run_started, 3),
    }
    with (args.out_dir / "ppr_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
