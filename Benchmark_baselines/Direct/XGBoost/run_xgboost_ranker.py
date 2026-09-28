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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
EMBEDDING_DIM = 64
QUERY_DIM = 2 * EMBEDDING_DIM
PAIR_FEATURE_DIM = QUERY_DIM + 3 * EMBEDDING_DIM
TOP_KS = (5, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a direct XGBoost disease ranker over all PrimeKG disease candidates."
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
        default=Path("Benchmark_baselines/Direct/XGBoost/outputs/profile_v1"),
    )
    parser.add_argument("--max_train_rows", type=int, default=5000)
    parser.add_argument("--max_valid_rows", type=int, default=1000)
    parser.add_argument("--max_test_rows", type=int, default=None)
    parser.add_argument("--max_eval_queries", type=int, default=200)
    parser.add_argument("--negative_samples", type=int, default=20)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--candidate_chunk_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--run_test", action="store_true", help="Also score test queries; off for validation profiling.")
    return parser.parse_args()


def rows(path: Path, limit: int | None, description: str) -> Iterator[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        iterable = islice(reader, limit) if limit is not None else reader
        yield from tqdm(iterable, total=limit, desc=description, unit="query")


def count_rows(path: Path, limit: int | None) -> int:
    with path.open("rb") as handle:
        if limit is None:
            return max(sum(1 for _ in handle) - 1, 0)
        return max(sum(1 for _ in islice(handle, limit + 1)) - 1, 0)


def load_graph_data(graph_dir: Path, disease_nodes_csv: Path | None) -> tuple[np.ndarray, list[str], dict[str, int]]:
    with (graph_dir / "mappings" / "node2id.json").open("r", encoding="utf-8") as handle:
        node_to_id = json.load(handle)
    all_embeddings = np.load(graph_dir / "node_embeddings.npy", mmap_mode="r")
    if all_embeddings.ndim != 2 or all_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected graph embeddings of dimension {EMBEDDING_DIM}, found {all_embeddings.shape}")

    disease_path = disease_nodes_csv or graph_dir / "disease_nodes.csv"
    candidate_nodes: list[str] = []
    candidate_ids: list[int] = []
    with disease_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in tqdm(csv.DictReader(handle), desc="Loading PrimeKG diseases", unit="node"):
            key = row.get("node_key", "")
            if row.get("node_type") != "disease" or not key.startswith("disease|"):
                raise ValueError(f"Non-disease candidate found: {key!r}")
            node_id = node_to_id.get(key)
            if node_id is None:
                raise ValueError(f"Candidate has no PrimeKG embedding: {key}")
            candidate_nodes.append(key)
            candidate_ids.append(int(node_id))
    if len(candidate_nodes) != len(set(candidate_nodes)) or not candidate_nodes:
        raise ValueError("Disease candidate file is empty or has duplicate node keys")
    disease_embeddings = np.asarray(all_embeddings[np.asarray(candidate_ids, dtype=np.int64)], dtype=np.float32)
    if disease_embeddings.shape != (len(candidate_nodes), EMBEDDING_DIM):
        raise ValueError(f"Unexpected disease embedding shape: {disease_embeddings.shape}")
    return disease_embeddings, candidate_nodes, node_to_id


def check_candidate_space(candidate_nodes: list[str], condition_map_path: Path) -> None:
    with condition_map_path.open("r", encoding="utf-8-sig") as handle:
        mapping = json.load(handle)
    mapped_targets = {key for entry in mapping.values() for key in entry.get("selected_primekg_nodes", [])}
    if len(candidate_nodes) <= max(50, len(mapped_targets)):
        raise RuntimeError(
            f"Candidate-space guard failed ({len(candidate_nodes)} candidates, {len(mapped_targets)} mapped targets); "
            "expected the full PrimeKG disease candidate list, not a DDXPlus closed set."
        )
    if any(not key.startswith("disease|") for key in candidate_nodes):
        raise RuntimeError("Candidate space contains non-disease nodes")


def query_representation(seed_keys: list[str], node_to_id: dict[str, int], all_embeddings: np.ndarray, patient_id: int) -> np.ndarray:
    graph_ids = [node_to_id[key] for key in seed_keys if key in node_to_id]
    if not graph_ids:
        raise ValueError(f"Query {patient_id} has no valid seed node embedding")
    seed_embeddings = np.asarray(all_embeddings[np.asarray(graph_ids, dtype=np.int64)], dtype=np.float32)
    if seed_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Invalid seed embedding dimension for query {patient_id}")
    return np.concatenate((seed_embeddings.mean(axis=0), seed_embeddings.max(axis=0))).astype(np.float32)


def make_pair_features(query_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    query_mean = query_emb[:EMBEDDING_DIM]
    repeated_query = np.broadcast_to(query_emb, (len(disease_emb), QUERY_DIM))
    return np.concatenate(
        (
            repeated_query,
            disease_emb,
            np.abs(query_mean[None, :] - disease_emb),
            query_mean[None, :] * disease_emb,
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def build_training_matrix(
    path: Path,
    limit: int | None,
    node_to_id: dict[str, int],
    all_embeddings: np.ndarray,
    candidate_to_id: dict[str, int],
    disease_embeddings: np.ndarray,
    negative_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    group_sizes: list[int] = []
    counts = {"queries_read": 0, "queries_with_positive": 0, "queries_without_candidate_target": 0, "positive_pairs": 0}
    disease_count = len(candidate_to_id)

    for row in rows(path, limit, "Sampling training query groups"):
        counts["queries_read"] += 1
        patient_id = int(row["patient_index"])
        seeds = [key for key in row.get("seed_node_keys", "").split(";") if key]
        query_emb = query_representation(seeds, node_to_id, all_embeddings, patient_id)
        targets = list(dict.fromkeys(key for key in row.get("target_node_keys", "").split(";") if key))
        positive_ids = sorted({candidate_to_id[key] for key in targets if key in candidate_to_id})
        if not positive_ids:
            counts["queries_without_candidate_target"] += 1
            continue

        counts["queries_with_positive"] += 1
        counts["positive_pairs"] += len(positive_ids)
        forbidden = set(positive_ids)
        sampled = rng.integers(0, disease_count, size=negative_count, dtype=np.int64)
        collisions = np.isin(sampled, list(forbidden))
        while collisions.any():
            sampled[collisions] = rng.integers(0, disease_count, size=int(collisions.sum()))
            collisions = np.isin(sampled, list(forbidden))
        pair_ids = np.asarray(positive_ids + sampled.tolist(), dtype=np.int64)
        pair_embeddings = disease_embeddings[pair_ids]
        pair_features = make_pair_features(query_emb, pair_embeddings)
        feature_rows.append(pair_features)
        labels.append(np.concatenate((np.ones(len(positive_ids), dtype=np.float32), np.zeros(negative_count, dtype=np.float32))))
        group_sizes.append(len(pair_ids))

    if not feature_rows:
        raise ValueError("No train queries have a mapped target in PrimeKG's disease candidate set")
    matrix = np.concatenate(feature_rows, axis=0)
    label_vector = np.concatenate(labels)
    group_vector = np.asarray(group_sizes, dtype=np.uint32)
    counts["negative_pairs"] = int(sum(group_sizes) - counts["positive_pairs"])
    counts["training_rows"] = int(len(label_vector))
    return matrix, label_vector, group_vector, counts


def rank_split(
    model,
    path: Path,
    limit: int | None,
    node_to_id: dict[str, int],
    all_embeddings: np.ndarray,
    candidate_nodes: list[str],
    disease_embeddings: np.ndarray,
    candidate_to_id: dict[str, int],
    chunk_size: int,
    max_eval_queries: int,
    out_dir: Path,
    split_name: str,
) -> dict[str, object]:
    query_count = count_rows(path, min(limit, max_eval_queries) if limit is not None else max_eval_queries)
    predictions_path = out_dir / f"{split_name}_predictions.csv"
    by_patient_path = out_dir / f"{split_name}_by_patient.csv"
    candidate_count = len(candidate_nodes)
    rr_sum = 0.0
    hit_counts = {k: 0 for k in (5, 10, 20, 50)}
    covered_queries = 0
    target_count = 0
    covered_target_count = 0
    absent_targets = 0
    prediction_pairs_per_second = 0.0
    start = time.perf_counter()

    with predictions_path.open("w", newline="", encoding="utf-8") as pred_handle, by_patient_path.open(
        "w", newline="", encoding="utf-8"
    ) as patient_handle:
        pred_writer = csv.DictWriter(pred_handle, fieldnames=["patient_index", "candidate", "score", "rank"])
        pred_writer.writeheader()
        patient_columns = [
            "patient_index", "num_targets", "targets_in_candidates", "target_candidate_coverage",
            "first_hit_rank", "mrr", "candidate_recall", "recall@5", "recall@10", "recall@20", "recall@50",
        ]
        patient_writer = csv.DictWriter(patient_handle, fieldnames=patient_columns)
        patient_writer.writeheader()

        for row in rows(path, min(limit, max_eval_queries) if limit is not None else max_eval_queries, f"Ranking {split_name}"):
            patient_id = int(row["patient_index"])
            seeds = [key for key in row.get("seed_node_keys", "").split(";") if key]
            query_emb = query_representation(seeds, node_to_id, all_embeddings, patient_id)
            scores = np.empty(candidate_count, dtype=np.float32)
            for start_id in range(0, candidate_count, chunk_size):
                stop_id = min(start_id + chunk_size, candidate_count)
                chunk_features = make_pair_features(query_emb, disease_embeddings[start_id:stop_id])
                scores[start_id:stop_id] = model.predict(chunk_features, validate_features=False)
                prediction_pairs_per_second += stop_id - start_id

            order = np.argsort(-scores, kind="stable")
            targets = list(dict.fromkeys(key for key in row.get("target_node_keys", "").split(";") if key))
            target_ids = sorted({candidate_to_id[key] for key in targets if key in candidate_to_id})
            n_targets = len(targets)
            n_covered = len(target_ids)
            candidate_hit = n_covered > 0
            target_count += n_targets
            covered_target_count += n_covered
            absent_targets += n_targets - n_covered
            covered_queries += int(candidate_hit)
            first_rank = None
            if target_ids:
                ranks = []
                for target_id in target_ids:
                    rank = int(np.searchsorted(-scores[order], -scores[target_id], side="left"))
                    rank += int(np.count_nonzero((scores == scores[target_id]) & (order < target_id)))
                    ranks.append(rank + 1)
                first_rank = min(ranks)
            reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
            rr_sum += reciprocal_rank
            row_metrics: dict[str, object] = {
                "patient_index": patient_id,
                "num_targets": n_targets,
                "targets_in_candidates": n_covered,
                "target_candidate_coverage": n_covered / max(n_targets, 1),
                "first_hit_rank": first_rank if first_rank is not None else "",
                "mrr": reciprocal_rank,
                "candidate_recall": int(candidate_hit),
            }
            for k in TOP_KS:
                hit = int(first_rank is not None and first_rank <= k)
                hit_counts[k] += hit
                row_metrics[f"recall@{k}"] = hit
            patient_writer.writerow(row_metrics)
            for rank, candidate_id in enumerate(order[: min(50, candidate_count)], start=1):
                pred_writer.writerow({
                    "patient_index": patient_id,
                    "candidate": candidate_nodes[int(candidate_id)],
                    "score": f"{float(scores[candidate_id]):.10g}",
                    "rank": rank,
                })

    elapsed = max(time.perf_counter() - start, 1e-9)
    return {
        "split": split_name,
        "num_evaluated": query_count,
        "candidate_space_size": candidate_count,
        "mrr": rr_sum / max(query_count, 1),
        **{f"recall@{k}": hit_counts[k] / max(query_count, 1) for k in TOP_KS},
        "candidate_recall": covered_queries / max(query_count, 1),
        "candidate_recall_at_k": None,
        "num_targets_total": target_count,
        "num_targets_in_candidates": covered_target_count,
        "num_target_nodes_absent_from_candidates": absent_targets,
        "mean_candidates_scored_per_query": candidate_count,
        "candidate_pairs_scored": prediction_pairs_per_second,
        "elapsed_seconds": round(elapsed, 3),
        "queries_per_second": query_count / elapsed,
        "candidate_pairs_per_second": prediction_pairs_per_second / elapsed,
        "predictions_csv": str(predictions_path),
        "by_patient_csv": str(by_patient_path),
    }


def main() -> None:
    args = parse_args()
    try:
        from xgboost import XGBRanker
    except Exception as exc:
        raise SystemExit(
            "XGBoost could not be loaded. Install it with `python -m pip install xgboost`; "
            "on macOS also install the OpenMP runtime (`brew install libomp`). "
            f"Loader error: {exc}"
        ) from exc
    if args.max_train_rows < 1 or args.max_valid_rows < 1 or args.max_eval_queries < 1 or args.negative_samples < 1:
        raise ValueError("row limits, max_eval_queries, and negative_samples must be positive")
    if args.n_estimators < 1 or args.max_depth < 1 or args.candidate_chunk_size < 1:
        raise ValueError("n_estimators, max_depth, and candidate_chunk_size must be positive")
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    disease_embeddings, candidate_nodes, node_to_id = load_graph_data(args.graph_dir, args.disease_nodes_csv)
    check_candidate_space(candidate_nodes, args.condition_map)
    candidate_to_id = {key: index for index, key in enumerate(candidate_nodes)}
    all_embeddings = np.load(args.graph_dir / "node_embeddings.npy", mmap_mode="r")
    x_train, y_train, group_train, training_counts = build_training_matrix(
        args.train_csv,
        args.max_train_rows,
        node_to_id,
        all_embeddings,
        candidate_to_id,
        disease_embeddings,
        args.negative_samples,
        rng,
    )

    model = XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    train_started = time.perf_counter()
    model.fit(x_train, y_train, group=group_train.tolist(), verbose=False)
    train_seconds = time.perf_counter() - train_started
    model_path = args.out_dir / "xgboost_ranker.json"
    model.save_model(model_path)

    valid_metrics = rank_split(
        model,
        args.valid_csv,
        args.max_valid_rows,
        node_to_id,
        all_embeddings,
        candidate_nodes,
        disease_embeddings,
        candidate_to_id,
        args.candidate_chunk_size,
        args.max_eval_queries,
        args.out_dir,
        "validation",
    )
    with (args.out_dir / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(valid_metrics, handle, indent=2)

    test_metrics = None
    if args.run_test:
        test_metrics = rank_split(
            model,
            args.test_csv,
            args.max_test_rows,
            node_to_id,
            all_embeddings,
            candidate_nodes,
            disease_embeddings,
            candidate_to_id,
            args.candidate_chunk_size,
            args.max_eval_queries,
            args.out_dir,
            "test",
        )

    total_summary = {
        "method": "direct XGBoost pairwise ranking",
        "candidate_source": str(args.disease_nodes_csv or args.graph_dir / "disease_nodes.csv"),
        "candidate_space_size": len(candidate_nodes),
        "embedding_dimension": EMBEDDING_DIM,
        "query_pooling": "seed embedding mean and max concatenated (128-D)",
        "pair_features": "query mean/max (128), disease embedding (64), abs(mean-disease) (64), mean*disease (64)",
        "pair_feature_dimension": PAIR_FEATURE_DIM,
        "ranking_objective": "rank:pairwise",
        "num_train_queries_requested": args.max_train_rows,
        "num_train_queries_read": training_counts["queries_read"],
        "num_train_queries_with_candidate_positive": training_counts["queries_with_positive"],
        "num_train_queries_without_candidate_positive": training_counts["queries_without_candidate_target"],
        "num_positive_pairs": training_counts["positive_pairs"],
        "num_negative_pairs_sampled": training_counts["negative_pairs"],
        "training_matrix_rows": training_counts["training_rows"],
        "num_valid_queries_in_split": count_rows(args.valid_csv, args.max_valid_rows),
        "num_valid_queries_ranked": valid_metrics["num_evaluated"],
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "negative_samples_per_query": args.negative_samples,
        "training_elapsed_seconds": round(train_seconds, 3),
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "model_file": str(model_path),
        "graph_traversal_used": False,
        "path_information_used": False,
        "pathology_used_as_input": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    with (args.out_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(total_summary, handle, indent=2)
    print(json.dumps(total_summary, indent=2))


if __name__ == "__main__":
    main()
