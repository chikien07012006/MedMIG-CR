from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import DirectMLP


EMBEDDING_DIM = 64
QUERY_DIM = EMBEDDING_DIM * 2
TOP_KS = (5, 10, 20, 50)


@dataclass
class QuerySplit:
    query_embeddings: np.ndarray
    positive_candidate_ids: list[list[int]]
    target_node_keys: list[list[str]]
    patient_indices: np.ndarray
    missing_seed_mentions: int

    @property
    def num_positive_pairs(self) -> int:
        return sum(len(targets) for targets in self.positive_candidate_ids)


class PairDataset(Dataset):
    def __init__(self, split: QuerySplit) -> None:
        self.query_embeddings = split.query_embeddings
        self.query_ids: list[int] = []
        self.positive_ids: list[int] = []
        for query_id, target_ids in enumerate(split.positive_candidate_ids):
            for target_id in target_ids:
                self.query_ids.append(query_id)
                self.positive_ids.append(target_id)

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        query_id = self.query_ids[index]
        return torch.from_numpy(self.query_embeddings[query_id]), self.positive_ids[index], query_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a direct pairwise disease ranker over PrimeKG disease embeddings."
    )
    parser.add_argument("--train_csv", type=Path, default=Path("data/processed/ddxplus_v2/train_queries.csv"))
    parser.add_argument("--valid_csv", type=Path, default=Path("data/processed/ddxplus_v2/valid_queries.csv"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/processed/ddxplus_v2/test_queries.csv"))
    parser.add_argument("--condition_map", type=Path, default=Path("data/mappings/ddxplus_v2/condition_to_primekg.json"))
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument("--disease_nodes_csv", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("Benchmark_baselines/Direct/MLP/outputs/direct_retrieval_v1"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--negative_samples", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_valid_rows", type=int, default=None)
    parser.add_argument("--max_test_rows", type=int, default=None)
    parser.add_argument("--eval_query_batch_size", type=int, default=8)
    parser.add_argument("--candidate_chunk_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available in this PyTorch environment.")
    return device


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


def load_graph_embeddings(graph_dir: Path, disease_nodes_csv: Path | None) -> tuple[np.ndarray, list[str], dict[str, int]]:
    disease_path = disease_nodes_csv or graph_dir / "disease_nodes.csv"
    with (graph_dir / "mappings" / "node2id.json").open("r", encoding="utf-8") as handle:
        node_to_id = json.load(handle)
    all_embeddings = np.load(graph_dir / "node_embeddings.npy", mmap_mode="r")
    if all_embeddings.ndim != 2 or all_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected PrimeKG embeddings with shape (num_nodes, {EMBEDDING_DIM}); got {all_embeddings.shape}"
        )

    candidate_nodes: list[str] = []
    candidate_ids: list[int] = []
    with disease_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in tqdm(csv.DictReader(handle), desc="Loading PrimeKG disease candidates", unit="node"):
            node_key = row.get("node_key", "")
            if row.get("node_type") != "disease" or not node_key.startswith("disease|"):
                raise ValueError(f"Candidate file contains a non-disease node: {node_key!r}")
            graph_id = node_to_id.get(node_key)
            if graph_id is None:
                raise ValueError(f"PrimeKG disease candidate has no graph embedding: {node_key}")
            candidate_nodes.append(node_key)
            candidate_ids.append(int(graph_id))

    if len(candidate_nodes) != len(set(candidate_nodes)):
        raise ValueError("PrimeKG disease candidate list contains duplicate node keys")
    candidate_embeddings = np.asarray(all_embeddings[np.asarray(candidate_ids, dtype=np.int64)], dtype=np.float32)
    if candidate_embeddings.shape != (len(candidate_nodes), EMBEDDING_DIM):
        raise ValueError(f"Unexpected disease candidate embedding shape: {candidate_embeddings.shape}")
    return candidate_embeddings, candidate_nodes, node_to_id


def mapped_target_count(condition_map_path: Path) -> int:
    with condition_map_path.open("r", encoding="utf-8-sig") as handle:
        mapping = json.load(handle)
    return len({node for entry in mapping.values() for node in entry.get("selected_primekg_nodes", [])})


def validate_candidate_space(candidate_nodes: list[str], mapped_target_count_value: int) -> None:
    candidate_count = len(candidate_nodes)
    if candidate_count <= max(50, mapped_target_count_value):
        raise RuntimeError(
            "Invalid candidate space: expected the full PrimeKG disease-node list, got "
            f"{candidate_count} candidates for {mapped_target_count_value} mapped targets. "
            "Refusing to run a closed-set DDXPlus-class classifier."
        )
    if any(not node.startswith("disease|") for node in candidate_nodes):
        raise RuntimeError("Candidate space must contain only PrimeKG disease nodes")


def build_query_split(
    path: Path,
    limit: int | None,
    candidate_to_id: dict[str, int],
    node_to_id: dict[str, int],
    all_embeddings: np.ndarray,
    description: str,
) -> QuerySplit:
    row_count = csv_row_count(path, limit)
    query_embeddings = np.empty((row_count, QUERY_DIM), dtype=np.float32)
    positive_candidate_ids: list[list[int]] = []
    target_node_keys: list[list[str]] = []
    patient_indices = np.empty(row_count, dtype=np.int64)
    missing_seed_mentions = 0

    for row_index, row in enumerate(query_rows(path, limit, description)):
        patient_index = int(row["patient_index"])
        patient_indices[row_index] = patient_index
        seed_keys = [key for key in row.get("seed_node_keys", "").split(";") if key]
        seed_ids: list[int] = []
        for key in seed_keys:
            graph_id = node_to_id.get(key)
            if graph_id is None:
                missing_seed_mentions += 1
            else:
                seed_ids.append(int(graph_id))
        if not seed_ids:
            raise ValueError(f"Query {patient_index} has no valid PrimeKG seed embedding")

        seed_vectors = np.asarray(all_embeddings[np.asarray(seed_ids, dtype=np.int64)], dtype=np.float32)
        if seed_vectors.ndim != 2 or seed_vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Invalid seed embedding shape for query {patient_index}: {seed_vectors.shape}")
        query_embeddings[row_index, :EMBEDDING_DIM] = seed_vectors.mean(axis=0)
        query_embeddings[row_index, EMBEDDING_DIM:] = seed_vectors.max(axis=0)

        targets = [key for key in row.get("target_node_keys", "").split(";") if key]
        if not targets:
            raise ValueError(f"Query {patient_index} has no target_node_keys")
        target_node_keys.append(targets)
        positive_candidate_ids.append(sorted({candidate_to_id[key] for key in targets if key in candidate_to_id}))

    if len(positive_candidate_ids) != row_count:
        raise RuntimeError(f"Read {len(positive_candidate_ids)} rows from {path}, expected {row_count}")
    return QuerySplit(
        query_embeddings,
        positive_candidate_ids,
        target_node_keys,
        patient_indices,
        missing_seed_mentions,
    )


def sample_negative_ids(
    query_ids: np.ndarray,
    positive_ids: np.ndarray,
    split: QuerySplit,
    candidate_count: int,
    negative_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, candidate_count, size=(len(query_ids), negative_count), dtype=np.int64)
    for row_index, query_id in enumerate(query_ids.tolist()):
        forbidden = set(split.positive_candidate_ids[query_id])
        collisions = np.isin(negatives[row_index], list(forbidden))
        while collisions.any():
            negatives[row_index, collisions] = rng.integers(0, candidate_count, size=int(collisions.sum()))
            collisions = np.isin(negatives[row_index], list(forbidden))
    if np.any(negatives == positive_ids[:, None]):
        raise RuntimeError("Negative sampler emitted a positive disease candidate")
    return negatives


def train_epoch(
    model: DirectMLP,
    split: QuerySplit,
    pair_loader: DataLoader,
    disease_embeddings: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    negative_count: int,
    rng: np.random.Generator,
    epoch: int,
    epochs: int,
) -> tuple[float, int, int, float]:
    model.train()
    total_loss = 0.0
    positive_pairs_seen = 0
    negatives_seen = 0
    started_at = time.perf_counter()
    progress = tqdm(pair_loader, desc=f"Train {epoch}/{epochs}", unit="pair-batch", leave=False)
    for query_embeddings, positive_ids, query_ids in progress:
        query_embeddings = query_embeddings.to(device=device, dtype=torch.float32)
        positive_ids_np = positive_ids.numpy()
        positive_ids = positive_ids.to(device=device, dtype=torch.long)
        query_ids_np = query_ids.numpy()
        negative_ids_np = sample_negative_ids(
            query_ids_np,
            positive_ids_np,
            split,
            len(disease_embeddings),
            negative_count,
            rng,
        )
        negative_ids = torch.as_tensor(negative_ids_np, dtype=torch.long, device=device)
        positive_emb = disease_embeddings[positive_ids]
        negative_emb = disease_embeddings[negative_ids]
        positive_scores = model(query_embeddings, positive_emb)
        negative_queries = query_embeddings[:, None, :].expand(-1, negative_count, -1)
        negative_scores = model(negative_queries, negative_emb)
        loss = F.softplus(-(positive_scores[:, None] - negative_scores)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_pairs = int(len(positive_ids))
        total_loss += float(loss.detach().item()) * batch_pairs
        positive_pairs_seen += batch_pairs
        negatives_seen += batch_pairs * negative_count
        progress.set_postfix(bpr=f"{float(loss.detach().item()):.4f}")

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    return total_loss / max(positive_pairs_seen, 1), positive_pairs_seen, negatives_seen, elapsed


def rank_split(
    model: DirectMLP,
    split: QuerySplit,
    disease_embeddings: torch.Tensor,
    candidate_nodes: list[str],
    device: torch.device,
    query_batch_size: int,
    candidate_chunk_size: int,
    predictions_path: Path | None = None,
    by_patient_path: Path | None = None,
    description: str = "Ranking candidates",
) -> dict[str, object]:
    model.eval()
    query_count = len(split.patient_indices)
    candidate_count = len(candidate_nodes)
    predictions_handle = predictions_path.open("w", newline="", encoding="utf-8") if predictions_path else None
    patient_handle = by_patient_path.open("w", newline="", encoding="utf-8") if by_patient_path else None
    predictions_writer = None
    patient_writer = None
    if predictions_handle:
        predictions_writer = csv.DictWriter(
            predictions_handle, fieldnames=["patient_index", "candidate", "score", "rank"]
        )
        predictions_writer.writeheader()
    patient_columns = [
        "patient_index", "num_targets", "num_targets_in_candidates",
        "target_candidate_coverage", "num_predictions", "first_hit_rank", "mrr",
        "candidate_recall", "recall@5", "recall@10", "recall@20", "recall@50",
    ]
    if patient_handle:
        patient_writer = csv.DictWriter(patient_handle, fieldnames=patient_columns)
        patient_writer.writeheader()

    reciprocal_ranks: list[float] = []
    hit_counts = {k: 0 for k in TOP_KS}
    covered_queries = 0
    target_count = 0
    covered_target_count = 0
    missing_target_nodes = 0
    query_results: list[dict[str, object]] = []
    started_at = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, query_count, query_batch_size), desc=description, unit="query-batch"):
            batch_end = min(batch_start + query_batch_size, query_count)
            query_batch = torch.as_tensor(
                split.query_embeddings[batch_start:batch_end], dtype=torch.float32, device=device
            )
            batch_size = len(query_batch)
            all_scores = torch.empty((batch_size, candidate_count), dtype=torch.float32, device=device)
            for candidate_start in range(0, candidate_count, candidate_chunk_size):
                candidate_end = min(candidate_start + candidate_chunk_size, candidate_count)
                disease_batch = disease_embeddings[candidate_start:candidate_end]
                all_scores[:, candidate_start:candidate_end] = model(
                    query_batch[:, None, :], disease_batch[None, :, :]
                )

            top_scores, top_ids = torch.topk(all_scores, k=min(50, candidate_count), dim=1)
            top_scores_np = top_scores.cpu().numpy()
            top_ids_np = top_ids.cpu().numpy()
            scores_np = all_scores.cpu().numpy()
            for local_index, absolute_index in enumerate(range(batch_start, batch_end)):
                targets = split.target_node_keys[absolute_index]
                candidate_target_ids = split.positive_candidate_ids[absolute_index]
                candidate_target_set = set(candidate_target_ids)
                total_targets = len(set(targets))
                target_count += total_targets
                covered_target_count += len(candidate_target_set)
                missing_target_nodes += total_targets - len(candidate_target_set)
                query_covered = bool(candidate_target_set)
                covered_queries += int(query_covered)

                first_hit_rank: int | None = None
                if candidate_target_ids:
                    row_scores = scores_np[local_index]
                    ranks = []
                    for target_id in candidate_target_ids:
                        target_score = row_scores[target_id]
                        higher = int(np.count_nonzero(row_scores > target_score))
                        earlier_ties = int(np.count_nonzero((row_scores == target_score) & (np.arange(candidate_count) < target_id)))
                        ranks.append(higher + earlier_ties + 1)
                    first_hit_rank = min(ranks)
                reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0
                reciprocal_ranks.append(reciprocal_rank)
                row_metrics = {
                    "patient_index": int(split.patient_indices[absolute_index]),
                    "num_targets": total_targets,
                    "num_targets_in_candidates": len(candidate_target_set),
                    "target_candidate_coverage": len(candidate_target_set) / max(total_targets, 1),
                    "num_predictions": min(50, candidate_count),
                    "first_hit_rank": first_hit_rank if first_hit_rank is not None else "",
                    "mrr": reciprocal_rank,
                    "candidate_recall": int(query_covered),
                }
                for k in TOP_KS:
                    hit = int(first_hit_rank is not None and first_hit_rank <= k)
                    hit_counts[k] += hit
                    row_metrics[f"recall@{k}"] = hit
                query_results.append(row_metrics)

                if predictions_writer:
                    for rank, (candidate_id, score) in enumerate(
                        zip(top_ids_np[local_index], top_scores_np[local_index]), start=1
                    ):
                        predictions_writer.writerow({
                            "patient_index": int(split.patient_indices[absolute_index]),
                            "candidate": candidate_nodes[int(candidate_id)],
                            "score": f"{float(score):.8f}",
                            "rank": rank,
                        })
                if patient_writer:
                    patient_writer.writerow(row_metrics)
    if predictions_handle:
        predictions_handle.close()
    if patient_handle:
        patient_handle.close()

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    metrics: dict[str, object] = {
        "num_evaluated": query_count,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        **{f"recall@{k}": hit_counts[k] / max(query_count, 1) for k in TOP_KS},
        "candidate_recall": covered_queries / max(query_count, 1),
        "candidate_recall_definition": "Fraction of queries with at least one mapped target in the full PrimeKG disease-node candidate set; no ground-truth candidates are inserted.",
        "candidate_recall_at_k": None,
        "candidate_recall_at_k_note": "Not applicable: this baseline scores the complete disease candidate set and has no separate pre-ranking candidate generator.",
        "num_queries_with_target_in_candidates": covered_queries,
        "num_targets_total": target_count,
        "num_targets_in_candidates": covered_target_count,
        "num_target_nodes_absent_from_candidates": missing_target_nodes,
        "elapsed_seconds": round(elapsed, 3),
        "queries_per_second": round(query_count / elapsed, 3),
    }
    return {"summary": metrics, "by_patient": query_results}


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.negative_samples < 1:
        raise ValueError("epochs, batch_size, and negative_samples must all be positive")
    if args.eval_query_batch_size < 1 or args.candidate_chunk_size < 1:
        raise ValueError("evaluation batch and candidate chunk sizes must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    run_started = time.perf_counter()

    disease_embeddings_np, candidate_nodes, node_to_id = load_graph_embeddings(args.graph_dir, args.disease_nodes_csv)
    mapped_targets = mapped_target_count(args.condition_map)
    validate_candidate_space(candidate_nodes, mapped_targets)
    candidate_to_id = {node: index for index, node in enumerate(candidate_nodes)}
    test_count = csv_row_count(args.test_csv, args.max_test_rows)
    disease_embeddings = torch.as_tensor(disease_embeddings_np, dtype=torch.float32, device=device)
    feature_memory = np.load(args.graph_dir / "node_embeddings.npy", mmap_mode="r")

    train_split = build_query_split(
        args.train_csv, args.max_train_rows, candidate_to_id, node_to_id, feature_memory, "Building train representations"
    )
    valid_split = build_query_split(
        args.valid_csv, args.max_valid_rows, candidate_to_id, node_to_id, feature_memory, "Building validation representations"
    )
    train_dataset = PairDataset(train_split)
    if len(train_dataset) == 0:
        raise ValueError("No train ground-truth disease targets occur in the PrimeKG disease candidate set")
    if len(valid_split.patient_indices) == 0:
        raise ValueError("Validation split is empty; validation MRR is required for checkpoint selection")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = DirectMLP(
        query_dim=QUERY_DIM,
        disease_dim=EMBEDDING_DIM,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_valid_mrr = float("-inf")
    best_valid_candidate_recall = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    total_negatives_sampled = 0
    pair_count = len(train_dataset)
    for epoch in range(1, args.epochs + 1):
        train_loss, positive_pairs_seen, negatives_seen, epoch_seconds = train_epoch(
            model,
            train_split,
            train_loader,
            disease_embeddings,
            optimizer,
            device,
            args.negative_samples,
            rng,
            epoch,
            args.epochs,
        )
        total_negatives_sampled += negatives_seen
        validation_result = rank_split(
            model,
            valid_split,
            disease_embeddings,
            candidate_nodes,
            device,
            args.eval_query_batch_size,
            args.candidate_chunk_size,
            description="Validation full-candidate ranking",
        )
        valid_mrr = float(validation_result["summary"]["mrr"])
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_bpr_loss": train_loss,
            "valid_mrr": valid_mrr,
            "valid_candidate_recall": float(validation_result["summary"]["candidate_recall"]),
            "train_positive_pairs": positive_pairs_seen,
            "negative_pairs_sampled": negatives_seen,
            "epoch_seconds": epoch_seconds,
            "positive_pairs_per_second": positive_pairs_seen / max(epoch_seconds, 1e-9),
        }
        history.append(record)
        print(json.dumps(record))

        if valid_mrr > best_valid_mrr + args.early_stopping_min_delta:
            best_valid_mrr = valid_mrr
            best_valid_candidate_recall = float(validation_result["summary"]["candidate_recall"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(json.dumps({"early_stopped": True, "selection_metric": "validation_mrr", "best_epoch": best_epoch}))
                break

    if best_state is None:
        raise RuntimeError("Training produced no validation-selected checkpoint")
    model.load_state_dict(best_state)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "mlp_baseline.pt"
    model_config = {
        "query_dim": QUERY_DIM,
        "disease_dim": EMBEDDING_DIM,
        "hidden_dim": args.hidden_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "dropout": args.dropout,
        "pooling": "concatenated mean and max over seed-node embeddings",
        "ranking_loss": "pairwise BPR logistic loss",
        "candidate_source": str(args.disease_nodes_csv or (args.graph_dir / "disease_nodes.csv")),
        "candidate_space_size": len(candidate_nodes),
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.save({
        "model_state": best_state,
        "model_config": model_config,
        "candidate_nodes": candidate_nodes,
        "best_epoch": best_epoch,
        "best_valid_mrr": best_valid_mrr,
        "embedding_dim": EMBEDDING_DIM,
    }, checkpoint_path)

    training_summary: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "train_csv": str(args.train_csv),
        "valid_csv": str(args.valid_csv),
        "device": str(device),
        "candidate_source": model_config["candidate_source"],
        "candidate_space_size": len(candidate_nodes),
        "num_train_queries": len(train_split.patient_indices),
        "num_valid_queries": len(valid_split.patient_indices),
        "num_test_queries": test_count,
        "num_positive_pairs": pair_count,
        "positive_pairs_per_epoch": pair_count,
        "num_valid_positive_pairs": valid_split.num_positive_pairs,
        "negative_samples_per_positive": args.negative_samples,
        "number_of_negatives_sampled": total_negatives_sampled,
        "embedding_dimension": EMBEDDING_DIM,
        "query_representation_dimension": QUERY_DIM,
        "pooling_method": "mean(seed embeddings) concatenated with max(seed embeddings)",
        "ranking_loss": "pairwise BPR logistic loss: mean(softplus(-(s(q,d+) - s(q,d-))))",
        "validation_mrr": best_valid_mrr,
        "validation_candidate_recall": best_valid_candidate_recall,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "trainable_parameter_count": parameter_count,
        "candidate_space_validation": "passed: full PrimeKG disease_nodes.csv, larger than 50 and mapped-target set",
        "pathology_used_as_input": False,
        "graph_traversal_used": False,
        "history": history,
        "unknown_train_seed_mentions": train_split.missing_seed_mentions,
        "unknown_valid_seed_mentions": valid_split.missing_seed_mentions,
    }
    with (args.out_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps({
        "stage": "direct_disease_ranker_training",
        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
        "candidate_space_size": len(candidate_nodes),
        "train_queries": len(train_split.patient_indices),
        "positive_pairs": pair_count,
        "negatives_sampled": total_negatives_sampled,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if device.type == "cuda" else None,
    }))

    del train_loader, train_dataset, train_split, valid_split
    test_split = build_query_split(args.test_csv, args.max_test_rows, candidate_to_id, node_to_id, feature_memory, "Building test representations")
    test_result = rank_split(
        model,
        test_split,
        disease_embeddings,
        candidate_nodes,
        device,
        args.eval_query_batch_size,
        args.candidate_chunk_size,
        predictions_path=args.out_dir / "predictions.csv",
        by_patient_path=args.out_dir / "by_patient.csv",
        description="Test full-candidate ranking",
    )
    summary = dict(test_result["summary"])
    summary.update({
        "candidate_source": model_config["candidate_source"],
        "candidate_space_size": len(candidate_nodes),
        "num_train_queries": training_summary["num_train_queries"],
        "num_valid_queries": training_summary["num_valid_queries"],
        "num_test_queries": len(test_split.patient_indices),
        "num_positive_pairs": pair_count,
        "number_of_negatives_sampled": total_negatives_sampled,
        "embedding_dimension": EMBEDDING_DIM,
        "query_representation_dimension": QUERY_DIM,
        "pooling_method": model_config["pooling"],
        "ranking_loss": model_config["ranking_loss"],
        "parameter_count": parameter_count,
        "trainable_parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "unknown_test_seed_mentions": test_split.missing_seed_mentions,
        "pathology_used_as_input": False,
        "graph_traversal_used": False,
        "candidate_space_validation": "passed: full PrimeKG disease_nodes.csv, larger than 50 and mapped-target set",
    })
    with (args.out_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
