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
from torch_geometric.nn import RGCNConv
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_DIM = 64
NUM_RELATIONS = 18
TOP_KS = (5, 10, 20, 50)


@dataclass
class QuerySplit:
    seed_ids: np.ndarray
    seed_mask: np.ndarray
    target_ids: np.ndarray
    target_mask: np.ndarray
    target_keys: list[list[str]]
    patient_indices: np.ndarray
    seed_width: int
    target_width: int
    missing_seed_mentions: int

    @property
    def size(self) -> int:
        return int(len(self.patient_indices))


class RGCNRanker(nn.Module):
    def __init__(self, initial_embeddings: np.ndarray, num_relations: int = NUM_RELATIONS, num_bases: int = 9) -> None:
        super().__init__()
        initial = torch.as_tensor(initial_embeddings, dtype=torch.float32)
        self.node_features = nn.Embedding.from_pretrained(initial, freeze=False)
        self.conv1 = RGCNConv(EMBEDDING_DIM, 128, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(128, EMBEDDING_DIM, num_relations=num_relations, num_bases=num_bases)
        self.scorer = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def encode_graph(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        features = self.node_features.weight
        hidden = F.relu(self.conv1(features, edge_index, edge_type))
        return self.conv2(hidden, edge_index, edge_type)

    def score_pairs(self, query_embeddings: torch.Tensor, disease_embeddings: torch.Tensor) -> torch.Tensor:
        interaction = query_embeddings * disease_embeddings
        pair_features = torch.cat((query_embeddings, disease_embeddings, interaction), dim=-1)
        return self.scorer(pair_features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a full-graph R-GCN disease ranker.")
    parser.add_argument("--train_csv", type=Path, default=Path("data/processed/ddxplus_v2/train_queries.csv"))
    parser.add_argument("--valid_csv", type=Path, default=Path("data/processed/ddxplus_v2/valid_queries.csv"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/processed/ddxplus_v2/test_queries.csv"))
    parser.add_argument("--condition_map", type=Path, default=Path("data/mappings/ddxplus_v2/condition_to_primekg.json"))
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument("--disease_nodes_csv", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("Benchmark_baselines/Relational_GNN/outputs/rgcn_v1"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--query_batch_size", type=int, default=512)
    parser.add_argument("--negative_samples", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_train_queries", type=int, default=None)
    parser.add_argument("--max_valid_queries", type=int, default=None)
    parser.add_argument("--max_test_queries", type=int, default=None)
    parser.add_argument("--eval_query_batch_size", type=int, default=8)
    parser.add_argument("--candidate_chunk_size", type=int, default=2048)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--num_bases", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_test", action="store_true", help="Score test queries after training; disabled by default.")
    return parser.parse_args()


def read_rows(path: Path, limit: int | None, description: str) -> Iterator[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        iterable = islice(reader, limit) if limit is not None else reader
        yield from tqdm(iterable, total=limit, desc=description, unit="query")


def row_count(path: Path, limit: int | None) -> int:
    with path.open("rb") as handle:
        if limit is None:
            return max(sum(1 for _ in handle) - 1, 0)
        return max(sum(1 for _ in islice(handle, limit + 1)) - 1, 0)


def load_graph(graph_dir: Path, device: torch.device) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, dict[str, int]]:
    node_to_id: dict[str, int] = json.loads((graph_dir / "mappings" / "node2id.json").read_text(encoding="utf-8"))
    relation_to_id: dict[str, int] = json.loads((graph_dir / "mappings" / "relation2id.json").read_text(encoding="utf-8"))
    if len(relation_to_id) != NUM_RELATIONS:
        raise ValueError(f"Expected {NUM_RELATIONS} PrimeKG relation types, found {len(relation_to_id)}")
    embeddings = np.load(graph_dir / "node_embeddings.npy").astype(np.float32, copy=False)
    if embeddings.shape != (len(node_to_id), EMBEDDING_DIM):
        raise ValueError(f"Expected node embeddings shape ({len(node_to_id)}, {EMBEDDING_DIM}), got {embeddings.shape}")
    with np.load(graph_dir / "graph_csr.npz") as data:
        indptr = data["indptr"].astype(np.int64, copy=False)
        indices = data["indices"].astype(np.int64, copy=False)
        edge_type = data["edge_relids"].astype(np.int64, copy=False)
    source = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
    if len(source) != len(indices) or len(edge_type) != len(indices):
        raise ValueError("PrimeKG CSR indices and relation IDs do not align")
    if edge_type.size and (edge_type.min() < 0 or edge_type.max() >= NUM_RELATIONS):
        raise ValueError(f"edge_type must be in 0..{NUM_RELATIONS - 1}")
    edge_index = torch.as_tensor(np.stack((source, indices)), dtype=torch.long, device=device)
    edge_type_tensor = torch.as_tensor(edge_type, dtype=torch.long, device=device)
    return embeddings, edge_index, edge_type_tensor, node_to_id


def load_candidates(graph_dir: Path, custom_path: Path | None, node_to_id: dict[str, int]) -> tuple[list[str], np.ndarray]:
    path = custom_path or graph_dir / "disease_nodes.csv"
    keys: list[str] = []
    graph_ids: list[int] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in tqdm(csv.DictReader(handle), desc="Loading disease candidates", unit="node"):
            key = row.get("node_key", "")
            if row.get("node_type") != "disease" or not key.startswith("disease|"):
                raise ValueError(f"Non-disease candidate in disease_nodes.csv: {key!r}")
            if key not in node_to_id:
                raise ValueError(f"Disease candidate is absent from graph node index: {key}")
            keys.append(key)
            graph_ids.append(int(node_to_id[key]))
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("PrimeKG disease candidate list is empty or has duplicate keys")
    return keys, np.asarray(graph_ids, dtype=np.int64)


def validate_candidates(candidate_nodes: list[str], condition_map_path: Path) -> None:
    mapping = json.loads(condition_map_path.read_text(encoding="utf-8-sig"))
    mapped = {key for entry in mapping.values() for key in entry.get("selected_primekg_nodes", [])}
    if len(candidate_nodes) <= max(50, len(mapped)):
        raise RuntimeError("Candidate set is not the full PrimeKG disease space; refusing closed-set DDXPlus classification")


def load_query_split(path: Path, limit: int | None, node_to_id: dict[str, int], candidate_to_id: dict[str, int], label: str) -> QuerySplit:
    count = row_count(path, limit)
    rows_cache: list[dict[str, object]] = []
    max_seeds = 1
    max_targets = 1
    missing_seeds = 0
    for row in read_rows(path, limit, f"Loading {label} query IDs"):
        patient_id = int(row["patient_index"])
        raw_seeds = [key for key in row.get("seed_node_keys", "").split(";") if key]
        seed_ids = [node_to_id[key] for key in raw_seeds if key in node_to_id]
        missing_seeds += len(raw_seeds) - len(seed_ids)
        if not seed_ids:
            raise ValueError(f"Query {patient_id} has no valid seed node IDs")
        targets = list(dict.fromkeys(key for key in row.get("target_node_keys", "").split(";") if key))
        if not targets:
            raise ValueError(f"Query {patient_id} has no ground-truth target")
        positive_ids = [candidate_to_id[key] for key in targets if key in candidate_to_id]
        rows_cache.append({"patient": patient_id, "seeds": seed_ids, "targets": targets, "positives": positive_ids})
        max_seeds = max(max_seeds, len(seed_ids))
        max_targets = max(max_targets, len(positive_ids))
    if len(rows_cache) != count:
        raise RuntimeError(f"Loaded {len(rows_cache)} rows from {path}, expected {count}")

    seed_ids_array = np.zeros((count, max_seeds), dtype=np.int64)
    seed_mask = np.zeros((count, max_seeds), dtype=np.bool_)
    target_ids_array = np.zeros((count, max_targets), dtype=np.int64)
    target_mask = np.zeros((count, max_targets), dtype=np.bool_)
    target_keys: list[list[str]] = []
    patients = np.empty(count, dtype=np.int64)
    for index, item in enumerate(rows_cache):
        seeds = item["seeds"]
        positives = item["positives"]
        seed_ids_array[index, :len(seeds)] = seeds
        seed_mask[index, :len(seeds)] = True
        target_ids_array[index, :len(positives)] = positives
        target_mask[index, :len(positives)] = True
        patients[index] = item["patient"]
        target_keys.append(item["targets"])
    return QuerySplit(seed_ids_array, seed_mask, target_ids_array, target_mask, target_keys, patients, max_seeds, max_targets, missing_seeds)


def sample_negatives(positive_ids: np.ndarray, valid_positive: np.ndarray, candidate_count: int, negative_count: int, rng: np.random.Generator) -> np.ndarray:
    rows, width = positive_ids.shape
    negatives = rng.integers(candidate_count, size=(rows, width, negative_count), dtype=np.int64)
    for row in range(rows):
        forbidden = set(positive_ids[row, valid_positive[row]].tolist())
        for positive_slot in range(width):
            if not valid_positive[row, positive_slot]:
                continue
            collisions = np.isin(negatives[row, positive_slot], list(forbidden))
            while collisions.any():
                negatives[row, positive_slot, collisions] = rng.integers(candidate_count, size=int(collisions.sum()))
                collisions = np.isin(negatives[row, positive_slot], list(forbidden))
    return negatives


def rank_split(
    model: RGCNRanker,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    split: QuerySplit,
    candidate_nodes: list[str],
    candidate_graph_ids: np.ndarray,
    candidate_to_id: dict[str, int],
    device: torch.device,
    eval_query_batch_size: int,
    candidate_chunk_size: int,
    out_dir: Path,
    split_name: str,
) -> dict[str, object]:
    model.eval()
    started = time.perf_counter()
    rr_total = 0.0
    hits = {k: 0 for k in TOP_KS}
    covered_queries = 0
    target_count = 0
    covered_targets = 0
    absent_targets = 0
    candidate_count = len(candidate_nodes)
    candidate_graph_ids_tensor = torch.as_tensor(candidate_graph_ids, dtype=torch.long, device=device)
    predictions_path = out_dir / f"{split_name}_predictions.csv"
    by_patient_path = out_dir / f"{split_name}_by_patient.csv"
    with torch.no_grad():
        node_embeddings = model.encode_graph(edge_index, edge_type)
        with predictions_path.open("w", newline="", encoding="utf-8") as pred_handle, by_patient_path.open("w", newline="", encoding="utf-8") as patient_handle:
            pred_writer = csv.DictWriter(pred_handle, fieldnames=["patient_index", "candidate", "score", "rank"])
            pred_writer.writeheader()
            patient_writer = csv.DictWriter(patient_handle, fieldnames=["patient_index", "num_targets", "targets_in_candidates", "candidate_recall", "first_hit_rank", "mrr", "recall@5", "recall@10", "recall@20", "recall@50"])
            patient_writer.writeheader()
            for batch_start in tqdm(range(0, split.size, eval_query_batch_size), desc=f"Ranking {split_name}", unit="batch"):
                batch_end = min(batch_start + eval_query_batch_size, split.size)
                seed_ids = torch.as_tensor(split.seed_ids[batch_start:batch_end], dtype=torch.long, device=device)
                seed_mask = torch.as_tensor(split.seed_mask[batch_start:batch_end], dtype=torch.bool, device=device)
                query_emb = (node_embeddings[seed_ids] * seed_mask.unsqueeze(-1)).sum(dim=1) / seed_mask.sum(dim=1, keepdim=True)
                score_parts: list[torch.Tensor] = []
                for candidate_start in range(0, candidate_count, candidate_chunk_size):
                    candidate_end = min(candidate_start + candidate_chunk_size, candidate_count)
                    disease_ids = torch.as_tensor(candidate_graph_ids[candidate_start:candidate_end], dtype=torch.long, device=device)
                    disease_emb = node_embeddings[disease_ids]
                    batch_size = len(query_emb)
                    q = query_emb[:, None, :].expand(-1, len(disease_ids), -1).reshape(-1, EMBEDDING_DIM)
                    d = disease_emb[None, :, :].expand(batch_size, -1, -1).reshape(-1, EMBEDDING_DIM)
                    scores = model.score_pairs(q, d).view(batch_size, -1)
                    score_parts.append(scores.cpu())
                all_scores = torch.cat(score_parts, dim=1).numpy()
                for local, absolute in enumerate(range(batch_start, batch_end)):
                    order = np.argsort(-all_scores[local], kind="stable")
                    targets = split.target_keys[absolute]
                    positive_ids = sorted({candidate_to_id[key] for key in targets if key in candidate_to_id})
                    n_covered = len(positive_ids)
                    n_targets = len(set(targets))
                    candidate_hit = n_covered > 0
                    covered_queries += int(candidate_hit)
                    target_count += n_targets
                    covered_targets += n_covered
                    absent_targets += n_targets - n_covered
                    first_rank = None
                    if positive_ids:
                        inverse = np.empty(candidate_count, dtype=np.int64)
                        inverse[order] = np.arange(candidate_count)
                        first_rank = int(inverse[positive_ids].min()) + 1
                    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
                    rr_total += reciprocal_rank
                    row = {"patient_index": int(split.patient_indices[absolute]), "num_targets": n_targets, "targets_in_candidates": n_covered, "candidate_recall": int(candidate_hit), "first_hit_rank": first_rank if first_rank else "", "mrr": reciprocal_rank}
                    for k in TOP_KS:
                        hit = int(first_rank is not None and first_rank <= k)
                        hits[k] += hit
                        row[f"recall@{k}"] = hit
                    patient_writer.writerow(row)
                    for rank, candidate_id in enumerate(order[:min(50, candidate_count)], start=1):
                        pred_writer.writerow({"patient_index": int(split.patient_indices[absolute]), "candidate": candidate_nodes[int(candidate_id)], "score": f"{all_scores[local, candidate_id]:.10g}", "rank": rank})
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {"num_evaluated": split.size, "candidate_space_size": candidate_count, "mrr": rr_total / max(split.size, 1), **{f"recall@{k}": hits[k] / max(split.size, 1) for k in TOP_KS}, "candidate_recall": covered_queries / max(split.size, 1), "num_targets_total": target_count, "num_targets_in_candidates": covered_targets, "num_target_nodes_absent_from_candidates": absent_targets, "elapsed_seconds": round(elapsed, 3), "queries_per_second": split.size / elapsed, "candidate_pairs_scored": split.size * candidate_count, "predictions_csv": str(predictions_path), "by_patient_csv": str(by_patient_path)}


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.query_batch_size < 1 or args.negative_samples < 1:
        raise ValueError("epochs, query_batch_size, and negative_samples must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()

    initial_embeddings, edge_index, edge_type, node_to_id = load_graph(args.graph_dir, device)
    candidate_nodes, candidate_graph_ids = load_candidates(args.graph_dir, args.disease_nodes_csv, node_to_id)
    validate_candidates(candidate_nodes, args.condition_map)
    candidate_to_id = {key: idx for idx, key in enumerate(candidate_nodes)}
    candidate_graph_ids_tensor = torch.as_tensor(candidate_graph_ids, dtype=torch.long, device=device)
    train = load_query_split(args.train_csv, args.max_train_queries, node_to_id, candidate_to_id, "train")
    valid = load_query_split(args.valid_csv, args.max_valid_queries, node_to_id, candidate_to_id, "validation")
    train_query_indices = np.flatnonzero(train.target_mask.any(axis=1))
    if len(train_query_indices) == 0:
        raise ValueError("No training queries have disease targets in the candidate set")
    model = RGCNRanker(initial_embeddings, num_bases=args.num_bases).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_valid_mrr = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience = 0
    history: list[dict[str, float | int]] = []
    negative_pairs_sampled = 0
    total_train_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        node_embeddings = model.encode_graph(edge_index, edge_type)
        batches = [train_query_indices[i:i + args.query_batch_size] for i in range(0, len(train_query_indices), args.query_batch_size)]
        total_positive_pairs = int(train.target_mask[train_query_indices].sum())
        epoch_loss = 0.0
        for batch_number, query_indices in enumerate(tqdm(batches, desc=f"R-GCN train epoch {epoch}/{args.epochs}", unit="batch")):
            seed_ids = torch.as_tensor(train.seed_ids[query_indices], dtype=torch.long, device=device)
            seed_mask = torch.as_tensor(train.seed_mask[query_indices], dtype=torch.bool, device=device)
            query_emb = (node_embeddings[seed_ids] * seed_mask.unsqueeze(-1)).sum(dim=1) / seed_mask.sum(dim=1, keepdim=True)
            pos_ids_np = train.target_ids[query_indices]
            pos_mask_np = train.target_mask[query_indices]
            neg_ids_np = sample_negatives(pos_ids_np, pos_mask_np, len(candidate_nodes), args.negative_samples, rng)
            pos_ids = torch.as_tensor(pos_ids_np, dtype=torch.long, device=device)
            pos_mask = torch.as_tensor(pos_mask_np, dtype=torch.bool, device=device)
            neg_ids = torch.as_tensor(neg_ids_np, dtype=torch.long, device=device)
            pos_emb = node_embeddings[candidate_graph_ids_tensor[pos_ids]]
            neg_emb = node_embeddings[candidate_graph_ids_tensor[neg_ids]]
            batch_size, target_width = pos_ids.shape
            q_pos = query_emb[:, None, :].expand(-1, target_width, -1)[pos_mask]
            positive_embeddings = pos_emb[pos_mask]
            positive_scores = model.score_pairs(q_pos, positive_embeddings)
            q_negative = query_emb[:, None, None, :].expand(-1, target_width, args.negative_samples, -1)[pos_mask]
            negative_embeddings = neg_emb[pos_mask]
            negative_scores = model.score_pairs(q_negative.reshape(-1, EMBEDDING_DIM), negative_embeddings.reshape(-1, EMBEDDING_DIM)).view(-1, args.negative_samples)
            per_positive_loss = F.softplus(-positive_scores[:, None] + negative_scores).mean(dim=1)
            loss = per_positive_loss.sum() / max(total_positive_pairs, 1)
            loss.backward(retain_graph=batch_number < len(batches) - 1)
            epoch_loss += float(per_positive_loss.detach().sum().cpu())
            negative_pairs_sampled += int(len(per_positive_loss) * args.negative_samples)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss = epoch_loss / max(total_positive_pairs, 1)

        valid_metrics = rank_split(model, edge_index, edge_type, valid, candidate_nodes, candidate_graph_ids, candidate_to_id, device, args.eval_query_batch_size, args.candidate_chunk_size, args.out_dir, "validation")
        record = {"epoch": epoch, "train_bpr_loss": train_loss, "valid_mrr": valid_metrics["mrr"], "valid_recall@10": valid_metrics["recall@10"], "epoch_seconds": time.perf_counter() - epoch_started, "train_positive_pairs": total_positive_pairs}
        history.append(record)
        print(json.dumps(record))
        if valid_metrics["mrr"] > best_valid_mrr + args.early_stopping_min_delta:
            best_valid_mrr = float(valid_metrics["mrr"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if args.early_stopping_patience > 0 and patience >= args.early_stopping_patience:
                print(json.dumps({"early_stopped": True, "best_epoch": best_epoch, "selection_metric": "validation MRR"}))
                break

    if best_state is None:
        raise RuntimeError("No best model selected from validation MRR")
    model.load_state_dict(best_state)
    checkpoint_path = args.out_dir / "rgcn_ranker.pt"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.save({"model_state": best_state, "num_relations": NUM_RELATIONS, "num_bases": args.num_bases, "candidate_nodes": candidate_nodes, "best_epoch": best_epoch, "best_valid_mrr": best_valid_mrr}, checkpoint_path)

    test_metrics = None
    if args.run_test:
        test = load_query_split(args.test_csv, args.max_test_queries, node_to_id, candidate_to_id, "test")
        test_metrics = rank_split(model, edge_index, edge_type, test, candidate_nodes, candidate_graph_ids, candidate_to_id, device, args.eval_query_batch_size, args.candidate_chunk_size, args.out_dir, "test")
    training_summary = {
        "graph_nodes": len(node_to_id), "graph_edges": int(edge_index.shape[1]), "relation_types": NUM_RELATIONS,
        "candidate_space_size": len(candidate_nodes), "embedding_dimension": EMBEDDING_DIM,
        "rgcn_layers": [{"in": 64, "out": 128, "num_bases": args.num_bases}, {"in": 128, "out": 64, "num_bases": args.num_bases}],
        "query_encoder": "mean pool of R-GCN output embeddings for seed IDs",
        "disease_scorer": "concat(q,d,q*d): 192 -> 128 -> 1",
        "ranking_loss": "mean(softplus(-score_positive + score_negative))",
        "optimizer": "Adam", "learning_rate": args.learning_rate,
        "num_train_queries": train.size, "num_train_queries_with_candidate_target": int(len(train_query_indices)),
        "num_valid_queries": valid.size, "num_test_queries": args.max_test_queries,
        "num_positive_pairs": int(train.target_mask.sum()), "num_negative_pairs_sampled": negative_pairs_sampled,
        "parameter_count": parameter_count, "best_epoch": best_epoch, "best_valid_mrr": best_valid_mrr,
        "training_elapsed_seconds": round(time.perf_counter() - total_train_started, 3),
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if device.type == "cuda" else None,
        "candidate_space_check": "passed: PrimeKG disease_nodes.csv, not DDXPlus-only targets",
        "graph_traversal_at_query_time": False, "path_information_used": False, "pathology_used_as_input": False,
        "history": history, "checkpoint": str(checkpoint_path), "test_metrics": test_metrics,
        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
    }
    with (args.out_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2)
    print(json.dumps(training_summary, indent=2))


if __name__ == "__main__":
    main()
