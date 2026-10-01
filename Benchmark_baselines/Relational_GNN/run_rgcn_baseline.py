from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch_geometric.nn import RGCNConv  # noqa: E402
from tqdm import tqdm  # noqa: E402

from baseline_common import (  # noqa: E402
    CANDIDATE_SPACES,
    DEFAULT_CONDITION_MAP,
    DEFAULT_GRAPH_DIR,
    DEFAULT_SUBSET_DIR,
    CandidateSet,
    Query,
    candidate_coverage,
    load_candidates,
    load_directed_edges,
    load_node_index,
    load_queries,
    prediction_rows,
    print_metrics,
    rank_candidates,
    run_shared_evaluator,
    write_json,
    write_predictions,
)
from medmigcr_kg.parallel import recommended_workers  # noqa: E402

FULL_SOFTMAX_MAX_CANDIDATES = 1024


@dataclass
class EncodedSplit:
    """Padded seed/positive index tensors for one query split."""

    queries: List[Query]
    seed_ids: torch.Tensor  # [Q, S] graph node ids, padded with 0
    seed_mask: torch.Tensor  # [Q, S]
    positive_mask: torch.Tensor  # [Q, C] candidate positions that are targets
    usable: np.ndarray  # queries with at least one resolved seed

    def __len__(self) -> int:
        return len(self.queries)


class RGCNRanker(nn.Module):
    def __init__(self, node_features: np.ndarray, num_relations: int, hidden_dim: int, num_bases: int, dropout: float) -> None:
        super().__init__()
        feature_dim = node_features.shape[1]
        self.node_features = nn.Embedding.from_pretrained(torch.as_tensor(node_features, dtype=torch.float32), freeze=False)
        self.conv1 = RGCNConv(feature_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, feature_dim, num_relations=num_relations, num_bases=num_bases)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Sequential(nn.Linear(feature_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.conv1(self.node_features.weight, edge_index, edge_type))
        return self.conv2(self.dropout(hidden), edge_index, edge_type)

    @staticmethod
    def pool_queries(node_embeddings: torch.Tensor, seed_ids: torch.Tensor, seed_mask: torch.Tensor) -> torch.Tensor:
        weights = seed_mask.float().unsqueeze(-1)
        return (node_embeddings[seed_ids] * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def score(self, queries: torch.Tensor, diseases: torch.Tensor) -> torch.Tensor:
        """queries [B, D] x diseases [B, C, D] -> scores [B, C]."""
        q = queries.unsqueeze(1).expand_as(diseases)
        return self.scorer(torch.cat((q, diseases, q * diseases), dim=-1)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_csv", type=Path, default=DEFAULT_SUBSET_DIR / "train_50k.csv")
    parser.add_argument("--valid_csv", type=Path, default=DEFAULT_SUBSET_DIR / "valid_5k.csv")
    parser.add_argument("--test_csv", type=Path, default=DEFAULT_SUBSET_DIR / "test_10k.csv")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--graph_dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--condition_map", type=Path, default=DEFAULT_CONDITION_MAP)
    parser.add_argument("--candidate_space", choices=CANDIDATE_SPACES, default="closed")
    parser.add_argument("--no_inverse_edges", action="store_true", help="Use only the stored edge directions.")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_bases", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--negative_samples", type=int, default=32, help="Open space only: sampled negatives per positive.")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_threads", type=int, default=recommended_workers(), help="PyTorch CPU threads.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=None, help="Smoke tests only.")
    parser.add_argument("--limit_eval", type=int, default=None, help="Smoke tests only: cap validation and test queries.")
    parser.add_argument("--no_test", action="store_true", help="Train and validate only.")
    return parser.parse_args()


def load_graph_tensors(graph_dir: Path, add_inverse: bool, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    source, target, relation, _num_nodes = load_directed_edges(graph_dir)
    num_relations = int(relation.max()) + 1
    if add_inverse:
        source, target, relation = (
            np.concatenate((source, target)),
            np.concatenate((target, source)),
            np.concatenate((relation, relation + num_relations)),
        )
        num_relations *= 2
    edge_index = torch.as_tensor(np.stack((source, target)), dtype=torch.long, device=device)
    edge_type = torch.as_tensor(relation, dtype=torch.long, device=device)
    return edge_index, edge_type, num_relations


def encode_split(queries: Sequence[Query], node_index: Dict[str, int], candidates: CandidateSet, device: torch.device) -> EncodedSplit:
    position = {key: i for i, key in enumerate(candidates.keys)}
    seeds = [[node_index[key] for key in dict.fromkeys(query.seed_keys) if key in node_index] for query in queries]
    width = max((len(ids) for ids in seeds), default=1) or 1
    seed_ids = torch.zeros((len(queries), width), dtype=torch.long)
    seed_mask = torch.zeros((len(queries), width), dtype=torch.bool)
    positive_mask = torch.zeros((len(queries), len(candidates)), dtype=torch.bool)
    for row, (query, ids) in enumerate(zip(queries, seeds)):
        seed_ids[row, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        seed_mask[row, : len(ids)] = True
        for key in query.target_keys:
            if key in position:
                positive_mask[row, position[key]] = True
    usable = np.asarray([len(ids) > 0 for ids in seeds])
    return EncodedSplit(list(queries), seed_ids.to(device), seed_mask.to(device), positive_mask.to(device), usable)


def candidate_scores(model: RGCNRanker, node_embeddings: torch.Tensor, query_vectors: torch.Tensor, candidate_ids: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    parts = []
    for start in range(0, len(candidate_ids), chunk):
        diseases = node_embeddings[candidate_ids[start:start + chunk]]
        parts.append(model.score(query_vectors, diseases.unsqueeze(0).expand(len(query_vectors), -1, -1)))
    return torch.cat(parts, dim=1)


def batch_loss(model: RGCNRanker, node_embeddings: torch.Tensor, split: EncodedSplit, rows: torch.Tensor,
               candidate_ids: torch.Tensor, negative_samples: int, generator: torch.Generator) -> torch.Tensor:
    queries = RGCNRanker.pool_queries(node_embeddings, split.seed_ids[rows], split.seed_mask[rows])
    positives = split.positive_mask[rows]
    if len(candidate_ids) <= FULL_SOFTMAX_MAX_CANDIDATES:
        scores = candidate_scores(model, node_embeddings, queries, candidate_ids)
        log_norm = torch.logsumexp(scores, dim=1)
        log_pos = torch.logsumexp(scores.masked_fill(~positives, float("-inf")), dim=1)
        return (log_norm - log_pos).mean()
    # Open space: sampled pairwise softplus loss (resampling collisions with positives is unnecessary at 22k candidates).
    pos_rows, pos_cols = positives.nonzero(as_tuple=True)
    negatives = torch.randint(len(candidate_ids), (len(pos_rows), negative_samples), generator=generator).to(candidate_ids.device)
    q = queries[pos_rows]
    pos_score = model.score(q, node_embeddings[candidate_ids[pos_cols]].unsqueeze(1)).squeeze(1)
    neg_score = model.score(q, node_embeddings[candidate_ids[negatives]])
    return F.softplus(neg_score - pos_score.unsqueeze(1)).mean()


@torch.no_grad()
def score_split(model: RGCNRanker, edge_index: torch.Tensor, edge_type: torch.Tensor, split: EncodedSplit,
                candidate_ids: torch.Tensor, batch_size: int, desc: str) -> np.ndarray:
    model.eval()
    node_embeddings = model.encode(edge_index, edge_type)
    scores = np.full((len(split), len(candidate_ids)), np.nan, dtype=np.float32)
    usable_rows = np.flatnonzero(split.usable)
    for start in tqdm(range(0, len(usable_rows), batch_size), desc=desc, unit="batch", leave=False):
        rows = torch.as_tensor(usable_rows[start:start + batch_size], device=candidate_ids.device)
        queries = RGCNRanker.pool_queries(node_embeddings, split.seed_ids[rows], split.seed_mask[rows])
        scores[rows.cpu().numpy()] = candidate_scores(model, node_embeddings, queries, candidate_ids).cpu().numpy()
    return scores


def mean_reciprocal_rank(scores: np.ndarray, split: EncodedSplit) -> float:
    positives = split.positive_mask.cpu().numpy()
    total = 0.0
    for row in np.flatnonzero(split.usable & positives.any(axis=1)):
        best_positive = scores[row, positives[row]].max()
        total += 1.0 / (1 + int((scores[row] > best_positive).sum()))
    return total / max(len(split), 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()

    node_index = load_node_index(args.graph_dir)
    candidates = load_candidates(args.candidate_space, args.graph_dir, args.condition_map, node_index)
    candidate_ids = torch.as_tensor(candidates.graph_ids, dtype=torch.long, device=device)
    edge_index, edge_type, num_relations = load_graph_tensors(args.graph_dir, not args.no_inverse_edges, device)
    node_features = np.load(args.graph_dir / "node_embeddings.npy").astype(np.float32)

    train = encode_split(load_queries(args.train_csv, args.limit_train), node_index, candidates, device)
    valid = encode_split(load_queries(args.valid_csv, args.limit_eval), node_index, candidates, device)
    train_rows = np.flatnonzero(train.usable & train.positive_mask.any(dim=1).cpu().numpy())
    if len(train_rows) == 0:
        raise ValueError("No training query has both a resolved seed and a target in the candidate space")

    model = RGCNRanker(node_features, num_relations, args.hidden_dim, args.num_bases, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    history: List[dict] = []
    best_mrr, best_epoch, best_state, stale = float("-inf"), 0, None, 0
    train_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        order = rng.permutation(train_rows)
        losses = []
        progress = tqdm(range(0, len(order), args.batch_size), desc=f"Epoch {epoch}/{args.epochs}", unit="step")
        for start in progress:
            rows = torch.as_tensor(order[start:start + args.batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            node_embeddings = model.encode(edge_index, edge_type)
            loss = batch_loss(model, node_embeddings, train, rows, candidate_ids, args.negative_samples, generator)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        valid_mrr = mean_reciprocal_rank(score_split(model, edge_index, edge_type, valid, candidate_ids, args.eval_batch_size, "Validation"), valid)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_mrr": valid_mrr,
                  "steps": len(losses), "epoch_seconds": round(time.perf_counter() - epoch_started, 1)}
        history.append(record)
        tqdm.write(f"epoch {epoch}: loss={record['train_loss']:.4f} valid_mrr={valid_mrr:.4f} ({record['epoch_seconds']}s)")
        if valid_mrr > best_mrr:
            best_mrr, best_epoch, stale = valid_mrr, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                tqdm.write(f"Early stopping: best epoch {best_epoch} (valid MRR {best_mrr:.4f})")
                break

    model.load_state_dict(best_state)
    checkpoint = args.out_dir / "rgcn_ranker.pt"
    torch.save({"model_state": best_state, "args": {k: str(v) for k, v in vars(args).items()},
                "candidate_keys": candidates.keys, "best_epoch": best_epoch, "best_valid_mrr": best_mrr}, checkpoint)

    summary = {
        "method": "R-GCN",
        "graph": f"PrimeKG directed edges{'' if args.no_inverse_edges else ' + inverse edges'}",
        "num_edges": int(edge_index.shape[1]),
        "num_relations": num_relations,
        "candidate_space": args.candidate_space,
        "num_candidates": len(candidates),
        "loss": "multi-positive softmax over all candidates" if len(candidates) <= FULL_SOFTMAX_MAX_CANDIDATES else "sampled pairwise softplus",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "num_train_queries": len(train),
        "num_train_queries_used": int(len(train_rows)),
        "num_valid_queries": len(valid),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_valid_mrr": best_mrr,
        "history": history,
        "training_seconds": round(time.perf_counter() - train_started, 1),
        "checkpoint": str(checkpoint),
    }

    if not args.no_test:
        test = encode_split(load_queries(args.test_csv, args.limit_eval), node_index, candidates, device)
        test_started = time.perf_counter()
        scores = score_split(model, edge_index, edge_type, test, candidate_ids, args.eval_batch_size, "Test ranking")
        rows: List[dict] = []
        for row, query in enumerate(test.queries):
            if test.usable[row]:
                order = rank_candidates(scores[row].astype(np.float64), args.top_k, keep_zero_scores=True)
                rows.extend(prediction_rows(query.patient_index, candidates, scores[row], order))
        predictions_csv = args.out_dir / "predictions.csv"
        write_predictions(predictions_csv, rows)
        summary["num_test_queries"] = len(test)
        summary["test_seconds"] = round(time.perf_counter() - test_started, 1)
        summary["candidate_coverage"] = candidate_coverage(predictions_csv, test.queries)
        metrics = run_shared_evaluator(args.test_csv, args.condition_map, predictions_csv, args.out_dir / "evaluation", args.limit_eval)
        metrics["candidate_coverage"] = summary["candidate_coverage"]
        summary["metrics"] = metrics
        print_metrics("R-GCN", metrics)

    summary["elapsed_seconds"] = round(time.perf_counter() - run_started, 1)
    write_json(args.out_dir / "training_summary.json", summary)


if __name__ == "__main__":
    main()
