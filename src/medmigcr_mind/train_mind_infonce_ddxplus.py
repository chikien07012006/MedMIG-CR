from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medmigcr_kg.graph_store import GraphStore  # noqa: E402
from medmigcr_mind.contrastive_model import (  # noqa: E402
    ProjectedClinicalMIND,
    average_pairwise_cosine,
    contrastive_loss,
)


PAD_SYM = 0
UNK_SYM = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train graph-grounded MIND with InfoNCE on DDXPlus.")
    parser.add_argument("--train_csv", type=Path, default=Path("data/processed/ddxplus_v2/train_queries.csv"))
    parser.add_argument("--valid_csv", type=Path, default=Path("data/processed/ddxplus_v2/valid_queries.csv"))
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/checkpoints/ddxplus_infonce"))
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--evidence_weight", type=float, default=0.2)
    parser.add_argument("--diversity_weight", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)
    parser.add_argument("--cosine_export_limit", type=int, default=5000)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_nodes(cell: str) -> List[str]:
    return [token.strip() for token in str(cell or "").split(";") if token.strip()]


def read_queries(path: Path, max_rows: int | None) -> pd.DataFrame:
    columns = ["patient_index", "seed_node_keys", "target_node_keys", "pathology"]
    frame = pd.read_csv(path, usecols=lambda column: column in columns, nrows=max_rows)
    frame["seed_node_keys"] = frame["seed_node_keys"].fillna("")
    frame["target_node_keys"] = frame["target_node_keys"].fillna("")
    return frame


def build_seed_vocab(*frames: pd.DataFrame) -> Dict[str, int]:
    nodes = set()
    for frame in frames:
        for cell in frame["seed_node_keys"].astype(str):
            nodes.update(split_nodes(cell))
    vocab = {"<PAD>": PAD_SYM, "<UNK>": UNK_SYM}
    for node in sorted(nodes):
        vocab[node] = len(vocab)
    return vocab


def collect_candidate_nodes(frames: Iterable[pd.DataFrame], column: str) -> List[str]:
    nodes = set()
    for frame in frames:
        for cell in frame[column].astype(str):
            nodes.update(split_nodes(cell))
    return sorted(nodes)


def graph_resolvable_nodes(graph_store: GraphStore, nodes: Iterable[str]) -> List[str]:
    return [node for node in nodes if graph_store.lookup_node_id(node) is not None]


def embedding_matrix(graph_store: GraphStore, node_keys: List[str]) -> np.ndarray:
    ids = [graph_store.lookup_node_id(node) for node in node_keys]
    if any(node_id is None for node_id in ids):
        missing = [node for node, node_id in zip(node_keys, ids) if node_id is None][:10]
        raise ValueError(f"Some candidate nodes are missing from the graph: {missing}")
    matrix = graph_store.node_embeddings[np.asarray(ids, dtype=np.int64)].astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return matrix / norms


class ContrastiveQueryDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        seed_vocab: Dict[str, int],
        target_index: Dict[str, int],
        evidence_index: Dict[str, int],
        max_seq_len: int,
    ) -> None:
        kept_rows = []
        seeds = []
        target_masks = []
        evidence_masks = []
        for _, row in frame.iterrows():
            seed_nodes = split_nodes(row["seed_node_keys"])
            target_nodes = split_nodes(row["target_node_keys"])
            target_ids = [target_index[node] for node in target_nodes if node in target_index]
            evidence_ids = [evidence_index[node] for node in seed_nodes if node in evidence_index]
            if not target_ids or not evidence_ids:
                continue
            encoded = [seed_vocab.get(node, UNK_SYM) for node in seed_nodes][:max_seq_len]
            encoded.extend([PAD_SYM] * (max_seq_len - len(encoded)))
            target_mask = np.zeros(len(target_index), dtype=np.bool_)
            target_mask[target_ids] = True
            evidence_mask = np.zeros(len(evidence_index), dtype=np.bool_)
            evidence_mask[evidence_ids] = True
            kept_rows.append(int(row["patient_index"]))
            seeds.append(encoded)
            target_masks.append(target_mask)
            evidence_masks.append(evidence_mask)

        self.patient_indices = kept_rows
        self.seeds = torch.as_tensor(np.asarray(seeds, dtype=np.int64), dtype=torch.long)
        self.target_masks = torch.as_tensor(np.asarray(target_masks, dtype=np.bool_), dtype=torch.bool)
        self.evidence_masks = torch.as_tensor(np.asarray(evidence_masks, dtype=np.bool_), dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.seeds.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.seeds[idx], self.target_masks[idx], self.evidence_masks[idx], self.patient_indices[idx]


def collate_fn(batch):
    seeds = torch.stack([item[0] for item in batch], dim=0)
    target_masks = torch.stack([item[1] for item in batch], dim=0)
    evidence_masks = torch.stack([item[2] for item in batch], dim=0)
    patient_indices = torch.as_tensor([item[3] for item in batch], dtype=torch.long)
    return seeds, target_masks, evidence_masks, patient_indices


def evaluate(
    model: ProjectedClinicalMIND,
    loader: DataLoader,
    target_embeddings: torch.Tensor,
    evidence_embeddings: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals: List[float] = []
    targets: List[float] = []
    evidences: List[float] = []
    diversities: List[float] = []
    cosines: List[float] = []
    with torch.no_grad():
        for seeds, target_mask, evidence_mask, _patient_indices in loader:
            seeds = seeds.to(device)
            target_mask = target_mask.to(device)
            evidence_mask = evidence_mask.to(device)
            projected, _latent, active_mask = model(seeds)
            loss = contrastive_loss(
                projected_interests=projected,
                target_embeddings=target_embeddings,
                evidence_embeddings=evidence_embeddings,
                target_positive_mask=target_mask,
                evidence_positive_mask=evidence_mask,
                active_interest_mask=active_mask,
                temperature=args.temperature,
                evidence_weight=args.evidence_weight,
                diversity_weight=args.diversity_weight,
            )
            cosine_mean, _ = average_pairwise_cosine(projected, active_mask)
            totals.append(float(loss.total.item()))
            targets.append(float(loss.target.item()))
            evidences.append(float(loss.evidence.item()))
            diversities.append(float(loss.diversity.item()))
            cosines.append(cosine_mean)
    model.train()
    return {
        "loss": float(np.mean(totals)) if totals else 0.0,
        "target_loss": float(np.mean(targets)) if targets else 0.0,
        "evidence_loss": float(np.mean(evidences)) if evidences else 0.0,
        "diversity_loss": float(np.mean(diversities)) if diversities else 0.0,
        "latent_cosine_mean": float(np.mean(cosines)) if cosines else 0.0,
    }


def export_cosine_csv(
    path: Path,
    model: ProjectedClinicalMIND,
    dataset: ContrastiveQueryDataset,
    batch_size: int,
    device: torch.device,
    limit: int,
) -> Dict[str, object]:
    if limit <= 0:
        return {"path": None, "num_rows": 0}
    subset_size = min(limit, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    pair_sums: Dict[str, float] = {}
    pair_counts: Dict[str, int] = {}
    model.eval()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_index", "mean_pairwise_cosine", "pair_cosines"])
        writer.writeheader()
        with torch.no_grad():
            for seeds, _target_mask, _evidence_mask, patient_indices in loader:
                seeds = seeds.to(device)
                projected, _latent, active_mask = model(seeds)
                z = torch.nn.functional.normalize(projected, dim=-1)
                sim = torch.matmul(z, z.transpose(1, 2)).detach().cpu().numpy()
                active = active_mask.detach().cpu().numpy()
                for local_idx, patient_index in enumerate(patient_indices.tolist()):
                    pairs = {}
                    for i in range(projected.shape[1]):
                        for j in range(i + 1, projected.shape[1]):
                            if not (active[local_idx, i] and active[local_idx, j]):
                                continue
                            key = f"{i}-{j}"
                            value = float(sim[local_idx, i, j])
                            pairs[key] = value
                            pair_sums[key] = pair_sums.get(key, 0.0) + value
                            pair_counts[key] = pair_counts.get(key, 0) + 1
                    mean_value = float(np.mean(list(pairs.values()))) if pairs else 0.0
                    writer.writerow(
                        {
                            "patient_index": patient_index,
                            "mean_pairwise_cosine": f"{mean_value:.8f}",
                            "pair_cosines": json.dumps(pairs, sort_keys=True),
                        }
                    )
                    row_count += 1
    model.train()
    return {
        "path": str(path),
        "num_rows": row_count,
        "pair_means": {
            key: pair_sums[key] / pair_counts[key]
            for key in sorted(pair_sums)
            if pair_counts[key] > 0
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    graph_store = GraphStore.load(
        graph_npz=args.graph_dir / "graph_csr.npz",
        node_embeddings_npy=args.graph_dir / "node_embeddings.npy",
        out_degree_npy=args.graph_dir / "out_degree.npy",
        in_degree_npy=args.graph_dir / "in_degree.npy",
        mapping_dir=args.graph_dir / "mappings",
        device=None,
    )
    train_queries = read_queries(args.train_csv, args.max_train_samples)
    valid_queries = read_queries(args.valid_csv, args.max_valid_samples) if args.valid_csv.is_file() else train_queries.head(0)

    seed_vocab = build_seed_vocab(train_queries, valid_queries)
    target_nodes = graph_resolvable_nodes(graph_store, collect_candidate_nodes([train_queries, valid_queries], "target_node_keys"))
    evidence_nodes = graph_resolvable_nodes(graph_store, collect_candidate_nodes([train_queries, valid_queries], "seed_node_keys"))
    if not target_nodes:
        raise RuntimeError("No graph-resolvable target nodes found.")
    if not evidence_nodes:
        raise RuntimeError("No graph-resolvable evidence nodes found.")

    target_index = {node: idx for idx, node in enumerate(target_nodes)}
    evidence_index = {node: idx for idx, node in enumerate(evidence_nodes)}
    train_ds = ContrastiveQueryDataset(train_queries, seed_vocab, target_index, evidence_index, args.max_seq_len)
    valid_ds = ContrastiveQueryDataset(valid_queries, seed_vocab, target_index, evidence_index, args.max_seq_len)
    if len(train_ds) == 0:
        raise RuntimeError("No train rows survived target/evidence graph filtering.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = None
    if len(valid_ds) > 0:
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    target_embeddings = torch.as_tensor(embedding_matrix(graph_store, target_nodes), dtype=torch.float32, device=device)
    evidence_embeddings = torch.as_tensor(embedding_matrix(graph_store, evidence_nodes), dtype=torch.float32, device=device)
    graph_dim = int(target_embeddings.shape[1])
    model = ProjectedClinicalMIND(
        num_symptoms=len(seed_vocab),
        mind_dim=args.D,
        graph_dim=graph_dim,
        num_interests=args.K,
        max_seq_len=args.max_seq_len,
        num_routing_iters=args.R,
        symptom_padding_idx=PAD_SYM,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_state = None
    best_valid = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for seeds, target_mask, evidence_mask, _patient_indices in train_loader:
            seeds = seeds.to(device)
            target_mask = target_mask.to(device)
            evidence_mask = evidence_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            projected, _latent, active_mask = model(seeds)
            loss = contrastive_loss(
                projected_interests=projected,
                target_embeddings=target_embeddings,
                evidence_embeddings=evidence_embeddings,
                target_positive_mask=target_mask,
                evidence_positive_mask=evidence_mask,
                active_interest_mask=active_mask,
                temperature=args.temperature,
                evidence_weight=args.evidence_weight,
                diversity_weight=args.diversity_weight,
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.total.item()))

        valid_metrics = evaluate(model, valid_loader, target_embeddings, evidence_embeddings, args, device) if valid_loader else {}
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        valid_loss = float(valid_metrics.get("loss", train_loss))
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        record = {"epoch": epoch, "train_loss": train_loss, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)
    cosine_export = (
        export_cosine_csv(
            args.out_dir / f"latent_cosine_k{args.K}.csv",
            model,
            valid_ds,
            args.batch_size,
            device,
            args.cosine_export_limit,
        )
        if len(valid_ds) > 0 and args.K > 1
        else {"path": None, "num_rows": 0}
    )

    checkpoint_path = args.out_dir / f"clinical_mind_infonce_k{args.K}.pt"
    checkpoint = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "symptom_str2id": seed_vocab,
        "target_node_keys": target_nodes,
        "evidence_node_keys": evidence_nodes,
        "hparams": {
            "D": args.D,
            "graph_dim": graph_dim,
            "K": args.K,
            "R": args.R,
            "max_seq_len": args.max_seq_len,
            "num_symptoms": len(seed_vocab),
            "num_target_nodes": len(target_nodes),
            "num_evidence_nodes": len(evidence_nodes),
            "temperature": args.temperature,
            "evidence_weight": args.evidence_weight,
            "diversity_weight": args.diversity_weight,
            "input_space": "primekg_seed_node_keys",
            "output_space": "primekg_graph_embedding_space",
            "alignment": "trainable_infonce_projection",
        },
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "checkpoint": str(checkpoint_path),
        "train_csv": str(args.train_csv),
        "valid_csv": str(args.valid_csv),
        "graph_dir": str(args.graph_dir),
        "device": str(device),
        "num_train_rows": len(train_queries),
        "num_valid_rows": len(valid_queries),
        "num_train_examples": len(train_ds),
        "num_valid_examples": len(valid_ds),
        "num_symptoms": len(seed_vocab),
        "num_target_nodes": len(target_nodes),
        "num_evidence_nodes": len(evidence_nodes),
        "history": history,
        "best_valid_loss": best_valid,
        "cosine_export": cosine_export,
    }
    summary_path = args.out_dir / f"training_summary_k{args.K}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
