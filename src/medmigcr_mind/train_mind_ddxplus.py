"""
Train Clinical MIND on DDXPlus queries mapped to PrimeKG nodes.

Input rows come from scripts/preprocess/build_ddxplus_test_queries.py run on
DDXPlus train/valid CSV files. The symptom vocabulary is the PrimeKG seed-node
key space (Option B), and the disease vocabulary is the PrimeKG target disease
node key space.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medmigcr_mind.model import (
    ClinicalMIND,
    average_active_interest_cosine_similarity,
    training_bce_loss,
)


PAD_SYM = 0
UNK_SYM = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Clinical MIND on mapped DDXPlus queries.")
    parser.add_argument("--train_csv", type=Path, default=Path("data/processed/ddxplus/train_queries.csv"))
    parser.add_argument("--valid_csv", type=Path, default=Path("data/processed/ddxplus/valid_queries.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/checkpoints/ddxplus"))
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--n_neg", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_nodes(cell: str) -> List[str]:
    return [token.strip() for token in str(cell or "").split(";") if token.strip()]


def build_seed_vocab(*frames: pd.DataFrame) -> Dict[str, int]:
    nodes = set()
    for frame in frames:
        for cell in frame["seed_node_keys"].astype(str):
            nodes.update(split_nodes(cell))
    vocab = {"<PAD>": PAD_SYM, "<UNK>": UNK_SYM}
    for node in sorted(nodes):
        vocab[node] = len(vocab)
    return vocab


def build_disease_vocab(*frames: pd.DataFrame) -> Dict[str, int]:
    nodes = set()
    for frame in frames:
        for cell in frame["target_node_keys"].astype(str):
            nodes.update(split_nodes(cell))
    vocab = {"<PAD>": 0}
    for node in sorted(nodes):
        vocab[node] = len(vocab)
    return vocab


def encode_seed_nodes(cell: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(node, UNK_SYM) for node in split_nodes(cell)]
    ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(PAD_SYM)
    return ids


def positive_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[["patient_index", "seed_node_keys", "target_node_keys"]].copy()
    rows["target_node"] = rows["target_node_keys"].astype(str).str.split(";")
    rows = rows.explode("target_node", ignore_index=True)
    rows["target_node"] = rows["target_node"].astype(str).str.strip()
    return rows.loc[rows["target_node"] != "", ["patient_index", "seed_node_keys", "target_node"]]


class DDXPlusMindDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        seed_vocab: Dict[str, int],
        disease_vocab: Dict[str, int],
        max_seq_len: int,
    ) -> None:
        rows = frame.reset_index(drop=True)
        seeds = np.full((len(rows), max_seq_len), PAD_SYM, dtype=np.int64)
        for index, cell in enumerate(rows["seed_node_keys"].astype(str)):
            encoded = [seed_vocab.get(node, UNK_SYM) for node in split_nodes(cell)][:max_seq_len]
            seeds[index, : len(encoded)] = encoded
        positives = rows["target_node"].map(disease_vocab).to_numpy(dtype=np.int64)
        self.seeds = torch.from_numpy(seeds)
        self.positives = torch.from_numpy(positives)

    def __len__(self) -> int:
        return len(self.positives)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.seeds[idx], self.positives[idx]


def collate_fn(batch):
    seeds = torch.stack([item[0] for item in batch], dim=0)
    pos = torch.stack([item[1] for item in batch], dim=0)
    return seeds, pos


def sample_negatives(pos: torch.Tensor, n_neg: int, num_diseases: int) -> torch.Tensor:
    if num_diseases <= 2:
        return torch.ones((pos.shape[0], n_neg), dtype=torch.long, device=pos.device)
    neg = torch.randint(1, num_diseases - 1, (pos.shape[0], n_neg), device=pos.device)
    return neg + (neg >= pos.unsqueeze(1)).long()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_queries = pd.read_csv(args.train_csv)
    valid_queries = pd.read_csv(args.valid_csv) if args.valid_csv.is_file() else train_queries.head(0)
    train_pos = positive_rows(train_queries)
    valid_pos = positive_rows(valid_queries)
    if args.max_train_samples is not None:
        train_pos = train_pos.head(args.max_train_samples).copy()
    if args.max_valid_samples is not None:
        valid_pos = valid_pos.head(args.max_valid_samples).copy()

    seed_vocab = build_seed_vocab(train_queries, valid_queries)
    disease_vocab = build_disease_vocab(train_queries, valid_queries)
    train_ds = DDXPlusMindDataset(
        train_pos,
        seed_vocab,
        disease_vocab,
        args.max_seq_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = None
    if not valid_pos.empty:
        valid_ds = DDXPlusMindDataset(valid_pos, seed_vocab, disease_vocab, args.max_seq_len)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    device = torch.device(args.device)
    model = ClinicalMIND(
        num_symptoms=len(seed_vocab),
        num_diseases=len(disease_vocab),
        dim=args.D,
        num_interests=args.K,
        max_seq_len=args.max_seq_len,
        num_routing_iters=args.R,
        symptom_padding_idx=PAD_SYM,
        disease_padding_idx=0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def evaluate() -> Tuple[float, float]:
        if valid_loader is None:
            return 0.0, 0.0
        model.eval()
        losses: List[float] = []
        sims: List[float] = []
        with torch.no_grad():
            for seeds, pos in valid_loader:
                seeds = seeds.to(device)
                pos = pos.to(device)
                neg = sample_negatives(pos, args.n_neg, len(disease_vocab))
                loss = training_bce_loss(model, seeds, pos, neg, temperature=args.temperature)
                z, _, active_mask = model.encode_symptoms(seeds)
                losses.append(float(loss.item()))
                sims.append(float(average_active_interest_cosine_similarity(z, active_mask).item()))
        model.train()
        return float(np.mean(losses)), float(np.mean(sims))

    best_state = None
    best_valid = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for seeds, pos in train_loader:
            seeds = seeds.to(device)
            pos = pos.to(device)
            neg = sample_negatives(pos, args.n_neg, len(disease_vocab))
            optimizer.zero_grad(set_to_none=True)
            loss = training_bce_loss(model, seeds, pos, neg, temperature=args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * seeds.shape[0]
            seen += seeds.shape[0]
        valid_loss, valid_sim = evaluate()
        if valid_pos.empty or valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch}/{args.epochs} train_loss={total / max(1, seen):.4f} "
            f"valid_loss={valid_loss:.4f} avg_interest_sim={valid_sim:.4f}"
        )

    checkpoint = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "symptom_str2id": seed_vocab,
        "disease_str2id": disease_vocab,
        "hparams": {
            "D": args.D,
            "K": args.K,
            "R": args.R,
            "max_seq_len": args.max_seq_len,
            "num_symptoms": len(seed_vocab),
            "num_diseases": len(disease_vocab),
            "temperature": args.temperature,
            "input_space": "primekg_seed_node_keys",
            "target_space": "primekg_disease_node_keys",
        },
    }
    ckpt_path = args.out_dir / f"clinical_mind_ddxplus_k{args.K}.pt"
    torch.save(checkpoint, ckpt_path)
    with (args.out_dir / f"clinical_mind_ddxplus_k{args.K}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(checkpoint["hparams"], handle, indent=2, ensure_ascii=False)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
