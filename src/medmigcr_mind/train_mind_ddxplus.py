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
    rows = []
    for row in frame.itertuples(index=False):
        for target in split_nodes(str(row.target_node_keys)):
            rows.append(
                {
                    "patient_index": int(row.patient_index),
                    "seed_node_keys": str(row.seed_node_keys),
                    "target_node": target,
                }
            )
    return pd.DataFrame(rows)


class DDXPlusMindDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        seed_vocab: Dict[str, int],
        disease_vocab: Dict[str, int],
        disease_ids_for_neg: np.ndarray,
        max_seq_len: int,
        n_neg: int,
    ) -> None:
        self.rows = frame.reset_index(drop=True)
        self.seed_vocab = seed_vocab
        self.disease_vocab = disease_vocab
        self.disease_ids_for_neg = disease_ids_for_neg
        self.max_seq_len = max_seq_len
        self.n_neg = n_neg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[idx]
        seed_ids = encode_seed_nodes(str(row["seed_node_keys"]), self.seed_vocab, self.max_seq_len)
        pos = int(self.disease_vocab[str(row["target_node"])])
        pool = self.disease_ids_for_neg[self.disease_ids_for_neg != pos]
        if len(pool) == 0:
            pool = self.disease_ids_for_neg
        neg = np.random.choice(pool, size=self.n_neg, replace=True)
        return (
            torch.tensor(seed_ids, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def collate_fn(batch):
    seeds = torch.stack([item[0] for item in batch], dim=0)
    pos = torch.stack([item[1] for item in batch], dim=0)
    neg = torch.stack([item[2] for item in batch], dim=0)
    return seeds, pos, neg


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
    disease_ids_for_neg = np.asarray([idx for idx in range(1, len(disease_vocab))], dtype=np.int64)

    train_ds = DDXPlusMindDataset(
        train_pos,
        seed_vocab,
        disease_vocab,
        disease_ids_for_neg,
        args.max_seq_len,
        args.n_neg,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
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
        if valid_pos.empty:
            return 0.0, 0.0
        valid_ds = DDXPlusMindDataset(
            valid_pos,
            seed_vocab,
            disease_vocab,
            disease_ids_for_neg,
            args.max_seq_len,
            args.n_neg,
        )
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        model.eval()
        losses: List[float] = []
        sims: List[float] = []
        with torch.no_grad():
            for seeds, pos, neg in valid_loader:
                seeds = seeds.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
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
        for seeds, pos, neg in train_loader:
            seeds = seeds.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
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
