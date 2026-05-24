"""
Train Clinical MIND (multi-interest symptom encoder) on MedMIG-CR interactions.

Reads:
  - processed_medmigcr_dataset/query_nodes.csv
  - processed_medmigcr_dataset/interactions_train.csv
  - processed_medmigcr_dataset/interactions_valid.csv (optional validation)

Saves:
  - multi-interest/checkpoints/clinical_mind.pt (weights + vocab + hparams)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mind_medmigcr_model import (
    ClinicalMIND,
    average_active_interest_cosine_similarity,
    training_bce_loss,
)


PAD_SYM = 0
UNK_SYM = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Clinical MIND on MedMIG-CR")
    p.add_argument("--data_dir", type=str, default="processed_medmigcr_dataset")
    p.add_argument("--out_dir", type=str, default="multi-interest/checkpoints")
    p.add_argument("--max_seq_len", type=int, default=16)
    p.add_argument("--D", type=int, default=64, help="Embedding / capsule dimension")
    p.add_argument("--K", type=int, default=5, help="Number of interest capsules")
    p.add_argument("--R", type=int, default=3, help="Dynamic routing iterations")
    p.add_argument("--n_neg", type=int, default=8, help="Random negatives per positive")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_train_samples", type=int, default=None, help="Cap rows for debugging")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_symptom_vocab(query_df: pd.DataFrame) -> Dict[str, int]:
    syms: set[str] = set()
    for cell in query_df["symptom_entity_ids"].astype(str):
        for s in cell.split(";"):
            s = s.strip()
            if s:
                syms.add(s)
    symptom_str2id: Dict[str, int] = {"<PAD>": PAD_SYM, "<UNK>": UNK_SYM}
    nxt = UNK_SYM + 1
    for s in sorted(syms):
        symptom_str2id[s] = nxt
        nxt += 1
    return symptom_str2id


def build_disease_vocab(pos_df: pd.DataFrame) -> Dict[str, int]:
    diseases = sorted(pos_df["disease_id"].astype(str).unique())
    disease_str2id: Dict[str, int] = {"<PAD>": 0}
    for i, d in enumerate(diseases, start=1):
        disease_str2id[d] = i
    return disease_str2id


def encode_symptom_row(cell: str, symptom_str2id: Dict[str, int], max_len: int) -> List[int]:
    ids: List[int] = []
    for s in cell.split(";"):
        s = s.strip()
        if not s:
            continue
        ids.append(symptom_str2id.get(s, UNK_SYM))
    if len(ids) > max_len:
        ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(PAD_SYM)
    return ids


class QueryDiseaseDataset(Dataset):
    """One row = one (query_id, positive disease) from label==1 interactions."""

    def __init__(
        self,
        pos_df: pd.DataFrame,
        query_symptoms: Dict[str, List[int]],
        disease_str2id: Dict[str, int],
        disease_ids_for_neg: np.ndarray,
        n_neg: int,
    ) -> None:
        self.rows = pos_df.reset_index(drop=True)
        self.query_symptoms = query_symptoms
        self.disease_str2id = disease_str2id
        self.disease_ids_for_neg = disease_ids_for_neg
        self.n_neg = n_neg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.rows.iloc[idx]
        qid = str(r["query_id"])
        dis = str(r["disease_id"])
        sym = torch.tensor(self.query_symptoms[qid], dtype=torch.long)
        pos = torch.tensor(self.disease_str2id[dis], dtype=torch.long)
        # random negatives (exclude padding index 0)
        pos_idx = int(pos.item())
        neg = []
        pool = self.disease_ids_for_neg[self.disease_ids_for_neg != pos_idx]
        if len(pool) == 0:
            pool = self.disease_ids_for_neg
        for _ in range(self.n_neg):
            neg.append(int(np.random.choice(pool)))
        neg_t = torch.tensor(neg[: self.n_neg], dtype=torch.long)
        return sym, pos, neg_t


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    syms = torch.stack([b[0] for b in batch], dim=0)
    pos = torch.stack([b[1] for b in batch], dim=0)
    neg = torch.stack([b[2] for b in batch], dim=0)
    return syms, pos, neg


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query_path = data_dir / "query_nodes.csv"
    train_path = data_dir / "interactions_train.csv"
    if not query_path.is_file() or not train_path.is_file():
        raise FileNotFoundError(f"Need {query_path} and {train_path}")

    query_df = pd.read_csv(query_path)
    train_int = pd.read_csv(train_path)
    train_pos = train_int[train_int["label"] == 1].copy()
    if args.max_train_samples:
        train_pos = train_pos.head(int(args.max_train_samples)).copy()

    valid_path = data_dir / "interactions_valid.csv"
    if valid_path.is_file():
        valid_int = pd.read_csv(valid_path)
        valid_pos = valid_int[valid_int["label"] == 1].copy()
    else:
        valid_pos = train_pos.head(0)

    symptom_str2id = build_symptom_vocab(query_df)
    disease_str2id = build_disease_vocab(
        pd.concat([train_pos, valid_pos], axis=0, ignore_index=True)
        if len(valid_pos) > 0
        else train_pos
    )
    num_symptoms = max(symptom_str2id.values()) + 1
    num_diseases = max(disease_str2id.values()) + 1

    query_symptoms: Dict[str, List[int]] = {}
    for r in query_df.itertuples(index=False):
        qid = str(r.query_id)
        query_symptoms[qid] = encode_symptom_row(str(r.symptom_entity_ids), symptom_str2id, args.max_seq_len)

    disease_ids_for_neg = np.array([i for i in range(1, num_diseases)], dtype=np.int64)

    train_ds = QueryDiseaseDataset(
        train_pos, query_symptoms, disease_str2id, disease_ids_for_neg, n_neg=args.n_neg
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
        num_symptoms=num_symptoms,
        num_diseases=num_diseases,
        dim=args.D,
        num_interests=args.K,
        max_seq_len=args.max_seq_len,
        num_routing_iters=args.R,
        symptom_padding_idx=PAD_SYM,
        disease_padding_idx=0,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_eval(valid_pos: pd.DataFrame) -> tuple[float, float]:
        model.eval()
        losses: List[float] = []
        sims: List[float] = []
        vds = QueryDiseaseDataset(
            valid_pos, query_symptoms, disease_str2id, disease_ids_for_neg, n_neg=args.n_neg
        )
        loader = DataLoader(vds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        with torch.no_grad():
            for syms, pos, neg in loader:
                syms = syms.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                loss = training_bce_loss(model, syms, pos, neg, temperature=args.temperature)
                z, _, active_mask = model.encode_symptoms(syms)
                sim = average_active_interest_cosine_similarity(z, active_mask)
                losses.append(float(loss.item()))
                sims.append(float(sim.item()))
        model.train()
        return (
            float(np.mean(losses)) if losses else 0.0,
            float(np.mean(sims)) if sims else 0.0,
        )

    best_val_loss = float("inf")
    best_state: dict = {}
    best_epoch = -1
    prev_val_loss = float("inf")
    prev_val_sim = 0.0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        total = 0.0
        n = 0
        for syms, pos, neg in train_loader:
            syms = syms.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            opt.zero_grad(set_to_none=True)
            loss = training_bce_loss(model, syms, pos, neg, temperature=args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * syms.size(0)
            n += syms.size(0)
        train_loss = total / max(n, 1)
        msg = f"epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f}"
        early_stop = False
        if len(valid_pos) > 0:
            valid_loss, valid_sim = run_eval(valid_pos)
            msg += f" valid_loss={valid_loss:.4f} avg_interest_sim={valid_sim:.4f}"
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
            if (
                epoch >= 6
                and valid_loss > prev_val_loss
                and valid_sim > 0.9
                and valid_sim > prev_val_sim
            ):
                early_stop = True
                msg += " EARLY_STOP_TRIGGERED"
            prev_val_loss = valid_loss
            prev_val_sim = valid_sim
        print(msg)
        if early_stop:
            print(f"Stopping early at epoch {epoch+1}; best epoch={best_epoch} with valid_loss={best_val_loss:.4f}")
            break

    if best_state:
        ckpt_model_state = best_state
    else:
        ckpt_model_state = model.state_dict()

    ckpt = {
        "model_state": ckpt_model_state,
        "symptom_str2id": symptom_str2id,
        "disease_str2id": disease_str2id,
        "hparams": {
            "D": args.D,
            "K": args.K,
            "R": args.R,
            "max_seq_len": args.max_seq_len,
            "num_symptoms": num_symptoms,
            "num_diseases": num_diseases,
            "temperature": args.temperature,
        },
    }
    ckpt_path = out_dir / "clinical_mind.pt"
    torch.save(ckpt, ckpt_path)
    with open(out_dir / "clinical_mind_meta.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in ckpt["hparams"].items()}, f, indent=2)
    print(f"Saved best checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
