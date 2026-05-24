"""
Load a trained Clinical MIND checkpoint and print multi-interest vectors Z ∈ R^{B×K×D}
for one or more clinical queries (symptom entity ID strings as in MedMIG-CR query_nodes.csv).

Example (from repo root):

  python multi-interest/test_multi_model.py \\
    --checkpoint multi-interest/checkpoints/clinical_mind.pt \\
    --symptoms "effect/phenotype|HPO|4322;effect/phenotype|HPO|962"

  python multi-interest/test_multi_model.py \\
    --checkpoint multi-interest/checkpoints/clinical_mind.pt \\
    --query_csv processed_medmigcr_dataset/query_nodes.csv \\
    --query_ids Q0000000 Q0000001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from mind_medmigcr_model import ClinicalMIND


PAD_SYM = 0
UNK_SYM = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Clinical MIND multi-interest encoder")
    p.add_argument("--checkpoint", type=str, default="multi-interest/checkpoints/clinical_mind.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--symptoms",
        type=str,
        default=None,
        help="Semicolon-separated symptom keys (same format as query_nodes symptom_entity_ids)",
    )
    p.add_argument("--query_csv", type=str, default=None, help="Optional query_nodes.csv path")
    p.add_argument(
        "--query_ids",
        nargs="*",
        default=None,
        help="query_id values to load from --query_csv (e.g. Q0000000)",
    )
    p.add_argument("--max_seq_len", type=int, default=None, help="Override if missing from checkpoint")
    return p.parse_args()


def encode_symptom_string(
    cell: str,
    symptom_str2id: Dict[str, int],
    max_len: int,
) -> List[int]:
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


def load_model(ckpt_path: Path, device: torch.device) -> tuple[ClinicalMIND, dict, Dict[str, int]]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt["hparams"]
    symptom_str2id: Dict[str, int] = ckpt["symptom_str2id"]
    model = ClinicalMIND(
        num_symptoms=hp["num_symptoms"],
        num_diseases=hp["num_diseases"],
        dim=hp["D"],
        num_interests=hp["K"],
        max_seq_len=hp["max_seq_len"],
        num_routing_iters=hp["R"],
        symptom_padding_idx=PAD_SYM,
        disease_padding_idx=0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, hp, symptom_str2id


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, hp, symptom_str2id = load_model(ckpt_path, device)
    max_len = int(args.max_seq_len or hp["max_seq_len"])

    batches_sym: List[torch.Tensor] = []
    labels: List[str] = []

    if args.symptoms:
        row = encode_symptom_string(args.symptoms, symptom_str2id, max_len)
        batches_sym.append(torch.tensor([row], dtype=torch.long, device=device))
        labels.append("CLI")

    if args.query_csv and args.query_ids:
        qdf = pd.read_csv(args.query_csv)
        qdf = qdf[qdf["query_id"].astype(str).isin(set(args.query_ids))]
        for r in qdf.itertuples(index=False):
            row = encode_symptom_string(str(r.symptom_entity_ids), symptom_str2id, max_len)
            batches_sym.append(torch.tensor([row], dtype=torch.long, device=device))
            labels.append(str(r.query_id))

    if not batches_sym:
        raise SystemExit("Provide --symptoms ... and/or --query_csv with --query_ids ...")

    x = torch.cat(batches_sym, dim=0)
    with torch.no_grad():
        z, _, active_mask = model.encode_symptoms(x)

    print("hparams:", json.dumps({k: hp[k] for k in ("D", "K", "R", "max_seq_len")}, indent=2))
    print("Z shape (B, K, D) =", tuple(z.shape))
    zn = torch.norm(z, p=2, dim=-1).cpu().numpy()
    for i, name in enumerate(labels):
        active_count = int(active_mask[i].sum().item())
        num_symptoms = int((x[i] != PAD_SYM).sum().item())
        print(f"\n--- {name} ---")
        print(f"  num symptoms = {num_symptoms}")
        print(f"  K_q' = {active_count}")
        print(f"  active interests = {list(range(active_count))}")
        if active_count > 0:
            z_i = z[i, :active_count, :]
            z_i_norm = torch.nn.functional.normalize(z_i, dim=-1)
            sim = torch.matmul(z_i_norm, z_i_norm.transpose(0, 1)).cpu().numpy()
            print("  cosine similarity matrix between active interests:")
            print(np.round(sim, 4))
        # Only print active interests
        for k in range(active_count):
            print(f"  interest {k}: L2 norm = {zn[i, k]:.4f}, first 8 dims = {z[i, k, :8].cpu().numpy().round(4)}")
        print(f"  full Z[{i}] (active K_q' x min(8,D)) sample:\n{z[i, :active_count, : min(8, z.shape[2])].cpu().numpy()}")


if __name__ == "__main__":
    main()
