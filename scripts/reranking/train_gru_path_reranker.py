from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medmigcr_path_reranker.gru_reranker import GRUPathReranker, GRUPathRerankerConfig  


Record = Dict[str, object]


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_records(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_patient(records: List[Record]) -> Dict[int, Dict[str, List[Record]]]:
    grouped: Dict[int, Dict[str, List[Record]]] = defaultdict(lambda: {"pos": [], "neg": []})
    for record in records:
        key = int(record["patient_index"])
        bucket = "pos" if int(record["label"]) == 1 else "neg"
        grouped[key][bucket].append(record)
    for buckets in grouped.values():
        buckets["pos"].sort(key=lambda record: float(record["additive_score"]), reverse=True)
        buckets["neg"].sort(key=lambda record: float(record["additive_score"]), reverse=True)
    return grouped


def eligible_patients(grouped: Dict[int, Dict[str, List[Record]]]) -> List[int]:
    return [
        patient_index
        for patient_index, buckets in grouped.items()
        if buckets["pos"] and buckets["neg"]
    ]


def direction_to_id(direction: int) -> int:
    if direction > 0:
        return 1
    if direction < 0:
        return 2
    return 0


def make_batch(records: List[Record], device: torch.device) -> Dict[str, torch.Tensor]:
    max_len = max(1, max(int(record["path_len"]) for record in records))
    batch_size = len(records)
    relation_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    direction_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    node_type_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    dst_node_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    additive_scores = torch.zeros(batch_size, dtype=torch.float32)

    for row_idx, record in enumerate(records):
        rels = [int(value) + 1 for value in record["relation_ids"]]
        dirs = [direction_to_id(int(value)) for value in record["direction_ids"]]
        types = [int(value) for value in record["node_type_ids"]]
        dst = [int(value) for value in record["dst_node_ids"]]
        length = min(len(rels), max_len)
        relation_ids[row_idx, :length] = torch.tensor(rels[:length], dtype=torch.long)
        direction_ids[row_idx, :length] = torch.tensor(dirs[:length], dtype=torch.long)
        node_type_ids[row_idx, :length] = torch.tensor(types[:length], dtype=torch.long)
        dst_node_ids[row_idx, :length] = torch.tensor(dst[:length], dtype=torch.long)
        lengths[row_idx] = max(1, length)
        additive_scores[row_idx] = float(record["additive_score"])

    return {
        "relation_ids": relation_ids.to(device),
        "direction_ids": direction_ids.to(device),
        "node_type_ids": node_type_ids.to(device),
        "dst_node_ids": dst_node_ids.to(device),
        "lengths": lengths.to(device),
        "additive_scores": additive_scores.to(device),
    }


def sample_pair_batch(
    grouped: Dict[int, Dict[str, List[Record]]],
    patient_ids: List[int],
    batch_size: int,
    rng: random.Random,
    hard_negative_top_k: int | None,
) -> Tuple[List[Record], List[Record]]:
    pos_records: List[Record] = []
    neg_records: List[Record] = []
    for _ in range(batch_size):
        patient_index = rng.choice(patient_ids)
        buckets = grouped[patient_index]
        pos_records.append(rng.choice(buckets["pos"]))
        negative_pool = buckets["neg"]
        if hard_negative_top_k is not None and hard_negative_top_k > 0:
            negative_pool = negative_pool[:hard_negative_top_k]
        neg_records.append(rng.choice(negative_pool))
    return pos_records, neg_records


def evaluate_pair_accuracy(
    model: GRUPathReranker,
    grouped: Dict[int, Dict[str, List[Record]]],
    patient_ids: List[int],
    device: torch.device,
    rng: random.Random,
    num_pairs: int = 2048,
    batch_size: int = 256,
) -> Dict[str, float]:
    if not patient_ids:
        return {"pair_accuracy": 0.0, "mean_margin": 0.0}
    margins: List[float] = []
    model.eval()
    with torch.no_grad():
        remaining = num_pairs
        while remaining > 0:
            current = min(batch_size, remaining)
            pos_records, neg_records = sample_pair_batch(grouped, patient_ids, current, rng, hard_negative_top_k=None)
            pos_batch = make_batch(pos_records, device)
            neg_batch = make_batch(neg_records, device)
            pos_scores = model(**pos_batch)
            neg_scores = model(**neg_batch)
            margins.extend((pos_scores - neg_scores).detach().cpu().tolist())
            remaining -= current
    margins_arr = np.asarray(margins, dtype=np.float32)
    return {
        "pair_accuracy": float(np.mean(margins_arr > 0.0)) if margins_arr.size else 0.0,
        "mean_margin": float(np.mean(margins_arr)) if margins_arr.size else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRU path reranker with BPR loss.")
    parser.add_argument("--train_jsonl", type=Path, default=Path("data/processed/reranker/infonce_k3_train_paths.jsonl"))
    parser.add_argument("--metadata_json", type=Path, default=None)
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument("--out_checkpoint", type=Path, default=Path("artifacts/checkpoints/path_gru_reranker/infonce_k3_gru.pt"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps_per_epoch", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hard_negative_top_k", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--relation_dim", type=int, default=32)
    parser.add_argument("--direction_dim", type=int, default=8)
    parser.add_argument("--node_type_dim", type=int, default=8)
    parser.add_argument("--node_projection_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_score_features", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    metadata_path = args.metadata_json or args.train_jsonl.with_suffix(".metadata.json")
    metadata = load_json(metadata_path)
    records = load_records(args.train_jsonl)
    grouped = group_by_patient(records)
    patient_ids = eligible_patients(grouped)
    if not patient_ids:
        raise ValueError("No patient has both positive and negative paths; BPR training cannot run.")

    node_embeddings = torch.from_numpy(np.load(args.graph_dir / "node_embeddings.npy"))
    config = GRUPathRerankerConfig(
        num_relations=int(metadata["num_relations"]),
        num_node_types=len(metadata["node_type2id"]),
        graph_dim=int(node_embeddings.shape[1]),
        relation_dim=args.relation_dim,
        direction_dim=args.direction_dim,
        node_type_dim=args.node_type_dim,
        node_projection_dim=args.node_projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_score_features=args.use_score_features,
    )
    model = GRUPathReranker(config, node_embeddings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for _step in range(args.steps_per_epoch):
            pos_records, neg_records = sample_pair_batch(
                grouped,
                patient_ids,
                args.batch_size,
                rng,
                hard_negative_top_k=args.hard_negative_top_k,
            )
            pos_batch = make_batch(pos_records, device)
            neg_batch = make_batch(neg_records, device)
            pos_scores = model(**pos_batch)
            neg_scores = model(**neg_batch)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        eval_stats = evaluate_pair_accuracy(model, grouped, patient_ids, device, rng)
        row = {
            "epoch": float(epoch),
            "loss": float(np.mean(losses)),
            **eval_stats,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "node_type2id": metadata["node_type2id"],
        "train_jsonl": str(args.train_jsonl),
        "metadata_json": str(metadata_path),
        "history": history,
        "num_records": len(records),
        "num_bpr_patients": len(patient_ids),
        "hard_negative_top_k": args.hard_negative_top_k,
    }
    torch.save(checkpoint, args.out_checkpoint)
    summary_path = args.out_checkpoint.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                key: value
                for key, value in checkpoint.items()
                if key != "model_state"
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved GRU path reranker to {args.out_checkpoint}")


if __name__ == "__main__":
    main()
