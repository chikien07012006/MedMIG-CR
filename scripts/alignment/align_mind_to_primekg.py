from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medmigcr_kg.graph_store import GraphStore  # noqa: E402


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def fit_linear_projection(x: np.ndarray, y: np.ndarray, ridge_alpha: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x_reg = np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])
    reg = ridge_alpha * np.eye(x_reg.shape[1], dtype=x_reg.dtype)
    reg[-1, -1] = 0.0
    w_plus = np.linalg.solve(x_reg.T @ x_reg + reg, x_reg.T @ y)
    weight = w_plus[:-1, :]
    bias = w_plus[-1, :]
    pred = x @ weight + bias
    mse = float(np.mean((pred - y) ** 2))
    before_cos = cosine_mean(x, y) if x.shape[1] == y.shape[1] else float("nan")
    after_cos = cosine_mean(pred, y)
    return weight.astype(np.float32), bias.astype(np.float32), {
        "mse": mse,
        "cosine_before": before_cos,
        "cosine_after": after_cos,
        "ridge_alpha": float(ridge_alpha),
    }


def cosine_mean(x: np.ndarray, y: np.ndarray) -> float:
    num = np.sum(x * y, axis=1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-9
    return float(np.mean(num / den))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a linear projection from MIND disease embeddings to PrimeKG node embeddings.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--min_overlap", type=int, default=10)
    parser.add_argument("--ridge_alpha", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    disease_vocab: Dict[str, int] = ckpt["disease_str2id"]
    disease_emb = ckpt["model_state"]["disease_emb.weight"].detach().cpu().numpy()
    graph_store = GraphStore.load(
        graph_npz=args.graph_dir / "graph_csr.npz",
        node_embeddings_npy=args.graph_dir / "node_embeddings.npy",
        out_degree_npy=args.graph_dir / "out_degree.npy",
        in_degree_npy=args.graph_dir / "in_degree.npy",
        mapping_dir=args.graph_dir / "mappings",
        device=None,
    )

    x_rows = []
    y_rows = []
    anchors = []
    for node_key, mind_id in disease_vocab.items():
        if node_key == "<PAD>":
            continue
        graph_id = graph_store.lookup_node_id(node_key)
        if graph_id is None:
            continue
        x_rows.append(disease_emb[int(mind_id)])
        y_rows.append(graph_store.node_embeddings[int(graph_id)])
        anchors.append(node_key)

    if len(anchors) < args.min_overlap:
        raise RuntimeError(f"Only {len(anchors)} overlapping disease anchors; need at least {args.min_overlap}.")

    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    weight, bias, metrics = fit_linear_projection(x, y, ridge_alpha=args.ridge_alpha)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, weight=weight, bias=bias)
    summary = {
        "checkpoint": str(args.checkpoint),
        "graph_dir": str(args.graph_dir),
        "output_npz": str(args.output_npz),
        "num_anchors": len(anchors),
        "mind_dim": int(x.shape[1]),
        "graph_dim": int(y.shape[1]),
        **metrics,
        "anchor_examples": anchors[:20],
    }
    summary_path = args.summary_json or args.output_npz.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
