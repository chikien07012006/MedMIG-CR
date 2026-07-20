from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medmigcr_kg.graph_store import GraphStore  # noqa: E402
from medmigcr_kg.retrieval_engine import RetrievalEngine  # noqa: E402
from medmigcr_mind.model import ClinicalMIND  # noqa: E402


PAD_SYM = 0
UNK_SYM = 1


def split_nodes(cell: str) -> List[str]:
    return [token.strip() for token in str(cell or "").split(";") if token.strip()]


def load_queries(path: Path, limit: int | None = None) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit is not None else rows


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_mind(path: Path, device: torch.device) -> Tuple[ClinicalMIND, dict, Dict[str, int]]:
    ckpt = load_checkpoint(path, device)
    hp = ckpt["hparams"]
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
    return model, hp, ckpt["symptom_str2id"]


def encode_seed_nodes(seed_node_keys: Sequence[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(key, UNK_SYM) for key in seed_node_keys]
    ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(PAD_SYM)
    return ids


def resolve_seed_ids(graph_store: GraphStore, seed_node_keys: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for key in seed_node_keys:
        node_id = graph_store.lookup_node_id(key)
        if node_id is not None:
            ids.append(node_id)
    return sorted(set(ids))


def is_disease_node(node_name: str | None) -> bool:
    return bool(node_name and node_name.startswith("disease|"))


def disease_endpoint_scores(result, graph_store: GraphStore, top_k: int) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for item in result.paths:
        node_name = graph_store.lookup_node_name(item.current_node)
        if not is_disease_node(node_name):
            continue
        current = scores.get(node_name)
        if current is None or item.score > current:
            scores[node_name] = float(item.score)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]


def write_predictions(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_index", "candidate", "score", "rank"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_projection(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = np.load(path)
    return {"weight": data["weight"], "bias": data["bias"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIND-driven beam-search retrieval on DDXPlus test queries.")
    parser.add_argument("--test_queries_csv", type=Path, default=Path("data/processed/ddxplus/test_queries.csv"))
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--projection", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--interest_count", type=int, default=None)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--beam_width", type=int, default=32)
    parser.add_argument("--paths_per_interest", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--limit_patients", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph_device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    mind, hp, seed_vocab = load_mind(args.checkpoint, device)
    use_interests = int(args.interest_count or hp["K"])
    if use_interests < 1 or use_interests > int(hp["K"]):
        raise ValueError(f"--interest_count must be between 1 and checkpoint K={hp['K']}")

    graph_store = GraphStore.load(
        graph_npz=args.graph_dir / "graph_csr.npz",
        node_embeddings_npy=args.graph_dir / "node_embeddings.npy",
        out_degree_npy=args.graph_dir / "out_degree.npy",
        in_degree_npy=args.graph_dir / "in_degree.npy",
        mapping_dir=args.graph_dir / "mappings",
        device=args.graph_device,
    )
    engine = RetrievalEngine(graph_store, projection=load_projection(args.projection))
    queries = load_queries(args.test_queries_csv, limit=args.limit_patients)

    rows: List[Dict[str, object]] = []
    skipped_missing_seed = 0
    no_disease_candidates = 0
    unk_seed_queries = 0
    for query in queries:
        patient_index = int(query["patient_index"])
        seed_keys = split_nodes(query.get("seed_node_keys", ""))
        seed_ids = resolve_seed_ids(graph_store, seed_keys)
        if not seed_ids:
            skipped_missing_seed += 1
            continue
        encoded = encode_seed_nodes(seed_keys, seed_vocab, int(hp["max_seq_len"]))
        if any(seed_vocab.get(key, UNK_SYM) == UNK_SYM for key in seed_keys):
            unk_seed_queries += 1
        x = torch.tensor([encoded], dtype=torch.long, device=device)
        with torch.no_grad():
            z = mind(x)[0, :use_interests, :].detach().cpu().numpy()
        result = engine.retrieve(
            seed_node_ids=seed_ids,
            max_hops=args.max_hops,
            beam_width=args.beam_width,
            topk_paths=args.paths_per_interest,
            alpha=args.alpha,
            beta=args.beta,
            interest_vectors=z,
            max_paths_per_interest=args.paths_per_interest,
        )
        ranked = disease_endpoint_scores(result, graph_store, top_k=args.top_k)
        if not ranked:
            no_disease_candidates += 1
            continue
        for rank, (candidate, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "patient_index": patient_index,
                    "candidate": candidate,
                    "score": f"{score:.8f}",
                    "rank": rank,
                }
            )

    write_predictions(args.output_csv, rows)
    summary = {
        "test_queries_csv": str(args.test_queries_csv),
        "graph_dir": str(args.graph_dir),
        "checkpoint": str(args.checkpoint),
        "projection": str(args.projection) if args.projection else None,
        "output_csv": str(args.output_csv),
        "num_queries_loaded": len(queries),
        "num_prediction_rows": len(rows),
        "skipped_missing_seed": skipped_missing_seed,
        "no_disease_candidates": no_disease_candidates,
        "queries_with_unknown_seed_tokens": unk_seed_queries,
        "checkpoint_k": int(hp["K"]),
        "interest_count_used": use_interests,
        "max_hops": args.max_hops,
        "beam_width": args.beam_width,
        "paths_per_interest": args.paths_per_interest,
        "top_k": args.top_k,
    }
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
