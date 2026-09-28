from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import islice
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medmigcr_kg.graph_store import GraphStore  # noqa: E402
from medmigcr_kg.beam_search import BeamScore  # noqa: E402
from medmigcr_kg.retrieval_engine import RetrievalEngine  # noqa: E402
from medmigcr_kg.telemetry import log_run_summary, reset_peak_memory  # noqa: E402
from medmigcr_mind.contrastive_model import ProjectedClinicalMIND, average_pairwise_cosine  # noqa: E402
from medmigcr_path_reranker.gru_reranker import GRUPathReranker, GRUPathRerankerConfig  # noqa: E402


PAD_SYM = 0
UNK_SYM = 1


def split_nodes(cell: str) -> List[str]:
    return [token.strip() for token in str(cell or "").split(";") if token.strip()]


def load_queries(path: Path, limit: int | None = None) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(tqdm(islice(reader, limit), total=limit, desc="Loading queries", unit="query")) if limit is not None else list(tqdm(reader, desc="Loading queries", unit="query"))
    return rows


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_mind_model(path: Path, device: torch.device) -> Tuple[ProjectedClinicalMIND, dict, Dict[str, int]]:
    ckpt = load_checkpoint(path, device)
    hp = ckpt["hparams"]
    model = ProjectedClinicalMIND(
        num_symptoms=hp["num_symptoms"],
        mind_dim=hp["D"],
        graph_dim=hp["graph_dim"],
        num_interests=hp["K"],
        max_seq_len=hp["max_seq_len"],
        num_routing_iters=hp["R"],
        symptom_padding_idx=PAD_SYM,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, hp, ckpt["symptom_str2id"]


def load_reranker(path: Path, graph_dir: Path, device: torch.device) -> Tuple[GRUPathReranker, dict]:
    ckpt = load_checkpoint(path, device)
    config = GRUPathRerankerConfig(**ckpt["config"])
    node_embeddings = torch.from_numpy(np.load(graph_dir / "node_embeddings.npy"))
    model = GRUPathReranker(config, node_embeddings)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def encode_seed_nodes(seed_node_keys: Sequence[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(key, UNK_SYM) for key in seed_node_keys]
    ids = ids[:max_len]
    ids.extend([PAD_SYM] * (max_len - len(ids)))
    return ids


def resolve_seed_ids(graph_store: GraphStore, seed_node_keys: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for key in seed_node_keys:
        node_id = graph_store.lookup_node_id(key)
        if node_id is not None:
            ids.append(node_id)
    return sorted(set(ids))


def load_target_universe(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8-sig") as handle:
        mapping = json.load(handle)
    return {
        node
        for entry in mapping.values()
        for node in entry.get("selected_primekg_nodes", [])
    }


def node_type(node_key: str) -> str:
    return node_key.split("|", 1)[0] if "|" in node_key else "unknown"


def endpoint_is_disease(node_name: str | None) -> bool:
    return bool(node_name and node_name.startswith("disease|"))


def direction_to_id(direction: int) -> int:
    if direction > 0:
        return 1
    if direction < 0:
        return 2
    return 0


def path_to_record(
    graph_store: GraphStore,
    node_type2id: Dict[str, int],
    target_universe: set[str] | None,
    item,
) -> Dict[str, object] | None:
    endpoint = graph_store.lookup_node_name(item.current_node)
    if not endpoint_is_disease(endpoint):
        return None
    if target_universe is not None and endpoint not in target_universe:
        return None
    features = beam_item_to_gru_features(graph_store, node_type2id, item)
    if features is None:
        return None
    return {"endpoint": endpoint, **features}


def beam_endpoint_scores(result, graph_store: GraphStore, target_universe: set[str] | None) -> List[Tuple[str, float]]:
    endpoint_scores: Dict[str, float] = {}
    for item in result.paths:
        endpoint = graph_store.lookup_node_name(item.current_node)
        if not endpoint:
            continue
        if target_universe is not None and endpoint not in target_universe:
            continue
        if target_universe is None and not endpoint_is_disease(endpoint):
            continue
        current = endpoint_scores.get(endpoint)
        if current is None or item.additive_score > current:
            endpoint_scores[endpoint] = float(item.additive_score)
    return sorted(endpoint_scores.items(), key=lambda pair: pair[1], reverse=True)


def beam_item_to_gru_features(
    graph_store: GraphStore,
    node_type2id: Dict[str, int],
    item,
) -> Dict[str, object] | None:
    """Convert any valid beam prefix into the feature format expected by the GRU."""
    if len(item.path) < 2 or len(item.relation_path) != len(item.path) - 1:
        return None
    if any(int(rel_id) < 0 for rel_id in item.relation_path):
        return None
    if any(int(direction) == 0 for direction in item.direction_path):
        return None

    dst_node_ids = list(item.path[1:])
    node_type_ids = [
        node_type2id.get(node_type(graph_store.lookup_node_name(node_id) or ""), 0)
        for node_id in dst_node_ids
    ]
    return {
        "additive_score": float(item.additive_score),
        "dst_node_ids": dst_node_ids,
        "relation_ids": list(item.relation_path),
        "direction_ids": list(item.direction_path),
        "node_type_ids": node_type_ids,
        "path_len": len(item.path) - 1,
    }


def make_batch(records: List[Dict[str, object]], device: torch.device) -> Dict[str, torch.Tensor]:
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


def score_records(
    model: GRUPathReranker,
    records: List[Dict[str, object]],
    device: torch.device,
    batch_size: int,
) -> List[float]:
    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = make_batch(records[start : start + batch_size], device)
            batch_scores = model(**batch).detach().cpu().tolist()
            scores.extend(float(score) for score in batch_scores)
    return scores


def score_beam_prefixes_with_gru(
    model: GRUPathReranker,
    graph_store: GraphStore,
    node_type2id: Dict[str, int],
    items,
    device: torch.device,
    batch_size: int,
) -> List[float | BeamScore]:
    """Score one new token per candidate and return its updated hidden state."""
    scores: List[object] = [float(item.score) for item in items]
    valid_indices: List[int] = []
    records: List[Dict[str, object]] = []
    for index, item in enumerate(items):
        features = beam_item_to_gru_features(graph_store, node_type2id, item)
        if features is not None:
            valid_indices.append(index)
            records.append(features)

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch_indices = valid_indices[start : start + batch_size]
        relation_ids = torch.tensor(
            [int(record["relation_ids"][-1]) + 1 for record in batch_records],
            dtype=torch.long,
            device=device,
        )
        direction_ids = torch.tensor(
            [direction_to_id(int(record["direction_ids"][-1])) for record in batch_records],
            dtype=torch.long,
            device=device,
        )
        node_type_ids = torch.tensor(
            [int(record["node_type_ids"][-1]) for record in batch_records],
            dtype=torch.long,
            device=device,
        )
        dst_node_ids = torch.tensor(
            [int(record["dst_node_ids"][-1]) for record in batch_records],
            dtype=torch.long,
            device=device,
        )
        lengths = torch.tensor(
            [int(record["path_len"]) for record in batch_records],
            dtype=torch.long,
            device=device,
        )
        additive_scores = torch.tensor(
            [float(record["additive_score"]) for record in batch_records],
            dtype=torch.float32,
            device=device,
        )
        parent_hidden = []
        for index in batch_indices:
            hidden = items[index].gru_hidden_state
            if hidden is None:
                hidden = torch.zeros(
                    (1, model.config.hidden_dim),
                    dtype=torch.float32,
                    device=device,
                )
            elif hidden.ndim == 3:
                hidden = hidden[:, 0, :]
            parent_hidden.append(hidden.to(device))
        hidden_batch = torch.stack(parent_hidden, dim=1)
        batch_scores, new_hidden = model.forward_incremental(
            relation_ids=relation_ids,
            direction_ids=direction_ids,
            node_type_ids=node_type_ids,
            dst_node_ids=dst_node_ids,
            hidden=hidden_batch,
            lengths=lengths,
            additive_scores=additive_scores,
            max_len=max(int(record["path_len"]) for record in batch_records),
        )
        for offset, (index, score) in enumerate(zip(batch_indices, batch_scores.detach().cpu().tolist())):
            scores[index] = BeamScore(
                score=float(score),
                gru_hidden_state=new_hidden[:, offset, :].detach(),
            )
    return scores


def reranked_endpoint_scores(
    model: GRUPathReranker,
    records: List[Dict[str, object]],
    device: torch.device,
    batch_size: int,
    top_k: int,
) -> List[Tuple[str, float]]:
    if not records:
        return []
    scores = score_records(model, records, device, batch_size)
    endpoint_scores: Dict[str, float] = {}
    for record, score in zip(records, scores):
        endpoint = str(record["endpoint"])
        current = endpoint_scores.get(endpoint)
        if current is None or score > current:
            endpoint_scores[endpoint] = score
    return sorted(endpoint_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]


def write_predictions(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_index", "candidate", "score", "rank"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InfoNCE MIND retrieval with a post-hoc GRU path reranker.")
    parser.add_argument("--test_queries_csv", type=Path, default=Path("data/processed/ddxplus_v2/test_queries.csv"))
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument(
        "--mind_checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ddxplus_infonce_e10_hop10/k3/clinical_mind_infonce_k3.pt"),
    )
    parser.add_argument(
        "--reranker_checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/path_gru_reranker/infonce_e10_k3_hop10_bw128_gru_100k.pt"),
    )
    parser.add_argument("--condition_map", type=Path, default=Path("data/mappings/ddxplus_v2/condition_to_primekg.json"))
    parser.add_argument("--output_csv", type=Path, default=Path("results/infonce_e10_hop10_beam128_gru_rerank_test5000/k3/predictions.csv"))
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--interest_count", type=int, default=None)
    parser.add_argument("--max_hops", type=int, default=10)
    parser.add_argument("--beam_width", type=int, default=128)
    parser.add_argument("--paths_per_interest", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--limit_patients", type=int, default=5000)
    parser.add_argument("--rerank_batch_size", type=int, default=512)
    parser.add_argument("--beam_guidance_batch_size", type=int, default=2048)
    parser.add_argument(
        "--disable_gru_beam_guidance",
        action="store_true",
        help="Use the additive score for beam selection; GRU remains enabled as a post-hoc reranker.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph_device", type=str, default="auto")
    parser.add_argument("--progress_every", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    device = torch.device(args.device)
    mind_model, hp, seed_vocab = load_mind_model(args.mind_checkpoint, device)
    reranker, reranker_ckpt = load_reranker(args.reranker_checkpoint, args.graph_dir, device)
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
    engine = RetrievalEngine(graph_store)
    queries = load_queries(args.test_queries_csv, limit=args.limit_patients)
    reset_peak_memory(device)
    target_universe = load_target_universe(args.condition_map)
    target_node_ids = None
    if target_universe is not None:
        target_node_ids = {
            node_id
            for node_key in target_universe
            for node_id in [graph_store.lookup_node_id(node_key)]
            if node_id is not None
        }
    node_type2id = reranker_ckpt["node_type2id"]
    beam_path_scorer = None
    if not args.disable_gru_beam_guidance:
        def beam_path_scorer(items):
            return score_beam_prefixes_with_gru(
                reranker,
                graph_store,
                node_type2id,
                items,
                device,
                args.beam_guidance_batch_size,
            )

    rows: List[Dict[str, object]] = []
    skipped_missing_seed = 0
    no_target_candidates = 0
    preserved_unreranked_candidates = 0
    unk_seed_queries = 0
    cosine_values: List[float] = []
    retrieve_latency_values: List[float] = []
    rerank_latency_values: List[float] = []

    progress = tqdm(queries, desc="Retrieving", unit="query")
    for query_idx, query in enumerate(progress, start=1):
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
            projected, _latent, active_mask = mind_model(x)
            interest_vectors = projected[0, :use_interests, :].detach().cpu().numpy()
            cosine_mean, _ = average_pairwise_cosine(projected[:, :use_interests, :], active_mask[:, :use_interests])
            cosine_values.append(cosine_mean)

        result = engine.retrieve(
            seed_node_ids=seed_ids,
            max_hops=args.max_hops,
            beam_width=args.beam_width,
            topk_paths=args.paths_per_interest,
            alpha=args.alpha,
            beta=args.beta,
            interest_vectors=interest_vectors,
            max_paths_per_interest=args.paths_per_interest,
            target_node_ids=target_node_ids,
            path_scorer=beam_path_scorer,
        )
        retrieve_latency_values.append(float(result.latency_seconds))
        beam_ranked = beam_endpoint_scores(result, graph_store, target_universe)

        records = [
            record
            for item in result.paths
            for record in [path_to_record(graph_store, node_type2id, target_universe, item)]
            if record is not None
        ]

        rerank_start = time.perf_counter()
        reranked = reranked_endpoint_scores(
            reranker,
            records,
            device,
            args.rerank_batch_size,
            top_k=len(target_universe) if target_universe is not None else args.top_k,
        )
        rerank_latency_values.append(time.perf_counter() - rerank_start)
        reranked_endpoints = {candidate for candidate, _score in reranked}
        fallback = [
            (candidate, score)
            for candidate, score in beam_ranked
            if candidate not in reranked_endpoints
        ]
        preserved_unreranked_candidates += len(fallback)
        ranked = reranked + fallback
        if not ranked:
            no_target_candidates += 1
            continue
        for rank, (candidate, score) in enumerate(ranked[: args.top_k], start=1):
            rows.append(
                {
                    "patient_index": patient_index,
                    "candidate": candidate,
                    "score": f"{score:.8f}",
                    "rank": rank,
                }
            )

        if args.progress_every and query_idx % args.progress_every == 0:
            progress.set_postfix(
                predictions=len(rows),
                skipped_seed=skipped_missing_seed,
                no_candidates=no_target_candidates,
            )

    progress.close()

    write_predictions(args.output_csv, rows)
    log_run_summary("gru_rerank_retrieval", run_started, len(queries), "queries", device, prediction_rows=len(rows), retrieval_seconds=round(sum(retrieve_latency_values), 3), rerank_seconds=round(sum(rerank_latency_values), 3))
    summary = {
        "test_queries_csv": str(args.test_queries_csv),
        "graph_dir": str(args.graph_dir),
        "mind_checkpoint": str(args.mind_checkpoint),
        "reranker_checkpoint": str(args.reranker_checkpoint),
        "reranker_config": asdict(reranker.config),
        "gru_beam_guidance_enabled": not args.disable_gru_beam_guidance,
        "beam_guidance_batch_size": args.beam_guidance_batch_size,
        "condition_map": str(args.condition_map) if args.condition_map else None,
        "target_universe_size": len(target_universe) if target_universe is not None else None,
        "candidate_endpoint_filter": "all mapped target endpoints found by beam search; valid disease paths are GRU-reranked and remaining endpoints retain beam ordering",
        "target_aware_candidate_collection": target_node_ids is not None,
        "output_csv": str(args.output_csv),
        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if device.type == "cuda" else None,
        "num_queries_loaded": len(queries),
        "num_prediction_rows": len(rows),
        "skipped_missing_seed": skipped_missing_seed,
        "no_target_candidates": no_target_candidates,
        "preserved_unreranked_candidates": preserved_unreranked_candidates,
        "queries_with_unknown_seed_tokens": unk_seed_queries,
        "checkpoint_k": int(hp["K"]),
        "interest_count_used": use_interests,
        "latent_cosine_mean": float(np.mean(cosine_values)) if cosine_values else 0.0,
        "max_hops": args.max_hops,
        "beam_width": args.beam_width,
        "paths_per_interest": args.paths_per_interest,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "beta": args.beta,
        "mean_retrieve_latency_seconds": float(np.mean(retrieve_latency_values)) if retrieve_latency_values else 0.0,
        "mean_rerank_latency_seconds": float(np.mean(rerank_latency_values)) if rerank_latency_values else 0.0,
        "mean_total_latency_seconds": (
            float(np.mean(np.asarray(retrieve_latency_values) + np.asarray(rerank_latency_values)))
            if retrieve_latency_values and rerank_latency_values
            else 0.0
        ),
    }
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
