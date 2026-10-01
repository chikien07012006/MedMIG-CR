from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import islice
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medmigcr_kg.graph_store import GraphStore  # noqa: E402
from medmigcr_kg.parallel import recommended_workers, run_ordered  # noqa: E402
from medmigcr_kg.retrieval_engine import RetrievalEngine  # noqa: E402
from medmigcr_kg.target_distance import build_distance_guide  # noqa: E402
from medmigcr_kg.telemetry import log_run_summary, reset_peak_memory  # noqa: E402
from medmigcr_mind.contrastive_model import ProjectedClinicalMIND  # noqa: E402


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


def load_model(path: Path, device: torch.device) -> Tuple[ProjectedClinicalMIND, dict, Dict[str, int]]:
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


def build_node_type_vocab(graph_store: GraphStore) -> Dict[str, int]:
    types = sorted({node_type(name) for name in graph_store.node2id})
    return {name: index + 1 for index, name in enumerate(types)}


def endpoint_is_disease(node_name: str | None) -> bool:
    return bool(node_name and node_name.startswith("disease|"))


def path_record(
    graph_store: GraphStore,
    node_type2id: Dict[str, int],
    patient_index: int,
    pathology: str,
    target_keys: set[str],
    candidate_universe: set[str] | None,
    item,
) -> Dict[str, object] | None:
    endpoint = graph_store.lookup_node_name(item.current_node)
    if not endpoint_is_disease(endpoint):
        return None
    if candidate_universe is not None and endpoint not in candidate_universe:
        return None
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
    label = 1 if endpoint in target_keys else 0
    return {
        "patient_index": patient_index,
        "pathology": pathology,
        "endpoint": endpoint,
        "label": label,
        "additive_score": float(item.score),
        "path_node_ids": list(item.path),
        "dst_node_ids": dst_node_ids,
        "relation_ids": list(item.relation_path),
        "direction_ids": list(item.direction_path),
        "node_type_ids": node_type_ids,
        "path_len": len(item.path) - 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GRU path reranker JSONL data from InfoNCE MIND beam paths.")
    parser.add_argument("--queries_csv", type=Path, default=Path("data/processed/ddxplus_v2/train_queries.csv"))
    parser.add_argument("--graph_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ddxplus_infonce_e10_hop10/k3/clinical_mind_infonce_k3.pt"),
    )
    parser.add_argument("--condition_map", type=Path, default=Path("data/mappings/ddxplus_v2/condition_to_primekg.json"))
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("data/processed/reranker/infonce_e10_k3_hop10_bw128_train_paths_100k.jsonl"),
    )
    parser.add_argument("--metadata_json", type=Path, default=None)
    parser.add_argument("--limit_patients", type=int, default=100000)
    parser.add_argument("--interest_count", type=int, default=None)
    parser.add_argument("--max_hops", type=int, default=10)
    parser.add_argument("--beam_width", type=int, default=128)
    parser.add_argument("--paths_per_interest", type=int, default=4096)
    parser.add_argument("--max_paths_per_patient", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph_device", type=str, default="auto")
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--distance_pruning", action="store_true",
                        help="Drop neighbours that cannot reach any target within the remaining hops.")
    parser.add_argument("--distance_weight", type=float, default=0.0,
                        help="Subtract weight * hop distance to the nearest target from each step score.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=f"CPU processes for beam search; output is identical to 1 worker. Suggested on this machine: {recommended_workers()}.",
    )
    parser.add_argument("--chunksize", type=int, default=8)
    return parser.parse_args()


def load_graph_store(args: argparse.Namespace) -> GraphStore:
    return GraphStore.load(
        graph_npz=args.graph_dir / "graph_csr.npz",
        node_embeddings_npy=args.graph_dir / "node_embeddings.npy",
        out_degree_npy=args.graph_dir / "out_degree.npy",
        in_degree_npy=args.graph_dir / "in_degree.npy",
        mapping_dir=args.graph_dir / "mappings",
        device=args.graph_device,
    )


def target_ids_for(graph_store: GraphStore, target_universe: set[str] | None) -> set[int] | None:
    if not target_universe:
        return None
    return {
        node_id
        for node_key in target_universe
        for node_id in [graph_store.lookup_node_id(node_key)]
        if node_id is not None
    }


# Per-process state, filled once by init_state() in the main process or each worker.
_STATE: Dict[str, object] = {}


def init_state(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, hp, seed_vocab = load_model(args.checkpoint, device)
    graph_store = load_graph_store(args)
    target_universe = load_target_universe(args.condition_map)
    _STATE.update(
        args=args,
        device=device,
        model=model,
        hp=hp,
        seed_vocab=seed_vocab,
        use_interests=int(args.interest_count or hp["K"]),
        graph_store=graph_store,
        engine=RetrievalEngine(graph_store),
        target_universe=target_universe,
        target_universe_ids=target_ids_for(graph_store, target_universe),
        node_type2id=build_node_type_vocab(graph_store),
    )
    _STATE["distance_guide"] = build_distance_guide(
        _STATE["graph_store"], _STATE["target_universe_ids"], args.distance_pruning, args.distance_weight
    )


def process_query(query: Dict[str, str]) -> List[Dict[str, object]] | None:
    """Return the path records for one query, or None when no seed resolves."""
    args = _STATE["args"]
    graph_store = _STATE["graph_store"]
    hp = _STATE["hp"]
    device = _STATE["device"]
    patient_index = int(query["patient_index"])
    pathology = query.get("pathology", "")
    seed_keys = split_nodes(query.get("seed_node_keys", ""))
    target_keys = set(split_nodes(query.get("target_node_keys", "")))
    seed_ids = resolve_seed_ids(graph_store, seed_keys)
    if not seed_ids:
        return None

    encoded = encode_seed_nodes(seed_keys, _STATE["seed_vocab"], int(hp["max_seq_len"]))
    x = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        projected, _latent, _active_mask = _STATE["model"](x)
        interest_vectors = projected[0, : _STATE["use_interests"], :].detach().cpu().numpy()

    result = _STATE["engine"].retrieve(
        seed_node_ids=seed_ids,
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        topk_paths=args.paths_per_interest,
        alpha=args.alpha,
        beta=args.beta,
        interest_vectors=interest_vectors,
        max_paths_per_interest=args.paths_per_interest,
        target_node_ids=_STATE["target_universe_ids"],
        distance_guide=_STATE["distance_guide"],
    )

    records = [
        record
        for item in result.paths
        for record in [
            path_record(
                graph_store,
                _STATE["node_type2id"],
                patient_index,
                pathology,
                target_keys,
                _STATE["target_universe"],
                item,
            )
        ]
        if record is not None
    ]
    records.sort(key=lambda record: float(record["additive_score"]), reverse=True)
    if args.max_paths_per_patient is not None:
        records = records[: args.max_paths_per_patient]
    return records


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    device = torch.device(args.device)
    _model, hp, _seed_vocab = load_model(args.checkpoint, device)
    use_interests = int(args.interest_count or hp["K"])
    if use_interests < 1 or use_interests > int(hp["K"]):
        raise ValueError(f"--interest_count must be between 1 and checkpoint K={hp['K']}")

    graph_store = load_graph_store(args)
    queries = load_queries(args.queries_csv, limit=args.limit_patients)
    reset_peak_memory(device)
    target_universe = load_target_universe(args.condition_map)
    node_type2id = build_node_type_vocab(graph_store)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata_json or args.output_jsonl.with_suffix(".metadata.json")

    counts: Counter[str] = Counter()
    results = run_ordered(
        queries,
        process_query,
        init_fn=init_state,
        init_args=(args,),
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for query_idx, records in enumerate(tqdm(results, total=len(queries), desc="Generating reranker paths", unit="query"), start=1):
            if records is None:
                counts["skipped_missing_seed"] += 1
                continue

            has_positive = any(int(record["label"]) == 1 for record in records)
            counts["queries_with_positive"] += int(has_positive)
            counts["queries_without_positive"] += int(not has_positive)
            for record in records:
                record["query_has_positive_path"] = has_positive
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["positive_paths" if int(record["label"]) == 1 else "negative_paths"] += 1
            counts["queries_processed"] += 1

            if args.progress_every and query_idx % args.progress_every == 0:
                print(f"processed {query_idx}/{len(queries)} queries; positives={counts['positive_paths']} negatives={counts['negative_paths']}")

    metadata = {
        "queries_csv": str(args.queries_csv),
        "checkpoint": str(args.checkpoint),
        "graph_dir": str(args.graph_dir),
        "condition_map": str(args.condition_map) if args.condition_map else None,
        "candidate_universe_size": len(target_universe) if target_universe else None,
        "candidate_endpoint_filter": "disease endpoints in condition_map target universe",
        "output_jsonl": str(args.output_jsonl),
        "num_relations": len(graph_store.relation2id),
        "node_type2id": node_type2id,
        "checkpoint_k": int(hp["K"]),
        "interest_count_used": use_interests,
        "max_hops": args.max_hops,
        "beam_width": args.beam_width,
        "paths_per_interest": args.paths_per_interest,
        "max_paths_per_patient": args.max_paths_per_patient,
        "alpha": args.alpha,
        "beta": args.beta,
        "counts": dict(counts),
        "num_workers": args.num_workers,
        "distance_pruning": args.distance_pruning,
        "distance_weight": args.distance_weight,
        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    log_run_summary("gru_path_generation", run_started, counts["queries_processed"], "queries", device, paths_written=counts["positive_paths"] + counts["negative_paths"])


if __name__ == "__main__":
    main()
