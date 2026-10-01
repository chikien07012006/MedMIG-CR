from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

EVALUATOR = REPO_ROOT / "scripts" / "evaluation" / "evaluate_ddxplus_retrieval.py"
METRIC_KS = (1, 5, 10, 20, 50)
CANDIDATE_SPACES = ("closed", "open")

DEFAULT_GRAPH_DIR = Path("data/processed/primekg_graph")
DEFAULT_CONDITION_MAP = Path("data/mappings/ddxplus_v2/condition_to_primekg.json")
DEFAULT_SUBSET_DIR = Path("data/processed/ddxplus_v2_subsets")


@dataclass(frozen=True)
class Query:
    patient_index: int
    pathology: str
    seed_keys: tuple[str, ...]
    target_keys: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSet:
    space: str
    keys: list[str]
    graph_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.keys)


def split_keys(cell: str | None) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(cell or "").split(";") if token.strip())


def load_queries(path: Path, limit: int | None = None) -> List[Query]:
    queries: List[Query] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in tqdm(csv.DictReader(handle), desc=f"Loading {path.name}", unit="query"):
            if limit is not None and len(queries) >= limit:
                break
            queries.append(
                Query(
                    patient_index=int(row["patient_index"]),
                    pathology=row.get("pathology", ""),
                    seed_keys=split_keys(row.get("seed_node_keys")),
                    target_keys=split_keys(row.get("target_node_keys")),
                )
            )
    return queries


def load_node_index(graph_dir: Path) -> Dict[str, int]:
    with (graph_dir / "mappings" / "node2id.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_relation_index(graph_dir: Path) -> Dict[str, int]:
    with (graph_dir / "mappings" / "relation2id.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_directed_edges(graph_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (source, target, relation_id, num_nodes) for the directed PrimeKG edges."""
    with np.load(graph_dir / "graph_csr.npz") as arrays:
        indptr = arrays["indptr"].astype(np.int64, copy=False)
        target = arrays["indices"].astype(np.int64, copy=False)
        relation = arrays["edge_relids"].astype(np.int64, copy=False)
    num_nodes = len(indptr) - 1
    source = np.repeat(np.arange(num_nodes, dtype=np.int64), np.diff(indptr))
    return source, target, relation, num_nodes


def load_undirected_adjacency(graph_dir: Path, exclude_relations: Sequence[str] = ()) -> sp.csr_matrix:
    """Binary, symmetric adjacency (edge union with its transpose, no self-loops).

    Both directions are used because the proposed beam search expands incoming
    and outgoing neighbours.
    """
    source, target, relation, num_nodes = load_directed_edges(graph_dir)
    if exclude_relations:
        relation_index = load_relation_index(graph_dir)
        unknown = [name for name in exclude_relations if name not in relation_index]
        if unknown:
            raise ValueError(f"Unknown relation names: {unknown}")
        keep = ~np.isin(relation, [relation_index[name] for name in exclude_relations])
        source, target = source[keep], target[keep]
    adjacency = sp.csr_matrix(
        (np.ones(len(source), dtype=np.float64), (source, target)),
        shape=(num_nodes, num_nodes),
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1.0
    return adjacency


def load_target_universe(condition_map: Path) -> List[str]:
    with condition_map.open("r", encoding="utf-8-sig") as handle:
        mapping = json.load(handle)
    return sorted({key for entry in mapping.values() for key in entry.get("selected_primekg_nodes", [])})


def load_candidates(space: str, graph_dir: Path, condition_map: Path, node_index: Dict[str, int]) -> CandidateSet:
    """closed: the mapped DDXPlus target universe; open: every PrimeKG disease node."""
    if space == "closed":
        keys = load_target_universe(condition_map)
    elif space == "open":
        with (graph_dir / "disease_nodes.csv").open("r", newline="", encoding="utf-8-sig") as handle:
            keys = [row["node_key"] for row in csv.DictReader(handle)]
    else:
        raise ValueError(f"candidate space must be one of {CANDIDATE_SPACES}")
    missing = [key for key in keys if key not in node_index]
    if missing:
        raise ValueError(f"{len(missing)} candidates are missing from the graph, e.g. {missing[:3]}")
    if len(keys) != len(set(keys)):
        raise ValueError("Candidate list contains duplicates")
    return CandidateSet(space, list(keys), np.asarray([node_index[key] for key in keys], dtype=np.int64))


def resolve_seed_ids(seed_keys: Iterable[str], node_index: Dict[str, int]) -> np.ndarray:
    return np.asarray(sorted({node_index[key] for key in seed_keys if key in node_index}), dtype=np.int64)


def rank_candidates(scores: np.ndarray, top_k: int, keep_zero_scores: bool) -> np.ndarray:
    """Candidate positions ordered by descending score; ties broken by candidate order."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    if not keep_zero_scores:
        order = order[scores[order] > 0]
    return order[:top_k]


def prediction_rows(patient_index: int, candidates: CandidateSet, scores: np.ndarray, order: np.ndarray) -> List[dict]:
    return [
        {
            "patient_index": patient_index,
            "candidate": candidates.keys[int(position)],
            "score": f"{float(scores[position]):.10g}",
            "rank": rank,
        }
        for rank, position in enumerate(order, start=1)
    ]


def write_predictions(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_index", "candidate", "score", "rank"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def run_shared_evaluator(
    queries_csv: Path,
    condition_map: Path,
    predictions_csv: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict:
    """Score predictions with the evaluator used by every method in the paper."""
    command = [
        sys.executable,
        str(EVALUATOR),
        "--queries_csv", str(queries_csv),
        "--condition_map", str(condition_map),
        "--predictions", str(predictions_csv),
        "--output_dir", str(output_dir),
        "--target_mode", "pathology",
        "--topk", *[str(k) for k in METRIC_KS],
    ]
    if limit is not None:
        command += ["--limit_patients", str(limit)]
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    with (output_dir / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_coverage(predictions_csv: Path, queries: Sequence[Query]) -> float:
    """Fraction of queries whose prediction list contains at least one target node."""
    predicted: Dict[int, set[str]] = {}
    with predictions_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            predicted.setdefault(int(row["patient_index"]), set()).add(row["candidate"])
    hits = sum(bool(predicted.get(query.patient_index, set()) & set(query.target_keys)) for query in queries)
    return hits / max(len(queries), 1)


def print_metrics(name: str, metrics: dict) -> None:
    keys = ["num_evaluated", "mrr", *[f"recall@{k}" for k in METRIC_KS], "candidate_coverage"]
    shown = {key: (round(metrics[key], 4) if isinstance(metrics.get(key), float) else metrics.get(key)) for key in keys if key in metrics}
    print(f"[{name}] " + json.dumps(shown))
