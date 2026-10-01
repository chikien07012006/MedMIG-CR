from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from baseline_common import (
    CANDIDATE_SPACES,
    DEFAULT_CONDITION_MAP,
    DEFAULT_GRAPH_DIR,
    DEFAULT_SUBSET_DIR,
    CandidateSet,
    Query,
    candidate_coverage,
    load_candidates,
    load_node_index,
    load_queries,
    load_undirected_adjacency,
    prediction_rows,
    print_metrics,
    rank_candidates,
    resolve_seed_ids,
    run_shared_evaluator,
    write_json,
    write_predictions,
)
from medmigcr_kg.parallel import recommended_workers, run_ordered

METHODS = ("ppr", "katz")
KATZ_NORMALIZATIONS = ("none", "sym", "rw")
PPR_SCORES = ("raw", "degree")


@dataclass(frozen=True)
class PropagationConfig:
    method: str
    graph_dir: Path
    condition_map: Path
    candidate_space: str
    exclude_relations: Tuple[str, ...]
    top_k: int
    keep_zero_scores: bool
    # Personalized PageRank
    restart_probability: float = 0.15
    max_iterations: int = 200
    tolerance: float = 1e-6
    ppr_score: str = "raw"
    # Truncated Katz
    katz_beta: float = 0.5
    katz_max_length: int = 4
    katz_normalization: str = "sym"


@dataclass
class BatchResult:
    rows: List[dict]
    num_queries: int
    skipped_missing_seed: int
    iterations: List[int]
    converged: int


_STATE: Dict[str, object] = {}


def column_stochastic(adjacency: sp.csr_matrix) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Return A D^-1 (the transpose of the random-walk matrix for a symmetric A) and the dangling mask."""
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return (adjacency @ sp.diags(inverse)).tocsr(), degree == 0


def katz_operator(adjacency: sp.csr_matrix, normalization: str) -> sp.csr_matrix:
    if normalization == "none":
        return adjacency
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if normalization == "sym":
        inv_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
        return (sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)).tocsr()
    if normalization == "rw":
        return column_stochastic(adjacency)[0]
    raise ValueError(f"katz_normalization must be one of {KATZ_NORMALIZATIONS}")


def init_state(config: PropagationConfig) -> None:
    node_index = load_node_index(config.graph_dir)
    adjacency = load_undirected_adjacency(config.graph_dir, config.exclude_relations)
    candidates = load_candidates(config.candidate_space, config.graph_dir, config.condition_map, node_index)
    state: Dict[str, object] = {"config": config, "node_index": node_index, "candidates": candidates}
    if config.method == "ppr":
        operator, dangling = column_stochastic(adjacency)
        state.update(operator=operator, dangling=dangling, degree=np.asarray(adjacency.sum(axis=1)).ravel())
    elif config.method == "katz":
        state.update(operator=katz_operator(adjacency, config.katz_normalization))
    else:
        raise ValueError(f"method must be one of {METHODS}")
    _STATE.clear()
    _STATE.update(state)


def seed_matrix(seed_lists: Sequence[np.ndarray], num_nodes: int) -> np.ndarray:
    """Uniform personalization: column j puts mass 1/|S_j| on each seed of query j."""
    seeds = np.zeros((num_nodes, len(seed_lists)), dtype=np.float64)
    for column, seed_ids in enumerate(seed_lists):
        seeds[seed_ids, column] = 1.0 / len(seed_ids)
    return seeds


def personalized_pagerank(seeds: np.ndarray, config: PropagationConfig) -> Tuple[np.ndarray, List[int], int]:
    """Power iteration r <- a*s + (1-a)*(P^T r + dangling_mass*s), run per column until L1 change <= tol."""
    operator: sp.csr_matrix = _STATE["operator"]
    dangling: np.ndarray = _STATE["dangling"]
    alpha = config.restart_probability
    rank = seeds.copy()
    iterations = np.full(seeds.shape[1], config.max_iterations, dtype=np.int64)
    done = np.zeros(seeds.shape[1], dtype=bool)
    for step in range(1, config.max_iterations + 1):
        spread = operator @ rank
        if dangling.any():
            spread += seeds * rank[dangling].sum(axis=0, keepdims=True)
        updated = alpha * seeds + (1.0 - alpha) * spread
        delta = np.abs(updated - rank).sum(axis=0)
        rank = updated
        newly_done = (delta <= config.tolerance) & ~done
        iterations[newly_done] = step
        done |= newly_done
        if done.all():
            break
    if config.ppr_score == "degree":
        degree: np.ndarray = _STATE["degree"]
        rank = rank / np.maximum(degree, 1.0)[:, None]
    return rank, iterations.tolist(), int(done.sum())


def truncated_katz(seeds: np.ndarray, config: PropagationConfig) -> np.ndarray:
    """score = sum_{l=1..L} beta^l * M^l s, i.e. weighted walk counts of length 1..L from the seeds."""
    operator: sp.csr_matrix = _STATE["operator"]
    walk = seeds
    score = np.zeros_like(seeds)
    weight = 1.0
    for _ in range(config.katz_max_length):
        walk = operator @ walk
        weight *= config.katz_beta
        score += weight * walk
    return score


def process_batch(batch: Sequence[Query]) -> BatchResult:
    config: PropagationConfig = _STATE["config"]
    node_index: Dict[str, int] = _STATE["node_index"]
    candidates: CandidateSet = _STATE["candidates"]
    operator: sp.csr_matrix = _STATE["operator"]

    usable: List[Tuple[Query, np.ndarray]] = []
    for query in batch:
        seed_ids = resolve_seed_ids(query.seed_keys, node_index)
        if len(seed_ids):
            usable.append((query, seed_ids))
    result = BatchResult([], len(batch), len(batch) - len(usable), [], 0)
    if not usable:
        return result

    seeds = seed_matrix([seed_ids for _, seed_ids in usable], operator.shape[0])
    if config.method == "ppr":
        scores, iterations, converged = personalized_pagerank(seeds, config)
        result.iterations, result.converged = iterations, converged
    else:
        scores = truncated_katz(seeds, config)
    candidate_scores = scores[candidates.graph_ids, :]
    for column, (query, _seed_ids) in enumerate(usable):
        column_scores = candidate_scores[:, column]
        order = rank_candidates(column_scores, config.top_k, config.keep_zero_scores)
        result.rows.extend(prediction_rows(query.patient_index, candidates, column_scores, order))
    return result


def run_propagation(
    config: PropagationConfig,
    queries_csv: Path,
    out_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    evaluate: bool,
    limit: int | None = None,
) -> dict:
    """Score every query in queries_csv, write predictions/summary, and optionally evaluate."""
    started = time.perf_counter()
    queries = load_queries(queries_csv, limit)
    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    rows: List[dict] = []
    skipped = 0
    iterations: List[int] = []
    converged = 0
    progress = tqdm(total=len(queries), desc=f"{config.method.upper()} ({num_workers} workers)", unit="query")
    for result in run_ordered(batches, process_batch, init_fn=init_state, init_args=(config,), num_workers=num_workers, chunksize=1):
        rows.extend(result.rows)
        skipped += result.skipped_missing_seed
        iterations.extend(result.iterations)
        converged += result.converged
        progress.update(result.num_queries)
    progress.close()

    predictions_csv = out_dir / "predictions.csv"
    write_predictions(predictions_csv, rows)
    elapsed = time.perf_counter() - started
    summary = {
        "method": {"ppr": "Personalized PageRank", "katz": "Truncated Katz"}[config.method],
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(config).items()},
        "queries_csv": str(queries_csv),
        "num_queries": len(queries),
        "skipped_missing_seed": skipped,
        "num_prediction_rows": len(rows),
        "graph": "PrimeKG, undirected union of directed edges, unweighted",
        "personalization": "uniform over resolved seed nodes",
        "batch_size": batch_size,
        "num_workers": num_workers,
        "elapsed_seconds": round(elapsed, 3),
        "queries_per_second": round(len(queries) / max(elapsed, 1e-9), 3),
    }
    if config.method == "ppr":
        summary["ppr_convergence"] = {
            "converged_queries": converged,
            "converged_fraction": converged / max(len(iterations), 1),
            "mean_iterations": float(np.mean(iterations)) if iterations else 0.0,
            "max_iterations_used": int(max(iterations)) if iterations else 0,
        }
    summary["candidate_coverage"] = candidate_coverage(predictions_csv, queries)
    if evaluate:
        metrics = run_shared_evaluator(queries_csv, config.condition_map, predictions_csv, out_dir / "evaluation", limit)
        metrics["candidate_coverage"] = summary["candidate_coverage"]
        summary["metrics"] = metrics
        print_metrics(summary["method"], metrics)
    write_json(out_dir / "run_summary.json", summary)
    return summary


def add_common_arguments(parser) -> None:
    """CLI options shared by the PPR and Katz entry points."""
    parser.add_argument("--queries_csv", type=Path, default=DEFAULT_SUBSET_DIR / "test_10k.csv",
                        help="Queries to score (valid_5k.csv for tuning, test_10k.csv for the paper).")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--graph_dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--condition_map", type=Path, default=DEFAULT_CONDITION_MAP)
    parser.add_argument("--candidate_space", choices=CANDIDATE_SPACES, default="closed",
                        help="closed: the 47 mapped DDXPlus targets (main protocol); open: all 22,205 PrimeKG diseases.")
    parser.add_argument("--exclude_relations", nargs="*", default=[],
                        help="PrimeKG relation names to drop from the graph, e.g. disease_phenotype_negative.")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--keep_zero_scores", action="store_true",
                        help="Also rank candidates the propagation never reached (score 0). Off by default, like beam search.")
    parser.add_argument("--batch_size", type=int, default=64, help="Queries propagated together in one sparse-dense product.")
    parser.add_argument("--num_workers", type=int, default=recommended_workers(),
                        help="CPU processes; results are identical for any value.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N queries (smoke tests).")
    parser.add_argument("--no_eval", action="store_true", help="Skip the shared evaluator.")
