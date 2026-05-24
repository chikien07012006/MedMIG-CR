"""
Phase 2 Retrieval: Semantic beam search over PrimeKG with per-interest path extraction.

Usage:
  python phase2_retrieve_query.py <query_id> [--max_hops=5] [--beam_width=32] [--interest_count=3]

Example:
  python phase2_retrieve_query.py Q0000005
  python phase2_retrieve_query.py Q0000001 --max_hops=5 --interest_count=3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from graph_store import GraphStore
from retrieval_engine import RetrievalEngine


def load_query_symptom_nodes(query_id: str, query_csv: Path) -> List[str]:
    """Load symptom entity IDs from query_nodes.csv"""
    df = pd.read_csv(query_csv, dtype=str)
    row = df.loc[df["query_id"] == query_id]
    if row.empty:
        raise ValueError(f"Query {query_id} not found in {query_csv}")
    symptom_ids = str(row.iloc[0]["symptom_entity_ids"]).strip()
    return [node.strip() for node in symptom_ids.split(";") if node.strip()]


def resolve_seed_node_ids(graph_store: GraphStore, raw_node_names: List[str]) -> List[int]:
    """Resolve symptom entity IDs to graph node IDs"""
    seed_ids = []
    missing = []
    for node_name in raw_node_names:
        node_id = graph_store.lookup_node_id(node_name)
        if node_id is None:
            fallback = None
            if "|" in node_name:
                token = node_name.split("|")[-1].strip()
                fallback = graph_store.lookup_node_id(token)
            if fallback is not None:
                seed_ids.append(fallback)
                continue
            missing.append(node_name)
        else:
            seed_ids.append(node_id)
    if missing:
        print("Warning: some symptom node names were not found in the graph mapping:")
        for node_name in missing:
            print("  -", node_name)
    return seed_ids


def extract_node_type(node_name: str) -> str:
    """Extract entity type from node name (e.g., 'disease|MONDO|123' -> 'MONDO disease')"""
    if not node_name:
        return "unknown"
    if "|" in node_name:
        parts = node_name.split("|")
        entity_type = parts[0].strip()
        database = parts[1].strip() if len(parts) > 1 else ""
        return f"{database}:{entity_type}" if database else entity_type
    return node_name


def format_path(path_nodes: tuple, graph_store: GraphStore) -> str:
    """Format path as readable node names with types"""
    formatted = []
    for node_id in path_nodes:
        node_name = graph_store.lookup_node_name(node_id)
        node_type = extract_node_type(node_name)
        formatted.append(f"{node_name} ({node_type})")
    return " -> ".join(formatted)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Retrieval for any MedMIG-CR query")
    parser.add_argument("query_id", type=str, help="Query ID (e.g., Q0000005)")
    parser.add_argument("--max_hops", type=int, default=5, help="Max traversal hops")
    parser.add_argument("--beam_width", type=int, default=32, help="Beam width per hop")
    parser.add_argument("--interest_count", type=int, default=3, help="Number of interest vectors")
    args = parser.parse_args()

    # Resolve paths (handles both direct run and exec context)
    if '__file__' in globals():
        base = Path(__file__).resolve().parent
    else:
        base = Path.cwd()
    processed_dir = base / "processed_test"
    query_csv = base.parent / "processed_medmigcr_dataset" / "query_nodes.csv"

    # Load projection if available
    projection_path = base.parent / "multi-interest" / "checkpoints" / "alignment_mind_to_primekg.npz"
    projection = None
    if projection_path.exists():
        print(f"Loading alignment projection from {projection_path}...")
        projection = RetrievalEngine.load_projection(projection_path)
    else:
        print("Warning: No alignment projection found. Retrieval might be less accurate.")

    # Load graph
    print(f"Loading graph from {processed_dir}...")
    graph_store = GraphStore.load(
        graph_npz=processed_dir / "graph_csr.npz",
        node_embeddings_npy=processed_dir / "node_embeddings.npy",
        out_degree_npy=processed_dir / "out_degree.npy",
        in_degree_npy=processed_dir / "in_degree.npy",
        mapping_dir=processed_dir / "mappings",
        device="auto",
    )

    # Load query seeds
    print(f"Loading query {args.query_id}...")
    symptom_nodes = load_query_symptom_nodes(args.query_id, query_csv)
    seed_ids = resolve_seed_node_ids(graph_store, symptom_nodes)
    if not seed_ids:
        raise RuntimeError(f"No seed symptom nodes could be resolved for query {args.query_id}")

    print(f"Query seeds resolved: {len(seed_ids)} nodes")
    for node_id in seed_ids:
        node_name = graph_store.lookup_node_name(node_id)
        node_type = extract_node_type(node_name)
        print(f"  - {node_name} ({node_type})")

    # Run retrieval
    print(f"\nRunning retrieval (max_hops={args.max_hops}, beam_width={args.beam_width}, interest_count={args.interest_count})...")
    engine = RetrievalEngine(graph_store, projection=projection)
    result = engine.retrieve(
        seed_node_ids=seed_ids,
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        topk_paths=100,
        alpha=1.0,
        beta=0.4,
        interest_count=args.interest_count,
        max_paths_per_interest=3,
    )

    # Print results
    print(f"\nRetrieval completed in {result.latency_seconds:.3f}s")
    print(f"Unique paths found: {len(result.paths)}")
    print(f"Unique nodes: {len(result.unique_nodes)}")
    print(f"Path diversity: {result.path_diversity:.4f}")
    print(f"Hub ratio: {result.hub_ratio:.4f}")

    print("\nTop 20 candidate diseases:")
    for rank, (node_name, score) in enumerate(result.candidate_scores[:20], start=1):
        node_type = extract_node_type(node_name)
        print(f"{rank:2d}. {node_name:40s} ({node_type:20s}) score={score:8.4f}")

    print("\nSample retrieved paths (first 10):")
    for rank, path_item in enumerate(result.paths[:10], start=1):
        formatted_path = format_path(path_item.path, graph_store)
        print(f"{rank:2d}. score={path_item.score:8.4f}")
        print(f"    {formatted_path}")

    # Save outputs
    output_json = base / f"phase2_retrieval_{args.query_id}.json"
    engine.save_subgraph(result, output_json)
    print(f"\nSaved detailed results to {output_json}")

    # Save summary
    output_summary = base / f"phase2_retrieval_{args.query_id}_summary.txt"
    with output_summary.open("w", encoding="utf-8") as f:
        f.write(f"Query: {args.query_id}\n")
        f.write(f"Seed symptom nodes: {len(seed_ids)}\n")
        f.write(f"Seed node details:\n")
        for node_id in seed_ids:
            node_name = graph_store.lookup_node_name(node_id)
            node_type = extract_node_type(node_name)
            f.write(f"  - {node_name} ({node_type})\n")
        f.write(f"\nRetrieval parameters:\n")
        f.write(f"  max_hops: {args.max_hops}\n")
        f.write(f"  beam_width: {args.beam_width}\n")
        f.write(f"  interest_count: {args.interest_count}\n")
        f.write(f"\nResults:\n")
        f.write(f"  unique_paths: {len(result.paths)}\n")
        f.write(f"  unique_nodes: {len(result.unique_nodes)}\n")
        f.write(f"  top_candidates: {len(result.candidate_scores)}\n")
        f.write(f"  latency_seconds: {result.latency_seconds:.4f}\n")
        f.write(f"  path_diversity: {result.path_diversity:.4f}\n")
        f.write(f"  hub_ratio: {result.hub_ratio:.4f}\n")
        f.write(f"\nTop 50 candidate diseases:\n")
        for rank, (node_name, score) in enumerate(result.candidate_scores[:50], start=1):
            node_type = extract_node_type(node_name)
            f.write(f"{rank:2d}. {node_name:40s} ({node_type:20s}) {score:.6f}\n")
        f.write(f"\nAll retrieved paths ({len(result.paths)} total):\n")
        for rank, path_item in enumerate(result.paths, start=1):
            f.write(f"{rank:3d}. score={path_item.score:.6f}\n")
            formatted_path = format_path(path_item.path, graph_store)
            f.write(f"     {formatted_path}\n")

    print(f"Saved summary to {output_summary}")


if __name__ == "__main__":
    main()
