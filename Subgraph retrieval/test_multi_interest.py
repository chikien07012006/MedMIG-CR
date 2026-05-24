import argparse
import json
import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch

# Add parent to sys.path to allow imports from neighbor directories
# Assuming we run from 'Subgraph retrieval' or project root
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / "multi-interest"))

from graph_store import GraphStore
from retrieval_engine import RetrievalEngine
from beam_search import SemanticBeamSearch
from mind_medmigcr_model import ClinicalMIND

def load_symptoms(query_id: str, query_csv: Path) -> str:
    df = pd.read_csv(query_csv, dtype=str)
    row = df.loc[df["query_id"] == query_id]
    if row.empty:
        raise ValueError(f"Query {query_id} not found in {query_csv}")
    return str(row.iloc[0]["symptom_entity_ids"]).strip()

def encode_symptoms(symptoms_str: str, model: ClinicalMIND, symptom_str2id: dict, max_len: int, device: torch.device):
    ids = []
    for s in symptoms_str.split(";"):
        s = s.strip()
        if not s: continue
        ids.append(symptom_str2id.get(s, 1)) # 1 = UNK
    
    if len(ids) > max_len:
        ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(0) # 0 = PAD
        
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        z, _, active_mask = model.encode_symptoms(x)
    
    active_count = int(active_mask[0].sum().item())
    return z[0, :active_count, :], active_count

def format_path(path_nodes: tuple, graph_store: GraphStore) -> str:
    formatted = []
    for node_id in path_nodes:
        node_name = graph_store.lookup_node_name(node_id)
        formatted.append(f"{node_name}")
    return " -> ".join(formatted)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_id", type=str)
    parser.add_argument("--max_hops", type=int, default=5)
    parser.add_argument("--beam_width", type=int, default=32)
    parser.add_argument("--k_per_interest", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(__file__).resolve().parent
    project_root = base.parent
    
    # Paths
    ckpt_path = project_root / "multi-interest" / "checkpoints" / "clinical_mind.pt"
    query_csv = project_root / "processed_medmigcr_dataset" / "query_nodes.csv"
    processed_dir = base / "processed_test"
    projection_path = project_root / "multi-interest" / "checkpoints" / "alignment_mind_to_primekg.npz"

    # 1. Load Model
    print(f"Loading ClinicalMIND from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt["hparams"]
    model = ClinicalMIND(
        num_symptoms=hp["num_symptoms"],
        num_diseases=hp["num_diseases"],
        dim=hp["D"],
        num_interests=hp["K"],
        max_seq_len=hp["max_seq_len"],
        num_routing_iters=hp["R"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    # 2. Get Interests for Query
    symptoms_str = load_symptoms(args.query_id, query_csv)
    interest_vectors, active_count = encode_symptoms(
        symptoms_str, model, ckpt["symptom_str2id"], hp["max_seq_len"], device
    )
    
    # Requirement 1: Print number of interest vectors
    print(f"\n[QUERY {args.query_id}]")
    print(f"Number of active interest vectors: {active_count}")
    
    # 3. Load Graph and Projection
    print(f"Loading Graph from {processed_dir}...")
    graph_store = GraphStore.load(
        graph_npz=processed_dir / "graph_csr.npz",
        node_embeddings_npy=processed_dir / "node_embeddings.npy",
        out_degree_npy=processed_dir / "out_degree.npy",
        in_degree_npy=processed_dir / "in_degree.npy",
        mapping_dir=processed_dir / "mappings",
        device=device,
    )
    
    projection = None
    if projection_path.exists():
        print(f"Loading alignment projection from {projection_path}...")
        projection = RetrievalEngine.load_projection(projection_path)
    
    # 4. Run Separate Beam Searches
    # Requirement 2 & 3: Run separately for each vector and print
    raw_seed_names = [s.strip() for s in symptoms_str.split(";") if s.strip()]
    seed_ids = []
    
    for node_name in raw_seed_names:
        nid = graph_store.lookup_node_id(node_name)
        if nid is None:
            if "|" in node_name:
                token = node_name.split("|")[-1].strip()
                nid = graph_store.lookup_node_id(token)
        
        if nid is not None:
            seed_ids.append(nid)
    
    if not seed_ids:
        print(f"Error: No seed nodes found in graph for query {args.query_id}.")
        print(f"Raw names: {raw_seed_names}")
        return

    all_retrieved_paths = []
    
    for i in range(active_count):
        print(f"\n--- Running Beam Search for Interest Vector {i} ---")
        z_i = interest_vectors[i]
        
        search = SemanticBeamSearch(
            graph_store, 
            z_i, 
            alpha=1.0, 
            beta=0.4, 
            projection=projection
        )
        
        # Keep maximum k=3 paths per interest
        paths = search.search(
            seed_ids, 
            max_hops=args.max_hops, 
            beam_width=args.beam_width, 
            topk_paths=args.k_per_interest
        )
        
        if not paths:
            print("  No paths found.")
        else:
            for rank, item in enumerate(paths, 1):
                path_str = format_path(item.path, graph_store)
                print(f"  Path {rank} (score={item.score:.4f}): {path_str}")
                all_retrieved_paths.append(item)
    
    print(f"\nTotal paths across all interests: {len(all_retrieved_paths)}")

if __name__ == "__main__":
    main()
