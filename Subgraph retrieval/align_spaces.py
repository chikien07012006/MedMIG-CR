import torch
import numpy as np
import json
from pathlib import Path
import sys

# Add paths to sys.path to import local modules
sys.path.append(r"d:\MedMIG-CR\Subgraph retrieval")
sys.path.append(r"d:\MedMIG-CR\multi-interest")

from graph_store import GraphStore

def align():
    # 1. Load ClinicalMIND
    ckpt_path = Path(r"d:\MedMIG-CR\multi-interest\checkpoints\clinical_mind.pt")
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mind_disease2id = ckpt["disease_str2id"]
    mind_disease_emb = ckpt["model_state"]["model_state"]["disease_emb.weight"].numpy() if "model_state" in ckpt["model_state"] else ckpt["model_state"]["disease_emb.weight"].numpy()
    
    # 2. Load GraphStore
    processed_dir = Path(r"d:\MedMIG-CR\Subgraph retrieval\processed_test")
    graph_store = GraphStore.load(
        graph_npz=processed_dir / "graph_csr.npz",
        node_embeddings_npy=processed_dir / "node_embeddings.npy",
        out_degree_npy=processed_dir / "out_degree.npy",
        in_degree_npy=processed_dir / "in_degree.npy",
        mapping_dir=processed_dir / "mappings",
        device="cpu",
    )
    
    # 3. Find Overlap using tokens
    mind_id_to_token = {}
    for k, vid in mind_disease2id.items():
        if "|" in k:
            token = k.split("|")[-1].strip()
            mind_id_to_token[vid] = token
            
    primekg_entities = set(graph_store.node2id.keys())
    
    x_pairs = []
    y_pairs = []
    
    overlap_count = 0
    for mind_id, token in mind_id_to_token.items():
        if token in graph_store.node2id:
            primekg_id = graph_store.node2id[token]
            x_pairs.append(mind_disease_emb[mind_id])
            y_pairs.append(graph_store.node_embeddings[primekg_id])
            overlap_count += 1
            
    print(f"Overlap size: {overlap_count}")
    
    if overlap_count < 10:
        print("Too little overlap to align.")
        return

    # 4. Extract Pairs
    X = np.array(x_pairs)
    Y = np.array(y_pairs)
    
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    # 5. Solve for Alignment (Linear Regression Y = XW + b)
    # Adding bias term
    X_reg = np.hstack([X, np.ones((X.shape[0], 1))])
    W_plus, res, rank, s = np.linalg.lstsq(X_reg, Y, rcond=None)
    
    W = W_plus[:-1, :]
    b = W_plus[-1, :]
    
    # Check error
    Y_pred = X @ W + b
    mse = np.mean((Y - Y_pred)**2)
    print(f"Alignment MSE: {mse:.6f}")
    
    # Cosine similarity improvement check
    def cos_sim(a, b):
        return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    
    before_sim = np.mean(cos_sim(X, Y))
    after_sim = np.mean(cos_sim(Y_pred, Y))
    print(f"Average Cosine Similarity Before alignment: {before_sim:.4f}")
    print(f"Average Cosine Similarity After alignment: {after_sim:.4f}")
    
    # 6. Save Alignment
    save_path = Path(r"d:\MedMIG-CR\multi-interest\checkpoints\alignment_mind_to_primekg.npz")
    np.savez(save_path, weight=W, bias=b)
    print(f"Alignment saved to {save_path}")

if __name__ == "__main__":
    align()
