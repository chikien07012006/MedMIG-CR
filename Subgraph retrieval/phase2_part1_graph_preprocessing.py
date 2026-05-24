"""
Graph preprocessing and baseline node embedding preparation for MedMIG-CR Phase 2.

This script loads PrimeKG triples, builds retrieval-efficient sparse graph artifacts,
computes degree statistics, and trains baseline Node2Vec node embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


@dataclass
class GraphArtifacts:
    node2id: dict
    relation2id: dict
    id2node: dict
    id2relation: dict
    csr_indptr: np.ndarray
    csr_indices: np.ndarray
    csr_data: np.ndarray
    csr_edge_relids: np.ndarray
    out_degree: np.ndarray
    in_degree: np.ndarray
    num_nodes: int
    num_relations: int
    num_edges: int


def load_primekg(csv_path: Path) -> List[Tuple[str, str, str]]:
    df = pd.read_csv(csv_path, low_memory=False, dtype=str)
    cols = [c.strip().lower() for c in df.columns]
    if set(cols) >= {"subject", "relation", "object"}:
        subj_col = df.columns[cols.index("subject")]
        rel_col = df.columns[cols.index("relation")]
        obj_col = df.columns[cols.index("object")]
    elif set(cols) >= {"head", "relation", "tail"}:
        subj_col = df.columns[cols.index("head")]
        rel_col = df.columns[cols.index("relation")]
        obj_col = df.columns[cols.index("tail")]
    else:
        if len(df.columns) < 3:
            raise ValueError("PrimeKG CSV must have at least 3 columns for subject, relation, object")
        subj_col, rel_col, obj_col = df.columns[:3]

    triples = []
    for row in df.itertuples(index=False):
        s = str(getattr(row, subj_col)).strip()
        r = str(getattr(row, rel_col)).strip()
        o = str(getattr(row, obj_col)).strip()
        if not s or not r or not o:
            continue
        triples.append((s, r, o))
    return triples


def build_mappings(triples: List[Tuple[str, str, str]]) -> Tuple[dict, dict, dict, dict]:
    nodes = set()
    relations = set()
    for s, r, o in triples:
        nodes.add(s)
        nodes.add(o)
        relations.add(r)
    node2id = {node: idx for idx, node in enumerate(sorted(nodes))}
    relation2id = {rel: idx for idx, rel in enumerate(sorted(relations))}
    id2node = {idx: node for node, idx in node2id.items()}
    id2relation = {idx: rel for rel, idx in relation2id.items()}
    return node2id, relation2id, id2node, id2relation


def deduplicate_and_filter(triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    seen = set()
    cleaned = []
    for s, r, o in triples:
        if s == o:
            continue
        key = (s, r, o)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


def build_csr_graph(
    triples: List[Tuple[str, str, str]],
    node2id: dict,
    relation2id: dict,
) -> GraphArtifacts:
    row = []
    col = []
    data = []
    rel_ids = []
    for s, r, o in triples:
        row.append(node2id[s])
        col.append(node2id[o])
        data.append(1)
        rel_ids.append(relation2id[r])

    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    rel_ids = np.asarray(rel_ids, dtype=np.int32)

    order = np.lexsort((col, row))
    row = row[order]
    col = col[order]
    rel_ids = rel_ids[order]

    if len(row) == 0:
        num_nodes = len(node2id)
        csr = sp.csr_matrix(([], ([], [])), shape=(num_nodes, num_nodes))
        csr_indptr = csr.indptr.astype(np.int64)
        csr_indices = csr.indices.astype(np.int64)
        csr_data = csr.data.astype(np.float32)
        out_degree = np.diff(csr_indptr).astype(np.int64)
        in_degree = np.bincount(csr_indices, minlength=num_nodes).astype(np.int64)
        return GraphArtifacts(
            node2id=node2id,
            relation2id=relation2id,
            id2node={idx: node for node, idx in node2id.items()},
            id2relation={idx: rel for rel, idx in relation2id.items()},
            csr_indptr=csr_indptr,
            csr_indices=csr_indices,
            csr_data=csr_data,
            csr_edge_relids=np.asarray([], dtype=np.int32),
            out_degree=out_degree,
            in_degree=in_degree,
            num_nodes=num_nodes,
            num_relations=len(relation2id),
            num_edges=0,
        )

    unique_edges = np.empty(len(row), dtype=bool)
    unique_edges[0] = True
    unique_edges[1:] = (row[1:] != row[:-1]) | (col[1:] != col[:-1])

    row = row[unique_edges]
    col = col[unique_edges]
    rel_ids = rel_ids[unique_edges]
    data = np.ones_like(row, dtype=np.float32)

    num_nodes = len(node2id)
    csr = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    csr_indptr = csr.indptr.astype(np.int64)
    csr_indices = csr.indices.astype(np.int64)
    csr_data = csr.data.astype(np.float32)

    if csr_indices.shape[0] != rel_ids.shape[0]:
        raise RuntimeError("CSR edge count mismatch with relation ids")

    out_degree = np.diff(csr_indptr).astype(np.int64)
    in_degree = np.bincount(csr_indices, minlength=num_nodes).astype(np.int64)

    return GraphArtifacts(
        node2id=node2id,
        relation2id=relation2id,
        id2node={idx: node for node, idx in node2id.items()},
        id2relation={idx: rel for rel, idx in relation2id.items()},
        csr_indptr=csr_indptr,
        csr_indices=csr_indices,
        csr_data=csr_data,
        csr_edge_relids=rel_ids,
        out_degree=out_degree,
        in_degree=in_degree,
        num_nodes=num_nodes,
        num_relations=len(relation2id),
        num_edges=len(rel_ids),
    )


def save_graph_artifacts(artifacts: GraphArtifacts, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = out_dir / "mappings"
    mapping_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        out_dir / "graph_csr.npz",
        indptr=artifacts.csr_indptr,
        indices=artifacts.csr_indices,
        data=artifacts.csr_data,
        edge_relids=artifacts.csr_edge_relids,
        shape=np.array(artifacts.num_nodes, dtype=np.int64),
    )
    np.save(out_dir / "out_degree.npy", artifacts.out_degree)
    np.save(out_dir / "in_degree.npy", artifacts.in_degree)
    with open(mapping_dir / "node2id.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.node2id, f, indent=2, ensure_ascii=False)
    with open(mapping_dir / "relation2id.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.relation2id, f, indent=2, ensure_ascii=False)
    with open(mapping_dir / "id2node.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.id2node, f, indent=2, ensure_ascii=False)
    with open(mapping_dir / "id2relation.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.id2relation, f, indent=2, ensure_ascii=False)
    meta = {
        "num_nodes": artifacts.num_nodes,
        "num_relations": artifacts.num_relations,
        "num_edges": artifacts.num_edges,
        "graph_path": str((out_dir / "graph_csr.npz").resolve()),
        "mappings": {
            "node2id": str((mapping_dir / "node2id.json").resolve()),
            "relation2id": str((mapping_dir / "relation2id.json").resolve()),
        },
    }
    with open(out_dir / "graph_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


class RandomWalkCorpus:
    def __init__(self, artifacts: GraphArtifacts, walk_length: int, walks_per_node: int, seed: int = 42):
        self.artifacts = artifacts
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.seed = seed
        self.rng = random.Random(seed)
        self.node_neighbors = self._build_neighbors()

    def _build_neighbors(self) -> List[np.ndarray]:
        neighbors = []
        indptr = self.artifacts.csr_indptr
        indices = self.artifacts.csr_indices
        for node in range(self.artifacts.num_nodes):
            nbrs = indices[indptr[node] : indptr[node + 1]]
            neighbors.append(nbrs)
        return neighbors

    def generate_walks(self) -> List[List[int]]:
        all_walks: List[List[int]] = []
        for node in range(self.artifacts.num_nodes):
            for _ in range(self.walks_per_node):
                walk = [node]
                while len(walk) < self.walk_length:
                    curr = walk[-1]
                    nbrs = self.node_neighbors[curr]
                    if len(nbrs) == 0:
                        break
                    next_node = int(self.rng.choice(nbrs.tolist()))
                    walk.append(next_node)
                all_walks.append(walk)
        return all_walks


class SkipGramDataset(Dataset):
    def __init__(self, walks: List[List[int]], window_size: int):
        self.pairs: List[Tuple[int, int]] = []
        for walk in walks:
            for idx, center in enumerate(walk):
                start = max(0, idx - window_size)
                end = min(len(walk), idx + window_size + 1)
                for context_idx in range(start, end):
                    if idx == context_idx:
                        continue
                    self.pairs.append((center, walk[context_idx]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.pairs[idx]


class Node2VecModel(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int):
        super().__init__()
        self.target_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.context_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.target_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.context_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, center_nodes: torch.Tensor, context_nodes: torch.Tensor, negative_nodes: torch.Tensor) -> torch.Tensor:
        center_emb = self.target_embeddings(center_nodes)
        context_emb = self.context_embeddings(context_nodes)
        negative_emb = self.context_embeddings(negative_nodes)
        pos_score = torch.sum(center_emb * context_emb, dim=-1)
        neg_score = torch.bmm(negative_emb, center_emb.unsqueeze(2)).squeeze(2)
        loss_pos = F.logsigmoid(pos_score)
        loss_neg = F.logsigmoid(-neg_score).sum(dim=1)
        return -torch.mean(loss_pos + loss_neg)


class NegativeSampler:
    def __init__(self, degree: np.ndarray, power: float = 0.75):
        prob = np.power(degree.astype(np.float64), power)
        prob /= prob.sum()
        self.dist = torch.from_numpy(prob).float()

    def sample(self, batch_size: int, num_negatives: int, device: torch.device) -> torch.Tensor:
        samples = torch.multinomial(self.dist, batch_size * num_negatives, replacement=True)
        return samples.to(device).view(batch_size, num_negatives)


def train_node2vec_embeddings(
    artifacts: GraphArtifacts,
    walk_length: int,
    walks_per_node: int,
    window_size: int,
    num_negative: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    corpus = RandomWalkCorpus(artifacts, walk_length=walk_length, walks_per_node=walks_per_node, seed=seed)
    walks = corpus.generate_walks()
    dataset = SkipGramDataset(walks, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    model = Node2VecModel(artifacts.num_nodes, embedding_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    sampler = NegativeSampler(artifacts.out_degree + 1)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for center, context in dataloader:
            center = center.to(device)
            context = context.to(device)
            negative = sampler.sample(center.shape[0], num_negative, device)
            loss = model(center, context, negative)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * center.shape[0]
        avg_loss = total_loss / max(len(dataset), 1)
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.6f}")

    embeddings = model.target_embeddings.weight.detach().cpu().numpy()
    return embeddings


def save_embeddings(embeddings: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "node_embeddings.npy", embeddings)
    with open(out_dir / "embedding_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"num_nodes": embeddings.shape[0], "embedding_dim": embeddings.shape[1]}, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PrimeKG preprocessing and Node2Vec embedding preparation for MedMIG-CR Phase 2")
    p.add_argument("--primekg_csv", type=str, default="PrimeKG.csv")
    p.add_argument("--output_dir", type=str, default="Subgraph retrieval/processed")
    p.add_argument("--walk_length", type=int, default=32)
    p.add_argument("--walks_per_node", type=int, default=10)
    p.add_argument("--window_size", type=int, default=5)
    p.add_argument("--num_negative", type=int, default=5)
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root_output = Path(args.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.primekg_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"PrimeKG CSV not found: {csv_path}")

    print(f"Loading PrimeKG triples from {csv_path}")
    triples = load_primekg(csv_path)
    print(f"Loaded {len(triples)} triples")

    print("Cleaning triples (remove self-loops and duplicates)")
    triples = deduplicate_and_filter(triples)
    print(f"After cleaning: {len(triples)} triples")

    print("Building node and relation mappings")
    node2id, relation2id, id2node, id2relation = build_mappings(triples)
    print(f"Nodes: {len(node2id)}, Relations: {len(relation2id)}")

    print("Building CSR sparse graph")
    artifacts = build_csr_graph(triples, node2id, relation2id)
    print(f"Graph edges: {artifacts.num_edges}, nodes: {artifacts.num_nodes}")

    print("Saving graph artifacts")
    save_graph_artifacts(artifacts, root_output)

    print("Training baseline Node2Vec embeddings")
    embeddings = train_node2vec_embeddings(
        artifacts,
        walk_length=args.walk_length,
        walks_per_node=args.walks_per_node,
        window_size=args.window_size,
        num_negative=args.num_negative,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=args.seed,
    )
    save_embeddings(embeddings, root_output)
    print(f"Saved node embeddings to {root_output / 'node_embeddings.npy'}")


if __name__ == "__main__":
    main()
