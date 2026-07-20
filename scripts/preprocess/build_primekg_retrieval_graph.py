from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


DEFAULT_NODE_TYPES = ("disease", "effect/phenotype", "drug", "gene/protein", "pathway", "anatomy")
DEFAULT_EDGE_CAPS = {"drug_drug": 500_000, "anatomy_protein_present": 500_000}


@dataclass
class GraphArtifacts:
    node2id: Dict[str, int]
    relation2id: Dict[str, int]
    id2node: Dict[int, str]
    id2relation: Dict[int, str]
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    edge_relids: np.ndarray
    out_degree: np.ndarray
    in_degree: np.ndarray


def entity_key(node_type: str, source: str, node_id: str) -> str:
    return f"{str(node_type).strip()}|{str(source).strip()}|{str(node_id).strip()}"


def parse_relation_caps(raw: str) -> Dict[str, int]:
    if not raw:
        return {}
    caps: Dict[str, int] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        name, value = part.split("=", 1)
        caps[name.strip()] = int(value)
    return caps


def collect_triples(
    primekg_csv: Path,
    allowed_types: set[str],
    relation_caps: Dict[str, int],
    chunksize: int,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, Dict[str, str]], Counter[str]]:
    required = {
        "relation",
        "x_id",
        "x_type",
        "x_name",
        "x_source",
        "y_id",
        "y_type",
        "y_name",
        "y_source",
    }
    triples: List[Tuple[str, str, str]] = []
    nodes: Dict[str, Dict[str, str]] = {}
    relation_counts: Counter[str] = Counter()
    kept_per_relation: Counter[str] = Counter()

    for chunk in pd.read_csv(primekg_csv, chunksize=chunksize, dtype=str, low_memory=False):
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"PrimeKG CSV is missing required columns: {sorted(missing)}")

        mask = chunk["x_type"].isin(allowed_types) & chunk["y_type"].isin(allowed_types)
        for row in chunk.loc[mask].itertuples(index=False):
            row_data = row._asdict()
            relation = str(row_data["relation"]).strip()
            relation_counts[relation] += 1
            cap = relation_caps.get(relation, 0)
            if cap > 0 and kept_per_relation[relation] >= cap:
                continue

            source = entity_key(row_data["x_type"], row_data["x_source"], row_data["x_id"])
            target = entity_key(row_data["y_type"], row_data["y_source"], row_data["y_id"])
            if source == target:
                continue
            triples.append((source, relation, target))
            kept_per_relation[relation] += 1
            nodes.setdefault(
                source,
                {
                    "node_key": source,
                    "node_type": str(row_data["x_type"]).strip(),
                    "source": str(row_data["x_source"]).strip(),
                    "node_id": str(row_data["x_id"]).strip(),
                    "name": str(row_data["x_name"]).strip(),
                },
            )
            nodes.setdefault(
                target,
                {
                    "node_key": target,
                    "node_type": str(row_data["y_type"]).strip(),
                    "source": str(row_data["y_source"]).strip(),
                    "node_id": str(row_data["y_id"]).strip(),
                    "name": str(row_data["y_name"]).strip(),
                },
            )

    return triples, nodes, relation_counts


def build_graph(triples: List[Tuple[str, str, str]]) -> GraphArtifacts:
    edge_relation: Dict[Tuple[str, str], str] = {}
    for source, relation, target in sorted(set(triples)):
        edge_relation.setdefault((source, target), relation)
    triples = [(source, relation, target) for (source, target), relation in sorted(edge_relation.items())]
    node_keys = sorted({node for h, _, t in triples for node in (h, t)})
    relation_keys = sorted({relation for _, relation, _ in triples})
    node2id = {node: idx for idx, node in enumerate(node_keys)}
    relation2id = {relation: idx for idx, relation in enumerate(relation_keys)}

    rows = np.asarray([node2id[h] for h, _, _ in triples], dtype=np.int64)
    cols = np.asarray([node2id[t] for _, _, t in triples], dtype=np.int64)
    relids = np.asarray([relation2id[r] for _, r, _ in triples], dtype=np.int32)
    data = np.ones(rows.shape[0], dtype=np.float32)

    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    relids = relids[order]
    data = data[order]

    csr = sp.csr_matrix((data, (rows, cols)), shape=(len(node2id), len(node2id)))
    out_degree = np.diff(csr.indptr).astype(np.int64)
    in_degree = np.bincount(csr.indices, minlength=len(node2id)).astype(np.int64)

    if csr.indices.shape[0] != relids.shape[0]:
        raise RuntimeError("CSR edge count mismatch with relation ids")

    return GraphArtifacts(
        node2id=node2id,
        relation2id=relation2id,
        id2node={idx: node for node, idx in node2id.items()},
        id2relation={idx: relation for relation, idx in relation2id.items()},
        indptr=csr.indptr.astype(np.int64),
        indices=csr.indices.astype(np.int64),
        data=csr.data.astype(np.float32),
        edge_relids=relids,
        out_degree=out_degree,
        in_degree=in_degree,
    )


class RandomWalkCorpus:
    def __init__(self, artifacts: GraphArtifacts, walk_length: int, walks_per_node: int, seed: int) -> None:
        self.artifacts = artifacts
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.rng = random.Random(seed)

    def generate_walks(self) -> List[List[int]]:
        walks: List[List[int]] = []
        for node in range(len(self.artifacts.node2id)):
            start = int(self.artifacts.indptr[node])
            end = int(self.artifacts.indptr[node + 1])
            if start == end:
                continue
            for _ in range(self.walks_per_node):
                walk = [node]
                while len(walk) < self.walk_length:
                    curr = walk[-1]
                    s = int(self.artifacts.indptr[curr])
                    e = int(self.artifacts.indptr[curr + 1])
                    if s == e:
                        break
                    walk.append(int(self.rng.choice(self.artifacts.indices[s:e].tolist())))
                walks.append(walk)
        return walks


class SkipGramDataset(Dataset):
    def __init__(self, walks: List[List[int]], window_size: int) -> None:
        self.pairs: List[Tuple[int, int]] = []
        for walk in walks:
            for idx, center in enumerate(walk):
                for context_idx in range(max(0, idx - window_size), min(len(walk), idx + window_size + 1)):
                    if idx != context_idx:
                        self.pairs.append((center, walk[context_idx]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.pairs[idx]


class SkipGramModel(nn.Module):
    def __init__(self, num_nodes: int, dim: int) -> None:
        super().__init__()
        self.target = nn.Embedding(num_nodes, dim)
        self.context = nn.Embedding(num_nodes, dim)
        nn.init.normal_(self.target.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.context.weight, mean=0.0, std=0.02)

    def forward(self, center: torch.Tensor, context: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        center_emb = self.target(center)
        context_emb = self.context(context)
        negative_emb = self.context(negative)
        pos_score = torch.sum(center_emb * context_emb, dim=-1)
        neg_score = torch.bmm(negative_emb, center_emb.unsqueeze(2)).squeeze(2)
        return -torch.mean(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(dim=1))


def train_node_embeddings(
    artifacts: GraphArtifacts,
    dim: int,
    walk_length: int,
    walks_per_node: int,
    window_size: int,
    num_negative: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    walks = RandomWalkCorpus(artifacts, walk_length, walks_per_node, seed).generate_walks()
    dataset = SkipGramDataset(walks, window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = SkipGramModel(len(artifacts.node2id), dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    degree = artifacts.out_degree + artifacts.in_degree + 1
    prob = np.power(degree.astype(np.float64), 0.75)
    prob = prob / prob.sum()
    neg_dist = torch.from_numpy(prob).float()

    for epoch in range(1, epochs + 1):
        total = 0.0
        for center, context in loader:
            center = center.to(device)
            context = context.to(device)
            negative = torch.multinomial(neg_dist, center.shape[0] * num_negative, replacement=True)
            negative = negative.to(device).view(center.shape[0], num_negative)
            loss = model(center, context, negative)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * center.shape[0]
        print(f"epoch={epoch} loss={total / max(1, len(dataset)):.6f}")
    return model.target.weight.detach().cpu().numpy()


def save_artifacts(
    artifacts: GraphArtifacts,
    nodes: Dict[str, Dict[str, str]],
    embeddings: np.ndarray,
    out_dir: Path,
    metadata: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = out_dir / "mappings"
    mapping_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        out_dir / "graph_csr.npz",
        indptr=artifacts.indptr,
        indices=artifacts.indices,
        data=artifacts.data,
        edge_relids=artifacts.edge_relids,
    )
    np.save(out_dir / "out_degree.npy", artifacts.out_degree)
    np.save(out_dir / "in_degree.npy", artifacts.in_degree)
    np.save(out_dir / "node_embeddings.npy", embeddings.astype(np.float32))
    for filename, payload in {
        "node2id.json": artifacts.node2id,
        "relation2id.json": artifacts.relation2id,
        "id2node.json": artifacts.id2node,
        "id2relation.json": artifacts.id2relation,
    }.items():
        with (mapping_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    node_rows = [nodes[key] for key in artifacts.node2id if key in nodes]
    pd.DataFrame(node_rows).to_csv(out_dir / "node_metadata.csv", index=False)
    pd.DataFrame([row for row in node_rows if row["node_type"] == "disease"]).to_csv(
        out_dir / "disease_nodes.csv", index=False
    )
    with (out_dir / "graph_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **metadata,
                "num_nodes": len(artifacts.node2id),
                "num_relations": len(artifacts.relation2id),
                "num_edges": int(artifacts.indices.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval-ready PrimeKG graph artifacts.")
    parser.add_argument("--primekg_csv", type=Path, default=Path("kg_giant.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/primekg_graph"))
    parser.add_argument("--node_types", nargs="*", default=list(DEFAULT_NODE_TYPES))
    parser.add_argument(
        "--relation_caps",
        type=str,
        default=",".join(f"{key}={value}" for key, value in DEFAULT_EDGE_CAPS.items()),
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--embedding_method", choices=("node2vec", "random"), default="node2vec")
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--walk_length", type=int, default=16)
    parser.add_argument("--walks_per_node", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--num_negative", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relation_caps = parse_relation_caps(args.relation_caps)
    triples, nodes, relation_counts = collect_triples(
        primekg_csv=args.primekg_csv,
        allowed_types=set(args.node_types),
        relation_caps=relation_caps,
        chunksize=args.chunksize,
    )
    print(f"Collected {len(triples)} triples across {len(nodes)} nodes")
    artifacts = build_graph(triples)
    if args.embedding_method == "node2vec":
        embeddings = train_node_embeddings(
            artifacts=artifacts,
            dim=args.embedding_dim,
            walk_length=args.walk_length,
            walks_per_node=args.walks_per_node,
            window_size=args.window_size,
            num_negative=args.num_negative,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
    else:
        rng = np.random.default_rng(args.seed)
        embeddings = rng.normal(0.0, 0.02, size=(len(artifacts.node2id), args.embedding_dim)).astype(np.float32)
    save_artifacts(
        artifacts=artifacts,
        nodes=nodes,
        embeddings=embeddings,
        out_dir=args.output_dir,
        metadata={
            "primekg_csv": str(args.primekg_csv),
            "node_types": list(args.node_types),
            "relation_caps": relation_caps,
            "relation_counts_raw": dict(relation_counts),
            "embedding_method": args.embedding_method,
        },
    )
    print(f"Wrote retrieval graph artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
