from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class GraphStore:
    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        edge_relids: np.ndarray,
        out_degree: np.ndarray,
        in_degree: np.ndarray,
        node_embeddings: np.ndarray,
        node2id: Dict[str, int],
        id2node: Dict[int, str],
        relation2id: Dict[str, int],
        id2relation: Dict[int, str],
        device: Optional[Union[str, "torch.device"]] = None,
    ):
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.edge_relids = edge_relids
        self.out_degree = out_degree
        self.in_degree = in_degree
        self.node_embeddings = node_embeddings
        self.node2id = node2id
        self.id2node = id2node
        self.relation2id = relation2id
        self.id2relation = id2relation
        self._device = self._resolve_device(device)
        self.node_embeddings_torch = self._build_torch_embeddings()
        self._reverse_indptr: Optional[np.ndarray] = None
        self._reverse_indices: Optional[np.ndarray] = None
        self._reverse_edge_relids: Optional[np.ndarray] = None

    @classmethod
    def load(
        cls,
        graph_npz: Path,
        node_embeddings_npy: Path,
        out_degree_npy: Path,
        in_degree_npy: Path,
        mapping_dir: Path,
        device: Optional[Union[str, "torch.device"]] = None,
    ) -> "GraphStore":
        arrays = np.load(graph_npz)
        indptr = arrays["indptr"].astype(np.int64)
        indices = arrays["indices"].astype(np.int64)
        data = arrays["data"].astype(np.float32)
        edge_relids = arrays["edge_relids"].astype(np.int32)

        node_embeddings = np.load(node_embeddings_npy)
        out_degree = np.load(out_degree_npy).astype(np.int64)
        in_degree = np.load(in_degree_npy).astype(np.int64)

        node2id = cls._load_json(mapping_dir / "node2id.json")
        relation2id = cls._load_json(mapping_dir / "relation2id.json")
        id2node = {int(k): v for k, v in cls._load_json(mapping_dir / "id2node.json").items()}
        id2relation = {int(k): v for k, v in cls._load_json(mapping_dir / "id2relation.json").items()}

        return cls(
            indptr=indptr,
            indices=indices,
            data=data,
            edge_relids=edge_relids,
            out_degree=out_degree,
            in_degree=in_degree,
            node_embeddings=node_embeddings,
            node2id=node2id,
            id2node=id2node,
            relation2id=relation2id,
            id2relation=id2relation,
            device=device,
        )

    @staticmethod
    def _load_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _resolve_device(self, device: Optional[Union[str, "torch.device"]]) -> Optional["torch.device"]:
        if torch is None:
            return None
        if device is None or device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            return torch.device(device)
        return device

    def _build_torch_embeddings(self) -> Optional["torch.Tensor"]:
        if torch is None:
            return None
        if self._device is None:
            return None
        return torch.from_numpy(self.node_embeddings).float().to(self._device)

    @property
    def num_nodes(self) -> int:
        return int(self.indptr.shape[0] - 1)

    @property
    def use_torch(self) -> bool:
        return torch is not None and self.node_embeddings_torch is not None

    @property
    def device(self) -> Optional["torch.device"]:
        return self._device

    def get_neighbors(self, node_id: int, direction: str = "out") -> np.ndarray:
        if node_id < 0 or node_id >= self.num_nodes:
            return np.array([], dtype=np.int64)
        if direction == "out":
            start = int(self.indptr[node_id])
            end = int(self.indptr[node_id + 1])
            return self.indices[start:end]
        if direction == "in":
            self._ensure_reverse_csr()
            assert self._reverse_indptr is not None and self._reverse_indices is not None
            start = int(self._reverse_indptr[node_id])
            end = int(self._reverse_indptr[node_id + 1])
            return self._reverse_indices[start:end]
        raise ValueError(f"Unknown direction: {direction}")

    def _ensure_reverse_csr(self) -> None:
        if self._reverse_indptr is not None and self._reverse_indices is not None:
            return
        row_counts = np.diff(self.indptr).astype(np.int64)
        row_ids = np.repeat(np.arange(self.num_nodes, dtype=np.int64), row_counts)
        if row_ids.size != self.indices.size:
            row_ids = np.repeat(np.arange(self.num_nodes, dtype=np.int64), row_counts, axis=0)
        order = np.argsort(self.indices, kind="stable")
        reverse_indices = row_ids[order]
        edge_relids = self.edge_relids[order]
        counts = np.bincount(self.indices, minlength=self.num_nodes).astype(np.int64)
        reverse_indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
        reverse_indptr[1:] = np.cumsum(counts)
        self._reverse_indptr = reverse_indptr
        self._reverse_indices = reverse_indices
        self._reverse_edge_relids = edge_relids

    def get_embeddings(self, ids: Sequence[int], as_tensor: bool = False) -> Union[np.ndarray, "torch.Tensor"]:
        if self.use_torch and (as_tensor or self.device is not None):
            indices = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
            return self.node_embeddings_torch[indices]
        return self.node_embeddings[np.asarray(ids, dtype=np.int64)]

    def lookup_node_id(self, node_name: str) -> Optional[int]:
        return self.node2id.get(node_name)

    def lookup_node_name(self, node_id: int) -> Optional[str]:
        return self.id2node.get(int(node_id))

    def get_edge_relations(self, source_node: int, target_node: int) -> List[str]:
        row_start = int(self.indptr[source_node])
        row_end = int(self.indptr[source_node + 1])
        if row_start >= row_end:
            return []
        positions = np.where(self.indices[row_start:row_end] == target_node)[0]
        if positions.size == 0:
            return []
        rel_ids = self.edge_relids[row_start:row_end][positions]
        return [self.id2relation[int(rel)] for rel in rel_ids]
