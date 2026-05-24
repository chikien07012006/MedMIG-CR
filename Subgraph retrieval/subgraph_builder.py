from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from beam_search import BeamItem
from graph_store import GraphStore


def build_subgraph_from_paths(paths: Iterable[BeamItem], graph_store: GraphStore) -> Dict[str, object]:
    nodes: Set[int] = set()
    edges: Set[Tuple[int, int]] = set()
    relations: Dict[Tuple[int, int], List[str]] = {}

    for item in paths:
        for source, target in zip(item.path[:-1], item.path[1:]):
            nodes.add(source)
            nodes.add(target)
            edge = (source, target)
            edges.add(edge)
            relations.setdefault(edge, [])
            relations[edge].extend(graph_store.get_edge_relations(source, target))

    unique_relations = {
        edge: sorted(set(relation_list)) for edge, relation_list in relations.items()
    }
    return {
        "nodes": sorted(nodes),
        "edges": sorted(edges),
        "edge_relations": unique_relations,
    }