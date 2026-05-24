"""Pipeline configuration (reproducible defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Optional


@dataclass
class PipelineConfig:
    kg_path: Path = Path("kg_giant.csv")
    output_dir: Path = Path("processed_medmigcr_dataset")
    seed: int = 42

    # Node types to retain in the biomedical subgraph
    allowed_node_types: FrozenSet[str] = field(
        default_factory=lambda: frozenset(
            {
                "disease",
                "effect/phenotype",
                "gene/protein",
                "pathway",
                "anatomy",
                "drug",
            }
        )
    )

    # Optional: drop ultra-dense relation types for smaller/faster builds (still valid RS-KG)
    exclude_relations: FrozenSet[str] = field(default_factory=frozenset)

    # Cap edges for a relation (0 = no cap). Applied after exclude, reproducible subsample.
    max_edges_per_relation: dict[str, int] = field(default_factory=dict)

    # Query synthesis
    min_symptoms_per_query: int = 2
    max_symptoms_per_query: int = 8
    queries_per_disease: int = 5
    max_queries_total: Optional[int] = None  # cap for smoke tests

    # Interactions per query (excluding positives)
    num_hard_negatives: int = 8  # overlapping symptoms, not true positive
    num_random_negatives: int = 8

    # Data split (by query_id)
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    # test_ratio = remainder

    # Multi-hop cache depth for query nodes only
    multihop_depth: int = 2

    # Interaction relation name in unified KG
    interaction_relation: str = "interacts_with"
