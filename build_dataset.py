from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from medmigcr.config import PipelineConfig
from medmigcr.exporters import (
    build_adjacency_list,
    build_id_maps,
    build_neighbor_sampler,
    multihop_neighbors,
    pyg_edge_index,
    save_pickle,
    save_triplets_txt,
    to_id_triplets,
)
from medmigcr.interactions import build_interactions, interaction_matrix
from medmigcr.kg_loader import build_disease_phenotype_map, collect_filtered_triples
from medmigcr.query_builder import attach_phenotype_names, build_queries
from medmigcr.stats import compute_stats


def main() -> None:
    # Defaults: build a RS-friendly subgraph; cap ultra-dense relations for practicality.
    cfg = PipelineConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)

    # Recommended defaults for this PrimeKG export:
    # - keep disease/phenotype/gene/pathway/drug/anatomy edges
    # - cap extremely dense anatomy_protein_present and drug_drug if needed
    if not cfg.max_edges_per_relation:
        cfg.max_edges_per_relation = {
            "anatomy_protein_present": 500_000,
            "drug_drug": 500_000,
        }

    print("STEP1: building disease->phenotype map...")
    dis_to_phen = build_disease_phenotype_map(cfg)
    print(f"  diseases with >= {cfg.min_symptoms_per_query} phenotypes: {len(dis_to_phen)}")

    print("STEP2: building phenotype name lookup (for query_text)...")
    phen_names = attach_phenotype_names(cfg)
    print(f"  phenotype names: {len(phen_names)}")

    print("STEP2: generating synthetic patient queries...")
    query_df, query_positives = build_queries(dis_to_phen, cfg, rng, phen_names=phen_names)
    query_nodes_path = cfg.output_dir / "query_nodes.csv"
    query_df.to_csv(query_nodes_path, index=False)
    print(f"  queries: {len(query_df)} -> {query_nodes_path}")

    print("STEP3: generating interactions (train/valid/test)...")
    splits = build_interactions(query_df, query_positives, dis_to_phen, cfg, rng)
    splits.train.to_csv(cfg.output_dir / "interactions_train.csv", index=False)
    splits.valid.to_csv(cfg.output_dir / "interactions_valid.csv", index=False)
    splits.test.to_csv(cfg.output_dir / "interactions_test.csv", index=False)
    print(
        f"  interactions: train={len(splits.train)} valid={len(splits.valid)} test={len(splits.test)}"
    )

    print("STEP4: collecting filtered biomedical triples (chunked)...")
    bio_triples, bio_entities, rel_kept = collect_filtered_triples(cfg, rng)
    print(f"  kept biomedical triples: {len(bio_triples)} (entities: {len(bio_entities)})")

    # Add query nodes + interaction edges to form the collaborative KG
    query_entities = [f"query|synthetic|{qid}" for qid in query_df["query_id"].astype(str).tolist()]
    disease_entities = sorted(dis_to_phen.keys())
    all_entities = set(bio_entities) | set(query_entities)

    # Use ONLY positive diagnosis edges as query→disease links (recommendation graph core)
    interaction_triples = []
    pos_all = pd.concat([splits.train, splits.valid, splits.test], axis=0, ignore_index=True)
    pos_all = pos_all[pos_all["label"] == 1]
    for r in pos_all.itertuples(index=False):
        q = f"query|synthetic|{r.query_id}"
        d = r.disease_id
        interaction_triples.append((q, cfg.interaction_relation, d))
        all_entities.add(q)
        all_entities.add(d)

    unified_triples = bio_triples + interaction_triples
    all_relations = sorted(set([r for _, r, _ in unified_triples]))

    print("STEP5: building id maps and RS artifacts...")
    maps = build_id_maps(all_entities, all_relations)
    save_pickle(maps.entity2id, cfg.output_dir / "entity2id.pkl")
    save_pickle(maps.relation2id, cfg.output_dir / "relation2id.pkl")

    id_triplets = to_id_triplets(unified_triples, maps.entity2id, maps.relation2id)
    save_triplets_txt(id_triplets, cfg.output_dir / "triplets.txt")

    # Interaction matrix for training (positives only)
    mat = interaction_matrix(pos_all, query_df["query_id"].astype(str).tolist(), disease_entities)
    from scipy import sparse

    sparse.save_npz(cfg.output_dir / "interaction_matrix.npz", mat)

    # Adjacency list + neighbor sampler (KGCN/KGAT utilities)
    adj = build_adjacency_list(id_triplets, num_entities=len(maps.entity2id))
    save_pickle(adj, cfg.output_dir / "adjacency_list.pkl")
    sampler = build_neighbor_sampler(adj, seed=cfg.seed)
    save_pickle(sampler, cfg.output_dir / "neighbor_sampler.pkl")

    # PyG edge_index + edge_type
    edge_index, edge_type = pyg_edge_index(id_triplets)
    np.save(cfg.output_dir / "edge_index.npy", edge_index)
    np.save(cfg.output_dir / "edge_type.npy", edge_type)

    # Multi-hop cache for query nodes (RippleNet-style)
    query_entity_ids = [maps.entity2id[q] for q in query_entities if q in maps.entity2id]
    mh = multihop_neighbors(query_entity_ids, adj, depth=cfg.multihop_depth)
    save_pickle(mh, cfg.output_dir / "multihop_cache.pkl")

    # Candidate disease retrieval index (simple list + id map; your retriever can build ANN on top)
    save_pickle(disease_entities, cfg.output_dir / "disease_entities.pkl")
    save_pickle({d: maps.entity2id[d] for d in disease_entities if d in maps.entity2id}, cfg.output_dir / "disease_entity2eid.pkl")

    print("STEP7: computing statistics...")
    stats = compute_stats(
        query_df=query_df,
        disease_entity_keys=disease_entities,
        entity2id=maps.entity2id,
        relation2id=maps.relation2id,
        id_triplets=id_triplets,
        interactions_all=pd.concat([splits.train, splits.valid, splits.test], axis=0, ignore_index=True),
        interaction_matrix_shape=mat.shape,
        adjacency_list=adj,
        query_entity_ids=query_entity_ids,
        max_hop_stat_depth=max(3, cfg.multihop_depth),
    )
    with open(cfg.output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, indent=2)

    with open(cfg.output_dir / "relation_kept_counts.json", "w", encoding="utf-8") as f:
        json.dump(rel_kept, f, indent=2)

    print("DONE.")
    print(f"Outputs written to: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()

