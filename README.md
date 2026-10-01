# MedMIG-CR

**Graph-grounded multi-interest retrieval for differential diagnosis**

MedMIG-CR retrieves plausible disease concepts for a patient case by connecting mapped clinical evidence to a large biomedical knowledge graph. The project combines DDXPlus clinical cases, PrimeKG graph structure and node representations, a contrastively aligned MIND query encoder, semantic beam search with optional target-distance guidance, and an optional path-level GRU reranker.

> **Research question.** Can a query encoder learn one or more graph-space clinical interests that guide knowledge-graph search toward the patient's target diagnosis, and can relation-aware path reranking improve the order of the retrieved candidates?

## Overview

The system maps a case's observed evidence to PrimeKG seed nodes. ClinicalMIND encodes the seed sequence into `K` interest vectors and projects them into the frozen PrimeKG node2vec space. Semantic beam search then expands paths from the seed nodes in both graph directions. Optionally, the search is guided by each node's hop distance to the fixed target set, and a path scorer reranks the completed candidates.

```mermaid
flowchart LR
    A[DDXPlus evidence] --> B[PrimeKG seed nodes]
    B --> C[ClinicalMIND multi-interest encoder]
    C --> D[Graph-space projection]
    D --> E[Semantic beam search over PrimeKG]
    T[Target-distance guide] -. optional .-> E
    E --> F[Ranked disease candidates]
    E -. optional .-> G[GRU path reranker]
    G -. post-hoc ranking .-> F
```

Every method compared here starts from the same mapped seed nodes and reaches disease candidates through the PrimeKG graph. Supervised direct rankers (MLP, XGBoost) that score the disease set without using the graph were removed from the benchmark, because they bypass candidate discovery.

## Dataset and Knowledge Graph

### DDXPlus

DDXPlus supplies patient cases, observed evidence, pathology labels, and differential diagnoses. The v2 query cohort contains 1,023,037 training queries, 132,190 validation queries, and 134,236 held-out test queries. Query records store observed evidence as `seed_node_keys` and mapped targets as `target_node_keys`. Target nodes are removed from the seed evidence to prevent label leakage.

### PrimeKG

The processed graph contains 82,240 nodes and 2,734,346 directed edges across 18 relation types. Frozen 64-dimensional node2vec vectors provide graph-space representations for the contrastive encoder, beam scoring, and selected baselines. The DDXPlus condition map resolves 49 pathology labels to 47 distinct PrimeKG target nodes; two label pairs collapse to shared ontology concepts, and three targets are phenotype nodes. The open disease candidate list contains 22,205 disease nodes.

### Mapping and query construction

The v2 mapping adds clinical aliases, token-indexed fuzzy matching, negative categorical-value filtering, and limited multi-mapping for evidence. Query construction keeps broader evidence coverage, reports mapping statistics, and excludes target nodes from the seed set. The processed test cohort retains 134,236 of 134,529 source test patients. All 49 pathology labels are mapped, although some labels share a PrimeKG target concept.

### Experimental protocol

Per-query graph search is CPU-bound, so every method, ablation, and baseline uses the same fixed, pathology-stratified subsets (seed 42, all 49 pathologies, total-variation distance to the source split ≤ 0.0014):

| Role | File | Size |
|---|---|---:|
| Training (MIND, GRU path generation, learned baselines) | `data/processed/ddxplus_v2_subsets/train_50k.csv` | 50,000 |
| Validation (early stopping and all hyperparameter tuning) | `data/processed/ddxplus_v2_subsets/valid_5k.csv` | 5,000 |
| Test (reported numbers) | `data/processed/ddxplus_v2_subsets/test_10k.csv` | 10,000 |

Create the subsets with `scripts/preprocess/build_query_subsets.py`; the output is deterministic. `train_50k.csv` is not version-controlled. Its `patient_index` list is stored in `subsets_summary.json`, and running the script on `train_queries.csv` rebuilds it.

## Method

### InfoNCE MIND

ClinicalMIND uses capsule-style Behavior-to-Interest dynamic routing to turn an ordered, padded sequence of evidence-node IDs into `K` latent interests. Each interest passes through a trainable projection and is normalized in the 64-dimensional PrimeKG graph space. `K=1` gives a single search direction; larger K allows several query-specific directions.

The objective aligns the predicted interests with the graph embeddings of target nodes and observed evidence. For multi-interest models, a diversity penalty discourages duplicated directions:

```text
L = L_target-InfoNCE + 0.2 * L_evidence-InfoNCE + 0.05 * L_diversity
```

The E10 configuration uses embedding dimension 64, three routing iterations, maximum input length 32, and temperature 0.07.

### Semantic beam search

For each interest vector, the search expands incoming and outgoing neighbors of the mapped seed nodes. Each extension is scored by its cosine similarity to the interest vector, minus a degree penalty:

```text
step_score = alpha * cosine(neighbor_embedding, interest_vector)
             - beta * log(degree(neighbor) + 1)
path_score = sum(step_score over path extensions)
```

The default setting uses 10 hops, beam width 128, up to 4,096 paths per interest, `alpha=1.0`, and `beta=0.1`. Candidate collection uses the fixed mapped target universe. Per-query labels are used only by evaluation, never by search.

### Target-distance guided search (optional)

A multi-source BFS from the fixed target set `T` gives `d_T(v)`, the hop distance from every node to its nearest target. It is computed once in about 2 s and is identical for all queries. With `H` hops and `r` hops remaining:

- `--distance_pruning` drops every neighbor with `d_T(v) > r`. Such nodes cannot reach a target within the remaining hops, so the rule never removes a feasible path and frees beam slots for paths that can still end at a diagnosis.
- `--distance_weight λ` orders the beam by an A*-style priority `f = g − λ·d_T(end(p))`, where `g` is the semantic path score. Candidates are still ranked by `g`, because `d_T = 0` at targets.

The guide uses only the label space shared by all queries, which every method also receives under the closed candidate protocol. It does not use per-query labels. On 1,000 validation queries (E10 K=3, no reranker), `--distance_pruning --distance_weight 0.3` raises MRR from 0.404 to 0.749 and candidate coverage from 0.584 to 0.977, at about 3.5× the search time. The GRU reranker must be retrained on guided paths before the two are combined. See [pruning_search.md](pruning_search.md) for the derivation, the leakage analysis, and all runs.

### GRU path reranker

The optional reranker encodes each path as a sequence of hop tokens. A token combines:

- a learned relation embedding (32-D);
- a traversal-direction embedding (8-D);
- a node-type embedding (8-D);
- a projection of the frozen endpoint node embedding (64-D to 32-D).

A one-layer GRU with 64 hidden units summarizes the path, and an MLP maps the final state to a path score. Training uses pairwise BPR with positive target paths and hard negative disease paths. The reranker does not condition on the query.

The reranker has two modes, and results are labeled by mode:

- **Post-hoc reranking** reorders the candidates found by the ordinary beam.
- **GRU-guided beam search** uses the GRU during beam pruning.

## Evaluation

All methods write predictions in a shared format (`patient_index, candidate, score, rank`) and are scored by `scripts/evaluation/evaluate_ddxplus_retrieval.py`. The metrics are MRR and Recall@1/5/10/20/50, macro-averaged over queries. A query without a prediction scores zero.

- **Closed protocol (main).** Candidates are the 47 mapped targets. Methods that score every candidate (PPR, Katz, R-GCN) reach Recall@50 = 1 by construction, so the main comparison uses MRR, Recall@1/5/10, and **candidate coverage** (the share of queries whose target appears in the prediction list). Random ranking of 47 candidates gives MRR ≈ 0.094.
- **Open protocol (optional).** Candidates are all 22,205 PrimeKG disease nodes.

Beam search returns only the candidates it reaches, so reachability and ranking are reported as separate factors.

## Results

### Legacy main runs (first 5,000 test rows)

These runs predate the fixed-subset protocol. They use the first 5,000 rows of the v2 test file and checkpoints trained on the full training set. All final numbers will be re-run on `test_10k.csv` under the protocol above.

| Method | Query representation | GRU mode | MRR | R@1 | R@5 | R@10 | R@20 | R@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Seed-only + beam | Mean seed embedding | None | 0.0843 | 0.0158 | 0.1484 | 0.2490 | 0.3738 | 0.4106 |
| InfoNCE MIND E10, K=1 + beam | One graph-space interest | None | 0.4212 | 0.3468 | 0.5386 | 0.5600 | 0.5620 | 0.5620 |
| InfoNCE MIND E10, K=1 + beam | One graph-space interest | Post-hoc | 0.4971 | 0.4834 | 0.5154 | 0.5172 | 0.5176 | 0.5178 |
| InfoNCE MIND E10, K=3 + beam | Three graph-space interests | None | 0.4180 | 0.3660 | 0.4966 | 0.5322 | 0.5722 | 0.6038 |
| InfoNCE MIND E10, K=3 + beam | Three graph-space interests | Post-hoc | 0.4738 | 0.4362 | 0.5250 | 0.5428 | 0.5442 | 0.5442 |

The last row uses the checkpoint `ddxplus_infonce_e10_hop10/k3` and a GRU trained on 100k K=3 paths. Its result directory is named `no_gru_guided` because GRU guidance was disabled during beam pruning; post-hoc reranking was still enabled.

### Baselines (fixed-subset protocol, closed candidates)

| Baseline | Setting (selected on `valid_5k`) | Test queries | MRR | R@1 | R@5 | R@10 | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| Personalized PageRank | restart 0.15, converged on 100% of queries | 10,000 | 0.1455 | 0.0674 | 0.1686 | 0.2764 | 1.000 |
| Truncated Katz | pending | | | | | | |
| Seed-only beam | pending | | | | | | |
| R-GCN (train_50k) | pending | | | | | | |

Earlier PPR and R-GCN numbers (`PPR/outputs/tune_restart_*`, `Relational_GNN/outputs/full_train_test5000`, `profile_1k`) came from superseded implementations and must not be reported:

- The old PPR ran only in the open space and was capped at 50 iterations without converging.
- The old R-GCN took one optimizer step per epoch.

## Running the Pipeline

Run every command from the repository root. On a 10-core Apple M1 Pro, use `--num_workers 6` while the machine is in use and 8 for unattended runs. Search is CPU-bound, so a GPU does not speed it up. Results are byte-identical for any worker count.

1. Build the fixed query subsets (once):

```bash
.venv/bin/python scripts/preprocess/build_query_subsets.py
```

2. Train InfoNCE MIND (K=3) on the training subset:

```bash
.venv/bin/python src/medmigcr_mind/train_mind_infonce_ddxplus.py \
    --train_csv data/processed/ddxplus_v2_subsets/train_50k.csv \
    --valid_csv data/processed/ddxplus_v2_subsets/valid_5k.csv \
    --graph_dir data/processed/primekg_graph \
    --out_dir experiments/main/k3/checkpoints/mind \
    --K 3 --epochs 10 --batch_size 128 --lr 1e-3 --weight_decay 1e-5 \
    --seed 42 --device cpu
```

3. Generate GRU training paths on the same training subset. To match a guided search at test time, add the same guidance flags here:

```bash
.venv/bin/python scripts/reranking/build_gru_path_reranker_dataset.py \
    --queries_csv data/processed/ddxplus_v2_subsets/train_50k.csv \
    --graph_dir data/processed/primekg_graph \
    --checkpoint experiments/main/k3/checkpoints/mind/clinical_mind_infonce_k3.pt \
    --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
    --output_jsonl experiments/main/k3/data/train50k_paths.jsonl \
    --limit_patients 50000 --interest_count 3 \
    --max_hops 10 --beam_width 128 --paths_per_interest 4096 \
    --alpha 1.0 --beta 0.1 --device cpu --graph_device cpu --num_workers 6
```

4. Train the GRU reranker:

```bash
.venv/bin/python scripts/reranking/train_gru_path_reranker.py \
    --train_jsonl experiments/main/k3/data/train50k_paths.jsonl \
    --metadata_json experiments/main/k3/data/train50k_paths.metadata.json \
    --graph_dir data/processed/primekg_graph \
    --out_checkpoint experiments/main/k3/checkpoints/gru_k3.pt \
    --epochs 12 --steps_per_epoch 2000 --batch_size 256 \
    --hard_negative_top_k 32 --lr 1e-3 --weight_decay 1e-5 \
    --validation_fraction 0.1 --validation_pairs 2048 \
    --early_stopping_patience 2 --seed 13 --device cpu
```

5. Retrieve on the test subset. Use `run_ddxplus_infonce_retrieval.py` with `--checkpoint` instead for the no-reranker variant:

```bash
.venv/bin/python scripts/retrieval/run_ddxplus_infonce_gru_rerank.py \
    --test_queries_csv data/processed/ddxplus_v2_subsets/test_10k.csv \
    --graph_dir data/processed/primekg_graph \
    --mind_checkpoint experiments/main/k3/checkpoints/mind/clinical_mind_infonce_k3.pt \
    --reranker_checkpoint experiments/main/k3/checkpoints/gru_k3.pt \
    --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
    --output_csv experiments/main/k3/results/test10k/predictions.csv \
    --summary_json experiments/main/k3/results/test10k/retrieval_summary.json \
    --interest_count 3 --max_hops 10 --beam_width 128 --paths_per_interest 4096 \
    --top_k 50 --alpha 1.0 --beta 0.1 \
    --disable_gru_beam_guidance --device cpu --graph_device cpu --num_workers 6
```

To enable the target-distance guide, add `--distance_pruning --distance_weight 0.3`. Choose λ on `valid_5k.csv`.

6. Evaluate:

```bash
.venv/bin/python scripts/evaluation/evaluate_ddxplus_retrieval.py \
    --queries_csv data/processed/ddxplus_v2_subsets/test_10k.csv \
    --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
    --predictions experiments/main/k3/results/test10k/predictions.csv \
    --output_dir experiments/main/k3/results/test10k/evaluation \
    --target_mode pathology --topk 1 5 10 20 50
```

The baselines (PPR, Katz, seed-only beam, R-GCN) have their own entry points under `Benchmark_baselines/`. [guide.md](guide.md) gives the tuning grids, test commands, expected runtimes, and reporting notes.

## Documentation and Artifacts

| Resource | Location |
|---|---|
| Baseline run guide (tuning, test, runtimes, reporting caveats) | [guide.md](guide.md) |
| Target-distance guided search: method, leakage analysis, results | [pruning_search.md](pruning_search.md) |
| DDXPlus-to-PrimeKG mapping and query construction study | [reports/mapping_query_v2_experiment.md](reports/mapping_query_v2_experiment.md) |
| InfoNCE K sweep, hop 6 | [reports/infonce_hop6_beam64_experiment.md](reports/infonce_hop6_beam64_experiment.md) |
| E6 beam and GRU reranking study | [reports/infonce_e6_hop8_gru_reranker_report.md](reports/infonce_e6_hop8_gru_reranker_report.md) |
| Weekly baseline and ablation report (historical) | [report.md](report.md) |
| Full v2 train/validation/test query files | `data/processed/ddxplus_v2/` |
| Fixed evaluation subsets | `data/processed/ddxplus_v2_subsets/` |
| PrimeKG graph and embeddings | `data/processed/primekg_graph/` |
| Legacy ablation outputs | `experiments/ablations/` |

Datasets, checkpoints, path dumps, and prediction files are excluded from version control. Small JSON summaries are tracked.

### Code map

| Component | Location |
|---|---|
| Graph storage, scoring, beam search, retrieval engine | `src/medmigcr_kg/` |
| Target-distance guide (multi-source BFS, pruning, A* priority) | `src/medmigcr_kg/target_distance.py` |
| Multi-process query execution | `src/medmigcr_kg/parallel.py` |
| ClinicalMIND and contrastive training | `src/medmigcr_mind/` |
| GRU path reranker | `src/medmigcr_path_reranker/` |
| Query subsets, retrieval, reranking, and evaluation entry points | `scripts/preprocess/`, `scripts/retrieval/`, `scripts/reranking/`, `scripts/evaluation/` |
| Runtime, CPU/RAM/GPU, and output-size profiling | `scripts/profiling/profile_command.py` |
| Shared baseline utilities and batched PPR/Katz propagation | `Benchmark_baselines/baseline_common.py`, `Benchmark_baselines/graph_propagation.py` |
| PPR, Katz, and seed-only beam baselines | `Benchmark_baselines/Graph_Retrieval/` |
| R-GCN baseline | `Benchmark_baselines/Relational_GNN/` |
