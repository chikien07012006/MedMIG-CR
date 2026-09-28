# Personalized PageRank Baseline

This is a non-parametric graph-retrieval baseline over the existing PrimeKG graph. It does not train a model, use node embeddings, or use path features.

## Algorithm

For a DDXPlus query, map `seed_node_keys` to PrimeKG graph IDs and distribute personalization mass uniformly across valid seeds. The transition graph is undirected: it is the union of the PrimeKG adjacency and its transpose, consistent with the project's beam search expanding both incoming and outgoing neighbors. Rows are normalized by degree.

Starting from the seed distribution $p_0=s$, iterate:

```text
p_(t+1) = r*s + (1-r)*(P^T*p_t + dangling_mass*s)
```

Here `r` is the restart probability (default 0.15), `P` is the row-normalized adjacency, and `dangling_mass` is the score mass at zero-degree nodes, returned to the seed distribution. Stop when the L1 change is below tolerance or the maximum iteration count is reached. Rank all PrimeKG disease nodes by their converged PPR score; retain the top 50 predictions.

The candidate list is `data/processed/primekg_graph/disease_nodes.csv`; ground-truth nodes are never inserted. If a mapped target is a phenotype rather than a disease, it is absent from this disease-only candidate space and counts as a candidate miss. Metrics report query-level candidate recall and target-node coverage separately from MRR/Recall@K. `candidate_recall@K` is marked not applicable because PPR scores the entire candidate space rather than using a separate pre-ranking candidate generator.

## Run Full Validation and Test Metrics

Run from the repository root. There is no model training stage. First compare restart settings using `--validation_only`, then run the selected setting on full validation and full test once.

Validation-only tuning example:

```bash
python Benchmark_baselines/Graph_Retrieval/PPR/run_ppr_baseline.py \
  --train_csv data/processed/ddxplus_v2/train_queries.csv \
  --valid_csv data/processed/ddxplus_v2/valid_queries.csv \
  --test_csv data/processed/ddxplus_v2/test_queries.csv \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --graph_dir data/processed/primekg_graph \
  --restart_probability 0.15 --max_iterations 50 --tolerance 1e-8 \
  --validation_only \
  --out_dir Benchmark_baselines/Graph_Retrieval/PPR/outputs/tune_restart_015
```

Repeat with a few candidate restart probabilities (for example 0.10, 0.15, 0.20), each in a distinct output folder, and select using validation MRR. Then run the final selected value without `--validation_only`:

```bash
python Benchmark_baselines/Graph_Retrieval/PPR/run_ppr_baseline.py \
  --train_csv data/processed/ddxplus_v2/train_queries.csv \
  --valid_csv data/processed/ddxplus_v2/valid_queries.csv \
  --test_csv data/processed/ddxplus_v2/test_queries.csv \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --graph_dir data/processed/primekg_graph \
  --restart_probability 0.15 \
  --max_iterations 50 \
  --tolerance 1e-8 \
  --top_k 50 \
  --out_dir Benchmark_baselines/Graph_Retrieval/PPR/outputs/ppr_v1
```

Use `--max_train_rows`, `--max_valid_rows`, and `--max_test_rows` for a small smoke/profile run. PPR cost is per query; the progress bars show queries per second. Lower `--max_iterations` for a faster preliminary profile, but report the value used and monitor the fraction of queries that converged.

## Outputs

- `predictions.csv`: up to 50 ranked PrimeKG disease node keys for each test patient.
- `by_patient.csv`: target count, how many targets are in the disease candidate set, per-query candidate recall, first hit rank, MRR, and Recall@K.
- `evaluation_summary.json`: full test MRR/Recall@5/10/20/50, candidate coverage/recall, absent target count, convergence, and throughput.
- `ppr_summary.json`: graph/candidate configuration and query counts plus full validation and test metrics.
