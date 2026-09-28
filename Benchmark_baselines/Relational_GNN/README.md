# PrimeKG R-GCN Disease Retrieval Baseline

This baseline encodes the **entire PrimeKG graph once per epoch** with relation-aware R-GCN layers. It does not build one graph/model per query and does not use path features or pathology labels as inputs.

- Graph: existing `data/processed/primekg_graph` CSR graph, 82,240 nodes, about 2.73M directed edges, 18 relation IDs.
- Initial features: existing frozen-version node2vec vectors used as trainable 64-D initial node features (the embedding table is optimized jointly with R-GCN).
- Encoder: `RGCNConv(64,128,num_relations=18,num_bases=9)`, ReLU, then `RGCNConv(128,64,num_relations=18,num_bases=9)`.
- Query: mean of the final embeddings for that query's seed node IDs.
- Pair scorer: `[q,d,q*d]` then `192 -> 128 -> 1`.
- Objective: mean pairwise `softplus(-positive_score + negative_score)`; 20 sampled disease negatives per positive by default.
- Optimizer: Adam, learning rate `1e-3`. Best checkpoint is selected by validation MRR.
- Candidates: complete `disease_nodes.csv` PrimeKG disease set. Targets outside the disease set remain candidate misses; they are never inserted.

The implementation computes one full graph encoder forward per epoch and accumulates minibatch ranking gradients through that shared forward before one optimizer update. This respects the transductive full-graph model, but graph activations can consume substantial GPU memory. Validation also recomputes and scores the full disease candidate set per query, so subset profiling is important.

## Command 1: Subset Timing/Sanity Run

From repository root. This trains on up to 1,000 train queries for one epoch, validates on up to 100 queries, and scores 100 test queries. It is a runtime/sanity profile only, not paper metrics.

```bash
.venv/bin/python Benchmark_baselines/Relational_GNN/run_rgcn_baseline.py \
  --train_csv data/processed/ddxplus_v2/train_queries.csv \
  --valid_csv data/processed/ddxplus_v2/valid_queries.csv \
  --test_csv data/processed/ddxplus_v2/test_queries.csv \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --graph_dir data/processed/primekg_graph \
  --max_train_queries 1000 \
  --max_valid_queries 100 \
  --max_test_queries 100 \
  --epochs 1 \
  --query_batch_size 128 \
  --negative_samples 20 \
  --eval_query_batch_size 4 \
  --candidate_chunk_size 1024 \
  --device cpu \
  --run_test \
  --out_dir Benchmark_baselines/Relational_GNN/outputs/profile_1k
```

For CUDA profiling use `--device cuda`. If memory is tight, reduce `--query_batch_size` and `--eval_query_batch_size`; these bound query/negative and candidate scoring working tensors, though R-GCN still encodes the whole graph.

## Command 2: Full Train, Validation, and Test

Uses full official train and validation splits, early-stopping by validation MRR, then full test metrics. Test is only evaluated after model selection.

```bash
.venv/bin/python Benchmark_baselines/Relational_GNN/run_rgcn_baseline.py \
  --train_csv data/processed/ddxplus_v2/train_queries.csv \
  --valid_csv data/processed/ddxplus_v2/valid_queries.csv \
  --test_csv data/processed/ddxplus_v2/test_queries.csv \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --graph_dir data/processed/primekg_graph \
  --epochs 5 \
  --early_stopping_patience 2 \
  --query_batch_size 512 \
  --negative_samples 20 \
  --learning_rate 1e-3 \
  --eval_query_batch_size 4 \
  --candidate_chunk_size 1024 \
  --device cuda \
  --run_test \
  --out_dir Benchmark_baselines/Relational_GNN/outputs/full_v1
```

Use `--device cpu` if CUDA is unavailable. The exact full run may require a high-memory GPU because full-graph activations are retained while ranking minibatches accumulate gradients. The subset profile should establish feasibility first. Full official split sizes are approximately 1,023,037 train, 132,190 validation, and 134,236 test queries.

## Outputs

- `rgcn_ranker.pt`: selected model checkpoint.
- `training_summary.json`: graph sizes, architecture, parameter count, per-epoch loss/MRR, train time, sampled positive/negative counts, peak CUDA allocation when available, and test metrics when `--run_test` is set.
- `validation_predictions.csv`, `validation_by_patient.csv`.
- With `--run_test`: `test_predictions.csv`, `test_by_patient.csv`, and test metrics embedded in `training_summary.json`.

Metrics include MRR, Recall@5/10/20/50 and candidate recall. Candidate recall measures whether at least one mapped target is a disease in the PrimeKG candidate space; R-GCN ranks the full set, so candidate-recall-at-K is not applicable.
