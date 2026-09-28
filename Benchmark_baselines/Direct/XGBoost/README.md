# Direct XGBoost Disease Ranker

This is a direct learning-to-rank baseline. It ranks all disease nodes in the PrimeKG `disease_nodes.csv` candidate set and does not perform graph traversal or consume path features.

## Features and Objective

For each query, load the 64-D PrimeKG embeddings of its `seed_node_keys` and concatenate mean and max pooling to make a fixed 128-D query vector. For every query/disease pair, build a 320-D feature:

```text
[query_mean_and_max (128), disease_embedding (64),
 abs(query_mean - disease_embedding) (64),
 query_mean * disease_embedding (64)]
```

`XGBRanker(objective="rank:pairwise")` learns to order known target disease nodes above sampled PrimeKG disease negatives. All mapped positive disease targets are retained; `--negative_samples` negatives are sampled per query group, excluding positives. Targets absent from the disease-only candidate universe are not inserted. Training queries with no mapped target in the candidate set are counted and skipped as ranking groups.

Validation/test inference scores the full disease candidate set in chunks and computes MRR, Recall@5/10/20/50, and candidate recall separately. Candidate recall is target coverage in the disease-node universe; candidate-recall-at-K is not applicable because this ranker scores every candidate. `--max_eval_queries` bounds validation/test query count while each evaluated query still scores all 22,205 disease candidates.

## Install

From repository root:

```bash
source .venv/bin/activate
python -m pip install xgboost
```

On macOS, XGBoost also requires the OpenMP runtime if it is not already installed:

```bash
brew install libomp
```

## Validation Profile Command

This initial profile trains on up to 5,000 train queries, samples 20 negatives per positive query group, and scores all PrimeKG disease candidates for 200 validation queries. It does not evaluate test data.

```bash
python Benchmark_baselines/Direct/XGBoost/run_xgboost_ranker.py \
  --train_csv data/processed/ddxplus_v2/train_queries.csv \
  --valid_csv data/processed/ddxplus_v2/valid_queries.csv \
  --test_csv data/processed/ddxplus_v2/test_queries.csv \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --graph_dir data/processed/primekg_graph \
  --max_train_rows 5000 \
  --max_valid_rows 2000 \
  --max_eval_queries 200 \
  --negative_samples 20 \
  --n_estimators 200 \
  --max_depth 6 \
  --candidate_chunk_size 2048 \
  --out_dir Benchmark_baselines/Direct/XGBoost/outputs/profile_train5k_valid200
```

The profile writes `xgboost_ranker.json`, `training_summary.json`, `validation_summary.json`, `validation_predictions.csv`, and `validation_by_patient.csv`. Summaries include training duration, candidate pairs scored, queries/second and candidate-pairs/second. No full train/test experiment is run by the profile command.

For another profile point, vary `--max_eval_queries` (for example 50, 100, 200) while holding model settings fixed. Validation queries are the first rows of the split, not a random sample. Use a fixed validation subset only for preliminary runtime/quality checks; choose final settings using the full validation split.

## Optional Test Profile

Once the validation profile is acceptable, append `--run_test --max_test_rows 1000 --max_eval_queries 1000` and use a new output folder. This scores 1,000 test rows only as a runtime smoke test; do not report that subset as the official benchmark.
