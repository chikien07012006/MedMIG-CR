# Direct Disease-Ranking MLP

This baseline ranks the complete disease-node candidate set in the existing PrimeKG graph, without graph traversal or paths.

## Formulation

Each seed node is looked up in `data/processed/primekg_graph/node_embeddings.npy` through the graph's `mappings/node2id.json`. Its 64-D embedding contributes to permutation-invariant pooling:

```text
query_emb = [mean(seed embeddings); max(seed embeddings)]  # 128-D
```

A trainable linear layer projects the query to 64 dimensions so it can be compared with a 64-D PrimeKG disease embedding. Pair features are `[q, d, |q-d|, q*d]` (256-D), scored by `256 -> 256 -> 128 -> 1` with ReLU/dropout. Only seed embeddings form the query representation; pathology strings are never model inputs.

Candidates come from `data/processed/primekg_graph/disease_nodes.csv` and their embeddings from the same frozen PrimeKG graph artifact. The current list has 22,205 disease nodes, not the 47 mapped DDXPlus targets. A validation guard rejects a candidate set of 50 or fewer nodes, one no larger than the mapped target set, or any non-disease candidate. A query with no valid seed embedding raises an error. Ground-truth nodes absent from the disease list are not inserted and count as candidate misses.

Training examples are positive `(query, mapped disease target)` pairs. Each positive gets `--negative_samples` randomly sampled disease candidates, excluding all known positive candidates for that query. Training uses pairwise BPR logistic loss; the checkpoint is selected by validation MRR with early stopping. Queries whose mapped target is not a disease candidate still count in validation/test metrics with zero ranking score and are included in candidate-coverage reporting; they do not create invalid positive training pairs.

Because this ranker scores the entire candidate set, candidate recall means the fraction of queries whose mapped target set overlaps the PrimeKG disease candidate set. There is no pre-ranking candidate generator here, so candidate-recall-at-K is marked not applicable; ranked Recall@K measures the model's ordering over all candidates.

## Profile Run

Run from repository root. This is a pipeline/performance profile, not a reportable result:

```bash
python Benchmark_baselines/Direct/MLP/run_mlp_baseline.py \
  --epochs 1 --max_train_rows 4096 --max_valid_rows 512 --max_test_rows 256 \
  --batch_size 256 --negative_samples 4 \
  --eval_query_batch_size 4 --candidate_chunk_size 1024 --device cpu \
  --out_dir Benchmark_baselines/Direct/MLP/outputs/profile_direct_retrieval
```


## Full Run

```bash
python Benchmark_baselines/Direct/MLP/run_mlp_baseline.py \
  --train_csv data/processed/ddxplus_v2/train_queries.csv \
  --valid_csv data/processed/ddxplus_v2/valid_queries.csv \
  --test_csv data/processed/ddxplus_v2/test_queries.csv \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --graph_dir data/processed/primekg_graph \
  --epochs 30 --early_stopping_patience 3 --batch_size 1024 \
  --negative_samples 8 --device auto \
  --out_dir Benchmark_baselines/Direct/MLP/outputs/direct_retrieval_v1
```

Set `--device cuda` on a CUDA machine if `auto` does not select it. Reduce `--eval_query_batch_size` or `--candidate_chunk_size` if device memory is limited. Validation MRR is computed by scoring every disease candidate each epoch, so validation can be a substantial fraction of total runtime.

## Outputs

- `mlp_baseline.pt`: scorer state, architecture, and ordered disease candidate list.
- `predictions.csv`: up to 50 ranked PrimeKG disease nodes per test query.
- `evaluation_summary.json`: MRR, Recall@5/10/20/50, candidate recall/coverage and absent target counts, split sizes, candidate count, pair/negative counts, dimensions, pooling, loss, and parameter count.
- `by_patient.csv`: per-query target presence, candidate coverage, rank, reciprocal rank, and Recall@K.
- `training_summary.json`: train/validation counts, positive pairs, negatives actually sampled, per-epoch losses/times, best validation MRR, and parameter count.

The summary explicitly reports that graph traversal and pathology inputs are disabled. The model/evaluator outside this Direct/MLP directory are not modified by this baseline.
