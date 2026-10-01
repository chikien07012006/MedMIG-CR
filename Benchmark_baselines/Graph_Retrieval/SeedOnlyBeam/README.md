# Seed-only Beam Search

Ablation baseline that removes the InfoNCE MIND encoder and the GRU reranker. The single search direction is the mean node2vec embedding of the query's seed nodes. Beam search, step scoring, target-aware candidate collection, and the best-path candidate score are the same as in the main method's no-reranker branch, so the gap to the main method measures what the learned interests add.

Queries run in parallel CPU processes (`--num_workers`). On the first 100 test queries, the output was checked to be byte-identical to the original `experiments/ablations/seed_only_beam/run_seed_only_beam.py`. See [guide.md](../../../guide.md).
