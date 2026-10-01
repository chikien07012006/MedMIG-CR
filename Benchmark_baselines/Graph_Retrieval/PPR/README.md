# Personalized PageRank (PPR)

Non-parametric baseline. A random walk with restart starts from the query's seed nodes (uniform personalization) on the undirected PrimeKG graph. The undirected graph is the union of each edge with its reverse, matching the beam search, which expands both directions. Candidates are ranked by stationary probability.

```text
r <- a * s + (1 - a) * (P^T r + dangling_mass * s)      until ||r_new - r||_1 <= tol per query
```

- Queries are propagated in batches (one sparse x dense product per iteration), and batches run in parallel CPU processes (`--num_workers`). The output does not depend on the worker count.
- `--candidate_space closed` (default) ranks the 47 mapped targets; `open` ranks all 22,205 PrimeKG diseases.
- `--ppr_score degree` divides by node degree to reduce hub bias. `--exclude_relations` drops relation types.
- Writes `predictions.csv`, `run_summary.json` (config, convergence, timing, coverage), and `evaluation/` (shared evaluator).

See [guide.md](../../../guide.md) for tuning and test commands. The previous implementation (sequential, open space only, capped at 50 iterations without convergence) produced `outputs/tune_restart_*_valid2000`. Those outputs are kept only for reference.
