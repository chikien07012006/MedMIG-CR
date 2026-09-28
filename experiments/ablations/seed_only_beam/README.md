# Seed-Only + Beam Ablation

This experiment removes MIND entirely. The query vector is the mean PrimeKG
embedding of the resolved seed nodes; beam search, graph scoring, target
universe, and evaluation protocol are otherwise kept aligned with the current
InfoNCE retrieval setup.

Run `run_seed_only_beam.py` from the repository root. By default it evaluates
the first 5,000 rows of `data/processed/ddxplus_v2/test_queries.csv` and writes:

- `results/test5000/predictions.csv`
- `results/test5000/retrieval_summary.json`
- `results/test5000/evaluation/summary.json`
- `results/test5000/evaluation/by_patient.csv`

The evaluator uses the query CSV's per-patient target nodes for metrics. The
retrieval runner only uses the fixed condition-map target universe to ensure
target endpoints encountered during beam expansion are collected; it does not
read query labels when searching or ranking.