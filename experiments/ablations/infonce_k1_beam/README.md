# InfoNCE MIND K=1 + Beam Ablation

This ablation tests whether multiple MIND interests improve retrieval. It uses
InfoNCE-trained MIND with one interest vector, ordinary semantic beam search,
and no GRU. The existing training, retrieval, and evaluation scripts are reused.

For a controlled comparison with K=3, train K=1 for 10 epochs with the same
training defaults, then run both K=1 and K=3 through the same non-GRU retrieval
script, graph, test-query cohort, target map, and beam settings (`10` hops,
width `128`, `4096` paths per interest, `alpha=1.0`, `beta=0.1`). The existing
K=3 `no_gru_guided` result still applies post-hoc GRU reranking, so it is not a
valid no-GRU comparison.

Run from the repository root:

```bash
.venv/bin/python src/medmigcr_mind/train_mind_infonce_ddxplus.py \
  --train_csv data/processed/ddxplus_v2/train_queries.csv \
  --valid_csv data/processed/ddxplus_v2/valid_queries.csv \
  --graph_dir data/processed/primekg_graph \
  --out_dir experiments/ablations/infonce_k1_beam/checkpoints \
  --K 1 \
  --epochs 12 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --seed 42 \
  --device cpu
```

```bash
.venv/bin/python scripts/retrieval/run_ddxplus_infonce_retrieval.py \
  --test_queries_csv data/processed/ddxplus_v2/test_queries.csv \
  --graph_dir data/processed/primekg_graph \
  --checkpoint experiments/ablations/infonce_k1_beam/checkpoints/clinical_mind_infonce_k1.pt \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --output_csv experiments/ablations/infonce_k1_beam/results/test5000/predictions.csv \
  --summary_json experiments/ablations/infonce_k1_beam/results/test5000/retrieval_summary.json \
  --interest_count 1 \
  --max_hops 10 \
  --beam_width 128 \
  --paths_per_interest 4096 \
  --top_k 50 \
  --alpha 1.0 \
  --beta 0.1 \
  --limit_patients 5000 \
  --device cpu \
  --graph_device auto
```

Run the K=3 no-GRU control with its existing E10 checkpoint and matching
retrieval settings:

```bash
.venv/bin/python scripts/retrieval/run_ddxplus_infonce_retrieval.py \
  --test_queries_csv data/processed/ddxplus_v2/test_queries.csv \
  --graph_dir data/processed/primekg_graph \
  --checkpoint artifacts/checkpoints/ddxplus_infonce_e10_hop10/k3/clinical_mind_infonce_k3.pt \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --output_csv experiments/ablations/infonce_k1_beam/results/k3_no_gru/test5000/predictions.csv \
  --summary_json experiments/ablations/infonce_k1_beam/results/k3_no_gru/test5000/retrieval_summary.json \
  --interest_count 3 \
  --max_hops 10 \
  --beam_width 128 \
  --paths_per_interest 4096 \
  --top_k 50 \
  --alpha 1.0 \
  --beta 0.1 \
  --limit_patients 5000 \
  --device cpu \
  --graph_device auto
```

Evaluate those K=3 predictions with the same evaluator command as below,
replacing the `--predictions` path with
`experiments/ablations/infonce_k1_beam/results/k3_no_gru/test5000/predictions.csv`
and the `--output_dir` with
`experiments/ablations/infonce_k1_beam/results/k3_no_gru/test5000/evaluation`.

```bash
.venv/bin/python scripts/evaluation/evaluate_ddxplus_retrieval.py \
  --queries_csv data/processed/ddxplus_v2/test_queries.csv \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --predictions experiments/ablations/infonce_k1_beam/results/test5000/predictions.csv \
  --output_dir experiments/ablations/infonce_k1_beam/results/test5000/evaluation \
  --target_mode pathology \
  --topk 1 5 10 20 50 \
  --limit_patients 5000
```

The K=1 checkpoint must be trained with `--K 1`; setting
`--interest_count 1` on a K=3 checkpoint only selects its first interest and
does not create a K=1-trained model. Existing K=1 checkpoints are available,
but their training runs used different epoch counts, so use them only for a
quick smoke comparison rather than the primary paper ablation.