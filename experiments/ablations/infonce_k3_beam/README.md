.venv/bin/python src/medmigcr_mind/train_mind_infonce_ddxplus.py \
  --train_csv data/processed/ddxplus_v2/train_queries.csv \
  --valid_csv data/processed/ddxplus_v2/valid_queries.csv \
  --graph_dir data/processed/primekg_graph \
  --out_dir experiments/ablations/infonce_k3_beam/checkpoints \
  --K 3 \
  --epochs 12 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --seed 42 \
  --device cpu


  .venv/bin/python scripts/retrieval/run_ddxplus_infonce_retrieval.py \
  --test_queries_csv data/processed/ddxplus_v2/test_queries.csv \
  --graph_dir data/processed/primekg_graph \
  --checkpoint experiments/ablations/infonce_k3_beam/checkpoints/clinical_mind_infonce_k3.pt \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --output_csv experiments/ablations/infonce_k3_beam/results/test5000/predictions.csv \
  --summary_json experiments/ablations/infonce_k3_beam/results/test5000/retrieval_summary.json \
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


  .venv/bin/python scripts/evaluation/evaluate_ddxplus_retrieval.py \
  --queries_csv data/processed/ddxplus_v2/test_queries.csv \
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --predictions experiments/ablations/infonce_k3_beam/results/test5000/predictions.csv \
  --output_dir experiments/ablations/infonce_k3_beam/results/test5000/evaluation \
  --target_mode pathology \
  --topk 1 5 10 20 50 \
  --limit_patients 5000