# MedMIG-CR

MedMIG-CR is a research workspace for clinical knowledge-graph retrieval over
PrimeKG using DDXPlus patient cases. The current project focus is evaluating
whether a graph-grounded multi-interest query encoder improves disease retrieval.

## Current Pipeline

The active workflow is:

```text
DDXPlus patient evidences
-> map evidences/pathologies to PrimeKG nodes
-> build train/valid/test query CSVs
-> train ClinicalMIND query encoder
-> run PrimeKG beam search
-> evaluate MRR and Recall@5/10/20/50
```

There are two main model branches:

- `MIND + OLS baseline`: trains ClinicalMIND in its own embedding space, then fits
  a ridge/OLS projection from MIND disease embeddings to PrimeKG node embeddings.
- `InfoNCE MIND proposed`: trains ClinicalMIND outputs directly into the frozen
  PrimeKG node2vec space using target-node InfoNCE, evidence-node InfoNCE, and
  multi-interest diversity regularization.

Latest reports:

- `reports/mapping_query_v2_experiment.md`
- `reports/infonce_hop6_beam64_experiment.md`

## Repository Layout

```text
Benchmark data/DDXPlus/       DDXPlus metadata and patient CSVs
data/mappings/                DDXPlus-to-PrimeKG mapping JSONs
data/processed/               Generated query files and PrimeKG graph artifacts
src/medmigcr_kg/              GraphStore, scoring, beam search, retrieval engine
src/medmigcr_mind/            ClinicalMIND, InfoNCE model, training scripts
scripts/mapping/              DDXPlus-to-PrimeKG mapping builders
scripts/preprocess/           PrimeKG and DDXPlus preprocessing utilities
scripts/alignment/            OLS/ridge MIND-to-PrimeKG alignment baseline
scripts/retrieval/            Retrieval runners that write prediction CSVs
scripts/evaluation/           Recall@K and MRR evaluation harness
scripts/experiments/          End-to-end experiment runners
scripts/reporting/            Markdown report builders
artifacts/checkpoints/        Generated model checkpoints
results/                      Generated predictions and metric outputs
reports/                      Human-readable experiment reports
```

Generated data, checkpoints, and results are ignored by Git. Keep raw data local.

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Expected local files:

```text
kg_giant.csv
Benchmark data/DDXPlus/release_conditions.json
Benchmark data/DDXPlus/release_evidences.json
Benchmark data/DDXPlus/release_train_patients.csv
Benchmark data/DDXPlus/release_validate_patients.csv
Benchmark data/DDXPlus/release_test_patients.csv
```

## Preprocessing

Build PrimeKG node index:

```bash
python scripts/preprocess/build_primekg_node_index.py ^
  --primekg_csv kg_giant.csv ^
  --output_dir data/processed/primekg
```

Build retrieval-ready graph artifacts:

```bash
python scripts/preprocess/build_primekg_retrieval_graph.py ^
  --primekg_csv kg_giant.csv ^
  --output_dir data/processed/primekg_graph ^
  --embedding_method node2vec
```

Build DDXPlus-to-PrimeKG mappings:

```bash
python scripts/mapping/build_ddxplus_primekg_mappings.py ^
  --conditions_json "Benchmark data/DDXPlus/release_conditions.json" ^
  --evidences_json "Benchmark data/DDXPlus/release_evidences.json" ^
  --primekg_index_dir data/processed/primekg ^
  --output_dir data/mappings/ddxplus_v2
```

Build train/valid/test query files:

```bash
python scripts/preprocess/build_ddxplus_test_queries.py ^
  --patients_csv "Benchmark data/DDXPlus/release_train_patients.csv" ^
  --evidence_map data/mappings/ddxplus_v2/evidence_to_primekg.json ^
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json ^
  --output_csv data/processed/ddxplus_v2/train_queries.csv ^
  --summary_json data/processed/ddxplus_v2/train_query_summary.json

python scripts/preprocess/build_ddxplus_test_queries.py ^
  --patients_csv "Benchmark data/DDXPlus/release_validate_patients.csv" ^
  --evidence_map data/mappings/ddxplus_v2/evidence_to_primekg.json ^
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json ^
  --output_csv data/processed/ddxplus_v2/valid_queries.csv ^
  --summary_json data/processed/ddxplus_v2/valid_query_summary.json

python scripts/preprocess/build_ddxplus_test_queries.py ^
  --patients_csv "Benchmark data/DDXPlus/release_test_patients.csv" ^
  --evidence_map data/mappings/ddxplus_v2/evidence_to_primekg.json ^
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json ^
  --output_csv data/processed/ddxplus_v2/test_queries.csv ^
  --summary_json data/processed/ddxplus_v2/test_query_summary.json
```

The optional 5,000-query comparison cohort can be regenerated with `scripts/preprocess/build_common_query_cohort.py` if old-vs-new mapping comparisons are needed:

```text
data/processed/comparison_cohort/new_queries.csv
```

Use `data/processed/ddxplus_v2/test_queries.csv` for the official held-out DDXPlus v2 evaluation.

## Proposed InfoNCE Experiment

Run the current full experiment for `K=1,2,3`:

```bash
python scripts/experiments/run_infonce_full_pipeline.py ^
  --ks 1 2 3 ^
  --epochs 3 ^
  --batch_size 256 ^
  --test_queries_csv data/processed/ddxplus_v2/test_queries.csv ^
  --max_hops 6 ^
  --beam_width 64 ^
  --paths_per_interest 4096 ^
  --alpha 1.0 ^
  --beta 0.1 ^
  --device cpu ^
  --graph_device auto ^
  --checkpoint_root artifacts/checkpoints/ddxplus_infonce_beam64_hop6 ^
  --results_root results/infonce_hop6_beam64_eval5000 ^
  --report_md reports/infonce_hop6_beam64_experiment.md
```

PowerShell shortcut:

```powershell
.\scripts\experiments\run_infonce_k123_cpu.ps1
```

Main outputs:

```text
artifacts/checkpoints/ddxplus_infonce_beam64_hop6/k1
artifacts/checkpoints/ddxplus_infonce_beam64_hop6/k2
artifacts/checkpoints/ddxplus_infonce_beam64_hop6/k3
results/infonce_hop6_beam64_eval5000/k1
results/infonce_hop6_beam64_eval5000/k2
results/infonce_hop6_beam64_eval5000/k3
reports/infonce_hop6_beam64_experiment.md
```

Current 5,000-query results:

| Method | MRR | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| InfoNCE hop6 K=1 | 0.3214 | 0.3602 | 0.3606 | 0.3606 | 0.3606 |
| InfoNCE hop6 K=2 | 0.2876 | 0.3222 | 0.3500 | 0.3646 | 0.3646 |
| InfoNCE hop6 K=3 | 0.2931 | 0.3274 | 0.3800 | 0.4182 | 0.4202 |

## MIND + OLS Baseline

Train ClinicalMIND:

```bash
python src/medmigcr_mind/train_mind_ddxplus.py ^
  --train_csv data/processed/ddxplus_v2/train_queries.csv ^
  --valid_csv data/processed/ddxplus_v2/valid_queries.csv ^
  --out_dir artifacts/checkpoints/ddxplus_v2 ^
  --K 3 ^
  --epochs 3
```

Fit OLS/ridge projection:

```bash
python scripts/alignment/align_mind_to_primekg.py ^
  --checkpoint artifacts/checkpoints/ddxplus_v2/clinical_mind_ddxplus_k3.pt ^
  --graph_dir data/processed/primekg_graph ^
  --output_npz artifacts/checkpoints/ddxplus_v2/alignment_k3_to_primekg.npz
```

Run retrieval:

```bash
python scripts/retrieval/run_ddxplus_mind_retrieval.py ^
  --test_queries_csv data/processed/ddxplus_v2/test_queries.csv ^
  --graph_dir data/processed/primekg_graph ^
  --checkpoint artifacts/checkpoints/ddxplus_v2/clinical_mind_ddxplus_k3.pt ^
  --projection artifacts/checkpoints/ddxplus_v2/alignment_k3_to_primekg.npz ^
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json ^
  --output_csv results/mind_v2_targetaware_k3_eval5000/predictions.csv ^
  --interest_count 3
```

## Evaluation Contract

Any retrieval method can be evaluated if it writes:

```csv
patient_index,candidate,score,rank
0,disease|MONDO|1234,0.98,1
0,disease|MONDO|5678,0.91,2
```

Evaluate predictions:

```bash
python scripts/evaluation/evaluate_ddxplus_retrieval.py ^
  --queries_csv data/processed/ddxplus_v2/test_queries.csv ^
  --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json ^
  --predictions results/my_run/predictions.csv ^
  --output_dir results/my_run/evaluation ^
  --topk 5 10 20 50
```

Outputs:

```text
summary.json
by_patient.csv
```

## Active Research Next Steps

- Run controlled retrieval comparisons to separate model improvements from hop/scoring changes.
- Run InfoNCE ablations: target-only, target+evidence, target+evidence+diversity.
- Add depth-stratified reporting for hop 2/3/4/5/6.
- Improve ranking after candidate generation with a lightweight reranker.
- Run final evaluation on the full DDXPlus test set after hyperparameters are fixed.
