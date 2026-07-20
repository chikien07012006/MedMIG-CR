# MedMIG-CR

MedMIG-CR is a research workspace for evaluating knowledge-graph retrieval methods for clinical diagnosis. The current benchmark direction is:

- **DDXPlus** provides patient cases, observed evidences, and ground-truth diagnoses.
- **PrimeKG** provides biomedical graph knowledge for retrieval.
- A retrieval agent produces ranked disease candidates for each DDXPlus patient.
- The evaluation harness computes Recall@5, Recall@10, Recall@20, Recall@50, and MRR.

The repository now treats retrieval as a replaceable component. Proposed methods and baselines only need to emit a common prediction format; the evaluator handles DDXPlus targets and metrics.

## Current Research Task

RQ3 asks whether multi-interest KG retrieval improves clinical disease retrieval compared with single-vector and other baseline methods. The immediate engineering goal is a reliable evaluation harness before adding new retrieval models.

The first benchmark target is DDXPlus test-set diagnosis retrieval:

- Input query: `EVIDENCES` from `release_test_patients.csv`
- Ground truth: `PATHOLOGY`
- Knowledge source: PrimeKG
- Metrics: Recall@5, Recall@10, Recall@20, Recall@50, MRR

## Repository Layout

```text
Benchmark data/DDXPlus/        DDXPlus release files
src/medmigcr_kg/               Active KG graph store, scoring, and beam search modules
src/medmigcr_mind/             Active Clinical MIND model and DDXPlus trainer
scripts/preprocess/            PrimeKG indexing utilities
scripts/mapping/               DDXPlus to PrimeKG mapping builders
scripts/evaluation/            Retrieval metric scripts
scripts/retrieval/             Retrieval runners that emit prediction CSV files
scripts/alignment/             MIND-to-PrimeKG vector-space alignment
artifacts/checkpoints/         Trained checkpoints and projection files
legacy/                        Older prototype code kept for reference
medmigcr/                      Legacy synthetic MedMIG-CR dataset utilities
Papers/                        Local research papers
data/processed/                Generated PrimeKG indexes (ignored)
data/mappings/                 DDXPlus-to-PrimeKG mapping artifacts
results/                       Evaluation outputs (ignored)
```

Legacy generated artifacts from the previous synthetic-query pipeline are no longer part of the primary DDXPlus evaluation path.

## Setup

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the PrimeKG CSV at the repository root as:

```text
kg_giant.csv
```

The DDXPlus files are expected under:

```text
Benchmark data/DDXPlus/
```

## 1. Build a PrimeKG Node Index

Create a typed node index from `kg_giant.csv`. This keeps stable keys such as `disease|MONDO|4979` and `effect/phenotype|HPO|2090`, which are safer than raw IDs.

```bash
python scripts/preprocess/build_primekg_node_index.py ^
  --primekg_csv kg_giant.csv ^
  --output_dir data/processed/primekg
```

Outputs:

- `data/processed/primekg/node_metadata.csv`
- `data/processed/primekg/disease_nodes.csv`
- `data/processed/primekg/phenotype_nodes.csv`
- `data/processed/primekg/index_metadata.json`

This node index is enough for DDXPlus concept mapping, but not enough for beam search. Beam search also needs CSR graph arrays, degrees, mappings, and node embeddings.

## 1b. Build Retrieval-Ready PrimeKG Graph Artifacts

Create graph artifacts for the legacy seed-SVD beam-search baseline:

```bash
python scripts/preprocess/build_primekg_retrieval_graph.py ^
  --primekg_csv kg_giant.csv ^
  --output_dir data/processed/primekg_graph ^
  --embedding_method node2vec
```

For quick smoke tests, use deterministic random embeddings:

```bash
python scripts/preprocess/build_primekg_retrieval_graph.py ^
  --primekg_csv kg_giant.csv ^
  --output_dir data/processed/primekg_graph ^
  --embedding_method random
```

Outputs:

- `data/processed/primekg_graph/graph_csr.npz`
- `data/processed/primekg_graph/node_embeddings.npy`
- `data/processed/primekg_graph/out_degree.npy`
- `data/processed/primekg_graph/in_degree.npy`
- `data/processed/primekg_graph/mappings/*.json`

## 2. Map DDXPlus Concepts to PrimeKG

Build candidate mappings from DDXPlus pathologies and evidences to PrimeKG disease and HPO phenotype nodes.

```bash
python scripts/mapping/build_ddxplus_primekg_mappings.py ^
  --conditions_json "Benchmark data/DDXPlus/release_conditions.json" ^
  --evidences_json "Benchmark data/DDXPlus/release_evidences.json" ^
  --primekg_index_dir data/processed/primekg ^
  --output_dir data/mappings/ddxplus
```

Outputs:

- `data/mappings/ddxplus/condition_to_primekg.json`
- `data/mappings/ddxplus/evidence_to_primekg.json`
- `data/mappings/ddxplus/mapping_summary.json`

Mappings with high fuzzy-match confidence are marked as `auto`; uncertain mappings are marked as `needs_review`.

## 3. Evaluate Retrieval Predictions

The evaluator does not run retrieval. It assumes a retrieval agent has already produced ranked candidates.

Recommended CSV prediction format:

```csv
patient_index,candidate,score
0,disease|MONDO|1234,0.98
0,disease|MONDO|5678,0.91
1,disease|MONDO|4979,0.87
```

`patient_index` is the zero-based row index in `release_test_patients.csv`. `candidate` can be a PrimeKG node key. A `rank` column may be used instead of `score`.

Run evaluation:

```bash
python scripts/evaluation/evaluate_ddxplus_retrieval.py ^
  --patients_csv "Benchmark data/DDXPlus/release_test_patients.csv" ^
  --condition_map data/mappings/ddxplus/condition_to_primekg.json ^
  --predictions results/my_retriever/predictions.csv ^
  --output_dir results/my_retriever/evaluation
```

Outputs:

- `summary.json`
- `by_patient.csv`

The summary includes Recall@5, Recall@10, Recall@20, Recall@50, MRR, and skip counts for missing mappings or missing predictions.

## Prediction Contract for Future Methods

Any baseline or proposed method can be evaluated if it emits ranked candidates per DDXPlus patient:

- Required: `patient_index`
- Required: one candidate column among `candidate`, `node_key`, `disease_node`, `prediction`, or `disease`
- Optional: `score`
- Optional: `rank`

This makes the evaluation harness independent from a specific retriever implementation.

## Legacy Baseline: Seed-SVD Beam Search

Build DDXPlus test queries from the mapped evidences and pathologies:

```bash
python scripts/preprocess/build_ddxplus_test_queries.py ^
  --patients_csv "Benchmark data/DDXPlus/release_test_patients.csv" ^
  --evidence_map data/mappings/ddxplus/evidence_to_primekg.json ^
  --condition_map data/mappings/ddxplus/condition_to_primekg.json ^
  --output_csv data/processed/ddxplus/test_queries.csv
```

Run the old beam-search method as a baseline:

```bash
python scripts/retrieval/run_ddxplus_seed_svd_retrieval.py ^
  --test_queries_csv data/processed/ddxplus/test_queries.csv ^
  --graph_dir data/processed/primekg_graph ^
  --output_csv results/seed_svd_k1/predictions.csv ^
  --interest_count 1
```

For a K=3 heuristic multi-interest run:

```bash
python scripts/retrieval/run_ddxplus_seed_svd_retrieval.py ^
  --test_queries_csv data/processed/ddxplus/test_queries.csv ^
  --graph_dir data/processed/primekg_graph ^
  --output_csv results/seed_svd_k3/predictions.csv ^
  --interest_count 3
```

In this baseline, K is selected at inference time because interest vectors are derived from seed-node embeddings with mean/SVD. For Clinical MIND, K is part of the model architecture and must be chosen before training.

## MIND Baseline: K=1, K=2, K=3

Build train/valid query files from DDXPlus before retraining MIND:

```bash
python scripts/preprocess/build_ddxplus_test_queries.py ^
  --patients_csv "Benchmark data/DDXPlus/release_train_patients.csv" ^
  --evidence_map data/mappings/ddxplus/evidence_to_primekg.json ^
  --condition_map data/mappings/ddxplus/condition_to_primekg.json ^
  --output_csv data/processed/ddxplus/train_queries.csv ^
  --summary_json data/processed/ddxplus/train_query_summary.json

python scripts/preprocess/build_ddxplus_test_queries.py ^
  --patients_csv "Benchmark data/DDXPlus/release_validate_patients.csv" ^
  --evidence_map data/mappings/ddxplus/evidence_to_primekg.json ^
  --condition_map data/mappings/ddxplus/condition_to_primekg.json ^
  --output_csv data/processed/ddxplus/valid_queries.csv ^
  --summary_json data/processed/ddxplus/valid_query_summary.json
```

Train separate MIND checkpoints for each K:

```bash
python src/medmigcr_mind/train_mind_ddxplus.py --K 1 --epochs 5 --out_dir artifacts/checkpoints/ddxplus
python src/medmigcr_mind/train_mind_ddxplus.py --K 2 --epochs 5 --out_dir artifacts/checkpoints/ddxplus
python src/medmigcr_mind/train_mind_ddxplus.py --K 3 --epochs 5 --out_dir artifacts/checkpoints/ddxplus
```

For quick smoke tests, add `--max_train_samples 5000 --max_valid_samples 1000 --epochs 1`.

Fit a ridge-regularized linear projection from each MIND disease embedding space into PrimeKG node embedding space:

```bash
python scripts/alignment/align_mind_to_primekg.py ^
  --checkpoint artifacts/checkpoints/ddxplus/clinical_mind_ddxplus_k3.pt ^
  --graph_dir data/processed/primekg_graph ^
  --output_npz artifacts/checkpoints/ddxplus/alignment_k3_to_primekg.npz
```

Run MIND-driven beam search:

```bash
python scripts/retrieval/run_ddxplus_mind_retrieval.py ^
  --test_queries_csv data/processed/ddxplus/test_queries.csv ^
  --graph_dir data/processed/primekg_graph ^
  --checkpoint artifacts/checkpoints/ddxplus/clinical_mind_ddxplus_k3.pt ^
  --projection artifacts/checkpoints/ddxplus/alignment_k3_to_primekg.npz ^
  --output_csv results/mind_k3/predictions.csv ^
  --interest_count 3
```

Then evaluate:

```bash
python scripts/evaluation/evaluate_ddxplus_retrieval.py ^
  --patients_csv "Benchmark data/DDXPlus/release_test_patients.csv" ^
  --condition_map data/mappings/ddxplus/condition_to_primekg.json ^
  --predictions results/mind_k3/predictions.csv ^
  --output_dir results/mind_k3/evaluation
```

## Notes

- Older prototype code has been moved under `legacy/`.
- Active importable code lives under `src/`; CLI-style experiment scripts live under `scripts/`.
- Generated model artifacts should live under `artifacts/checkpoints/`; generated retrieval outputs should live under `results/`.
