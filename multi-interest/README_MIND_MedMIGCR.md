# Clinical MIND (MedMIG-CR): multi-interest encoding

This folder adapts the **Multi-Interest Network with Dynamic Routing (MIND)** idea from the Tmall codebase to **MedMIG-CR**:

- **Input**: a clinical query as a **variable-length sequence of symptom entity strings** (same format as `processed_medmigcr_dataset/query_nodes.csv`, e.g. `effect/phenotype|HPO|4322`).
- **Output**: **K latent interest vectors** `Z ∈ ℝ^{B × K × D}` produced by **capsule dynamic routing** over the symptom sequence (no label-aware attention).

The encoder is intended as a **retrieval anchor** for later PrimeKG / LogosKG multi-hop reasoning.

## Files

| File | Role |
|------|------|
| `mind_medmigcr_model.py` | **Single module**: symptom embeddings, disease embeddings (for training), dynamic routing, squash, optional BCE training helper. |
| `train_mind_medmigcr.py` | **Single script**: builds vocabs from MedMIG-CR CSVs, trains with max-pool interest–disease scoring (no attention). |
| `test_multi_model.py` | Loads `clinical_mind.pt`, runs forward passes, prints `Z` for CLI symptoms or selected `query_id` rows. |

## Prerequisites

- Python 3.10+ recommended
- `torch`, `pandas`, `numpy`
- Trained data under `processed_medmigcr_dataset/`:
  - `query_nodes.csv`
  - `interactions_train.csv`
  - (optional) `interactions_valid.csv` for validation loss

Install PyTorch from [https://pytorch.org](https://pytorch.org) for your CUDA/CPU setup, then:

```bash
pip install pandas numpy
```

## Train

From the **repository root** (`MedMIG-CR/`):

```bash
python multi-interest/train_mind_medmigcr.py ^
  --data_dir processed_medmigcr_dataset ^
  --out_dir multi-interest/checkpoints ^
  --epochs 5 ^
  --batch_size 256 ^
  --D 64 ^
  --K 3 ^
  --R 3 ^
  --max_seq_len 16 ^
  --n_neg 8 ^
  --lr 1e-3
```

### Important hyperparameters

- **`--K`**: number of interest capsules (latent “hypotheses”).
- **`--D`**: embedding dimension (same for symptoms, diseases, capsules).
- **`--R`**: number of dynamic routing iterations (agreement updates).
- **`--max_seq_len`**: padded length; must match training when testing.
- **`--n_neg`**: random negative diseases per positive pair (BCE training).

Outputs:

- `multi-interest/checkpoints/clinical_mind.pt` — `model_state`, `symptom_str2id`, `disease_str2id`, `hparams`
- `multi-interest/checkpoints/clinical_mind_meta.json` — hyperparameters only

### Quick debug (subset of training rows)

```bash
python multi-interest/train_mind_medmigcr.py --max_train_samples 5000 --epochs 2 --batch_size 128
```

## Test / inspect multi-interest vectors

After training, from repo root:

### 1) Arbitrary symptom string (semicolon-separated)

```bash
python multi-interest/test_multi_model.py ^
  --checkpoint multi-interest/checkpoints/clinical_mind.pt ^
  --symptoms "effect/phenotype|HPO|4322;effect/phenotype|HPO|962"
```

### 2) Rows from `query_nodes.csv`

```bash
python multi-interest/test_multi_model.py ^
  --checkpoint multi-interest/checkpoints/clinical_mind.pt ^
  --query_csv processed_medmigcr_dataset/query_nodes.csv ^
  --query_ids Q0000000 Q0000001
```

The script prints:

- `Z` shape `(B, K, D)`
- per-interest L2 norms and a short numeric slice for sanity checks

## Training objective (no attention)

The original MIND uses **label-aware attention** between interests and items. This port **omits** that by design.

Instead, `training_bce_loss` in `mind_medmigcr_model.py`:

1. Computes `Z = model(symptom_ids)` → `(B, K, D)`.
2. Embeds positive / negative diseases → `(B, D)` and `(B, N_neg, D)`.
3. Scores each disease with **max over K** of dot products `⟨Z_{b,k}, d⟩` (scaled by temperature), then applies **binary cross-entropy** so positives are pushed above negatives.

This keeps the **multi-interest encoder** as the main learnable structure and remains compatible with downstream **retrieval / KG** modules that will consume `Z`.

## Relation to original MIND

| Original MIND | Clinical MIND (this folder) |
|----------------|----------------------------|
| User click sequence of **item IDs** | Clinical query sequence of **symptom entity IDs** |
| `itemEmbeds` for history | `symptom_emb` |
| `itemEmbeds` for target item | `disease_emb` (supervision only) |
| B2I routing over sequence | Same routing math, modular `DynamicRoutingB2I` |
| Label-aware attention + sampled softmax | **Removed**; max-pool over K + BCE |

## Notes for later KG integration

- Use `model(symptom_ids)` or `model.encode_symptoms(symptom_ids)[0]` as **query-side embedding** `(B, K, D)`.
- You can pool `Z` (e.g. mean over K, or keep all K for multi-vector retrieval) before nearest-neighbor over graph nodes or path encoders.
