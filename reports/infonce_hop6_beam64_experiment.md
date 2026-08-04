# InfoNCE MIND Hop-6 Retrieval Experiment Report

Run status: All requested K runs were found.

## Goal

This experiment tests trainable InfoNCE alignment for K=1, K=2, and K=3 with `max_hops=6` PrimeKG beam search over DDXPlus queries.

## Method

ClinicalMIND encodes mapped DDXPlus evidence nodes into K latent interest vectors. A trainable linear projection maps those vectors into the frozen PrimeKG node2vec space. Training combines target-node InfoNCE, evidence-node InfoNCE, and squared-cosine diversity regularization.

```text
L = L_target_InfoNCE + evidence_weight * L_evidence_InfoNCE + diversity_weight * L_diversity
```

## Training Summary

| K | Status | Train examples | Valid examples | Best valid loss | Latent cosine | Cosine export |
|---|---|---|---|---|---|---|
| 1 | OK | 1023037 | 132190 | 0.1159 | 0.0000 | - |
| 2 | OK | 1023037 | 132190 | 0.0945 | 0.2556 | artifacts/checkpoints/ddxplus_infonce_beam64_hop6/k2/latent_cosine_k2.csv |
| 3 | OK | 1023037 | 132190 | 0.0860 | 0.2471 | artifacts/checkpoints/ddxplus_infonce_beam64_hop6/k3/latent_cosine_k3.csv |

## Retrieval Summary

| K | Status | Queries | Prediction rows | No target candidates | Max hops | Beam width | Paths/interest | Alpha | Beta | Latent cosine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | OK | 5000 | 45718 | 0 | 6 | 64 | 4096 | 1.0000 | 0.1000 | 0.0000 |
| 2 | OK | 5000 | 62812 | 0 | 6 | 64 | 4096 | 1.0000 | 0.1000 | 0.2554 |
| 3 | OK | 5000 | 74618 | 0 | 6 | 64 | 4096 | 1.0000 | 0.1000 | 0.2685 |

## Evaluation

| K | Status | Evaluated | MRR | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Missing prediction |
|---|---|---|---|---|---|---|---|---|
| 1 | OK | 5000 | 0.3214 | 0.3602 | 0.3606 | 0.3606 | 0.3606 | 0 |
| 2 | OK | 5000 | 0.2876 | 0.3222 | 0.3500 | 0.3646 | 0.3646 | 0 |
| 3 | OK | 5000 | 0.2931 | 0.3274 | 0.3800 | 0.4182 | 0.4202 | 0 |

## Comparison With Previous Runs

| Method | Type | MRR | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---|---|---|---|---|---|
| Old K=1 | baseline | 0.0108 | 0.0112 | 0.0112 | 0.0112 | 0.0112 |
| Old K=2 | baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Old K=3 | baseline | 0.0086 | 0.0112 | 0.0112 | 0.0112 | 0.0112 |
| V2 K=1 | baseline | 0.0490 | 0.1008 | 0.1054 | 0.1054 | 0.1054 |
| V2 K=2 | baseline | 0.0905 | 0.1274 | 0.1398 | 0.1398 | 0.1398 |
| V2 K=3 | baseline | 0.0565 | 0.1036 | 0.1082 | 0.1082 | 0.1082 |
| InfoNCE hop6 K=1 | proposed | 0.3214 | 0.3602 | 0.3606 | 0.3606 | 0.3606 |
| InfoNCE hop6 K=2 | proposed | 0.2876 | 0.3222 | 0.3500 | 0.3646 | 0.3646 |
| InfoNCE hop6 K=3 | proposed | 0.2931 | 0.3274 | 0.3800 | 0.4182 | 0.4202 |

## Candidate Generation Audit

| K | Queries with prediction | Mean candidates | Median candidates | Max candidates | Top-K cap |
|---|---|---|---|---|---|
| 1 | 5000 / 5000 | 9.1436 | 9.0000 | 27 | 50 |
| 2 | 5000 / 5000 | 12.5624 | 12.0000 | 26 | 50 |
| 3 | 5000 / 5000 | 14.9236 | 15.0000 | 32 | 50 |

## Interpretation

- Low Recall@5 with higher Recall@20/50 means target diseases are reached but ranked too low.
- This run uses `beam_width=64` and `paths_per_interest=4096`; report these with the metrics.
- K=2/K=3 cosine exports diagnose whether multi-interest vectors still collapse.

## Artifacts

- Checkpoints root: `artifacts/checkpoints/ddxplus_infonce_beam64_hop6`
- Results root: `results/infonce_hop6_beam64_eval5000`
- Report: `reports/infonce_hop6_beam64_experiment.md`
