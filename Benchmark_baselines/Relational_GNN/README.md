# R-GCN

Relational graph convolutional network ([Schlichtkrull et al., 2018](Relational%20Graph%20Convolutional%20Network.pdf)) trained as a disease ranker on the full PrimeKG graph.

- **Graph:** all 2.73M directed edges plus their inverses (36 relation types, basis decomposition with 9 bases). The node features are node2vec vectors and are fine-tuned during training.
- **Encoder:** two `RGCNConv` layers, 64 -> 128 -> 64, ReLU, dropout 0.1.
- **Query and scorer:** the query is the mean R-GCN embedding of its seed nodes. A candidate is scored by an MLP over `[q, d, q*d]` (192 -> 128 -> 1).
- **Training:** one Adam step per batch of 512 queries, re-encoding the full graph each step.
  - Closed space (47 targets): multi-positive softmax over all candidates.
  - Open space: sampled pairwise softplus loss.
  - Early stopping on validation MRR.
- **Test:** the best checkpoint ranks the candidate set, and predictions are scored by the shared evaluator.
- **Runtime on an Apple M1 Pro (6 threads):** about 3.7 s per step and a peak of about 3 GB RAM, so about 6 minutes per epoch on 50k training queries.

**Fix compared with the previous version:** the old script accumulated gradients over the whole epoch and called `optimizer.step()` once per epoch. Ten epochs were therefore only ten parameter updates. It also supported only the open candidate space. The results in `outputs/full_train_test5000` and `outputs/profile_1k` come from that version and should not be reported.

See [guide.md](../../guide.md).
