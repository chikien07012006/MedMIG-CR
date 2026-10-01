# Truncated Katz

Non-parametric baseline that ranks candidates by weighted walk counts from the seed nodes:

```text
score = sum_{l=1..L} beta^l * M^l s
```

`s` is the uniform seed vector. `M` is the undirected PrimeKG adjacency, either raw (`none`), symmetrically normalized `D^-1/2 A D^-1/2` (`sym`, the default), or random-walk normalized `A D^-1` (`rw`). It shares the batching, multiprocessing, candidate protocol, and output format of the PPR baseline (see [graph_propagation.py](../../graph_propagation.py)).

Tune `--katz_beta`, `--katz_max_length`, and `--katz_normalization` on `valid_5k.csv` only. See [guide.md](../../../guide.md).
