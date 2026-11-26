# Experiments

This folder contains four experiments, each using [Hydra](https://hydra.cc/)
configurations for parameter management.

| Folder | Description |
|---|---|
| `neural_to` | Neural network-based topology optimization for a single task. |
| `sine_regression` | Proof-of-concept meta-learning on sine function regression. |
| `generate_datasets` | Generation of pseudorandom topology optimization tasks. |
| `meta_neural_to` | Meta-learning combined with neural topology optimization — the method described in the paper. |

`neural_to` and `sine_regression` are also covered in the package tutorials.

`meta_neural_to` is the entry point for reproducing the published results. It
reads checkpoints from `$METATOPIA_RUNS` and writes results to
`$METATOPIA_RESULTS` (default `./results`). See the repository README for
download and reproduction instructions.
