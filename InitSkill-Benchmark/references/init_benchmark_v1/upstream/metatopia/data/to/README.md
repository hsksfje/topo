# Topology optimization task datasets

Populate this directory by running, from the repository root:

```bash
pip install "metatopia[data]"
python data/download_data.py
```

## Published datasets

These are downloaded from Zenodo ([10.5281/zenodo.15172933](https://doi.org/10.5281/zenodo.15172933)):

| Dataset | Tasks | Description |
|---|---|---|
| `train/` | 30,000 | Meta-training tasks, 64×64, point loads and supports. |
| `valid/` | 100 | Validation tasks used for hyperparameter selection. |
| `test_id/` | 1,000 | In-distribution meta-test set. |
| `test_ood/` | 1,000 | Out-of-distribution: boundary conditions distributed along line segments. |
| `test_cr/` | 1,000 | Cross-resolution: point loads and supports on a 256×256 discretization. |

Each dataset has a matching `<name>.yaml` describing its TO properties.

## Task file format

Each task is stored as six `.npy` files sharing a `problem_<id>_` prefix:

| File | Contents |
|---|---|
| `*_forces.npy` | Applied loads, shape `(Ny+1, Nx+1, 2)` |
| `*_normals.npy` | Support conditions, shape `(Ny+1, Nx+1, 2)` |
| `*_strain_energy.npy` | Strain energy field of the uniform-density reference design |
| `*_reference_compliance.npy` | Compliance of the reference design (scalar) |
| `*_target_vf.npy` | Target volume fraction (scalar) |
| `*_id.npy` | Unique task id (scalar) |

This flat layout is designed for the NVIDIA DALI data loader in
`metatopia/dataset/to`.

## Appendix datasets (not distributed)

The paper's appendix additionally reports two datasets that are **not** on
Zenodo:

- `test_ar/` — cross-aspect: a rectangular design domain.
- `test_cr_ood/` — compound-shift: out-of-distribution boundary conditions at
  the cross-resolution discretization.

Their TO property files (`test_ar.yaml`, `test_cr_ood.yaml`) and Hydra dataset
configs are kept in the repository, but the task data itself is not, so
evaluating against them fails with a "No files found" error until they are
regenerated:

```bash
cd experiments/generate_datasets
python generate_dataset.py --config-name gen_ar
python generate_dataset.py --config-name gen_cr_ood
```

This writes `data/to/test_ar/` and `data/to/test_cr_ood/`, after which they can
be passed to `meta_test.py` like any other dataset.
