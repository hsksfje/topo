
<p align="center">
  <img src="docs/source/_static/metatopia_logo.svg" alt="metatopia">
</p>

# Summary
Metatopia is a [JAX](https://jax.readthedocs.io/en/latest/index.html#) library for meta-learning research in neural topology optimization. The key features include
- Neural network models for representing structural designs (`Siren`, `FourierMLP`, etc.).
- Differentiable compliance solver and filters for topology optimization.
- Optimization-based meta-learners including `MAML` and `Reptile`.
- Filtering system which enables learning a subset of the parameters during adaptation.
- Tools for creating topology optimization tasks with pseudorandom boundary conditions.
- Integration with [Hydra](https://hydra.cc/) to easily configure and run experiments.

This is the reference implementation for the paper *Meta-neural Topology Optimization: Knowledge Infusion with Meta-learning*. See [Citation](#citation) and [Reproducing the paper](#reproducing-the-paper).

# Installation
Clone the repository from GitHub:
```
git clone https://github.com/bessagroup/metatopia.git
cd metatopia
```

Set up a conda environment and install JAX. Detailed instructions for JAX installation can be found at [https://jax.readthedocs.io/en/latest/installation.html](https://jax.readthedocs.io/en/latest/installation.html). For best performance, we recommend the GPU build:
```
conda create --name metatopia_env python=3.12
conda activate metatopia_env
pip install "jax[cuda12]==0.6.2"
```

> **The JAX version is pinned deliberately.** JAX and its ecosystem (`equinox`, `lineax`, `optimistix`) introduce breaking changes frequently, and `requirements.txt` pins those to versions matching JAX 0.6.2. Installing a different JAX release is likely to break the library. CPU users can install `jax==0.6.2` instead.

Install the package with all requirements:
```
pip install -e .
```

Install NVIDIA DALI (optional), a GPU-accelerated library for data loading (assumes CUDA 12.x):
```
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
```
DALI is only needed to iterate over the topology optimization task datasets. The library imports and the models, solver, and filters all work without it.

To reproduce the exact environment used for the results in the paper, install `requirements-paper.txt` instead:
```
pip install -r requirements-paper.txt
pip install -e .
```

# Data and checkpoints
The task datasets and meta-learned model checkpoints are archived on Zenodo under the concept DOI [10.5281/zenodo.15172933](https://doi.org/10.5281/zenodo.15172933), which always resolves to the latest version.

> **Note:** the paper cites the version-specific record [`zenodo.org/records/15172934`](https://zenodo.org/records/15172934), which is the *first* version and does not contain `test_id` or the model checkpoints. Use the concept DOI above, or the latest version linked from that page.

Download and unpack everything with:
```
pip install "metatopia[data]"
python data/download_data.py
```
This downloads 1.3 GB and places the task datasets in `data/to/` and the model checkpoints in `checkpoints/`. See [`data/to/README.md`](data/to/README.md) for the dataset inventory and task file format.

Extraction falls back to the pure-Python `py7zr`, which needs no system packages but is slow (roughly 40 minutes for the full record, most of it unpacking the 180,000 files of `train/`). Installing the `7z` command line tool first (`apt install p7zip-full`, `brew install p7zip`) makes this dramatically faster, and the script will use it automatically when present.

# Quickstart
With the datasets and checkpoints downloaded, evaluate the meta-learned initialization against a randomly initialized baseline on a handful of tasks:

```bash
export METATOPIA_RUNS=$PWD/checkpoints

cd experiments/meta_neural_to
python meta_test.py dataset=test_id \
  meta_test.steps=15 meta_test.min_steps=5 \
  meta_test.batch_size=4 meta_test.sample_size=4 std_model.n_inits=1
```

This runs in well under a minute on a GPU. The meta-learned model should converge in fewer adaptation steps, at a lower loss, than the random initialization:

```
Average loss: 0.086     Average steps: 13.5     <- meta-learned
Average loss: 0.108     Average steps: 15.0     <- standard neural TO
```

Results are written to `$METATOPIA_RESULTS` (default `./results`), never back into the downloaded checkpoints.

# Reproducing the paper
Drop the smoke-test overrides to run the full evaluation protocol from the paper — minimum 10 and maximum 200 adaptation iterations, subject to a relative-change stopping criterion of `1e-5`, over all 1000 tasks in each dataset:

```bash
export METATOPIA_RUNS=$PWD/checkpoints
cd experiments/meta_neural_to
python meta_test.py dataset=test_id     # in-distribution
python meta_test.py dataset=test_ood    # out-of-distribution
python meta_test.py dataset=test_cr     # cross-resolution
```

The appendix additionally reports a cross-aspect (`test_ar`) and a compound-shift (`test_cr_ood`) dataset. These are not distributed on Zenodo and must be regenerated first — see [`data/to/README.md`](data/to/README.md).

Meta-training from scratch (`meta_train.py`) uses 30,000 tasks over 6000 outer steps and requires substantial GPU time; the published checkpoints let you skip it.

# Documentation
The documentation can be built from the repository root. The package itself must be installed, because the API reference is generated from the source:

```
pip install -e ".[cpu,docs]"
cd docs
make html
```

# Getting started
To get started with `metatopia`, check out the tutorials section of the documentation or the `experiments` folder, containing examples to get you familiar with the framework.

# Statement of need
Engineers learn from every design they create, building intuition that helps them quickly identify promising solutions for new problems. Topology optimization (TO) - a well-established computational method for designing structures with optimized performance - lacks this ability to learn from experience. Existing approaches treat design tasks in isolation, starting from a "blank canvas" design for each new problem and often requiring many computationally expensive iterations to converge, even when solving similar problems like optimizing structures with varying load conditions or material constraints. Metatopia bridges this gap by combining topology optimization with meta-learning techniques. Built on JAX's efficient automatic differentiation and acceleration capabilities, it enables optimization algorithms to learn general-purpose design strategies from collections of related problems, rapidly adapt to new design scenarios, and transfer knowledge between similar design tasks, thereby reducing computational overhead and accelerating the design process.

# Citation
If you use this software, please cite the paper:

```bibtex
@article{kuszczak2026metaneural,
  title   = {Meta-neural Topology Optimization: Knowledge Infusion with Meta-learning},
  author  = {Kuszczak, Igor and Bessa, Miguel and Ku{\'s}, Gawe{\l} and Bosi, Federico},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  doi     = {10.1016/j.engappai.2026.115763}
}
```

If you additionally need to reference a specific version of the datasets or model checkpoints, cite the Zenodo record [10.5281/zenodo.15172933](https://doi.org/10.5281/zenodo.15172933).

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

# Authorship
This package was created by [Igor Kuszczak](mailto:igor.kuszczak.18@ucl.ac.uk) and [Gaweł Kuś](mailto:gawel_kus@brown.edu).

# Community support

If you find any **issues, bugs or problems** in the code, please use the [GitHub issue tracker](https://github.com/bessagroup/metatopia/issues) to report them.

# License
This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/metatopia/blob/main/LICENSE) for the full license text.
