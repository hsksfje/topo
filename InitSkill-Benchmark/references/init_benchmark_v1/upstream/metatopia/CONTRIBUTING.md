# Contributing to metatopia

Contributions are welcome, whether that is a bug report, a documentation fix, or
a new feature.

## Reporting bugs

Please open an issue on the [GitHub issue tracker](https://github.com/bessagroup/metatopia/issues).
A useful report includes:

- what you ran (the command and, for Hydra experiments, any config overrides),
- what you expected and what happened instead, with the full traceback,
- your Python, JAX, and `metatopia` versions, and whether you are on CPU or GPU.

## Development setup

```bash
git clone https://github.com/bessagroup/metatopia.git
cd metatopia
conda create --name metatopia_dev python=3.12
conda activate metatopia_dev
pip install "jax[cuda12]==0.6.2"     # or jax==0.6.2 for CPU
pip install -e ".[test]"
```

The JAX ecosystem pins in `requirements.txt` are deliberate — JAX, `equinox`,
`lineax`, and `optimistix` introduce breaking changes in lockstep. Please do not
loosen them without running the full test suite.

## Before opening a pull request

Run the tests and the linter:

```bash
pytest tests
ruff check
```

The test suite runs on CPU without NVIDIA DALI and takes about a minute. CI runs
the same checks on Python 3.11 and 3.12.

If you change numerical behaviour, say so explicitly in the pull request. Some
tolerances in the test suite are set to match the configuration used to produce
the published results rather than the tightest achievable value; these are
documented at the relevant call sites.

## Style

Code is linted and formatted with [Ruff](https://docs.astral.sh/ruff/) using the
configuration in `pyproject.toml`. Match the surrounding code; there is no
separate style guide.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold it.
