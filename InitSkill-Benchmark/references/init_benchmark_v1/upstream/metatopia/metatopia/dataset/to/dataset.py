from pathlib import Path


_DALI_MISSING_MSG = (
    "NVIDIA DALI is required for the topology optimization data pipeline. "
    "Install it with:\n\n"
    "    pip install --extra-index-url https://pypi.nvidia.com "
    "--upgrade nvidia-dali-cuda120\n\n"
    "(assumes CUDA 12.x; see https://docs.nvidia.com/deeplearning/dali/ "
    "for other builds)"
)


def get_iterator_fn(path: Path, seed: int):
    # DALI is an optional dependency: it is only needed to iterate over the
    # generated TO task datasets, so it is imported lazily to keep
    # ``import metatopia`` working on installs without it.
    try:
        from nvidia.dali import fn
        from nvidia.dali.plugin.jax import data_iterator
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_DALI_MISSING_MSG) from exc

    # Convert input path to absolute path relative to cwd
    common_kwd = {
        "file_root": path,
        "seed": seed,
        "random_shuffle": True,
    }

    @data_iterator(
        output_map=[
            "forces",
            "normals",
            "target_vf",
            "reference_compliance",
            "strain_energy",
            "id",
        ],
        reader_name="forces",
        auto_reset=False,
    )
    def iterator_fn():
        forces = fn.readers.numpy(
            file_filter="*forces.npy", name="forces", **common_kwd
        )

        normals = fn.readers.numpy(
            file_filter="*normals.npy", name="normals", **common_kwd
        )
        target_vf = fn.readers.numpy(
            file_filter="*target_vf.npy", name="target_vf", **common_kwd
        )

        reference_compliance = fn.readers.numpy(
            file_filter="*reference_compliance.npy",
            name="reference_compliance",
            **common_kwd,
        )

        strain_energy = fn.readers.numpy(
            file_filter="*strain_energy.npy", name="strain_energy", **common_kwd
        )

        id = fn.readers.numpy(file_filter="*id.npy", name="id", **common_kwd)

        return (
            forces.gpu(),
            normals.gpu(),
            target_vf.gpu(),
            reference_compliance.gpu(),
            strain_energy.gpu(),
            id.gpu(),
        )

    return iterator_fn
