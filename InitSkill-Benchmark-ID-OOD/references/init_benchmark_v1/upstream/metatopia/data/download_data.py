"""Download the meta-neural TO datasets and checkpoints from Zenodo.

Usage:
    pip install "metatopia[data]"
    python data/download_data.py

Datasets are extracted to ``data/to/`` and model checkpoints to
``checkpoints/`` at the repository root.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from zenodo_get import download


# Concept DOI: always resolves to the LATEST version of the record. Do not
# replace this with a version-specific DOI (e.g. 10.5281/zenodo.15172934), or
# the download will silently pin to an outdated release.
CONCEPT_DOI = "10.5281/zenodo.15172933"

DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
TARGET_DIR = DATA_DIR / "to"
STAGING_DIR = DATA_DIR / "to_zenodo"

# Archives that hold model checkpoints rather than task data. These extract to
# the repository root so that METATOPIA_RUNS can point straight at them.
CHECKPOINT_ARCHIVES = {"checkpoints.7z"}


def extract(archive: Path, dest: Path) -> None:
    """Extract a .7z archive into ``dest``.

    Prefers the ``7z`` binary when available because it is considerably faster,
    and otherwise falls back to py7zr so that no system package is required.
    """
    dest.mkdir(parents=True, exist_ok=True)

    if shutil.which("7z") is not None:
        subprocess.run(
            ["7z", "x", str(archive), f"-o{dest}", "-aoa", "-bsp1"],
            check=True,
        )
        return

    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            "Extracting the archives needs either the '7z' command line tool "
            "or the py7zr package. Install py7zr with:\n\n"
            "    pip install py7zr\n"
        ) from exc

    with py7zr.SevenZipFile(archive, "r") as handle:
        handle.extractall(path=dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the downloaded .7z archives after extraction.",
    )
    args = parser.parse_args()

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading record {CONCEPT_DOI} into {STAGING_DIR} ...")
    download(CONCEPT_DOI, output_dir=str(STAGING_DIR))

    archives = sorted(STAGING_DIR.glob("*.7z"))
    if not archives:
        print("Error: no .7z archives were downloaded.", file=sys.stderr)
        return 1

    # Each archive carries its own top-level directory (train/, test_id/, ...),
    # so dataset archives extract directly into data/to/ while the checkpoint
    # archive extracts to the repository root.
    for archive in archives:
        dest = REPO_ROOT if archive.name in CHECKPOINT_ARCHIVES else TARGET_DIR
        print(f"Extracting {archive.name} -> {dest}")
        try:
            extract(archive, dest)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"Error: failed to extract {archive.name}: {exc}", file=sys.stderr)
            return 1

    # The per-dataset .yaml descriptors sit alongside the archives.
    for yaml_file in STAGING_DIR.glob("*.yaml"):
        shutil.copy2(yaml_file, TARGET_DIR / yaml_file.name)

    if not args.keep_archives:
        for archive in archives:
            archive.unlink()

    print(f"\nDone. Datasets are in {TARGET_DIR}")
    for entry in sorted(TARGET_DIR.iterdir()):
        if entry.is_dir():
            n = sum(1 for _ in entry.iterdir())
            print(f"  {entry.name}/  ({n} files, {n // 6} tasks)")

    ckpt_dir = REPO_ROOT / "checkpoints"
    if ckpt_dir.is_dir():
        print(f"\nCheckpoints are in {ckpt_dir}")
        for entry in sorted(ckpt_dir.iterdir()):
            if entry.is_dir():
                print(f"  {entry.name}/")
        print(f"\nNext: export METATOPIA_RUNS={ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
