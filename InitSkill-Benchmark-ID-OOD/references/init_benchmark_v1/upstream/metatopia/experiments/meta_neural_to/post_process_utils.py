from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import jax.tree as jt
import numpy as np
import pandas as pd
from metatopia.dataset.to import get_iterator_fn


@dataclass
class MetaTestResults:
    """Container for meta-testing results and designs with analysis methods."""

    results_path: Path
    dataset_name: str
    steps: int
    _cached_problems: Dict[int, Dict] = field(default_factory=dict)
    _cached_designs: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    _cached_metrics: Optional[pd.DataFrame] = None
    _iterator_fn: Optional[Callable] = None

    def __post_init__(self):
        """Setup dataset iterator and validate paths."""
        self.base_path = Path(self.results_path) / self.dataset_name
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")

        # Get dataset root path from config
        current_file = Path(__file__).resolve()
        root_path = current_file.parents[2]
        data_path = root_path / "data" / "to"
        self.dataset_root = data_path / self.dataset_name

    def _get_model_path(self, model_name: str) -> Path:
        return self.base_path / model_name

    @property
    def iterator_fn(self) -> Callable:
        """Lazy loading of iterator function."""
        if self._iterator_fn is None:
            self._iterator_fn = get_iterator_fn(
                path=self.dataset_root,
                seed=0,  # You might want to make this configurable
            )
        return self._iterator_fn

    def _load_model_results(self, model_name: str) -> pd.DataFrame:
        """Load results for a specific model."""
        model_base = self._get_model_path(model_name)
        metrics_path = model_base / f"{self.steps}_steps" / "metrics.csv"

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"No results found for {model_name} at {metrics_path}"
            )
        return pd.read_csv(metrics_path)

    def _load_model_design(self, model_name: str, id_: int) -> np.ndarray:
        """Load a specific design for a model."""
        model_base = self._get_model_path(model_name)
        design_path = model_base / f"{self.steps}_steps" / "designs" / f"{id_}.npy"
        if not design_path.exists():
            raise FileNotFoundError(f"No design found for {model_name}, id {id_}")
        return np.load(design_path)

    def _load_model_losses(self, model_name: str, id_: int) -> Dict:
        """Load optimization trajectory for a specific design."""
        model_base = self._get_model_path(model_name)
        loss_path = model_base / f"{self.steps}_steps" / "losses" / f"{id_}.npy"
        if not loss_path.exists():
            raise FileNotFoundError(f"No trajectory found for {model_name}, id {id_}")
        return np.load(loss_path)

    def get_available_models(self) -> List[str]:
        """Get list of available model results."""
        models = []

        # Check regular model directories
        if self.base_path.exists():
            for model_dir in self.base_path.iterdir():
                if not model_dir.is_dir():
                    continue

                steps_dir = model_dir / f"{self.steps}_steps"
                metrics_file = steps_dir / "metrics.csv"

                if steps_dir.exists() and metrics_file.exists():
                    models.append(model_dir.name)

        return sorted(models)

    # Rest of the methods remain unchanged
    def to_df(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Merge results into a single DataFrame with distinct column names.

        Parameters:
        -----------
        model_names : Optional[List[str]]
            If provided, only include data from these specific models

        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with results from specified models
        """
        # If we're requesting specific models, don't use the cache
        use_cache = model_names is None and self._cached_metrics is not None

        if use_cache:
            return self._cached_metrics

        try:
            available_models = self.get_available_models()
        except FileNotFoundError as e:
            print(f"Error finding models: {e}")
            print(f"Checking path: {self.base_path}")
            if self.base_path.exists():
                print("Contents of base directory:")
                for p in self.base_path.iterdir():
                    print(f"  {p}")
            raise

        if not available_models:
            msg = f"No model results found in {self.base_path}."
            raise FileNotFoundError(msg)

        # Filter models if requested
        if model_names is not None:
            models = [m for m in model_names if m in available_models]
            if not models:
                raise ValueError(
                    f"None of the requested models {model_names} were found. "
                    f"Available models: {available_models}"
                )
        else:
            models = available_models

        # Start with first model's results
        merged_df = self._load_model_results(models[0]).copy()
        merged_df.columns = [
            f"{col}_{models[0]}" if col != "id" else col for col in merged_df.columns
        ]

        # Add other models
        for model in models[1:]:
            model_df = self._load_model_results(model).copy()
            model_df.columns = [
                f"{col}_{model}" if col != "id" else col for col in model_df.columns
            ]
            merged_df = pd.merge(merged_df, model_df, on="id")

        # Only cache when we're retrieving all models
        if model_names is None:
            self._cached_metrics = merged_df

        return merged_df

    def get_solver_names(self) -> List[str]:
        """Get list of available solver names."""
        return self.get_available_models()

    def get_design(self, model_name: str, id_: int) -> np.ndarray:
        """Get a specific design."""
        if model_name not in self._cached_designs:
            self._cached_designs[model_name] = {}

        if id_ not in self._cached_designs[model_name]:
            self._cached_designs[model_name][id_] = self._load_model_design(
                model_name, id_
            )

        return self._cached_designs[model_name][id_]

    def get_losses(self, model_name: str, id_: int) -> Dict:
        """Get optimization trajectory for a specific design."""
        return self._load_model_losses(model_name, id_)

    def get_problem(self, id_: int) -> Optional[Dict]:
        """Get problem definition for a specific ID."""
        if id_ not in self._cached_problems:
            # Create iterator with batch size 1
            iterator = self.iterator_fn(batch_size=1)

            for sample in iterator:
                if sample["id"][0] == id_:
                    # Extract the first item from each batch dimension
                    problem = jt.map(lambda x: x[0], sample)
                    self._cached_problems[id_] = problem
                    return problem

            return None

        return self._cached_problems[id_]

    def get_problems(self, ids: List[int]) -> Dict[int, Dict]:
        """Get problem definitions for a list of IDs."""
        problems = {}
        for id_ in ids:
            problem = self.get_problem(id_)
            if problem is not None:
                problems[id_] = problem
            else:
                print(f"Warning: Problem with ID {id_} not found")
        return problems

    def summarize_statistics(self) -> pd.DataFrame:
        """Compute summary statistics for all models."""
        models = self.get_available_models()
        stats = []

        for model in models:
            df = self._load_model_results(model)
            stats.append(
                {
                    "model": model,
                    "mean_loss": df["loss"].mean(),
                    "std_loss": df["loss"].std(),
                    "mean_steps": df["steps"].mean(),
                    "mean_compliance": df["compliance"].mean(),
                    "mean_volume": df["volume"].mean(),
                }
            )

        return pd.DataFrame(stats)

    @classmethod
    def from_path(
        cls,
        results_path: Path,
        dataset_name: str,
        steps: int,
    ) -> "MetaTestResults":
        """Create instance from path configuration."""
        return cls(
            results_path=results_path,
            dataset_name=dataset_name,
            steps=steps,
        )


def extract_multi_trial_df(
    trials: List[MetaTestResults],
    model_selections: List[List[str]],
    trial_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Merge specific models from multiple trials into a single DataFrame.

    Parameters:
    -----------
    trials : List[MetaTestResults]
        List of MetaTestResults objects from different trials
    model_selections : List[List[str]]
        List of model names to include from each trial
    trial_names : Optional[List[str]]
        Names to identify each trial (defaults to trial0, trial1, etc.)

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with results from selected models across trials
    """
    if trial_names is None:
        trial_names = [f"trial{i}" for i in range(len(trials))]

    if len(trials) != len(model_selections) or len(trials) != len(trial_names):
        raise ValueError(
            "Number of trials, model selections, and trial names must match"
        )

    combined_df = None

    for trial, models, name in zip(trials, model_selections, trial_names):
        # Get DataFrame for just the selected models from this trial
        trial_df = trial.to_df(model_names=models)

        # Rename columns to include trial name (except for 'id')
        trial_df.columns = [
            f"{col}_{name}" if col != "id" else col for col in trial_df.columns
        ]

        # Merge with existing results
        if combined_df is None:
            combined_df = trial_df
        else:
            combined_df = pd.merge(combined_df, trial_df, on="id", how="outer")

    return combined_df
