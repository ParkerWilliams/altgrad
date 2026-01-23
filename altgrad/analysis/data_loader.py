"""W&B API data loading for experiment analysis.

Provides ExperimentDataLoader class for fetching experiment runs from W&B,
filtering by format type, optimizer configuration, and extracting run histories
for comparison analysis.

Key features:
  - Format comparison runs (filter by fp8_format config)
  - Manifold comparison runs (filter by use_manifold_aware config)
  - Per-run metric history extraction
  - Graceful handling of missing metrics

Example:
    >>> from altgrad.analysis import ExperimentDataLoader
    >>> loader = ExperimentDataLoader(project="altgrad")
    >>> df = loader.get_format_comparison_runs(formats=["E5M2", "E3M4", "E7M0"])
    >>> print(df[["format", "final_loss", "completed"]])
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import pandas as pd

# Lazy import for wandb to allow testing without credentials
_wandb_api = None


def _get_wandb_api():
    """Lazy import of wandb.Api().

    Returns:
        wandb.Api instance
    """
    global _wandb_api
    if _wandb_api is None:
        import wandb
        _wandb_api = wandb.Api()
    return _wandb_api


class ExperimentDataLoader:
    """Load and filter experiment data from W&B.

    Provides methods for fetching experiment runs from Weights & Biases,
    filtering by format type and optimizer configuration, and extracting
    per-step metric histories.

    Attributes:
        project: W&B project name (default "altgrad")
        entity: W&B entity (username or team, None for default)

    Example:
        >>> loader = ExperimentDataLoader(project="altgrad")
        >>> runs = loader.get_format_comparison_runs()
        >>> manifold_runs = loader.get_manifold_comparison_runs()
    """

    def __init__(
        self,
        project: str = "altgrad",
        entity: Optional[str] = None,
    ):
        """Initialize data loader.

        Args:
            project: W&B project name
            entity: W&B entity (username or team). None uses default.
        """
        self.project = project
        self.entity = entity

    @property
    def _project_path(self) -> str:
        """Get W&B project path.

        Returns:
            Project path as "entity/project" or just "project"
        """
        if self.entity:
            return f"{self.entity}/{self.project}"
        return self.project

    def _extract_run_data(self, run) -> dict:
        """Extract relevant data from a W&B run.

        Args:
            run: W&B run object

        Returns:
            Dictionary with run metadata and metrics
        """
        config = run.config
        summary = run.summary

        return {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,  # "finished", "crashed", "running"
            # Config fields
            "format": config.get("fp8_format", "BF16"),
            "use_fp8": config.get("use_fp8", False),
            "use_manifold_aware": config.get("use_manifold_aware", False),
            "manifold_mantissa_bits": config.get("manifold_mantissa_bits", 2),
            "learning_rate": config.get("learning_rate", float("nan")),
            "batch_size": config.get("batch_size", 64),
            "max_steps": config.get("max_steps", 5000),
            "seed": config.get("seed", 42),
            # Summary metrics (final values)
            "final_loss": summary.get("loss", float("nan")),
            "best_loss": summary.get("best_loss", summary.get("loss", float("nan"))),
            "final_perplexity": summary.get("perplexity", float("nan")),
            "bit_stall_rate": summary.get("quantization/bit_stall_rate",
                                          summary.get("bit_stall_rate", float("nan"))),
            "overflow_rate": summary.get("quantization/overflow_rate",
                                         summary.get("overflow_rate", float("nan"))),
            "steps_completed": summary.get("_step", config.get("max_steps", 0)),
            "completed": run.state == "finished",
            # Tags for filtering
            "tags": run.tags,
            "created_at": run.created_at,
        }

    def get_format_comparison_runs(
        self,
        formats: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch runs for format comparison analysis.

        Retrieves all runs from the project, optionally filtering by FP8 format.
        Includes both completed and crashed runs for failure analysis.

        Args:
            formats: List of FP8 format names to include (e.g., ["E5M2", "E3M4"]).
                    If None, includes all formats.

        Returns:
            DataFrame with columns:
                - run_id, run_name, state
                - format, use_fp8, learning_rate, batch_size
                - final_loss, best_loss, final_perplexity
                - bit_stall_rate, overflow_rate
                - steps_completed, completed

        Example:
            >>> loader = ExperimentDataLoader()
            >>> df = loader.get_format_comparison_runs(["E5M2", "E3M4", "E7M0"])
            >>> print(df.groupby("format")["best_loss"].mean())
        """
        api = _get_wandb_api()

        # Build filters for W&B API
        filters = {}
        if formats:
            filters["config.fp8_format"] = {"$in": formats}

        # Fetch runs
        runs = api.runs(self._project_path, filters=filters)

        # Extract data from each run
        data = []
        for run in runs:
            try:
                run_data = self._extract_run_data(run)
                data.append(run_data)
            except Exception as e:
                # Skip runs with missing/corrupt data
                print(f"Warning: Skipping run {run.id}: {e}")
                continue

        if not data:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "run_id", "run_name", "state", "format", "use_fp8",
                "use_manifold_aware", "learning_rate", "batch_size", "max_steps",
                "final_loss", "best_loss", "final_perplexity",
                "bit_stall_rate", "overflow_rate", "steps_completed", "completed",
            ])

        return pd.DataFrame(data)

    def get_manifold_comparison_runs(self) -> pd.DataFrame:
        """Fetch E5M2 runs comparing standard vs manifold-aware optimizer.

        Retrieves E5M2 runs and filters by use_manifold_aware config to enable
        comparison between AdamW and ManifoldAdamW optimizers.

        Returns:
            DataFrame with manifold comparison data, including:
                - All columns from get_format_comparison_runs
                - Filtered to format="E5M2" only
                - Groups available: use_manifold_aware True/False

        Example:
            >>> loader = ExperimentDataLoader()
            >>> df = loader.get_manifold_comparison_runs()
            >>> standard = df[~df["use_manifold_aware"]]
            >>> manifold = df[df["use_manifold_aware"]]
        """
        api = _get_wandb_api()

        # Filter for E5M2 format runs
        filters = {
            "config.fp8_format": "E5M2",
            "config.use_fp8": True,
        }

        runs = api.runs(self._project_path, filters=filters)

        data = []
        for run in runs:
            try:
                run_data = self._extract_run_data(run)
                data.append(run_data)
            except Exception as e:
                print(f"Warning: Skipping run {run.id}: {e}")
                continue

        if not data:
            return pd.DataFrame(columns=[
                "run_id", "run_name", "state", "format", "use_fp8",
                "use_manifold_aware", "learning_rate", "batch_size", "max_steps",
                "final_loss", "best_loss", "final_perplexity",
                "bit_stall_rate", "overflow_rate", "steps_completed", "completed",
            ])

        return pd.DataFrame(data)

    def get_run_history(
        self,
        run_id: str,
        keys: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch per-step metric history for a specific run.

        Retrieves the full training history for detailed analysis,
        useful for loss curves, stall rate trends, etc.

        Args:
            run_id: W&B run ID
            keys: Specific metric keys to fetch. If None, fetches all.
                  Common keys: ["loss", "grad_norm", "quantization/bit_stall_rate"]

        Returns:
            DataFrame with step as index and metrics as columns.
            Missing values are represented as NaN.

        Example:
            >>> loader = ExperimentDataLoader()
            >>> history = loader.get_run_history("abc123", keys=["loss", "grad_norm"])
            >>> history.plot(y="loss")
        """
        api = _get_wandb_api()

        # Get run object
        run = api.run(f"{self._project_path}/{run_id}")

        # Fetch history
        if keys:
            # Filter to specific keys
            history_df = run.history(keys=keys, pandas=True)
        else:
            # Get all metrics
            history_df = run.history(pandas=True)

        # Set step as index if present
        if "_step" in history_df.columns:
            history_df = history_df.set_index("_step")

        return history_df

    def get_baseline_runs(self) -> pd.DataFrame:
        """Fetch BF16 baseline runs (no FP8 quantization).

        Returns:
            DataFrame with baseline run data (use_fp8=False)
        """
        api = _get_wandb_api()

        filters = {"config.use_fp8": False}
        runs = api.runs(self._project_path, filters=filters)

        data = []
        for run in runs:
            try:
                run_data = self._extract_run_data(run)
                data.append(run_data)
            except Exception as e:
                print(f"Warning: Skipping run {run.id}: {e}")
                continue

        if not data:
            return pd.DataFrame(columns=[
                "run_id", "run_name", "state", "format", "use_fp8",
                "final_loss", "best_loss", "completed",
            ])

        return pd.DataFrame(data)


__all__ = [
    "ExperimentDataLoader",
]
