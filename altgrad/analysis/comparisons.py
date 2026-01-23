"""Format and optimizer comparison logic.

Provides FormatComparator class for analyzing experiment results,
ranking formats by metrics, identifying optimal configurations,
and computing improvement percentages.

Key features:
  - Rank formats by any metric (loss, stall rate, etc.)
  - Identify format "sweet spot" balancing precision vs range
  - Compute relative improvement between formats
  - Compare performance by layer type

Example:
    >>> from altgrad.analysis import FormatComparator
    >>> comparator = FormatComparator()
    >>> rankings = comparator.rank_by_metric(runs_df, "best_loss")
    >>> print(rankings[["format", "best_loss", "rank"]])
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


class FormatComparator:
    """Compare FP8 format performance across experiments.

    Provides methods for ranking formats, identifying optimal configurations,
    and computing relative improvements. Works with DataFrames from
    ExperimentDataLoader.

    Example:
        >>> comparator = FormatComparator()
        >>> df = loader.get_format_comparison_runs()
        >>> rankings = comparator.rank_by_metric(df, "best_loss", ascending=True)
        >>> improvement = comparator.compute_improvement(df, "E5M2", "E3M4", "best_loss")
    """

    def rank_by_metric(
        self,
        df: pd.DataFrame,
        metric: str,
        ascending: bool = True,
        group_by: str = "format",
    ) -> pd.DataFrame:
        """Rank formats by a specified metric.

        Computes the mean metric value for each format/group and assigns ranks.
        Lower rank = better (rank 1 is best).

        Args:
            df: DataFrame with experiment runs
            metric: Column name to rank by (e.g., "best_loss", "bit_stall_rate")
            ascending: If True, lower values are better (default for loss).
                      If False, higher values are better.
            group_by: Column to group by before ranking (default "format")

        Returns:
            DataFrame with columns: [group_by, metric + "_mean", metric + "_std",
                                     "count", "rank"]

        Example:
            >>> rankings = comparator.rank_by_metric(df, "best_loss")
            >>> print(rankings.head())
            #   format   best_loss_mean  best_loss_std  count  rank
            # 0 E5M2     2.45            0.12           5      1
            # 1 E3M4     2.67            0.18           4      2
        """
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame columns: {df.columns.tolist()}")

        # Aggregate metrics by group
        agg_df = df.groupby(group_by).agg(
            **{
                f"{metric}_mean": (metric, "mean"),
                f"{metric}_std": (metric, "std"),
                "count": (metric, "count"),
            }
        ).reset_index()

        # Fill NaN std with 0 (single sample)
        agg_df[f"{metric}_std"] = agg_df[f"{metric}_std"].fillna(0)

        # Assign ranks
        agg_df["rank"] = agg_df[f"{metric}_mean"].rank(
            ascending=ascending, method="min"
        ).astype(int)

        # Sort by rank
        agg_df = agg_df.sort_values("rank").reset_index(drop=True)

        return agg_df

    def identify_sweet_spot(
        self,
        df: pd.DataFrame,
        primary_metric: str = "best_loss",
        stability_metric: str = "bit_stall_rate",
        stability_threshold: float = 0.3,
    ) -> Optional[str]:
        """Identify the format with the best balance of performance and stability.

        A "sweet spot" format achieves low loss while maintaining acceptable
        stability (bit stall rate below threshold).

        Args:
            df: DataFrame with experiment runs
            primary_metric: Metric to optimize (lower is better)
            stability_metric: Stability metric (lower is better)
            stability_threshold: Maximum acceptable stability metric value

        Returns:
            Format name of the sweet spot, or None if no format qualifies.

        Example:
            >>> sweet_spot = comparator.identify_sweet_spot(df)
            >>> print(f"Recommended format: {sweet_spot}")
        """
        # Filter to completed runs only
        completed = df[df["completed"] == True].copy()  # noqa: E712

        if completed.empty:
            return None

        # Aggregate by format
        agg = completed.groupby("format").agg({
            primary_metric: "mean",
            stability_metric: "mean",
        }).reset_index()

        # Filter by stability threshold
        stable_formats = agg[agg[stability_metric] <= stability_threshold]

        if stable_formats.empty:
            # If no format meets stability, return best performer anyway
            return agg.loc[agg[primary_metric].idxmin(), "format"]

        # Return format with best primary metric among stable ones
        best_idx = stable_formats[primary_metric].idxmin()
        return stable_formats.loc[best_idx, "format"]

    def compute_improvement(
        self,
        df: pd.DataFrame,
        baseline_format: str,
        comparison_format: str,
        metric: str,
    ) -> Dict[str, float]:
        """Compute relative improvement between two formats.

        Calculates the percentage improvement of comparison_format over
        baseline_format for the specified metric.

        Args:
            df: DataFrame with experiment runs
            baseline_format: Format to compare against (e.g., "E5M2")
            comparison_format: Format being evaluated (e.g., "E3M4")
            metric: Metric to compare (e.g., "best_loss")

        Returns:
            Dictionary with:
                - baseline_mean: Mean metric for baseline format
                - comparison_mean: Mean metric for comparison format
                - absolute_diff: comparison - baseline
                - relative_improvement: (baseline - comparison) / baseline * 100
                  Positive = comparison is better, Negative = baseline is better

        Example:
            >>> improvement = comparator.compute_improvement(df, "E5M2", "E3M4", "best_loss")
            >>> print(f"E3M4 is {improvement['relative_improvement']:.1f}% better than E5M2")
        """
        baseline_df = df[df["format"] == baseline_format]
        comparison_df = df[df["format"] == comparison_format]

        baseline_mean = baseline_df[metric].mean()
        comparison_mean = comparison_df[metric].mean()

        absolute_diff = comparison_mean - baseline_mean

        # For loss-like metrics, lower is better, so improvement is (baseline - comparison)
        if baseline_mean != 0:
            relative_improvement = (baseline_mean - comparison_mean) / abs(baseline_mean) * 100
        else:
            relative_improvement = 0.0

        return {
            "baseline_format": baseline_format,
            "comparison_format": comparison_format,
            "metric": metric,
            "baseline_mean": baseline_mean,
            "comparison_mean": comparison_mean,
            "absolute_diff": absolute_diff,
            "relative_improvement": relative_improvement,
        }

    def compare_by_layer_type(
        self,
        histories: Dict[str, pd.DataFrame],
        layer_patterns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare format performance by layer type.

        Analyzes per-layer metrics (if logged) to identify which layers
        benefit most from each format.

        Args:
            histories: Dict mapping format name to per-step history DataFrame
            layer_patterns: Layer name patterns to analyze (e.g., ["attn", "mlp"])
                           If None, defaults to ["attn", "mlp", "ln"]

        Returns:
            DataFrame comparing metrics across layer types and formats.

        Example:
            >>> histories = {"E5M2": loader.get_run_history(e5m2_run_id),
            ...             "E3M4": loader.get_run_history(e3m4_run_id)}
            >>> comparison = comparator.compare_by_layer_type(histories)
        """
        if layer_patterns is None:
            layer_patterns = ["attn", "mlp", "ln"]

        results = []

        for format_name, history in histories.items():
            for pattern in layer_patterns:
                # Find columns matching the layer pattern
                matching_cols = [c for c in history.columns if pattern in c.lower()]

                if not matching_cols:
                    continue

                for col in matching_cols:
                    mean_val = history[col].mean()
                    std_val = history[col].std()

                    results.append({
                        "format": format_name,
                        "layer_pattern": pattern,
                        "metric": col,
                        "mean": mean_val,
                        "std": std_val,
                    })

        return pd.DataFrame(results)

    def compute_completion_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute completion rates by format.

        Args:
            df: DataFrame with experiment runs

        Returns:
            DataFrame with format, total runs, completed runs, completion rate
        """
        agg = df.groupby("format").agg(
            total_runs=("completed", "count"),
            completed_runs=("completed", "sum"),
        ).reset_index()

        agg["completion_rate"] = agg["completed_runs"] / agg["total_runs"]

        return agg.sort_values("completion_rate", ascending=False).reset_index(drop=True)


__all__ = [
    "FormatComparator",
]
