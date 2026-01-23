"""Markdown report generation for analysis artifacts.

Provides ReportGenerator class for creating structured markdown reports
from experiment data, including format comparisons (ANAL-01), failure
mode summaries (ANAL-02), and manifold optimizer comparisons (ANAL-03).

Key features:
  - Format comparison tables using DataFrame.to_markdown()
  - Failure mode analysis with recommendations
  - Manifold vs standard optimizer comparison
  - Automatic reports/ directory creation

Example:
    >>> from altgrad.analysis import ReportGenerator
    >>> generator = ReportGenerator()
    >>> generator.generate_format_comparison(runs_df, "reports/ANAL-01.md")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class ReportGenerator:
    """Generate markdown analysis reports.

    Creates structured markdown reports from experiment data for
    documentation and analysis review.

    Attributes:
        reports_dir: Directory for output reports (created on init)

    Example:
        >>> generator = ReportGenerator(reports_dir="reports")
        >>> generator.generate_format_comparison(runs, "reports/ANAL-01.md")
        >>> generator.generate_failure_modes(failures, "reports/ANAL-02.md")
        >>> generator.generate_manifold_comparison(manifold_runs, "reports/ANAL-03.md")
    """

    def __init__(self, reports_dir: str = "reports"):
        """Initialize report generator.

        Args:
            reports_dir: Directory for output reports (created if missing)
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_format_comparison(
        self,
        runs_df: pd.DataFrame,
        output_path: Optional[str] = None,
        baseline_format: str = "E5M2",
    ) -> str:
        """Generate format comparison report (ANAL-01).

        Creates a comprehensive comparison of FP8 format performance
        including rankings, completion rates, and improvement metrics.

        Args:
            runs_df: DataFrame from ExperimentDataLoader.get_format_comparison_runs()
            output_path: Output file path (default: reports/ANAL-01-format-comparison.md)
            baseline_format: Format to use as baseline for improvement calculations

        Returns:
            Path to generated report

        Example:
            >>> path = generator.generate_format_comparison(runs)
            >>> print(f"Report written to: {path}")
        """
        if output_path is None:
            output_path = str(self.reports_dir / "ANAL-01-format-comparison.md")

        # Compute statistics
        from altgrad.analysis.comparisons import FormatComparator

        comparator = FormatComparator()

        # Rankings by different metrics
        loss_rankings = comparator.rank_by_metric(runs_df, "best_loss", ascending=True)
        stall_rankings = comparator.rank_by_metric(runs_df, "bit_stall_rate", ascending=True)
        completion_rates = comparator.compute_completion_rates(runs_df)

        # Sweet spot identification
        sweet_spot = comparator.identify_sweet_spot(runs_df)

        # Build report content
        report = f"""# ANAL-01: FP8 Format Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Runs analyzed:** {len(runs_df)}
**Formats tested:** {', '.join(runs_df['format'].unique())}

## Executive Summary

This report compares FP8 format performance across training experiments,
analyzing convergence quality, stability metrics, and completion rates.

**Recommended format:** {sweet_spot or "Unable to determine (insufficient data)"}

## Format Rankings by Loss

Lower loss = better convergence. Rankings based on mean best_loss across runs.

{loss_rankings.to_markdown(index=False) if not loss_rankings.empty else "No data available."}

## Format Rankings by Stability

Lower bit stall rate = more stable training. High stall rates indicate
gradients being quantized to zero due to insufficient precision.

{stall_rankings.to_markdown(index=False) if not stall_rankings.empty else "No data available."}

## Completion Rates by Format

Training completion indicates stability - crashed runs suggest numerical
issues with the format.

{completion_rates.to_markdown(index=False) if not completion_rates.empty else "No data available."}

## Detailed Format Statistics

"""
        # Add per-format details
        for format_name in runs_df["format"].unique():
            format_runs = runs_df[runs_df["format"] == format_name]
            completed = format_runs[format_runs["completed"] == True]  # noqa: E712

            report += f"""### {format_name}

| Metric | Value |
|--------|-------|
| Total runs | {len(format_runs)} |
| Completed | {len(completed)} |
| Completion rate | {len(completed)/len(format_runs)*100:.1f}% |
| Best loss (mean) | {completed['best_loss'].mean():.4f if len(completed) > 0 else 'N/A'} |
| Best loss (std) | {completed['best_loss'].std():.4f if len(completed) > 1 else 'N/A'} |
| Bit stall rate (mean) | {completed['bit_stall_rate'].mean()*100:.1f}% if len(completed) > 0 else 'N/A' |

"""
        # Improvements vs baseline
        if baseline_format in runs_df["format"].unique():
            report += f"""## Improvement vs {baseline_format} Baseline

"""
            for format_name in runs_df["format"].unique():
                if format_name == baseline_format:
                    continue

                try:
                    improvement = comparator.compute_improvement(
                        runs_df, baseline_format, format_name, "best_loss"
                    )
                    sign = "+" if improvement["relative_improvement"] > 0 else ""
                    report += f"- **{format_name}:** {sign}{improvement['relative_improvement']:.1f}% ({improvement['comparison_mean']:.4f} vs {improvement['baseline_mean']:.4f})\n"
                except Exception:
                    report += f"- **{format_name}:** Unable to compute (insufficient data)\n"

        report += f"""

## Conclusions

Based on the analysis:

1. **Best convergence:** {loss_rankings.iloc[0]['format'] if not loss_rankings.empty else 'Unknown'} achieved lowest mean loss
2. **Most stable:** {stall_rankings.iloc[0]['format'] if not stall_rankings.empty else 'Unknown'} had lowest bit stall rate
3. **Recommended:** {sweet_spot or 'More data needed'} balances convergence and stability

## Methodology

- Rankings computed from mean metric values across all runs
- Sweet spot identified as format with lowest loss among those with <30% bit stall rate
- Completion rate reflects training stability (crashed = incomplete)

---
*Generated by altgrad.analysis.ReportGenerator*
"""

        # Write report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)

        return output_path

    def generate_failure_modes(
        self,
        failures_df: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate failure mode analysis report (ANAL-02).

        Documents failure patterns across formats with classification
        and recommendations for addressing each failure mode.

        Args:
            failures_df: DataFrame from FailureAnalyzer.classify_failure_mode()
            output_path: Output file path (default: reports/ANAL-02-failure-modes.md)

        Returns:
            Path to generated report

        Example:
            >>> path = generator.generate_failure_modes(classified_failures)
        """
        if output_path is None:
            output_path = str(self.reports_dir / "ANAL-02-failure-modes.md")

        from altgrad.analysis.failure_analysis import FailureAnalyzer

        analyzer = FailureAnalyzer()

        # Ensure failures are classified
        if "failure_mode" not in failures_df.columns and not failures_df.empty:
            failures_df = analyzer.classify_failure_mode(failures_df)

        # Summary by format
        summary = analyzer.summarize_failures_by_format(failures_df)

        report = f"""# ANAL-02: Failure Mode Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total failures analyzed:** {len(failures_df)}

## Executive Summary

This report analyzes training failures across FP8 formats, classifying
failure modes and providing recommendations for mitigation.

## Failure Summary by Format

{summary.to_markdown(index=False) if not summary.empty else "No failures recorded."}

## Failure Mode Definitions

| Mode | Description | Indicators |
|------|-------------|------------|
| nan_loss | Training loss became NaN/Inf | Loss = NaN, Inf |
| bit_stall | Gradients quantized to zero | Bit stall rate > 50% |
| overflow | Gradient overflow | Overflow rate > 10% |
| early_stop | Stopped early | < 10% of max steps |
| gradient_vanishing | Gradients vanished | > 70% near-zero gradients |
| unknown | Could not classify | None of above |

## Failures by Mode

"""
        if not failures_df.empty and "failure_mode" in failures_df.columns:
            for mode in failures_df["failure_mode"].unique():
                mode_failures = failures_df[failures_df["failure_mode"] == mode]
                report += f"""### {mode.replace("_", " ").title()} ({len(mode_failures)} failures)

| Format | Run | Steps | Final Loss | Bit Stall |
|--------|-----|-------|------------|-----------|
"""
                for _, row in mode_failures.head(10).iterrows():
                    report += f"| {row.get('format', 'N/A')} | {row.get('run_name', row.get('run_id', 'N/A'))[:20]} | {row.get('steps_completed', 'N/A')} | {row.get('final_loss', 'N/A'):.4f if pd.notna(row.get('final_loss')) else 'NaN'} | {row.get('bit_stall_rate', 0)*100:.1f}% |\n"

                if len(mode_failures) > 10:
                    report += f"\n*... and {len(mode_failures) - 10} more*\n"

                report += "\n"

        report += """## Recommendations by Failure Mode

### NaN Loss
1. Enable partition-relative gradient clipping
2. Reduce learning rate
3. Consider format with larger dynamic range (more exponent bits)

### Bit Stall
1. Use format with more mantissa bits (E3M4, E5M2)
2. Enable stiffness-aware preconditioning
3. Increase learning rate to overcome quantization noise

### Overflow
1. Enable automatic scaling adjustment
2. Reduce gradient clipping threshold
3. Use format with larger exponent (E7M0, E5M2)

### Early Stop / Unknown
1. Review training logs for specific errors
2. Check for resource constraints (OOM, timeout)
3. Verify data pipeline stability

---
*Generated by altgrad.analysis.ReportGenerator*
"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)

        return output_path

    def generate_manifold_comparison(
        self,
        manifold_runs: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate manifold optimizer comparison report (ANAL-03).

        Compares ManifoldAdamW vs standard AdamW optimizer performance
        on E5M2 format to evaluate geometry-aware optimization benefits.

        Args:
            manifold_runs: DataFrame from ExperimentDataLoader.get_manifold_comparison_runs()
            output_path: Output file path (default: reports/ANAL-03-manifold-comparison.md)

        Returns:
            Path to generated report

        Example:
            >>> path = generator.generate_manifold_comparison(manifold_runs)
        """
        if output_path is None:
            output_path = str(self.reports_dir / "ANAL-03-manifold-comparison.md")

        # Split into standard and manifold runs
        standard_runs = manifold_runs[manifold_runs["use_manifold_aware"] == False]  # noqa: E712
        manifold_aware = manifold_runs[manifold_runs["use_manifold_aware"] == True]  # noqa: E712

        report = f"""# ANAL-03: Manifold-Aware Optimizer Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total E5M2 runs:** {len(manifold_runs)}
**Standard AdamW runs:** {len(standard_runs)}
**ManifoldAdamW runs:** {len(manifold_aware)}

## Executive Summary

This report compares the performance of ManifoldAdamW (geometry-aware optimizer)
versus standard AdamW on E5M2 format experiments. ManifoldAdamW uses stiffness
preconditioning to account for the non-uniform quantization grid.

## Performance Comparison

| Metric | Standard AdamW | ManifoldAdamW | Improvement |
|--------|---------------|---------------|-------------|
"""
        # Compute comparison metrics
        metrics = ["best_loss", "final_loss", "bit_stall_rate"]
        for metric in metrics:
            std_mean = standard_runs[metric].mean() if not standard_runs.empty else float("nan")
            mani_mean = manifold_aware[metric].mean() if not manifold_aware.empty else float("nan")

            if pd.notna(std_mean) and pd.notna(mani_mean) and std_mean != 0:
                # For loss metrics, lower is better
                if "loss" in metric:
                    improvement = (std_mean - mani_mean) / abs(std_mean) * 100
                else:
                    improvement = (std_mean - mani_mean) / abs(std_mean) * 100

                sign = "+" if improvement > 0 else ""
                report += f"| {metric} | {std_mean:.4f} | {mani_mean:.4f} | {sign}{improvement:.1f}% |\n"
            else:
                report += f"| {metric} | {std_mean:.4f if pd.notna(std_mean) else 'N/A'} | {mani_mean:.4f if pd.notna(mani_mean) else 'N/A'} | N/A |\n"

        # Completion rates
        std_completion = standard_runs["completed"].mean() * 100 if not standard_runs.empty else 0
        mani_completion = manifold_aware["completed"].mean() * 100 if not manifold_aware.empty else 0

        report += f"""
## Completion Rates

| Optimizer | Completion Rate |
|-----------|-----------------|
| Standard AdamW | {std_completion:.1f}% |
| ManifoldAdamW | {mani_completion:.1f}% |

## Detailed Run Statistics

### Standard AdamW Runs

{standard_runs[['run_name', 'best_loss', 'final_loss', 'bit_stall_rate', 'completed']].to_markdown(index=False) if not standard_runs.empty else "No standard AdamW runs found."}

### ManifoldAdamW Runs

{manifold_aware[['run_name', 'best_loss', 'final_loss', 'bit_stall_rate', 'completed']].to_markdown(index=False) if not manifold_aware.empty else "No ManifoldAdamW runs found."}

## Analysis

### Hypothesis

ManifoldAdamW should improve training stability and convergence on quantized
formats by:
1. Scaling gradients by local stiffness (inverse of ULP spacing)
2. Preventing updates smaller than quantization resolution
3. Adapting step sizes to the non-uniform FP8 grid

### Results Interpretation

"""
        # Interpret results
        if not standard_runs.empty and not manifold_aware.empty:
            std_loss = standard_runs["best_loss"].mean()
            mani_loss = manifold_aware["best_loss"].mean()

            if mani_loss < std_loss:
                report += """**ManifoldAdamW shows improved convergence.** The geometry-aware
optimizer achieved lower loss, suggesting that stiffness preconditioning
effectively accounts for the non-uniform quantization grid.
"""
            elif mani_loss > std_loss:
                report += """**Standard AdamW performed better.** This may indicate:
- The stiffness computation may need tuning
- E5M2 format may not benefit significantly from manifold awareness
- More experiments needed to draw conclusions
"""
            else:
                report += """**Performance is similar.** Both optimizers achieved comparable
results on E5M2 format.
"""
        else:
            report += """**Insufficient data for comparison.** Run both standard and
ManifoldAdamW experiments on E5M2 format to enable comparison.
"""

        report += """
## Recommendations

1. **If ManifoldAdamW improves loss:** Apply to other exotic formats (E3M4, E1M6)
   where non-uniform grids are more pronounced

2. **If no improvement:** Review stiffness computation, consider format-specific
   tuning of manifold_mantissa_bits parameter

3. **For production:** Use ManifoldAdamW when training with FP8 quantization,
   especially for formats with few mantissa bits

---
*Generated by altgrad.analysis.ReportGenerator*
"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)

        return output_path

    def generate_all_reports(
        self,
        format_runs: pd.DataFrame,
        failures: pd.DataFrame,
        manifold_runs: pd.DataFrame,
    ) -> Dict[str, str]:
        """Generate all three analysis reports.

        Args:
            format_runs: DataFrame for format comparison
            failures: DataFrame with classified failures
            manifold_runs: DataFrame for manifold comparison

        Returns:
            Dictionary mapping report ID to file path
        """
        paths = {}

        paths["ANAL-01"] = self.generate_format_comparison(format_runs)
        paths["ANAL-02"] = self.generate_failure_modes(failures)
        paths["ANAL-03"] = self.generate_manifold_comparison(manifold_runs)

        return paths


__all__ = [
    "ReportGenerator",
]
