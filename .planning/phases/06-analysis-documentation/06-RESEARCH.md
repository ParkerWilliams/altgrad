# Phase 6: Analysis & Documentation - Research

**Researched:** 2026-01-22
**Domain:** ML experiment analysis, FP8 quantization synthesis, failure mode documentation
**Confidence:** HIGH

## Summary

This phase synthesizes experiment results from Phases 4-5 to answer the core research question: "Which FP8 format most benefits from geometry-aware updates, and why?" The analysis covers three requirements: (ANAL-01) sweet-spot format per layer type, (ANAL-02) failure mode documentation, and (ANAL-03) manifold-aware vs standard comparison.

The research confirms that the existing infrastructure provides all necessary data: W&B logs contain per-step metrics (loss, gradient norms, bit-stall rates, overflow counts), the FormatExperimentRunner generates failure reports on collapse, and the experiment configs enable controlled comparison between formats and optimizer modes. The analysis phase does NOT require running new experiments -- it aggregates and synthesizes existing W&B data.

**Primary recommendation:** Use the W&B Public API to programmatically download experiment histories as pandas DataFrames, compute comparative statistics (mean loss, convergence rate, failure step), and generate markdown reports directly from Python. Use matplotlib for visualization with a consistent style across all comparison charts.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| wandb | 0.24+ | Experiment data retrieval | Public API provides `runs.histories()` returning pandas DataFrame |
| pandas | 2.0+ | Data aggregation and analysis | `DataFrame.to_markdown()` for direct report generation |
| matplotlib | 3.7+ | Comparison visualizations | Subplot grids for multi-format comparison |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tabulate | 0.9+ | Markdown table generation | Required by `DataFrame.to_markdown()` |
| scipy.stats | 1.11+ | Bootstrap confidence intervals | `scipy.stats.bootstrap()` for statistical rigor |
| seaborn | 0.12+ | Enhanced plot styling | Optional: better defaults for scientific plots |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| W&B Public API | Manual CSV export | API is programmatic, reproducible; CSV is manual |
| pandas | polars | pandas has `to_markdown()`, polars faster but less tooling |
| matplotlib | plotly | matplotlib static images easier for markdown; plotly needs HTML |

**Installation:**
```bash
pip install wandb pandas matplotlib tabulate scipy
# seaborn optional but recommended
pip install seaborn
```

## Architecture Patterns

### Recommended Project Structure
```
altgrad/
├── analysis/                    # NEW: Analysis module
│   ├── __init__.py
│   ├── data_loader.py          # W&B API data fetching
│   ├── comparisons.py          # Format/optimizer comparison logic
│   ├── failure_analysis.py     # Failure mode summarization
│   └── report_generator.py     # Markdown report generation
├── reports/                     # NEW: Generated reports
│   ├── format_comparison.md    # ANAL-01: Sweet-spot analysis
│   ├── failure_modes.md        # ANAL-02: Failure documentation
│   └── manifold_comparison.md  # ANAL-03: Manifold vs standard
└── scripts/
    └── generate_reports.py     # Entry point for analysis
```

### Pattern 1: W&B Public API Data Loading

**What:** Programmatically fetch experiment data from W&B
**When to use:** Any analysis requiring experiment metrics

```python
# Source: W&B Public API documentation
import wandb
import pandas as pd

def load_experiment_runs(
    project: str = "altgrad",
    filters: dict = None,
    keys: list = None,
) -> pd.DataFrame:
    """Load W&B runs as a pandas DataFrame.

    Args:
        project: W&B project name
        filters: MongoDB-style filter dict, e.g. {"tags": {"$in": ["fp8"]}}
        keys: Metric keys to include, e.g. ["loss", "bit_stall_rate"]

    Returns:
        DataFrame with columns: run_id, step, plus each metric key
    """
    api = wandb.Api()
    runs = api.runs(
        path=f"{api.default_entity}/{project}",
        filters=filters,
    )

    # Use histories() for bulk download
    df = runs.histories(
        samples=500,  # Downsample long runs
        keys=keys or ["loss", "perplexity", "grad_norm",
                      "quantization/bit_stall_rate", "quantization/overflow_rate"],
        x_axis="_step",
        format="pandas",
    )

    return df
```

**Key points:**
- `api.runs()` returns lazy iterator, data fetched on demand
- `histories()` bulk downloads with downsampling
- Filter by tags: `{"tags": {"$in": ["fp8", "e5m2"]}}`
- Filter by config: `{"config.fp8_format": "E5M2"}`

### Pattern 2: Per-Run Summary Extraction

**What:** Extract summary statistics from individual runs
**When to use:** Comparing final performance across formats

```python
# Source: W&B Public API documentation
def extract_run_summary(run_path: str) -> dict:
    """Extract summary metrics from a single run.

    Args:
        run_path: Format "entity/project/run_id"

    Returns:
        Dict with config and summary metrics
    """
    api = wandb.Api()
    run = api.run(run_path)

    return {
        "run_id": run.id,
        "name": run.name,
        "state": run.state,  # finished, crashed, etc.
        "format": run.config.get("fp8_format", "bf16"),
        "manifold_aware": run.config.get("use_manifold_aware", False),
        "final_loss": run.summary.get("loss"),
        "best_loss": run.summary.get("best_loss", run.summary.get("loss")),
        "steps_completed": run.summary.get("_step", 0),
        "bit_stall_rate_final": run.summary.get("quantization/bit_stall_rate"),
        "overflow_rate_final": run.summary.get("quantization/overflow_rate"),
    }
```

### Pattern 3: Layer-Type Analysis from Tags

**What:** Aggregate metrics by layer type (attention, MLP, classifier)
**When to use:** ANAL-01 sweet-spot format per layer type

```python
# Source: Derived from project architecture
def analyze_by_layer_type(
    df: pd.DataFrame,
    layer_patterns: dict = None,
) -> pd.DataFrame:
    """Aggregate metrics by layer type.

    Args:
        df: DataFrame with per-layer metrics (e.g., grad_norm_l2/layer.0.attn.c_proj)
        layer_patterns: Mapping of layer type to name patterns

    Returns:
        DataFrame grouped by layer type with aggregated metrics
    """
    layer_patterns = layer_patterns or {
        "attention": ["attn", "c_attn", "c_proj"],
        "mlp": ["mlp", "c_fc", "c_proj"],
        "embedding": ["wte", "wpe"],
        "classifier": ["lm_head"],
    }

    # Implementation: parse column names, classify, aggregate
    # ...
```

**Key points:**
- nanoGPT uses naming: `transformer.h.{N}.{attn|mlp}.{weight_name}`
- Attention weights: c_attn, c_proj
- MLP weights: c_fc, c_proj
- Group by layer type, then by format, compute mean metrics

### Pattern 4: Markdown Report Generation

**What:** Generate markdown reports directly from DataFrames
**When to use:** ANAL-01, ANAL-02, ANAL-03 reports

```python
# Source: pandas DataFrame.to_markdown() documentation
import pandas as pd
from datetime import datetime

def generate_comparison_report(
    summary_df: pd.DataFrame,
    title: str,
    output_path: str,
) -> str:
    """Generate markdown comparison report.

    Args:
        summary_df: DataFrame with comparison data
        title: Report title
        output_path: Path to write markdown file

    Returns:
        Path to generated report
    """
    report = f"""# {title}

**Generated:** {datetime.now().isoformat()}

## Summary

{summary_df.describe().to_markdown()}

## Full Comparison

{summary_df.to_markdown(index=False)}

## Key Findings

[Analysis text here based on data]

---
*Generated by AltGrad analysis pipeline*
"""

    with open(output_path, "w") as f:
        f.write(report)

    return output_path
```

**Key points:**
- `DataFrame.to_markdown()` requires `tabulate` package
- Use `index=False` for cleaner tables
- Include generation timestamp for reproducibility
- Combine with manual analysis text

### Pattern 5: Matplotlib Comparison Plots

**What:** Multi-format training curve comparison
**When to use:** Visualizing loss/metric trends across formats

```python
# Source: Matplotlib documentation and ML best practices
import matplotlib.pyplot as plt
import seaborn as sns

def plot_format_comparison(
    df: pd.DataFrame,
    metric: str = "loss",
    formats: list = None,
    output_path: str = None,
) -> None:
    """Plot training curves for multiple formats.

    Args:
        df: DataFrame with columns [run_id, _step, metric, format]
        metric: Metric column to plot
        formats: List of format names to include
        output_path: Path to save figure (None for display)
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(10, 6))

    formats = formats or df['format'].unique()
    colors = plt.cm.Set2(range(len(formats)))

    for i, fmt in enumerate(formats):
        fmt_data = df[df['format'] == fmt]
        ax.plot(
            fmt_data['_step'],
            fmt_data[metric],
            label=fmt,
            color=colors[i],
            linewidth=2,
        )

    ax.set_xlabel('Training Step')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.title()} by FP8 Format')
    ax.legend(loc='upper right')

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

**Key points:**
- Use `plt.style.use('seaborn-v0_8-whitegrid')` for clean scientific style
- Save as PNG for markdown embedding
- Include legend and axis labels
- Use consistent color palette across reports

### Pattern 6: Bootstrap Confidence Intervals

**What:** Statistical comparison with confidence intervals
**When to use:** Quantifying manifold vs standard benefit

```python
# Source: scipy.stats.bootstrap documentation
from scipy.stats import bootstrap
import numpy as np

def compute_metric_ci(
    values: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
) -> tuple:
    """Compute bootstrap confidence interval for metric.

    Args:
        values: Array of metric values (e.g., final losses across seeds)
        confidence_level: Confidence level (default 0.95)
        n_resamples: Number of bootstrap resamples

    Returns:
        Tuple of (mean, ci_low, ci_high)
    """
    # scipy.stats.bootstrap expects tuple of arrays
    data = (values,)

    result = bootstrap(
        data,
        statistic=np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method='BCa',  # Bias-corrected and accelerated
    )

    return (
        np.mean(values),
        result.confidence_interval.low,
        result.confidence_interval.high,
    )
```

**Key points:**
- BCa method (bias-corrected and accelerated) is most robust
- 1000 resamples sufficient for 95% CI
- Report as "mean [CI_low, CI_high]" in tables

### Anti-Patterns to Avoid

- **Running new experiments in Phase 6:** This is analysis-only; experiments run in Phases 4-5
- **Hardcoding run IDs:** Use filters by tags/config for reproducibility
- **Manual data entry:** All data should flow from W&B API
- **Generating reports without data validation:** Check for missing runs before analysis
- **Ignoring failed runs:** Failure data is valuable for ANAL-02

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Experiment data fetching | Custom CSV parsing | W&B Public API | Handles auth, pagination, filtering |
| Markdown tables | String concatenation | `DataFrame.to_markdown()` | Handles alignment, escaping |
| Statistical comparison | Manual mean/std | `scipy.stats.bootstrap()` | Proper CI estimation with BCa |
| Training curve plots | Raw matplotlib calls | Consistent wrapper function | Ensures uniform style |
| Failure report parsing | Regex on markdown | `ExperimentResult.failure_report_path` | Already structured in runner |

**Key insight:** The existing infrastructure (W&B logging, FormatExperimentRunner, ExperimentResult dataclass) already captures all necessary data. This phase aggregates and synthesizes, not collects.

## Common Pitfalls

### Pitfall 1: Missing Runs in W&B

**What goes wrong:** Analysis code assumes all runs exist and are finished
**Why it happens:** Experiments may crash, be interrupted, or not be uploaded
**How to avoid:** Filter by state: `{"state": "finished"}` and handle partial data
**Warning signs:** KeyError on summary metrics, NaN in aggregations

**Prevention strategy:**
```python
# Filter for finished runs only
runs = api.runs(project, filters={"state": "finished"})

# Handle missing metrics gracefully
final_loss = run.summary.get("loss", float("nan"))
```

### Pitfall 2: Inconsistent Step Counts

**What goes wrong:** Comparing runs with different max_steps
**Why it happens:** E7M0 crashes early, others run to completion
**How to avoid:** Normalize by step count or compare at common steps only
**Warning signs:** Loss curves with different lengths

**Prevention strategy:**
```python
# Compare at common steps only
min_steps = df.groupby('run_id')['_step'].max().min()
df_common = df[df['_step'] <= min_steps]
```

### Pitfall 3: Aggregating Across Seeds Without Noting It

**What goes wrong:** Reporting single run as representative
**Why it happens:** Limited compute budget (one run per config)
**How to avoid:** Clearly note n=1 in reports; use "observed" not "expected"
**Warning signs:** Overconfident conclusions from single data points

**Prevention strategy:**
```markdown
Note: Results from single runs (n=1) due to compute constraints.
Statistical confidence intervals not applicable.
```

### Pitfall 4: Cherry-Picking Failure Steps

**What goes wrong:** Reporting collapse step without context
**Why it happens:** E7M0 may collapse at step 10, but that's the interesting data
**How to avoid:** Document ALL failure modes, not just dramatic ones
**Warning signs:** Report says "E7M0 failed" without step-by-step breakdown

**Prevention strategy:**
Include in failure report:
- Exact collapse step
- Metrics trend in last N steps before collapse
- Gradient sparsity at collapse
- Comparison to successful formats at same step

### Pitfall 5: Ignoring Layer-Type Heterogeneity

**What goes wrong:** Reporting "E5M2 is best" without per-layer analysis
**Why it happens:** Averaging across layers hides important structure
**How to avoid:** ANAL-01 explicitly requires per-layer-type analysis
**Warning signs:** No mention of attention vs MLP differences

**Prevention strategy:**
Always report:
- Overall format ranking
- Format ranking for attention layers
- Format ranking for MLP layers
- Format ranking for classifier/embedding

### Pitfall 6: Not Linking to W&B Runs

**What goes wrong:** Reports without traceability to source data
**Why it happens:** Generated reports don't include run URLs
**How to avoid:** Include run links in reports for verification
**Warning signs:** Can't reproduce analysis from report

**Prevention strategy:**
```python
run_url = f"https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}"
# Include in report
```

## Code Examples

Verified patterns from official sources:

### Complete Data Loading Pipeline
```python
# Source: W&B Public API + pandas
import wandb
import pandas as pd
from typing import List, Dict

class ExperimentDataLoader:
    """Load and preprocess W&B experiment data."""

    def __init__(self, project: str = "altgrad"):
        self.api = wandb.Api()
        self.project = project
        self.entity = self.api.default_entity

    def get_format_comparison_runs(
        self,
        formats: List[str] = None,
    ) -> pd.DataFrame:
        """Load runs for format comparison (ANAL-01).

        Args:
            formats: FP8 formats to include (default: all)

        Returns:
            DataFrame with run summaries
        """
        formats = formats or ["E5M2", "E3M4", "E1M6", "E0M7", "E7M0"]

        summaries = []
        for fmt in formats:
            runs = self.api.runs(
                path=f"{self.entity}/{self.project}",
                filters={
                    "config.fp8_format": fmt,
                    "config.use_manifold_aware": False,
                    "state": {"$in": ["finished", "crashed"]},
                },
            )

            for run in runs:
                summaries.append({
                    "format": fmt,
                    "run_id": run.id,
                    "run_url": run.url,
                    "state": run.state,
                    "steps": run.summary.get("_step", 0),
                    "final_loss": run.summary.get("loss"),
                    "best_loss": run.summary.get("best_loss"),
                    "bit_stall_final": run.summary.get("quantization/bit_stall_rate"),
                    "overflow_final": run.summary.get("quantization/overflow_rate"),
                })

        return pd.DataFrame(summaries)

    def get_manifold_comparison_runs(self) -> pd.DataFrame:
        """Load runs for manifold vs standard comparison (ANAL-03).

        Returns:
            DataFrame with paired manifold/standard run summaries
        """
        summaries = []

        for manifold_aware in [True, False]:
            runs = self.api.runs(
                path=f"{self.entity}/{self.project}",
                filters={
                    "config.fp8_format": "E5M2",
                    "config.use_manifold_aware": manifold_aware,
                    "state": "finished",
                },
            )

            for run in runs:
                summaries.append({
                    "optimizer": "ManifoldAdamW" if manifold_aware else "AdamW",
                    "manifold_aware": manifold_aware,
                    "run_id": run.id,
                    "final_loss": run.summary.get("loss"),
                    "best_loss": run.summary.get("best_loss"),
                    "bit_stall_final": run.summary.get("quantization/bit_stall_rate"),
                })

        return pd.DataFrame(summaries)
```

### Complete Report Generator
```python
# Source: pandas to_markdown + project requirements
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

class ReportGenerator:
    """Generate markdown analysis reports."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_format_comparison(
        self,
        df: pd.DataFrame,
        output_name: str = "format_comparison.md",
    ) -> str:
        """Generate ANAL-01: Sweet-spot format per layer type.

        Args:
            df: DataFrame with format comparison data
            output_name: Output filename

        Returns:
            Path to generated report
        """
        path = self.output_dir / output_name

        # Rank formats by final loss
        ranked = df.sort_values("final_loss")

        # Identify sweet-spot (best performing non-crashed)
        finished = ranked[ranked["state"] == "finished"]
        sweet_spot = finished.iloc[0]["format"] if len(finished) > 0 else "None"

        report = f"""# FP8 Format Comparison Report

**Generated:** {datetime.now().isoformat()}
**Project:** AltGrad FP8 Quantization Study

## Executive Summary

**Sweet-spot format:** {sweet_spot}

This report compares {len(df)} experiment runs across {df['format'].nunique()} FP8 formats
to identify which format provides the best training stability for transformer models.

## Results by Format

{ranked[['format', 'state', 'steps', 'final_loss', 'best_loss', 'bit_stall_final']].to_markdown(index=False)}

## Format Rankings

### By Final Loss (lower is better)
{finished[['format', 'final_loss']].sort_values('final_loss').to_markdown(index=False)}

### By Bit-Stall Rate (lower is better)
{finished[['format', 'bit_stall_final']].sort_values('bit_stall_final').to_markdown(index=False)}

## Failure Analysis

Formats that crashed before completion:

{df[df['state'] == 'crashed'][['format', 'steps', 'final_loss']].to_markdown(index=False)}

## Layer-Type Analysis

[To be populated with per-layer metrics]

## Conclusions

1. **Best overall format:** {sweet_spot}
2. **Formats to avoid:** {', '.join(df[df['state'] == 'crashed']['format'].tolist()) or 'None'}
3. **Recommended layer-specific formats:** [TBD based on per-layer analysis]

---
*Generated by AltGrad analysis pipeline*

## Run Links

{chr(10).join(f"- [{row['format']}]({row['run_url']})" for _, row in df.iterrows())}
"""

        with open(path, "w") as f:
            f.write(report)

        return str(path)

    def generate_failure_report(
        self,
        failure_data: pd.DataFrame,
        output_name: str = "failure_modes.md",
    ) -> str:
        """Generate ANAL-02: Failure mode documentation.

        Documents where each format fails: forward, backward, optimizer.
        """
        path = self.output_dir / output_name

        report = f"""# FP8 Failure Mode Documentation

**Generated:** {datetime.now().isoformat()}

## Overview

This report documents failure modes observed during FP8 training experiments.
Understanding failure patterns helps identify format limitations.

## Failure Summary

| Format | Collapse Step | Reason | Last Good Loss |
|--------|---------------|--------|----------------|
"""

        for _, row in failure_data.iterrows():
            report += f"| {row['format']} | {row['steps']} | {row.get('reason', 'Unknown')} | {row['final_loss']:.4f} |\n"

        report += """
## Detailed Failure Analysis

### E7M0: Powers-of-Two Only (Expected Failure)

**Hypothesis:** E7M0 with 0 mantissa bits can only represent powers of 2.
This extreme quantization was expected to fail.

**Observed:**
- Collapse step: [from data]
- Gradient sparsity at collapse: [from failure report]
- Zero-update regions: [from failure report]

**Failure Mode Classification:**
- [ ] Forward pass overflow
- [ ] Backward pass gradient vanishing
- [ ] Optimizer state corruption

### Other Format Failures

[Additional format-specific analysis]

## Prevention Recommendations

Based on observed failures:

1. **For high overflow rates:** Use partition-relative clipping
2. **For gradient vanishing:** Consider format with more mantissa bits
3. **For bit-stall:** Manifold-aware optimizer may help

---
*Generated by AltGrad analysis pipeline*
"""

        with open(path, "w") as f:
            f.write(report)

        return str(path)

    def generate_manifold_comparison(
        self,
        df: pd.DataFrame,
        output_name: str = "manifold_comparison.md",
    ) -> str:
        """Generate ANAL-03: Manifold-aware vs standard comparison.

        Quantifies benefit (or lack thereof) of geometry-aware updates.
        """
        path = self.output_dir / output_name

        manifold = df[df["manifold_aware"] == True]
        standard = df[df["manifold_aware"] == False]

        # Compute improvement
        if len(manifold) > 0 and len(standard) > 0:
            manifold_loss = manifold["final_loss"].mean()
            standard_loss = standard["final_loss"].mean()
            improvement = (standard_loss - manifold_loss) / standard_loss * 100
        else:
            improvement = float("nan")

        report = f"""# Manifold-Aware vs Standard Optimizer Comparison

**Generated:** {datetime.now().isoformat()}

## Executive Summary

**Loss improvement:** {improvement:.1f}% {'better' if improvement > 0 else 'worse'} with manifold-aware

This report compares the ManifoldAdamW optimizer (with stiffness preconditioning)
against standard AdamW on E5M2 FP8 training.

## Hypothesis

The manifold-aware optimizer should benefit formats with variable grid spacing
(floating-point FP8) by scaling updates by local stiffness. This makes updates
move weights by consistent ULP counts rather than fixed real values.

## Results

### Optimizer Comparison

{df[['optimizer', 'final_loss', 'best_loss', 'bit_stall_final']].to_markdown(index=False)}

### Statistical Comparison

| Metric | ManifoldAdamW | Standard AdamW | Difference |
|--------|---------------|----------------|------------|
| Mean Final Loss | {manifold['final_loss'].mean():.4f} | {standard['final_loss'].mean():.4f} | {manifold['final_loss'].mean() - standard['final_loss'].mean():+.4f} |
| Bit-Stall Rate | {manifold['bit_stall_final'].mean():.2%} | {standard['bit_stall_final'].mean():.2%} | - |

## Analysis

### When Manifold-Aware Helps

[Based on data analysis]

### When Manifold-Aware Doesn't Help

[Based on data analysis]

## Conclusions

1. **Overall benefit:** {improvement:.1f}% loss improvement
2. **Recommended usage:** [Based on analysis]
3. **Future work:** [Suggestions]

---
*Generated by AltGrad analysis pipeline*
"""

        with open(path, "w") as f:
            f.write(report)

        return str(path)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual W&B inspection | Programmatic API analysis | 2024 | Reproducible, scriptable |
| Copy-paste tables | `DataFrame.to_markdown()` | pandas 1.0 (2020) | Error-free tables |
| Single metric comparison | Multi-metric + layer-type | Current best practice | Reveals layer-specific behavior |
| Point estimates only | Bootstrap confidence intervals | Always was best practice | Statistical rigor |

**Current practice in FP8 research:**
- Per-layer sensitivity analysis standard in quantization papers (ICLR 2025)
- Failure mode documentation expected for negative results
- Reproducibility via W&B or similar tracking is standard

**Deprecated/outdated:**
- Manual metric extraction from logs
- Screenshots of W&B dashboards in papers

## Open Questions

Things that couldn't be fully resolved:

1. **Number of seeds for statistical validity**
   - What we know: More seeds = better confidence
   - What's unclear: Budget allows ~1 run per config
   - Recommendation: Report n=1 clearly; avoid overclaiming

2. **Layer-type metric availability**
   - What we know: Per-layer gradient norms logged
   - What's unclear: Whether all layer types have sufficient samples
   - Recommendation: Verify metric coverage before analysis

3. **Failure report format consistency**
   - What we know: FormatExperimentRunner generates markdown reports
   - What's unclear: Whether all failure reports have same structure
   - Recommendation: Parse conservatively, handle missing fields

4. **W&B API rate limits**
   - What we know: API has rate limits for large queries
   - What's unclear: Exact limits for our data volume
   - Recommendation: Cache data locally after first fetch

## Sources

### Primary (HIGH confidence)
- [W&B Public API - Runs](https://docs.wandb.ai/ref/python/public-api/runs/) - `runs.histories()` API
- [W&B Public API - Run](https://docs.wandb.ai/ref/python/public-api/run) - Single run access patterns
- [pandas DataFrame.to_markdown](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html) - Table generation
- Existing codebase: `FormatExperimentRunner`, `ExperimentResult`, `DiagnosticSnapshot`

### Secondary (MEDIUM confidence)
- [scipy.stats.bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) - Confidence intervals
- [Layer-Wise Sensitivity Analysis (arXiv 2503.06518)](https://arxiv.org/html/2503.06518v1) - Attention 3-5x more sensitive than MLP
- [NVIDIA FP8 Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) - FP8 failure modes

### Tertiary (LOW confidence)
- [Machine Learning Mastery - Bootstrap CI](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/) - Bootstrap patterns
- Various blog posts on matplotlib subplot styling

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - pandas, wandb, matplotlib are industry standard
- Architecture patterns: HIGH - Based on official API documentation
- Pitfalls: MEDIUM - Based on general ML analysis best practices
- Code examples: HIGH - Derived from official documentation

**Research date:** 2026-01-22

**Valid until:** 90 days - Analysis patterns stable; W&B API may have minor updates

**Key takeaway for planner:**
1. Create `altgrad/analysis/` module with data loading, comparison, and report generation
2. Use W&B Public API to fetch experiment data (NOT new experiments)
3. Generate three markdown reports: format_comparison.md, failure_modes.md, manifold_comparison.md
4. Include visualizations saved as PNG for markdown embedding
5. All reports should link back to W&B runs for traceability
6. Handle missing/crashed runs gracefully
