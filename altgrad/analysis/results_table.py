"""Experiment results table generator.

Produces the master comparison table for the research question:
Which FP8 datatype benefits most from manifold-aware gradient updates?

Table Structure:
- Rows: FP8 formats (BF16, E5M2, E3M4, E1M6, E0M7, E7M0)
- Column Groups:
  1. Throughput (samples/sec, tokens/sec)
  2. Classifier Performance - AdamW (F1 micro, F1 macro, Precision, Recall, Subset Acc)
  3. Classifier Performance - ManifoldAdamW (same metrics)
  4. Rank Health (Stable Rank, Effective Rank - init→final)
  5. Training Efficiency - AdamW (Time, Flips, Stall %)
  6. Training Efficiency - ManifoldAdamW (same metrics)
- Derived: F1 Delta, Stall Delta, Rank Collapse flag
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class ThroughputMetrics:
    """Throughput measurements."""
    samples_per_sec: float = 0.0
    tokens_per_sec: float = 0.0


@dataclass
class ClassifierMetrics:
    """Classification performance metrics."""
    f1_micro: float = 0.0
    f1_macro: float = 0.0
    precision_micro: float = 0.0
    recall_micro: float = 0.0
    subset_accuracy: float = 0.0
    hamming_loss: float = 0.0
    roc_auc_micro: float = 0.0


@dataclass
class RankMetrics:
    """Rank health metrics."""
    stable_rank_init: float = 0.0
    stable_rank_final: float = 0.0
    effective_rank_init: float = 0.0
    effective_rank_final: float = 0.0
    collapse_detected: bool = False


@dataclass
class TrainingMetrics:
    """Training efficiency metrics."""
    total_time_sec: float = 0.0
    total_flips: int = 0
    mean_stall_ratio: float = 0.0
    total_steps: int = 0


@dataclass
class ExperimentRow:
    """Single row in results table (one format, both optimizers)."""
    format: str

    # Throughput (same for both optimizers on same format)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)

    # Classifier performance per optimizer
    classifier_adamw: ClassifierMetrics = field(default_factory=ClassifierMetrics)
    classifier_manifold: ClassifierMetrics = field(default_factory=ClassifierMetrics)

    # Rank health per optimizer
    rank_adamw: RankMetrics = field(default_factory=RankMetrics)
    rank_manifold: RankMetrics = field(default_factory=RankMetrics)

    # Training efficiency per optimizer
    training_adamw: TrainingMetrics = field(default_factory=TrainingMetrics)
    training_manifold: TrainingMetrics = field(default_factory=TrainingMetrics)

    # Derived metrics
    @property
    def f1_delta(self) -> float:
        """F1 micro improvement from ManifoldAdamW over AdamW."""
        return self.classifier_manifold.f1_micro - self.classifier_adamw.f1_micro

    @property
    def stall_delta(self) -> float:
        """Stall ratio change (negative = manifold reduces stalls)."""
        return self.training_manifold.mean_stall_ratio - self.training_adamw.mean_stall_ratio

    @property
    def rank_collapse_adamw(self) -> bool:
        """Did AdamW experience rank collapse?"""
        if self.rank_adamw.stable_rank_init == 0:
            return False
        drop = (self.rank_adamw.stable_rank_init - self.rank_adamw.stable_rank_final) / self.rank_adamw.stable_rank_init
        return drop > 0.3 or self.rank_adamw.collapse_detected

    @property
    def rank_collapse_manifold(self) -> bool:
        """Did ManifoldAdamW experience rank collapse?"""
        if self.rank_manifold.stable_rank_init == 0:
            return False
        drop = (self.rank_manifold.stable_rank_init - self.rank_manifold.stable_rank_final) / self.rank_manifold.stable_rank_init
        return drop > 0.3 or self.rank_manifold.collapse_detected


class ResultsTable:
    """Master results table for experiment comparison.

    Example:
        >>> table = ResultsTable()
        >>> table.add_result("FP8_E5M2", "adamw", classifier_metrics, rank_metrics, training_metrics, throughput)
        >>> table.add_result("FP8_E5M2", "manifold", classifier_metrics, rank_metrics, training_metrics, throughput)
        >>> print(table.to_plaintext())
        >>> table.to_csv("results.csv")
    """

    # All 8-bit formats (E+M=7)
    FP8_FORMATS = [
        "FP8_E0M7", "FP8_E1M6", "FP8_E2M5", "FP8_E3M4",
        "FP8_E4M3", "FP8_E5M2", "FP8_E6M1", "FP8_E7M0",
    ]

    # All 16-bit formats (E+M=15)
    FP16_FORMATS = [
        "FP16_E0M15", "FP16_E1M14", "FP16_E2M13", "FP16_E3M12",
        "FP16_E4M11", "FP16", "FP16_E6M9", "FP16_E7M8",
        "BF16", "FP16_E9M6", "FP16_E10M5", "FP16_E11M4",
        "FP16_E12M3", "FP16_E13M2", "FP16_E14M1", "FP16_E15M0",
    ]

    # Default: practical subset for experiments
    FORMATS = FP8_FORMATS + ["FP16", "BF16"]

    def __init__(self, formats: list = None):
        """Initialize results table.

        Args:
            formats: List of format names to track. Defaults to FORMATS.
        """
        self.rows: Dict[str, ExperimentRow] = {}
        formats = formats or self.FORMATS
        for fmt in formats:
            self.rows[fmt] = ExperimentRow(format=fmt)

    def _parse_format_name(self, name: str) -> tuple:
        """Parse format name to extract bits, exponent, mantissa.

        Returns:
            Tuple of (total_bits, exponent_bits, mantissa_bits)
        """
        import re
        # Handle standard names
        if name == "FP16":
            return (16, 5, 10)
        if name == "BF16":
            return (16, 8, 7)

        # Parse FP8_E*M* or FP16_E*M* pattern
        match = re.match(r"FP(\d+)_E(\d+)M(\d+)", name)
        if match:
            bits = int(match.group(1))
            exp = int(match.group(2))
            man = int(match.group(3))
            return (bits, exp, man)

        # Legacy E*M* pattern (8-bit)
        match = re.match(r"E(\d+)M(\d+)", name)
        if match:
            exp = int(match.group(1))
            man = int(match.group(2))
            return (8, exp, man)

        return (0, 0, 0)

    def add_result(
        self,
        format: str,
        optimizer: str,  # "adamw" or "manifold"
        classifier: ClassifierMetrics,
        rank: RankMetrics,
        training: TrainingMetrics,
        throughput: ThroughputMetrics,
    ) -> None:
        """Add experiment result to table.

        Args:
            format: FP8 format name
            optimizer: "adamw" or "manifold"
            classifier: Classification metrics
            rank: Rank health metrics
            training: Training efficiency metrics
            throughput: Throughput metrics
        """
        if format not in self.rows:
            self.rows[format] = ExperimentRow(format=format)

        row = self.rows[format]
        row.throughput = throughput

        if optimizer == "adamw":
            row.classifier_adamw = classifier
            row.rank_adamw = rank
            row.training_adamw = training
        else:
            row.classifier_manifold = classifier
            row.rank_manifold = rank
            row.training_manifold = training

    def to_plaintext(self, formats: list = None) -> str:
        """Generate plaintext table.

        Args:
            formats: List of format names to include. Defaults to self.FORMATS.
        """
        formats = formats or self.FORMATS
        lines = []

        # Header
        lines.append("=" * 200)
        lines.append("ALTGRAD EXPERIMENT RESULTS - Floating-Point Format Comparison with Manifold-Aware Optimization")
        lines.append("=" * 200)
        lines.append("")

        # Section 1: Throughput
        lines.append("THROUGHPUT")
        lines.append("-" * 50)
        lines.append(f"{'Format':<12} {'Bits':>4} {'E':>3} {'M':>3} {'sam/sec':>10} {'tok/sec':>12}")
        lines.append("-" * 50)
        for fmt in formats:
            if fmt not in self.rows:
                continue
            row = self.rows[fmt]
            # Extract E/M from format name
            bits, exp, man = self._parse_format_name(fmt)
            lines.append(f"{fmt:<12} {bits:>4} {exp:>3} {man:>3} {row.throughput.samples_per_sec:>10.1f} {row.throughput.tokens_per_sec:>12.1f}")
        lines.append("")

        # Section 2: Classifier Performance
        lines.append("CLASSIFIER PERFORMANCE")
        lines.append("-" * 130)
        lines.append(f"{'Format':<12} {'|':^3} {'F1-mi':>7} {'F1-ma':>7} {'Prec':>7} {'Rec':>7} {'Subset':>7} {'|':^3} {'F1-mi':>7} {'F1-ma':>7} {'Prec':>7} {'Rec':>7} {'Subset':>7} {'|':^3} {'Δ F1':>7}")
        lines.append(f"{'':12} {'|':^3} {'----------- AdamW -----------':^42} {'|':^3} {'------- ManifoldAdamW -------':^42} {'|':^3} {'':>7}")
        lines.append("-" * 130)
        for fmt in formats:
            if fmt not in self.rows:
                continue
            row = self.rows[fmt]
            a = row.classifier_adamw
            m = row.classifier_manifold
            delta = row.f1_delta
            delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
            lines.append(
                f"{fmt:<12} | {a.f1_micro:>7.4f} {a.f1_macro:>7.4f} {a.precision_micro:>7.4f} {a.recall_micro:>7.4f} {a.subset_accuracy:>7.4f} "
                f"| {m.f1_micro:>7.4f} {m.f1_macro:>7.4f} {m.precision_micro:>7.4f} {m.recall_micro:>7.4f} {m.subset_accuracy:>7.4f} "
                f"| {delta_str:>7}"
            )
        lines.append("")

        # Section 3: Rank Health
        lines.append("RANK HEALTH")
        lines.append("-" * 110)
        lines.append(f"{'Format':<12} {'|':^3} {'Stable init':>11} {'Stable fin':>11} {'Eff init':>11} {'Eff fin':>11} {'Collapse':>8} {'|':^3} {'Stable init':>11} {'Stable fin':>11} {'Collapse':>8}")
        lines.append(f"{'':12} {'|':^3} {'------------- AdamW -------------':^50} {'|':^3} {'------- ManifoldAdamW -------':^35}")
        lines.append("-" * 110)
        for fmt in formats:
            if fmt not in self.rows:
                continue
            row = self.rows[fmt]
            ra = row.rank_adamw
            rm = row.rank_manifold
            ca = "YES" if row.rank_collapse_adamw else "no"
            cm = "YES" if row.rank_collapse_manifold else "no"
            lines.append(
                f"{fmt:<12} | {ra.stable_rank_init:>11.2f} {ra.stable_rank_final:>11.2f} {ra.effective_rank_init:>11.2f} {ra.effective_rank_final:>11.2f} {ca:>8} "
                f"| {rm.stable_rank_init:>11.2f} {rm.stable_rank_final:>11.2f} {cm:>8}"
            )
        lines.append("")

        # Section 4: Training Efficiency
        lines.append("TRAINING EFFICIENCY")
        lines.append("-" * 110)
        lines.append(f"{'Format':<12} {'|':^3} {'Time(s)':>10} {'Flips':>12} {'Stall%':>8} {'|':^3} {'Time(s)':>10} {'Flips':>12} {'Stall%':>8} {'|':^3} {'Δ Stall':>8}")
        lines.append(f"{'':12} {'|':^3} {'---------- AdamW ----------':^35} {'|':^3} {'------- ManifoldAdamW -------':^35} {'|':^3}")
        lines.append("-" * 110)
        for fmt in formats:
            if fmt not in self.rows:
                continue
            row = self.rows[fmt]
            ta = row.training_adamw
            tm = row.training_manifold

            # N/A for 16-bit formats (no simulated quantization)
            bits, _, _ = self._parse_format_name(fmt)
            if bits == 16:
                flips_a = "N/A"
                flips_m = "N/A"
                stall_a = "N/A"
                stall_m = "N/A"
                delta_stall = "N/A"
            else:
                flips_a = f"{ta.total_flips:>12,}"
                flips_m = f"{tm.total_flips:>12,}"
                stall_a = f"{ta.mean_stall_ratio*100:>7.1f}%"
                stall_m = f"{tm.mean_stall_ratio*100:>7.1f}%"
                delta_stall = f"{row.stall_delta*100:+.1f}%"

            lines.append(
                f"{fmt:<12} | {ta.total_time_sec:>10.1f} {flips_a:>12} {stall_a:>8} "
                f"| {tm.total_time_sec:>10.1f} {flips_m:>12} {stall_m:>8} "
                f"| {delta_stall:>8}"
            )
        lines.append("")

        # Section 5: Summary / Research Answer
        lines.append("=" * 110)
        lines.append("RESEARCH QUESTION: Which format benefits most from ManifoldAdamW?")
        lines.append("-" * 110)

        # Find best delta per bit width
        best_8bit = (None, -float('inf'))
        best_16bit = (None, -float('inf'))

        for fmt in formats:
            if fmt not in self.rows:
                continue
            row = self.rows[fmt]
            bits, _, _ = self._parse_format_name(fmt)
            if bits == 8 and row.f1_delta > best_8bit[1]:
                best_8bit = (fmt, row.f1_delta)
            elif bits == 16 and row.f1_delta > best_16bit[1]:
                best_16bit = (fmt, row.f1_delta)

        lines.append(f"{'Format':<12} {'Bits':>4} {'E':>3} {'M':>3} {'F1 Delta':>10} {'Stall Delta':>12} {'Rank Collapse?':>15}")
        lines.append("-" * 65)
        for fmt in formats:
            if fmt not in self.rows:
                continue
            row = self.rows[fmt]
            bits, exp, man = self._parse_format_name(fmt)

            collapse = "AdamW" if row.rank_collapse_adamw and not row.rank_collapse_manifold else \
                       "Both" if row.rank_collapse_adamw and row.rank_collapse_manifold else \
                       "Manifold" if row.rank_collapse_manifold else "Neither"

            if bits == 16:
                stall_str = "N/A"
            else:
                stall_str = f"{row.stall_delta*100:+.1f}%"

            marker = ""
            if fmt == best_8bit[0] and best_8bit[1] > 0:
                marker = " <-- BEST 8-bit"
            elif fmt == best_16bit[0] and best_16bit[1] > 0:
                marker = " <-- BEST 16-bit"

            lines.append(f"{fmt:<12} {bits:>4} {exp:>3} {man:>3} {row.f1_delta:>+10.4f} {stall_str:>12} {collapse:>15}{marker}")

        lines.append("")
        lines.append("SUMMARY:")
        if best_8bit[0] and best_8bit[1] > 0:
            lines.append(f"  8-bit:  {best_8bit[0]} benefits most (F1 improvement: {best_8bit[1]:+.4f})")
        else:
            lines.append("  8-bit:  No format showed improvement from ManifoldAdamW")
        if best_16bit[0] and best_16bit[1] > 0:
            lines.append(f"  16-bit: {best_16bit[0]} benefits most (F1 improvement: {best_16bit[1]:+.4f})")
        else:
            lines.append("  16-bit: No format showed improvement from ManifoldAdamW")
        lines.append("=" * 110)

        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """Export to CSV."""
        import csv

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "Format",
                "Throughput_sam_sec", "Throughput_tok_sec",
                "AdamW_F1_micro", "AdamW_F1_macro", "AdamW_Precision", "AdamW_Recall", "AdamW_Subset",
                "Manifold_F1_micro", "Manifold_F1_macro", "Manifold_Precision", "Manifold_Recall", "Manifold_Subset",
                "AdamW_StableRank_init", "AdamW_StableRank_final", "AdamW_EffRank_init", "AdamW_EffRank_final", "AdamW_Collapse",
                "Manifold_StableRank_init", "Manifold_StableRank_final", "Manifold_Collapse",
                "AdamW_Time_sec", "AdamW_Flips", "AdamW_Stall_pct",
                "Manifold_Time_sec", "Manifold_Flips", "Manifold_Stall_pct",
                "F1_Delta", "Stall_Delta"
            ])

            for fmt in self.FORMATS:
                row = self.rows[fmt]
                writer.writerow([
                    fmt,
                    row.throughput.samples_per_sec, row.throughput.tokens_per_sec,
                    row.classifier_adamw.f1_micro, row.classifier_adamw.f1_macro,
                    row.classifier_adamw.precision_micro, row.classifier_adamw.recall_micro, row.classifier_adamw.subset_accuracy,
                    row.classifier_manifold.f1_micro, row.classifier_manifold.f1_macro,
                    row.classifier_manifold.precision_micro, row.classifier_manifold.recall_micro, row.classifier_manifold.subset_accuracy,
                    row.rank_adamw.stable_rank_init, row.rank_adamw.stable_rank_final,
                    row.rank_adamw.effective_rank_init, row.rank_adamw.effective_rank_final, row.rank_collapse_adamw,
                    row.rank_manifold.stable_rank_init, row.rank_manifold.stable_rank_final, row.rank_collapse_manifold,
                    row.training_adamw.total_time_sec, row.training_adamw.total_flips, row.training_adamw.mean_stall_ratio,
                    row.training_manifold.total_time_sec, row.training_manifold.total_flips, row.training_manifold.mean_stall_ratio,
                    row.f1_delta, row.stall_delta
                ])

    def to_json(self, path: str) -> None:
        """Export to JSON."""
        data = {}
        for fmt in self.FORMATS:
            row = self.rows[fmt]
            data[fmt] = {
                "throughput": {"samples_per_sec": row.throughput.samples_per_sec, "tokens_per_sec": row.throughput.tokens_per_sec},
                "adamw": {
                    "classifier": row.classifier_adamw.__dict__,
                    "rank": row.rank_adamw.__dict__,
                    "training": row.training_adamw.__dict__,
                },
                "manifold": {
                    "classifier": row.classifier_manifold.__dict__,
                    "rank": row.rank_manifold.__dict__,
                    "training": row.training_manifold.__dict__,
                },
                "derived": {
                    "f1_delta": row.f1_delta,
                    "stall_delta": row.stall_delta,
                    "rank_collapse_adamw": row.rank_collapse_adamw,
                    "rank_collapse_manifold": row.rank_collapse_manifold,
                }
            }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def demo_table(formats: list = None) -> str:
    """Generate demo table with placeholder data.

    Args:
        formats: List of formats to include. Defaults to practical subset.
    """
    # Default practical subset
    if formats is None:
        formats = [
            # 8-bit
            "FP8_E0M7", "FP8_E2M5", "FP8_E4M3", "FP8_E5M2", "FP8_E7M0",
            # 16-bit
            "FP16_E4M11", "FP16", "BF16",
        ]

    table = ResultsTable(formats=formats)

    # Add fake data to show format
    for fmt in formats:
        bits = 16 if "FP16" in fmt or fmt == "BF16" else 8
        is_16bit = (bits == 16)

        table.add_result(
            format=fmt,
            optimizer="adamw",
            classifier=ClassifierMetrics(f1_micro=0.65, f1_macro=0.42, precision_micro=0.70, recall_micro=0.61, subset_accuracy=0.12),
            rank=RankMetrics(stable_rank_init=45.2, stable_rank_final=38.1, effective_rank_init=32.5, effective_rank_final=28.9),
            training=TrainingMetrics(total_time_sec=1842.5, total_flips=0 if is_16bit else 1250000, mean_stall_ratio=0 if is_16bit else 0.35),
            throughput=ThroughputMetrics(samples_per_sec=125.4, tokens_per_sec=64205),
        )
        table.add_result(
            format=fmt,
            optimizer="manifold",
            classifier=ClassifierMetrics(f1_micro=0.68, f1_macro=0.45, precision_micro=0.72, recall_micro=0.64, subset_accuracy=0.14),
            rank=RankMetrics(stable_rank_init=45.2, stable_rank_final=42.1, effective_rank_init=32.5, effective_rank_final=31.2),
            training=TrainingMetrics(total_time_sec=1920.3, total_flips=0 if is_16bit else 980000, mean_stall_ratio=0 if is_16bit else 0.22),
            throughput=ThroughputMetrics(samples_per_sec=125.4, tokens_per_sec=64205),
        )

    return table.to_plaintext(formats=formats)


__all__ = [
    "ThroughputMetrics",
    "ClassifierMetrics",
    "RankMetrics",
    "TrainingMetrics",
    "ExperimentRow",
    "ResultsTable",
    "demo_table",
]
