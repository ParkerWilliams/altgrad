"""Analysis module for synthesizing W&B experiment data.

Provides tools for aggregating experiment results from Phases 4-5,
computing format comparisons, analyzing failure modes, and generating
markdown reports that answer the core research question.

Submodules:
    data_loader: W&B API data fetching with format/optimizer filtering
    comparisons: Format and optimizer comparison logic with ranking
    failure_analysis: Failure mode extraction and classification
    report_generator: Markdown report generation for analysis artifacts

Example:
    >>> from altgrad.analysis import ExperimentDataLoader, ReportGenerator
    >>> loader = ExperimentDataLoader(project="altgrad")
    >>> runs = loader.get_format_comparison_runs()
    >>> generator = ReportGenerator()
    >>> generator.generate_format_comparison(runs, "reports/ANAL-01.md")
"""

from altgrad.analysis.data_loader import ExperimentDataLoader
from altgrad.analysis.comparisons import FormatComparator
from altgrad.analysis.failure_analysis import FailureAnalyzer
from altgrad.analysis.report_generator import ReportGenerator

__all__ = [
    "ExperimentDataLoader",
    "FormatComparator",
    "FailureAnalyzer",
    "ReportGenerator",
]
