"""Multi-label classification metrics.

Comprehensive metrics for EUR-Lex document classification:
- F1 (micro, macro, weighted, samples)
- Precision/Recall (micro, macro, weighted)
- Hamming loss, subset accuracy
- Ranking metrics (coverage error, ranking loss, label ranking AP)
- ROC-AUC, PR-AUC

All metrics work with multi-hot label vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor
import numpy as np


@dataclass
class ClassificationMetrics:
    """Container for all classification metrics."""

    # Core F1 variants
    f1_micro: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    f1_samples: float = 0.0

    # Precision variants
    precision_micro: float = 0.0
    precision_macro: float = 0.0
    precision_weighted: float = 0.0

    # Recall variants
    recall_micro: float = 0.0
    recall_macro: float = 0.0
    recall_weighted: float = 0.0

    # Multi-label specific
    hamming_loss: float = 0.0
    subset_accuracy: float = 0.0  # exact match

    # Ranking metrics
    coverage_error: float = 0.0
    ranking_loss: float = 0.0
    label_ranking_ap: float = 0.0

    # AUC metrics
    roc_auc_micro: float = 0.0
    roc_auc_macro: float = 0.0
    pr_auc_micro: float = 0.0
    pr_auc_macro: float = 0.0

    # Sample counts
    num_samples: int = 0
    num_labels: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for logging."""
        return {
            "f1_micro": self.f1_micro,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "f1_samples": self.f1_samples,
            "precision_micro": self.precision_micro,
            "precision_macro": self.precision_macro,
            "precision_weighted": self.precision_weighted,
            "recall_micro": self.recall_micro,
            "recall_macro": self.recall_macro,
            "recall_weighted": self.recall_weighted,
            "hamming_loss": self.hamming_loss,
            "subset_accuracy": self.subset_accuracy,
            "coverage_error": self.coverage_error,
            "ranking_loss": self.ranking_loss,
            "label_ranking_ap": self.label_ranking_ap,
            "roc_auc_micro": self.roc_auc_micro,
            "roc_auc_macro": self.roc_auc_macro,
            "pr_auc_micro": self.pr_auc_micro,
            "pr_auc_macro": self.pr_auc_macro,
        }


class MetricsComputer:
    """Compute classification metrics from predictions and labels.

    Accumulates predictions over batches, then computes all metrics.
    Supports both hard predictions (thresholded) and soft scores (for AUC).

    Example:
        >>> computer = MetricsComputer(num_labels=100)
        >>> for batch in dataloader:
        ...     logits = model(batch)
        ...     computer.update(logits, batch["labels"])
        >>> metrics = computer.compute()
        >>> print(f"F1 micro: {metrics.f1_micro:.4f}")
    """

    def __init__(self, num_labels: int, threshold: float = 0.5):
        """Initialize metrics computer.

        Args:
            num_labels: Total number of labels
            threshold: Threshold for converting logits to binary predictions
        """
        self.num_labels = num_labels
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """Clear accumulated predictions."""
        self.all_preds: List[Tensor] = []
        self.all_labels: List[Tensor] = []
        self.all_scores: List[Tensor] = []  # For AUC computation

    def update(self, logits: Tensor, labels: Tensor) -> None:
        """Accumulate batch predictions.

        Args:
            logits: Raw model outputs (batch, num_labels)
            labels: Ground truth multi-hot labels (batch, num_labels)
        """
        # Sigmoid for multi-label
        scores = torch.sigmoid(logits).detach().cpu()
        preds = (scores >= self.threshold).float()

        self.all_scores.append(scores)
        self.all_preds.append(preds)
        self.all_labels.append(labels.detach().cpu())

    def compute(self) -> ClassificationMetrics:
        """Compute all metrics from accumulated predictions.

        Returns:
            ClassificationMetrics with all computed values
        """
        if not self.all_preds:
            return ClassificationMetrics()

        # Concatenate all batches
        preds = torch.cat(self.all_preds, dim=0)  # (N, L)
        labels = torch.cat(self.all_labels, dim=0)  # (N, L)
        scores = torch.cat(self.all_scores, dim=0)  # (N, L)

        n_samples, n_labels = preds.shape

        metrics = ClassificationMetrics(
            num_samples=n_samples,
            num_labels=n_labels,
        )

        # Compute confusion matrix components per label
        tp = (preds * labels).sum(dim=0)  # (L,)
        fp = (preds * (1 - labels)).sum(dim=0)  # (L,)
        fn = ((1 - preds) * labels).sum(dim=0)  # (L,)
        tn = ((1 - preds) * (1 - labels)).sum(dim=0)  # (L,)

        # Per-label metrics
        label_precision = tp / (tp + fp + 1e-10)
        label_recall = tp / (tp + fn + 1e-10)
        label_f1 = 2 * label_precision * label_recall / (label_precision + label_recall + 1e-10)

        # Label support (for weighted averaging)
        label_support = labels.sum(dim=0)  # (L,)
        total_support = label_support.sum()

        # ===== F1 Variants =====

        # Micro: aggregate TP/FP/FN across all labels
        micro_tp = tp.sum()
        micro_fp = fp.sum()
        micro_fn = fn.sum()
        micro_precision = micro_tp / (micro_tp + micro_fp + 1e-10)
        micro_recall = micro_tp / (micro_tp + micro_fn + 1e-10)
        metrics.f1_micro = (2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-10)).item()
        metrics.precision_micro = micro_precision.item()
        metrics.recall_micro = micro_recall.item()

        # Macro: average per-label metrics
        metrics.f1_macro = label_f1.mean().item()
        metrics.precision_macro = label_precision.mean().item()
        metrics.recall_macro = label_recall.mean().item()

        # Weighted: weighted by label support
        if total_support > 0:
            weights = label_support / total_support
            metrics.f1_weighted = (label_f1 * weights).sum().item()
            metrics.precision_weighted = (label_precision * weights).sum().item()
            metrics.recall_weighted = (label_recall * weights).sum().item()

        # Samples: average per-sample F1
        sample_tp = (preds * labels).sum(dim=1)  # (N,)
        sample_fp = (preds * (1 - labels)).sum(dim=1)
        sample_fn = ((1 - preds) * labels).sum(dim=1)
        sample_precision = sample_tp / (sample_tp + sample_fp + 1e-10)
        sample_recall = sample_tp / (sample_tp + sample_fn + 1e-10)
        sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall + 1e-10)
        metrics.f1_samples = sample_f1.mean().item()

        # ===== Multi-label Specific =====

        # Hamming loss: fraction of wrong labels
        metrics.hamming_loss = ((preds != labels).float().sum() / (n_samples * n_labels)).item()

        # Subset accuracy: exact match (all labels correct)
        exact_match = (preds == labels).all(dim=1).float()
        metrics.subset_accuracy = exact_match.mean().item()

        # ===== Ranking Metrics =====
        metrics.coverage_error = self._coverage_error(scores, labels)
        metrics.ranking_loss = self._ranking_loss(scores, labels)
        metrics.label_ranking_ap = self._label_ranking_ap(scores, labels)

        # ===== AUC Metrics =====
        roc_micro, roc_macro = self._compute_roc_auc(scores, labels)
        metrics.roc_auc_micro = roc_micro
        metrics.roc_auc_macro = roc_macro

        pr_micro, pr_macro = self._compute_pr_auc(scores, labels)
        metrics.pr_auc_micro = pr_micro
        metrics.pr_auc_macro = pr_macro

        return metrics

    def _coverage_error(self, scores: Tensor, labels: Tensor) -> float:
        """Compute coverage error.

        Coverage error measures how far we need to go in the ranked list
        to cover all true labels.
        """
        n_samples = scores.shape[0]
        coverage = 0.0

        for i in range(n_samples):
            if labels[i].sum() == 0:
                continue

            # Rank scores (descending)
            ranks = torch.argsort(torch.argsort(-scores[i])) + 1  # 1-indexed ranks

            # Coverage = max rank among true labels
            true_label_ranks = ranks[labels[i] == 1]
            if len(true_label_ranks) > 0:
                coverage += true_label_ranks.max().item()

        return coverage / max(n_samples, 1)

    def _ranking_loss(self, scores: Tensor, labels: Tensor) -> float:
        """Compute ranking loss.

        Fraction of reversely ordered pairs (positive ranked lower than negative).
        """
        n_samples = scores.shape[0]
        total_loss = 0.0
        total_pairs = 0

        for i in range(n_samples):
            pos_mask = labels[i] == 1
            neg_mask = labels[i] == 0

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_scores = scores[i][pos_mask]
            neg_scores = scores[i][neg_mask]

            # Count pairs where negative score >= positive score
            n_pairs = len(pos_scores) * len(neg_scores)
            n_incorrect = 0
            for ps in pos_scores:
                n_incorrect += (neg_scores >= ps).sum().item()

            total_loss += n_incorrect
            total_pairs += n_pairs

        return total_loss / max(total_pairs, 1)

    def _label_ranking_ap(self, scores: Tensor, labels: Tensor) -> float:
        """Compute label ranking average precision (LRAP).

        Average over samples of: average precision of true labels in ranking.
        """
        n_samples = scores.shape[0]
        ap_sum = 0.0
        valid_samples = 0

        for i in range(n_samples):
            if labels[i].sum() == 0:
                continue

            # Sort by score descending
            sorted_indices = torch.argsort(-scores[i])
            sorted_labels = labels[i][sorted_indices]

            # Compute precision at each true label position
            true_positions = torch.where(sorted_labels == 1)[0]
            if len(true_positions) == 0:
                continue

            precisions = []
            for j, pos in enumerate(true_positions):
                # Precision = (j+1) / (pos+1)  where j is count of true labels seen
                precisions.append((j + 1) / (pos.item() + 1))

            ap_sum += sum(precisions) / len(precisions)
            valid_samples += 1

        return ap_sum / max(valid_samples, 1)

    def _compute_roc_auc(self, scores: Tensor, labels: Tensor) -> tuple:
        """Compute ROC-AUC (micro and macro).

        Returns:
            Tuple of (micro_auc, macro_auc)
        """
        try:
            from sklearn.metrics import roc_auc_score

            scores_np = scores.numpy()
            labels_np = labels.numpy()

            # Filter labels with at least one positive and one negative
            valid_labels = []
            for j in range(labels.shape[1]):
                if labels_np[:, j].sum() > 0 and labels_np[:, j].sum() < len(labels_np):
                    valid_labels.append(j)

            if len(valid_labels) == 0:
                return 0.0, 0.0

            scores_valid = scores_np[:, valid_labels]
            labels_valid = labels_np[:, valid_labels]

            micro = roc_auc_score(labels_valid, scores_valid, average="micro")
            macro = roc_auc_score(labels_valid, scores_valid, average="macro")

            return float(micro), float(macro)
        except Exception:
            return 0.0, 0.0

    def _compute_pr_auc(self, scores: Tensor, labels: Tensor) -> tuple:
        """Compute PR-AUC (micro and macro).

        Returns:
            Tuple of (micro_auc, macro_auc)
        """
        try:
            from sklearn.metrics import average_precision_score

            scores_np = scores.numpy()
            labels_np = labels.numpy()

            # Filter labels with at least one positive
            valid_labels = []
            for j in range(labels.shape[1]):
                if labels_np[:, j].sum() > 0:
                    valid_labels.append(j)

            if len(valid_labels) == 0:
                return 0.0, 0.0

            scores_valid = scores_np[:, valid_labels]
            labels_valid = labels_np[:, valid_labels]

            micro = average_precision_score(labels_valid, scores_valid, average="micro")
            macro = average_precision_score(labels_valid, scores_valid, average="macro")

            return float(micro), float(macro)
        except Exception:
            return 0.0, 0.0


def compute_throughput_metrics(
    batch_size: int,
    seq_length: int,
    step_time_seconds: float,
) -> Dict[str, float]:
    """Compute throughput metrics.

    Args:
        batch_size: Number of samples in batch
        seq_length: Sequence length per sample
        step_time_seconds: Time taken for one step

    Returns:
        Dictionary with throughput metrics
    """
    samples_per_sec = batch_size / step_time_seconds
    tokens_per_sec = batch_size * seq_length / step_time_seconds

    return {
        "throughput/samples_per_sec": samples_per_sec,
        "throughput/tokens_per_sec": tokens_per_sec,
        "throughput/step_time_ms": step_time_seconds * 1000,
    }


__all__ = [
    "ClassificationMetrics",
    "MetricsComputer",
    "compute_throughput_metrics",
]
