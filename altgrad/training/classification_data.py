"""EUR-Lex multi-label classification data loading.

Loads the EUR-Lex dataset for multi-label document classification,
returning input_ids, attention_mask, and multi-hot label vectors.

This is the correct data pipeline for classification experiments,
replacing the language modeling data.py for classification tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class ClassificationBatch:
    """Batch of classification examples."""
    input_ids: Tensor          # (batch, seq_len)
    attention_mask: Tensor     # (batch, seq_len)
    labels: Tensor             # (batch, num_labels) multi-hot


class EURLexDataset(Dataset):
    """EUR-Lex multi-label classification dataset.

    Each example is a legal document with EUROVOC concept labels.
    Labels are encoded as multi-hot vectors.

    Attributes:
        encodings: Tokenized text (input_ids, attention_mask)
        labels: Multi-hot label tensors
        num_labels: Total number of unique labels
        label_to_idx: Mapping from label ID to index
    """

    def __init__(
        self,
        split: str = "train",
        max_length: int = 512,
        max_examples: Optional[int] = None,
        tokenizer_name: str = "distilbert-base-uncased",
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        """Load and prepare EUR-Lex dataset.

        Args:
            split: Dataset split ("train", "validation", "test")
            max_length: Maximum sequence length for tokenization
            max_examples: Limit number of examples (None = all)
            tokenizer_name: HuggingFace tokenizer to use
            label_to_idx: Pre-built label vocabulary (for val/test consistency)
        """
        # Load dataset
        dataset = load_dataset(
            "NLP-AUEB/eurlex",
            "eurlex57k",
            split=split,
            trust_remote_code=True,
        )

        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # EUR-Lex uses "eurovoc_concepts" as the label field
        label_field = "eurovoc_concepts"

        # Use provided vocabulary or build from this split
        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
        else:
            # Build label vocabulary from all labels in dataset
            all_labels = set()
            for example in dataset:
                all_labels.update(example[label_field])
            self.label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_labels = len(self.label_to_idx)

        # Tokenize texts
        texts = [example["text"] for example in dataset]
        self.encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        # Convert labels to multi-hot vectors using the vocabulary
        self.labels = torch.zeros(len(dataset), self.num_labels)
        for i, example in enumerate(dataset):
            for label in example[label_field]:
                if label in self.label_to_idx:
                    self.labels[i, self.label_to_idx[label]] = 1.0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def create_dataloaders(
    batch_size: int = 16,
    max_length: int = 512,
    max_examples: Optional[int] = None,
    tokenizer_name: str = "distilbert-base-uncased",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create train/val/test dataloaders for EUR-Lex.

    Args:
        batch_size: Batch size for all loaders
        max_length: Maximum sequence length
        max_examples: Limit examples per split (None = all)
        tokenizer_name: HuggingFace tokenizer name
        num_workers: DataLoader workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_labels)
    """
    # Train dataset defines the label vocabulary
    train_dataset = EURLexDataset(
        split="train",
        max_length=max_length,
        max_examples=max_examples,
        tokenizer_name=tokenizer_name,
        label_to_idx=None,  # Build vocabulary from train
    )

    # Val/test use train's vocabulary for consistent label indexing
    val_dataset = EURLexDataset(
        split="validation",
        max_length=max_length,
        max_examples=max_examples,
        tokenizer_name=tokenizer_name,
        label_to_idx=train_dataset.label_to_idx,  # Use train vocabulary
    )

    test_dataset = EURLexDataset(
        split="test",
        max_length=max_length,
        max_examples=max_examples,
        tokenizer_name=tokenizer_name,
        label_to_idx=train_dataset.label_to_idx,  # Use train vocabulary
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset.num_labels


__all__ = [
    "ClassificationBatch",
    "EURLexDataset",
    "create_dataloaders",
]
