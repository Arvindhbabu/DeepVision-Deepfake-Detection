"""
DeepVision AI

Balanced Sampler

Creates a WeightedRandomSampler to address class imbalance during training.
"""

import torch
from torch.utils.data import WeightedRandomSampler


def create_balanced_sampler(dataset):
    """
    Create a weighted random sampler for SequenceDataset.

    Args:
        dataset (SequenceDataset)

    Returns:
        WeightedRandomSampler or None
    """
    if len(dataset) == 0:
        return None

    if "label" not in dataset.data.columns:
        return None

    labels = dataset.data["label"].tolist()

    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    class_weights = {}
    for cls, count in class_counts.items():
        class_weights[cls] = 1.0 / count if count > 0 else 1.0

    sample_weights = [class_weights.get(label, 1.0) for label in labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


class BalancedSampler:
    """
    Class wrapper for WeightedRandomSampler creation.
    """
    def __new__(cls, dataset):
        return create_balanced_sampler(dataset)