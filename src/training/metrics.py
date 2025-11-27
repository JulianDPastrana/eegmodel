"""Evaluation metrics for model performance."""

from typing import Dict

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute accuracy score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return accuracy_score(y_true, y_pred)


def precision_recall_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    average: str = "weighted",
) -> Dict[str, float]:
    """Compute precision, recall, and F1-score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('binary', 'micro', 'macro', 'weighted')
    
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> np.ndarray:
    """Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return confusion_matrix(y_true, y_pred)


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    average: str = "weighted",
) -> Dict[str, float]:
    """Compute all common classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Accuracy
    metrics["accuracy"] = accuracy(y_true, y_pred)
    
    # Precision, Recall, F1
    prf = precision_recall_f1(y_true, y_pred, average=average)
    metrics.update(prf)
    
    return metrics


def print_classification_report(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    target_names: list = None,
) -> None:
    """Print detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: Optional list of class names
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    print(classification_report(y_true, y_pred, target_names=target_names))


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Create dummy predictions
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([0, 1, 2, 0, 2, 1, 0, 1, 2])
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    print(f"✓ Metrics: {metrics}")
    
    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    print(f"✓ Confusion Matrix:\n{cm}")
    
    # Classification report
    print("✓ Classification Report:")
    print_classification_report(y_true, y_pred, target_names=["Class 0", "Class 1", "Class 2"])
