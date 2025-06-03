import torch
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_cm
from typing import List, Tuple

def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 5
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices (N,)
        targets: True class indices (N,)
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    return sklearn_cm(targets.numpy(), predictions.numpy(), labels=list(range(num_classes)))

def compute_per_class_accuracy(confusion_matrix: np.ndarray) -> List[float]:
    """
    Compute per-class accuracy from confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix
        
    Returns:
        List of per-class accuracies
    """
    per_class_acc = []
    
    for i in range(confusion_matrix.shape[0]):
        if confusion_matrix[i].sum() > 0:
            acc = confusion_matrix[i, i] / confusion_matrix[i].sum()
        else:
            acc = 0.0
        per_class_acc.append(acc)
    
    return per_class_acc

def compute_rotation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 5
) -> dict:
    """
    Compute comprehensive rotation classification metrics.
    
    Args:
        predictions: Model predictions (N, num_classes)
        targets: True labels (N,)
        num_classes: Number of rotation classes
        
    Returns:
        Dictionary of metrics
    """
    # Get predicted classes
    pred_classes = predictions.argmax(dim=1)
    
    # Overall accuracy
    accuracy = (pred_classes == targets).float().mean().item()
    
    # Confusion matrix
    cm = compute_confusion_matrix(pred_classes, targets, num_classes)
    
    # Per-class metrics
    per_class_acc = compute_per_class_accuracy(cm)
    
    # Confidence statistics
    probs = torch.softmax(predictions, dim=1)
    confidence_correct = probs[torch.arange(len(targets)), targets].mean().item()
    confidence_incorrect = probs[pred_classes != targets].max(dim=1)[0].mean().item() if (pred_classes != targets).any() else 0.0
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc,
        'confidence_correct': confidence_correct,
        'confidence_incorrect': confidence_incorrect
    }