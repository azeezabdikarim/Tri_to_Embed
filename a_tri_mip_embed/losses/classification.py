import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class RotationClassificationLoss(nn.Module):
    """
    Cross-entropy loss for rotation classification.
    
    Supports:
    - Standard cross-entropy
    - Label smoothing
    - Class weighting
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification loss.
        
        Args:
            predictions: Logits of shape (B, num_classes)
            targets: Labels of shape (B,)
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        # Compute cross-entropy loss
        if self.label_smoothing > 0:
            # Create smoothed targets
            with torch.no_grad():
                true_dist = torch.zeros_like(predictions)
                true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # KL divergence loss
            loss = F.kl_div(
                F.log_softmax(predictions, dim=1),
                true_dist,
                reduction='batchmean'
            )
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(
                predictions,
                targets,
                weight=self.class_weights
            )
        
        # Compute accuracy
        with torch.no_grad():
            pred_classes = predictions.argmax(dim=1)
            accuracy = (pred_classes == targets).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }