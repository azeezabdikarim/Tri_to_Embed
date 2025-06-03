import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class EmbeddingMSELoss(nn.Module):
    """
    MSE loss for embeddings, useful for continuous metrics.
    
    Can be used for:
    - Embedding reconstruction
    - Continuous rotation prediction
    - Feature matching
    """
    
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        pred_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MSE loss between embeddings.
        
        Args:
            pred_embedding: Predicted embeddings (B, D)
            target_embedding: Target embeddings (B, D)
            
        Returns:
            Dictionary with 'loss' and 'l2_distance'
        """
        if self.normalize:
            # Normalize embeddings to unit sphere
            pred_embedding = F.normalize(pred_embedding, p=2, dim=1)
            target_embedding = F.normalize(target_embedding, p=2, dim=1)
        
        # MSE loss
        loss = F.mse_loss(pred_embedding, target_embedding)
        
        # L2 distance for monitoring
        with torch.no_grad():
            l2_distance = torch.norm(pred_embedding - target_embedding, p=2, dim=1).mean()
        
        return {
            'loss': loss,
            'l2_distance': l2_distance
        }