import torch
import torch.nn as nn
from typing import Tuple

class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion for plane embeddings.
    """
    
    def forward(self, embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            embeddings: Tuple of 3 tensors, each (B, embedding_dim)
            
        Returns:
            fused: Tensor of shape (B, 3 * embedding_dim)
        """
        return torch.cat(embeddings, dim=1)

class SumFusion(nn.Module):
    """
    Weighted sum fusion with learnable weights.
    """
    
    def __init__(self):
        super().__init__()
        # Learnable weights for each plane
        self.weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            embeddings: Tuple of 3 tensors, each (B, embedding_dim)
            
        Returns:
            fused: Tensor of shape (B, embedding_dim)
        """
        # Normalize weights
        weights = torch.softmax(self.weights, dim=0)
        
        # Weighted sum
        fused = sum(w * emb for w, emb in zip(weights, embeddings))
        
        return fused

class AttentionFusion(nn.Module):
    """
    Attention-based fusion for plane embeddings.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Final projection
        self.projection = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            embeddings: Tuple of 3 tensors, each (B, embedding_dim)
            
        Returns:
            fused: Tensor of shape (B, embedding_dim)
        """
        # Stack embeddings to create sequence
        # Shape: (B, 3, embedding_dim)
        x = torch.stack(embeddings, dim=1)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        
        # Residual connection and norm
        x = self.norm(x + attn_out)
        
        # Global pooling (mean across planes)
        x = x.mean(dim=1)  # (B, embedding_dim)
        
        # Final projection
        fused = self.projection(x)
        
        return fused

class LearnedTokenFusion(nn.Module):
    """
    Fusion using a learned global token (similar to CLS token in ViT).
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Learned fusion token
        self.fusion_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
    
    def forward(self, embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            embeddings: Tuple of 3 tensors, each (B, embedding_dim)
            
        Returns:
            fused: Tensor of shape (B, embedding_dim)
        """
        B = embeddings[0].shape[0]
        
        # Stack embeddings
        x = torch.stack(embeddings, dim=1)  # (B, 3, embedding_dim)
        
        # Add fusion token
        fusion_token = self.fusion_token.expand(B, -1, -1)  # (B, 1, embedding_dim)
        x = torch.cat([fusion_token, x], dim=1)  # (B, 4, embedding_dim)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Extract fusion token output
        fused = x[:, 0]  # (B, embedding_dim)
        
        return fused