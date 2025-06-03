import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple

class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for all plane encoders.
    """
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    @abstractmethod
    def encode_single_plane(self, plane: torch.Tensor) -> torch.Tensor:
        """
        Encode a single plane into features.
        
        Args:
            plane: Tensor of shape (B, 512, 512, 16)
            
        Returns:
            features: Tensor of shape (B, feature_dim)
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the dimension of features before final projection."""
        pass

class BasePairClassifier(nn.Module):
    """
    Base class for rotation classification from NeRF pairs.
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        plane_fusion: str = 'concat',
        pair_combination: str = 'subtract',
        num_classes: int = 5
    ):
        super().__init__()
        self.encoder = encoder
        self.plane_fusion = plane_fusion
        self.pair_combination = pair_combination
        self.num_classes = num_classes
        
        # Calculate dimensions based on fusion and combination strategies
        single_embed_dim = self._calculate_single_embed_dim()
        pair_embed_dim = self._calculate_pair_embed_dim(single_embed_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(pair_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def _calculate_single_embed_dim(self) -> int:
        """Calculate embedding dimension for a single NeRF based on plane fusion."""
        if self.plane_fusion == 'concat':
            return self.encoder.embedding_dim * 3
        else:  # sum or attention
            return self.encoder.embedding_dim
    
    def _calculate_pair_embed_dim(self, single_dim: int) -> int:
        """Calculate embedding dimension for the pair based on combination strategy."""
        if self.pair_combination == 'subtract':
            return single_dim
        elif self.pair_combination == 'concat':
            return single_dim * 2
        else:  # both
            return single_dim * 3
    
    @abstractmethod
    def fuse_planes(self, plane_embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Fuse embeddings from three planes into a single embedding.
        
        Args:
            plane_embeddings: Tuple of 3 tensors, each (B, embedding_dim)
            
        Returns:
            fused: Tensor of shape (B, fused_dim)
        """
        pass
    
    def combine_pair_embeddings(
        self,
        base_embed: torch.Tensor,
        aug_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine base and augmented embeddings based on strategy.
        
        Args:
            base_embed: Embedding of base model
            aug_embed: Embedding of augmented model
            
        Returns:
            combined: Combined embedding for classification
        """
        if self.pair_combination == 'subtract':
            return aug_embed - base_embed
        elif self.pair_combination == 'concat':
            return torch.cat([base_embed, aug_embed], dim=1)
        else:  # both
            return torch.cat([
                base_embed,
                aug_embed,
                aug_embed - base_embed
            ], dim=1)
    
    def encode_nerf(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Encode a full NeRF (all three planes) into an embedding.
        
        Args:
            planes: Tensor of shape (B, 3, 512, 512, 16)
            
        Returns:
            embedding: Tensor of shape (B, embed_dim)
        """
        # Encode each plane separately
        plane_embeddings = []
        for i in range(3):
            plane = planes[:, i]  # (B, 512, 512, 16)
            embed = self.encoder.encode_single_plane(plane)
            plane_embeddings.append(embed)
        
        # Fuse plane embeddings
        fused = self.fuse_planes(tuple(plane_embeddings))
        
        return fused
    
    def forward(
        self,
        base_planes: torch.Tensor,
        aug_planes: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for rotation classification.
        
        Args:
            base_planes: Base model planes (B, 3, 512, 512, 16)
            aug_planes: Augmented model planes (B, 3, 512, 512, 16)
            
        Returns:
            logits: Classification logits (B, num_classes)
        """
        # Encode both NeRFs
        base_embed = self.encode_nerf(base_planes)
        aug_embed = self.encode_nerf(aug_planes)
        
        # Combine embeddings
        combined = self.combine_pair_embeddings(base_embed, aug_embed)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits