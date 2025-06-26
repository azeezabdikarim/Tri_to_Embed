"""
MLP Base Encoder for Geometry/Density Network

Handles the base MLP that processes geometric features:
Architecture: 48 → 128 → 128 → 16
- Layer 0: 48 → 128 
- Layer 1: 128 → 128
- Layer 2: 128 → 16
"""

import torch
import torch.nn as nn
from typing import List

from .inr2vec_encoder import BaseINR2VecEncoder
from .weight_utils import validate_mlp_structure


class MLPBaseEncoder(BaseINR2VecEncoder):
    """
    Encoder for the geometry/density MLP (MLP Base).
    
    Expected architecture: 48 → 128 → 128 → 16
    """
    
    def __init__(
        self,
        projection_dim: int = 256,
        final_embedding_dim: int = 128,
        pooling_strategy: str = 'max',
        layer_fusion: str = 'concat',
        include_bias: bool = True,
        dropout: float = 0.1
    ):
        # Define expected layer dimensions for MLP Base
        layer_dimensions = [
            (48, 128),   # Layer 0: 48 → 128
            (128, 128),  # Layer 1: 128 → 128  
            (128, 16),   # Layer 2: 128 → 16
        ]
        
        super().__init__(
            layer_dimensions=layer_dimensions,
            projection_dim=projection_dim,
            final_embedding_dim=final_embedding_dim,
            pooling_strategy=pooling_strategy,
            layer_fusion=layer_fusion,
            include_bias=include_bias,
            dropout=dropout
        )
    
    def get_expected_architecture(self) -> List[int]:
        """Return expected architecture: [48, 128, 128, 16]"""
        return [48, 128, 128, 16]
    
    def forward(self, mlp_dict: dict) -> torch.Tensor:
        """
        Forward pass with validation for MLP Base structure.
        
        Args:
            mlp_dict: Structured MLP base dictionary
            
        Returns:
            embedding: Embedding tensor of shape (final_embedding_dim,) or (batch_size, final_embedding_dim)
        """
        # Validate structure
        if not validate_mlp_structure(mlp_dict, self.get_expected_architecture()):
            print(f"Warning: MLP Base structure doesn't match expected architecture {self.get_expected_architecture()}")
            # Continue anyway - might be a variant
        
        return super().forward(mlp_dict)


def create_mlp_base_encoder(config: dict) -> MLPBaseEncoder:
    """
    Create MLP Base encoder from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MLPBaseEncoder instance
    """
    encoder_config = config['model']['mlp_encoders']['base_mlp']
    
    return MLPBaseEncoder(
        projection_dim=encoder_config.get('projection_dim', 256),
        final_embedding_dim=encoder_config.get('embedding_dim', 128),
        pooling_strategy=config['model']['mlp_encoders'].get('pooling_strategy', 'max'),
        layer_fusion=config['model']['fusion'].get('layer_fusion', 'concat'),
        include_bias=encoder_config.get('include_bias', True),
        dropout=config['model']['fusion'].get('dropout', 0.1)
    )