"""
MLP Head Encoder for Appearance/Color Network

Handles the head MLP that processes appearance features:
Architecture: 40 → 128 → 128 → 128 → 128 → 3
- Layer 0: 40 → 128 (SH degree 4: 25 + geo_feat: 15 = 40)
- Layer 1: 128 → 128
- Layer 2: 128 → 128  
- Layer 3: 128 → 128
- Layer 4: 128 → 3
"""

import torch
import torch.nn as nn
from typing import List

from .inr2vec_encoder import BaseINR2VecEncoder
from .weight_utils import validate_mlp_structure, handle_extra_params


class MLPHeadEncoder(BaseINR2VecEncoder):
    """
    Encoder for the appearance/color MLP (MLP Head).
    
    Expected architecture: 40 → 128 → 128 → 128 → 128 → 3
    Handles extra parameters that might be present in tinycudann MLPs.
    """
    
    def __init__(
        self,
        projection_dim: int = 512,
        final_embedding_dim: int = 256,
        pooling_strategy: str = 'max',
        layer_fusion: str = 'concat',
        include_bias: bool = True,
        handle_extra_params: bool = True,
        dropout: float = 0.1
    ):
        # Define expected layer dimensions for MLP Head
        layer_dimensions = [
            (40, 128),   # Layer 0: 40 → 128 (25 SH + 15 geo_feat)
            (128, 128),  # Layer 1: 128 → 128
            (128, 128),  # Layer 2: 128 → 128
            (128, 128),  # Layer 3: 128 → 128
            (128, 3),    # Layer 4: 128 → 3 (RGB)
        ]
        
        self.handle_extra_params = handle_extra_params
        
        super().__init__(
            layer_dimensions=layer_dimensions,
            projection_dim=projection_dim,
            final_embedding_dim=final_embedding_dim,
            pooling_strategy=pooling_strategy,
            layer_fusion=layer_fusion,
            include_bias=include_bias,
            dropout=dropout
        )
        
        # Extra parameter handler if needed
        if handle_extra_params:
            self.extra_param_encoder = nn.Sequential(
                nn.Linear(125, 64),  # Handle the 125 extra parameters
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32)
            )
            
            # Adjust final fusion to include extra params
            if layer_fusion == 'concat':
                # Need to modify the fusion layer to account for extra params
                original_fusion_dim = self.projection_dim * self.num_layers
                new_fusion_dim = original_fusion_dim + 32  # Add extra param embedding
                
                self.fusion = nn.Sequential(
                    nn.Linear(new_fusion_dim, final_embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
    
    def get_expected_architecture(self) -> List[int]:
        """Return expected architecture: [40, 128, 128, 128, 128, 3]"""
        return [40, 128, 128, 128, 128, 3]
    
    def forward(self, mlp_dict: dict) -> torch.Tensor:
        """
        Forward pass with validation and extra parameter handling.
        
        Args:
            mlp_dict: Structured MLP head dictionary
            
        Returns:
            embedding: Embedding tensor of shape (final_embedding_dim,) or (batch_size, final_embedding_dim)
        """
        # Validate structure (but be flexible about input dimensions)
        expected_arch = self.get_expected_architecture()
        if not validate_mlp_structure(mlp_dict, expected_arch):
            # Try alternative architectures
            alt_archs = [
                [31, 128, 128, 128, 128, 3],  # SH degree 3
                [25, 128, 128, 128, 128, 3],  # Just SH, no geo_feat
            ]
            
            valid_alt = False
            for alt_arch in alt_archs:
                if validate_mlp_structure(mlp_dict, alt_arch):
                    print(f"Note: MLP Head using alternative architecture {alt_arch}")
                    valid_alt = True
                    break
            
            if not valid_alt:
                print(f"Warning: MLP Head structure doesn't match any expected architecture")
        
        # Get layer embeddings using parent method
        # But we need to handle this manually to incorporate extra params
        
        # Extract weight matrices
        from .weight_utils import prepare_weight_matrices
        weight_matrices = prepare_weight_matrices(mlp_dict, include_bias=self.include_bias)
        
        if len(weight_matrices) != self.num_layers:
            # Handle flexible number of layers
            if len(weight_matrices) > self.num_layers:
                print(f"Warning: Got {len(weight_matrices)} layers, expected {self.num_layers}. Using first {self.num_layers}.")
                weight_matrices = weight_matrices[:self.num_layers]
            else:
                raise ValueError(f"Expected at least {self.num_layers} layers, got {len(weight_matrices)}")
        
        layer_embeddings = []
        
        # Process each layer  
        for layer_idx, weight_matrix in enumerate(weight_matrices):
            # Handle batching
            if len(weight_matrix.shape) == 2:
                weight_matrix = weight_matrix.unsqueeze(0)
                is_single = True
            else:
                is_single = False
            
            # Project each neuron's weights
            projected = self.layer_projections[layer_idx](weight_matrix)
            
            # Apply batch norm
            batch_size, num_neurons, proj_dim = projected.shape
            projected_flat = projected.view(-1, proj_dim)
            normed_flat = self.layer_norms[layer_idx](projected_flat)
            normed = normed_flat.view(batch_size, num_neurons, proj_dim)
            
            # Apply activation and dropout
            activated = self.activation(normed)
            activated = self.dropout(activated)
            
            # Pool across neurons
            layer_embedding = self._pool_layer_embedding(activated)
            
            if is_single:
                layer_embedding = layer_embedding.squeeze(0)
            
            layer_embeddings.append(layer_embedding)
        
        # Handle extra parameters
        if self.handle_extra_params:
            extra_params = handle_extra_params(mlp_dict)
            if extra_params is not None:
                # Process extra parameters
                if len(extra_params.shape) == 1:
                    extra_params = extra_params.unsqueeze(0)  # Add batch dimension
                
                extra_embedding = self.extra_param_encoder(extra_params)
                
                # Include in fusion
                if self.layer_fusion == 'concat':
                    # Concatenate with layer embeddings
                    all_embeddings = torch.cat(layer_embeddings + [extra_embedding], dim=-1)
                    final_embedding = self.fusion(all_embeddings)
                else:
                    # For non-concat fusion, add extra embedding separately
                    layer_fused = self._fuse_layer_embeddings(layer_embeddings)
                    final_embedding = layer_fused + extra_embedding.squeeze(0) if len(extra_embedding.shape) > 1 else layer_fused + extra_embedding
            else:
                # No extra params, use standard fusion
                final_embedding = self._fuse_layer_embeddings(layer_embeddings)
        else:
            # Standard fusion without extra params
            final_embedding = self._fuse_layer_embeddings(layer_embeddings)
        
        return final_embedding


def create_mlp_head_encoder(config: dict) -> MLPHeadEncoder:
    """
    Create MLP Head encoder from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MLPHeadEncoder instance
    """
    encoder_config = config['model']['mlp_encoders']['head_mlp']
    
    return MLPHeadEncoder(
        projection_dim=encoder_config.get('projection_dim', 512),
        final_embedding_dim=encoder_config.get('embedding_dim', 256),
        pooling_strategy=config['model']['mlp_encoders'].get('pooling_strategy', 'max'),
        layer_fusion=config['model']['fusion'].get('layer_fusion', 'concat'),
        include_bias=encoder_config.get('include_bias', True),
        handle_extra_params=encoder_config.get('handle_extra_params', True),
        dropout=config['model']['fusion'].get('dropout', 0.1)
    )