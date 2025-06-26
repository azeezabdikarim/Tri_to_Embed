"""
Base INR2Vec Encoder Implementation

Implements the core inr2vec architecture for encoding MLP weights:
1. Layer-wise processing with separate projections for each layer
2. Configurable pooling strategies (max, mean, sum)
3. Layer fusion strategies (concat, mean, attention)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from .weight_utils import prepare_weight_matrices, get_layer_dimensions


class BaseINR2VecEncoder(nn.Module, ABC):
    """
    Base class for INR2Vec-style MLP weight encoders.
    
    Architecture:
    1. For each layer: weight_matrix → linear_projection → batch_norm → relu → pooling
    2. Combine layer embeddings using fusion strategy
    """
    
    def __init__(
        self,
        layer_dimensions: List[tuple],  # [(input_dim, output_dim), ...]
        projection_dim: int = 256,
        final_embedding_dim: int = 256,
        pooling_strategy: str = 'mean',
        layer_fusion: str = 'concat',
        include_bias: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            layer_dimensions: List of (input_dim, output_dim) for each layer
            projection_dim: Dimension for layer-wise projections
            final_embedding_dim: Final embedding dimension after fusion
            pooling_strategy: 'max', 'mean', 'sum'
            layer_fusion: 'concat', 'mean', 'attention'
            include_bias: Whether to include bias terms in processing
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layer_dimensions = layer_dimensions
        self.projection_dim = projection_dim
        self.final_embedding_dim = final_embedding_dim
        self.pooling_strategy = pooling_strategy
        self.layer_fusion = layer_fusion
        self.include_bias = include_bias
        self.dropout_rate = dropout  # Store as rate, not as nn.Dropout object
        self.num_layers = len(layer_dimensions)
        
        # Create layer-specific projections
        self.layer_projections = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for input_dim, output_dim in layer_dimensions:
            # Adjust input dimension if including bias
            actual_input_dim = input_dim + 1 if include_bias else input_dim
            
            # Linear projection for this layer
            projection = nn.Linear(actual_input_dim, projection_dim)
            self.layer_projections.append(projection)
            
            # Batch normalization for this layer
            norm = nn.BatchNorm1d(projection_dim)
            self.layer_norms.append(norm)
        
        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Create nn.Dropout object here
        
        # Layer fusion
        self._setup_layer_fusion()
    
    def _setup_layer_fusion(self):
        """Setup the layer fusion mechanism."""
        if self.layer_fusion == 'concat':
            # Concatenate all layer embeddings
            fusion_input_dim = self.projection_dim * self.num_layers
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, self.final_embedding_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)  # Use dropout_rate here, not self.dropout
            )
        
        elif self.layer_fusion == 'mean':
            # Simple mean, then project if needed
            if self.projection_dim != self.final_embedding_dim:
                self.fusion = nn.Linear(self.projection_dim, self.final_embedding_dim)
            else:
                self.fusion = nn.Identity()
        
        elif self.layer_fusion == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=self.projection_dim,
                num_heads=8,
                batch_first=True,
                dropout=self.dropout_rate  # Use dropout_rate here, not self.dropout
            )
            self.layer_norm = nn.LayerNorm(self.projection_dim)
            
            if self.projection_dim != self.final_embedding_dim:
                self.fusion = nn.Linear(self.projection_dim, self.final_embedding_dim)
            else:
                self.fusion = nn.Identity()
        
        else:
            raise ValueError(f"Unknown layer fusion strategy: {self.layer_fusion}")
    
    def _pool_layer_embedding(self, layer_features: torch.Tensor) -> torch.Tensor:
        """
        Pool features across neurons in a layer.
        
        Args:
            layer_features: Tensor of shape (batch_size, num_neurons, projection_dim)
            
        Returns:
            pooled: Tensor of shape (batch_size, projection_dim)
        """
        if self.pooling_strategy == 'max':
            pooled, _ = torch.max(layer_features, dim=1)
        elif self.pooling_strategy == 'mean':
            pooled = torch.mean(layer_features, dim=1)
        elif self.pooling_strategy == 'sum':
            pooled = torch.sum(layer_features, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
        return pooled
    
    def _fuse_layer_embeddings(self, layer_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse embeddings from all layers.
        
        Args:
            layer_embeddings: List of tensors, each (batch_size, projection_dim)
            
        Returns:
            fused: Tensor of shape (batch_size, final_embedding_dim)
        """
        if self.layer_fusion == 'concat':
            # Concatenate all embeddings
            concatenated = torch.cat(layer_embeddings, dim=1)  # (batch_size, num_layers * projection_dim)
            return self.fusion(concatenated)
        
        elif self.layer_fusion == 'mean':
            # Average all embeddings
            stacked = torch.stack(layer_embeddings, dim=1)  # (batch_size, num_layers, projection_dim)
            averaged = torch.mean(stacked, dim=1)  # (batch_size, projection_dim)
            return self.fusion(averaged)
        
        elif self.layer_fusion == 'attention':
            # Attention-based fusion
            stacked = torch.stack(layer_embeddings, dim=1)  # (batch_size, num_layers, projection_dim)
            
            # Self-attention
            attn_out, _ = self.attention(stacked, stacked, stacked)
            attended = self.layer_norm(stacked + attn_out)
            
            # Global pooling
            pooled = torch.mean(attended, dim=1)  # (batch_size, projection_dim)
            
            return self.fusion(pooled)
    
    def forward(self, mlp_dict: Dict) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            mlp_dict: Structured MLP dictionary from preprocessing
            
        Returns:
            embedding: Final embedding tensor (batch_size, final_embedding_dim)
        """
        # Extract weight matrices
        weight_matrices = prepare_weight_matrices(mlp_dict, include_bias=self.include_bias)
        
        if len(weight_matrices) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {len(weight_matrices)}")
        
        layer_embeddings = []
        
        # Process each layer
        for layer_idx, weight_matrix in enumerate(weight_matrices):
            # weight_matrix shape: (num_neurons, input_dim) or (num_neurons, input_dim + 1) with bias
            batch_size = weight_matrix.shape[0] if len(weight_matrix.shape) == 3 else 1
            
            # Handle batching - if weight_matrix is 2D, add batch dimension
            if len(weight_matrix.shape) == 2:
                weight_matrix = weight_matrix.unsqueeze(0)  # (1, num_neurons, input_dim)
                is_single = True
            else:
                is_single = False
            
            # Project each neuron's weights
            # weight_matrix: (batch_size, num_neurons, input_dim)
            projected = self.layer_projections[layer_idx](weight_matrix)  # (batch_size, num_neurons, projection_dim)
            
            # Apply batch norm (need to reshape for batch norm)
            batch_size, num_neurons, proj_dim = projected.shape
            projected_flat = projected.view(-1, proj_dim)  # (batch_size * num_neurons, projection_dim)
            normed_flat = self.layer_norms[layer_idx](projected_flat)
            normed = normed_flat.view(batch_size, num_neurons, proj_dim)
            
            # Apply activation and dropout
            activated = self.activation(normed)
            activated = self.dropout(activated)
            
            # Pool across neurons
            layer_embedding = self._pool_layer_embedding(activated)  # (batch_size, projection_dim)
            
            # Remove batch dimension if it was added
            if is_single:
                layer_embedding = layer_embedding.squeeze(0)  # (projection_dim,)
            
            layer_embeddings.append(layer_embedding)
        
        # Fuse layer embeddings
        final_embedding = self._fuse_layer_embeddings(layer_embeddings)
        
        return final_embedding
    
    @abstractmethod
    def get_expected_architecture(self) -> List[int]:
        """Return the expected MLP architecture for validation."""
        pass