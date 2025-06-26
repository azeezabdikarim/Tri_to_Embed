"""
Dual Stream Classifier for Hybrid NeRF Rotation Classification

Combines feature planes and MLP embeddings for rotation classification.
Supports three processing streams:
1. Feature planes (existing CNN-based processing)
2. MLP base embeddings (geometry network weights)
3. MLP head embeddings (appearance network weights)

Maintains backward compatibility with planes-only models.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from ..cnn_classification.resnet_encoder import ResNetPlaneEncoder
from ..cnn_classification.fusion_strategies import ConcatFusion, SumFusion, AttentionFusion
from ..mlp_encoders.mlp_base_encoder import create_mlp_base_encoder
from ..mlp_encoders.mlp_head_encoder import create_mlp_head_encoder


class StreamFusion(nn.Module):
    """Fusion module for combining different feature streams."""
    
    def __init__(self, input_dims: list, output_dim: int, fusion_type: str = 'concat', dropout: float = 0.5):
        super().__init__()
        self.fusion_type = fusion_type
        self.input_dims = input_dims
        
        if fusion_type == 'concat':
            total_dim = sum(input_dims)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == 'mean':
            # All inputs must have same dimension for mean fusion
            assert len(set(input_dims)) == 1, "All input dimensions must be equal for mean fusion"
            if input_dims[0] != output_dim:
                self.fusion = nn.Linear(input_dims[0], output_dim)
            else:
                self.fusion = nn.Identity()
        elif fusion_type == 'attention':
            # Use the first dimension as the common dimension
            common_dim = input_dims[0]
            self.projections = nn.ModuleList([
                nn.Linear(dim, common_dim) if dim != common_dim else nn.Identity()
                for dim in input_dims
            ])
            
            self.attention = nn.MultiheadAttention(
                embed_dim=common_dim,
                num_heads=8,
                batch_first=True,
                dropout=dropout
            )
            self.layer_norm = nn.LayerNorm(common_dim)
            
            if common_dim != output_dim:
                self.fusion = nn.Linear(common_dim, output_dim)
            else:
                self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, streams: list) -> torch.Tensor:
        """
        Fuse multiple streams.
        
        Args:
            streams: List of tensors to fuse
            
        Returns:
            fused: Fused tensor
        """
        if self.fusion_type == 'concat':
            concatenated = torch.cat(streams, dim=-1)
            return self.fusion(concatenated)
        
        elif self.fusion_type == 'mean':
            stacked = torch.stack(streams, dim=0)  # (num_streams, batch_size, dim)
            averaged = torch.mean(stacked, dim=0)  # (batch_size, dim)
            return self.fusion(averaged)
        
        elif self.fusion_type == 'attention':
            # Project all streams to common dimension
            projected_streams = []
            for stream, projection in zip(streams, self.projections):
                projected_streams.append(projection(stream))
            
            # Stack for attention
            stacked = torch.stack(projected_streams, dim=1)  # (batch_size, num_streams, common_dim)
            
            # Self-attention
            attn_out, _ = self.attention(stacked, stacked, stacked)
            attended = self.layer_norm(stacked + attn_out)
            
            # Global pooling
            pooled = torch.mean(attended, dim=1)  # (batch_size, common_dim)
            
            return self.fusion(pooled)


class DualStreamClassifier(nn.Module):
    """
    Hybrid classifier that combines feature planes and MLP embeddings.
    
    Supports three modes:
    1. Planes only (backward compatible)
    2. MLPs only  
    3. Hybrid (planes + MLPs)
    """
    
    def __init__(
        self,
        config: dict,
        num_classes: int = 5
    ):
        super().__init__()
        
        self.config = config
        self.num_classes = num_classes
        
        # Determine which streams to use
        self.use_planes = config['model'].get('use_planes', True)
        self.use_mlp_features = config['model'].get('use_mlp_features', False)
        
        if not self.use_planes and not self.use_mlp_features:
            raise ValueError("At least one of planes or MLP features must be enabled")
        
        # Stream dimensions for fusion
        stream_dims = []
        self.stream_names = []
        
        # Initialize plane encoder if needed
        if self.use_planes:
            self.plane_encoder = self._create_plane_encoder(config)
            plane_embed_dim = self._calculate_plane_embed_dim(config)
            stream_dims.append(plane_embed_dim)
            self.stream_names.append('planes')
        
        # Initialize MLP encoders if needed
        if self.use_mlp_features:
            self.mlp_base_encoder = create_mlp_base_encoder(config)
            self.mlp_head_encoder = create_mlp_head_encoder(config)
            
            mlp_base_dim = config['model']['mlp_encoders']['base_mlp']['embedding_dim']
            mlp_head_dim = config['model']['mlp_encoders']['head_mlp']['embedding_dim']
            
            stream_dims.extend([mlp_base_dim, mlp_head_dim])
            self.stream_names.extend(['mlp_base', 'mlp_head'])
        
        # Pair combination strategy
        self.pair_combination = config['model'].get('pair_combination', 'subtract')
        
        # Calculate final embedding dimension after pair combination
        if self.pair_combination == 'subtract':
            pair_embed_dim = sum(stream_dims)
        elif self.pair_combination == 'concat':
            pair_embed_dim = sum(stream_dims) * 2
        else:  # both
            pair_embed_dim = sum(stream_dims) * 3
        
        # Stream fusion (if multiple streams)
        if len(stream_dims) > 1:
            fusion_dim = config['model']['fusion'].get('final_dim', 512)
            self.stream_fusion = StreamFusion(
                input_dims=stream_dims,
                output_dim=fusion_dim,
                fusion_type=config['model']['fusion'].get('stream_fusion', 'concat'),
                dropout=config['model']['fusion'].get('dropout', 0.5)
            )
            
            # Update pair embed dim after stream fusion
            if self.pair_combination == 'subtract':
                pair_embed_dim = fusion_dim
            elif self.pair_combination == 'concat':
                pair_embed_dim = fusion_dim * 2
            else:  # both
                pair_embed_dim = fusion_dim * 3
        else:
            self.stream_fusion = None
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(pair_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def _create_plane_encoder(self, config: dict) -> nn.Module:
        """Create the plane encoder component."""
        plane_config = config['model']
        
        # Create ResNet encoder
        encoder = ResNetPlaneEncoder(
            resnet_type=plane_config['encoder_type'],
            embedding_dim=plane_config['embedding_dim'],
            pretrained=plane_config.get('pretrained', False)
        )
        
        # Create plane fusion
        plane_fusion = plane_config['plane_fusion']
        if plane_fusion == 'concat':
            fusion = ConcatFusion()
        elif plane_fusion == 'sum':
            fusion = SumFusion()
        elif plane_fusion == 'attention':
            fusion = AttentionFusion(plane_config['embedding_dim'])
        else:
            raise ValueError(f"Unknown plane fusion strategy: {plane_fusion}")
        
        return nn.ModuleDict({
            'encoder': encoder,
            'fusion': fusion
        })
    
    def _calculate_plane_embed_dim(self, config: dict) -> int:
        """Calculate the embedding dimension after plane fusion."""
        plane_fusion = config['model']['plane_fusion']
        embedding_dim = config['model']['embedding_dim']
        
        if plane_fusion == 'concat':
            return embedding_dim * 3
        else:  # sum or attention
            return embedding_dim
    
    def encode_planes(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Encode feature planes into embeddings.
        
        Args:
            planes: Tensor of shape (B, 3, 512, 512, 16)
            
        Returns:
            embedding: Plane embedding
        """
        if not self.use_planes:
            raise RuntimeError("Plane encoding not enabled")
        
        # Encode each plane separately
        plane_embeddings = []
        for i in range(3):
            plane = planes[:, i]  # (B, 512, 512, 16)
            embed = self.plane_encoder['encoder'].encode_single_plane(plane)
            plane_embeddings.append(embed)
        
        # Fuse plane embeddings
        fused = self.plane_encoder['fusion'](tuple(plane_embeddings))
        
        return fused
    
    def encode_mlps(self, mlp_base_dict: dict, mlp_head_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode MLP weights into embeddings.
        
        Args:
            mlp_base_dict: Structured MLP base dictionary
            mlp_head_dict: Structured MLP head dictionary
            
        Returns:
            mlp_base_embed, mlp_head_embed: MLP embeddings
        """
        if not self.use_mlp_features:
            raise RuntimeError("MLP encoding not enabled")
        
        # Move MLP dictionaries to the correct device
        device = next(self.mlp_base_encoder.parameters()).device
        mlp_base_dict = self._move_mlp_dict_to_device(mlp_base_dict, device)
        mlp_head_dict = self._move_mlp_dict_to_device(mlp_head_dict, device)
        
        mlp_base_embed = self.mlp_base_encoder(mlp_base_dict)
        mlp_head_embed = self.mlp_head_encoder(mlp_head_dict)
        
        return mlp_base_embed, mlp_head_embed
    
    def _move_mlp_dict_to_device(self, mlp_dict: dict, device: torch.device) -> dict:
        """
        Move all tensors in an MLP dictionary to the specified device.
        
        Args:
            mlp_dict: MLP dictionary with tensors
            device: Target device
            
        Returns:
            mlp_dict: Dictionary with all tensors moved to device
        """
        moved_dict = {}
        for key, value in mlp_dict.items():
            if isinstance(value, dict):
                # Recursively handle nested dictionaries (layer data)
                moved_dict[key] = self._move_mlp_dict_to_device(value, device)
            elif isinstance(value, torch.Tensor):
                # Move tensor to device
                moved_dict[key] = value.to(device)
            else:
                # Keep non-tensor values as-is
                moved_dict[key] = value
        
        return moved_dict
    
    def encode_nerf(
        self,
        planes: Optional[torch.Tensor] = None,
        mlp_base_dict: Optional[dict] = None,
        mlp_head_dict: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Encode a full NeRF into a single embedding.
        
        Args:
            planes: Feature planes (B, 3, 512, 512, 16)
            mlp_base_dict: MLP base weights
            mlp_head_dict: MLP head weights
            
        Returns:
            embedding: Combined NeRF embedding
        """
        stream_embeddings = []
        
        # Encode planes if enabled
        if self.use_planes:
            if planes is None:
                raise ValueError("Planes required when plane encoding is enabled")
            plane_embed = self.encode_planes(planes)
            stream_embeddings.append(plane_embed)
        
        # Encode MLPs if enabled
        if self.use_mlp_features:
            if mlp_base_dict is None or mlp_head_dict is None:
                raise ValueError("MLP dictionaries required when MLP encoding is enabled")
            mlp_base_embed, mlp_head_embed = self.encode_mlps(mlp_base_dict, mlp_head_dict)
            stream_embeddings.extend([mlp_base_embed, mlp_head_embed])
        
        # Fuse streams if multiple
        if self.stream_fusion is not None:
            fused_embedding = self.stream_fusion(stream_embeddings)
        else:
            fused_embedding = stream_embeddings[0]
        
        return fused_embedding
    
    def combine_pair_embeddings(
        self,
        base_embed: torch.Tensor,
        aug_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine base and augmented embeddings.
        
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
    
    def forward(
        self,
        base_planes: Optional[torch.Tensor] = None,
        aug_planes: Optional[torch.Tensor] = None,
        base_mlp_base: Optional[dict] = None,
        aug_mlp_base: Optional[dict] = None,
        base_mlp_head: Optional[dict] = None,
        aug_mlp_head: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Forward pass for rotation classification.
        
        Args:
            base_planes: Base model planes (B, 3, 512, 512, 16)
            aug_planes: Augmented model planes (B, 3, 512, 512, 16)
            base_mlp_base: Base model MLP base weights
            aug_mlp_base: Augmented model MLP base weights
            base_mlp_head: Base model MLP head weights
            aug_mlp_head: Augmented model MLP head weights
            
        Returns:
            logits: Classification logits (B, num_classes)
        """
        # Encode both NeRFs
        base_embed = self.encode_nerf(
            planes=base_planes,
            mlp_base_dict=base_mlp_base,
            mlp_head_dict=base_mlp_head
        )
        
        aug_embed = self.encode_nerf(
            planes=aug_planes,
            mlp_base_dict=aug_mlp_base,
            mlp_head_dict=aug_mlp_head
        )
        
        # Combine embeddings
        combined = self.combine_pair_embeddings(base_embed, aug_embed)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits


def create_dual_stream_classifier(config: dict) -> DualStreamClassifier:
    """
    Create a dual stream classifier from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DualStreamClassifier instance
    """
    return DualStreamClassifier(config=config, num_classes=5)