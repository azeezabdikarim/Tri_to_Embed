import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

from .base_encoder import BaseEncoder, BasePairClassifier
from .fusion_strategies import ConcatFusion, SumFusion, AttentionFusion

class ResNetPlaneEncoder(BaseEncoder):
    """
    ResNet-based encoder for individual feature planes.
    """
    
    def __init__(
        self,
        resnet_type: str = 'resnet18',
        embedding_dim: int = 256,
        pretrained: bool = False
    ):
        super().__init__(embedding_dim)
        
        # Create ResNet backbone
        if resnet_type == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif resnet_type == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif resnet_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet type: {resnet_type}")
        
        # Modify first conv layer to accept 16 channels
        self.backbone.conv1 = nn.Conv2d(
            16, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Remove the final FC layer
        self.backbone.fc = nn.Identity()
        
        # Add projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.feature_dim = feature_dim
    
    def encode_single_plane(self, plane: torch.Tensor) -> torch.Tensor:
        """
        Encode a single plane into features.
        
        Args:
            plane: Tensor of shape (B, 512, 512, 16)
            
        Returns:
            features: Tensor of shape (B, embedding_dim)
        """
        # Permute to (B, 16, 512, 512) for Conv2d
        plane = plane.permute(0, 3, 1, 2)
        
        # Extract features
        features = self.backbone(plane)  # (B, feature_dim)
        
        # Project to embedding dimension
        embedding = self.projection(features)  # (B, embedding_dim)
        
        return embedding
    
    def get_feature_dim(self) -> int:
        return self.feature_dim

class ResNetClassifier(BasePairClassifier):
    """
    Complete ResNet-based classifier for rotation prediction.
    """
    
    def __init__(
        self,
        resnet_type: str = 'resnet18',
        embedding_dim: int = 256,
        plane_fusion: str = 'concat',
        pair_combination: str = 'subtract',
        num_classes: int = 5,
        pretrained: bool = False
    ):
        # Create encoder
        encoder = ResNetPlaneEncoder(
            resnet_type=resnet_type,
            embedding_dim=embedding_dim,
            pretrained=pretrained
        )
        
        super().__init__(
            encoder=encoder,
            plane_fusion=plane_fusion,
            pair_combination=pair_combination,
            num_classes=num_classes
        )
        
        # Create fusion module based on strategy
        if plane_fusion == 'concat':
            self.fusion = ConcatFusion()
        elif plane_fusion == 'sum':
            self.fusion = SumFusion()
        elif plane_fusion == 'attention':
            self.fusion = AttentionFusion(embedding_dim)
        else:
            raise ValueError(f"Unknown fusion strategy: {plane_fusion}")
    
    def fuse_planes(self, plane_embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Fuse embeddings from three planes.
        
        Args:
            plane_embeddings: Tuple of 3 tensors, each (B, embedding_dim)
            
        Returns:
            fused: Fused embedding
        """
        return self.fusion(plane_embeddings)

def create_resnet_classifier(config: dict) -> ResNetClassifier:
    """
    Create a ResNet classifier from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ResNetClassifier instance
    """
    model_config = config['model']
    
    # Extract encoder type (resnet18, resnet34, etc.)
    encoder_type = model_config['encoder_type']
    
    return ResNetClassifier(
        resnet_type=encoder_type,
        embedding_dim=model_config['embedding_dim'],
        plane_fusion=model_config['plane_fusion'],
        pair_combination=model_config['pair_combination'],
        num_classes=5,  # Fixed for our task
        pretrained=model_config.get('pretrained', False)
    )