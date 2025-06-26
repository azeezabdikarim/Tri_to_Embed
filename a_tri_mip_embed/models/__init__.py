"""
Model factory and imports for NeRF rotation classification.

Provides a unified interface for creating different model architectures
based on configuration settings.
"""

from .cnn_classification.resnet_encoder import create_resnet_classifier
from .hybrid_models.dual_stream_classifier import create_dual_stream_classifier


def create_model(config: dict):
    """
    Factory function to create the appropriate model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: Instantiated model
    """
    # Check if MLP features are enabled
    use_mlp_features = config['model'].get('use_mlp_features', False)
    use_planes = config['model'].get('use_planes', True)
    
    if use_mlp_features or not use_planes:
        # Use hybrid model (supports planes-only, MLP-only, or hybrid modes)
        return create_dual_stream_classifier(config)
    else:
        # Use original ResNet classifier for backward compatibility
        return create_resnet_classifier(config)


__all__ = [
    'create_model',
    'create_resnet_classifier', 
    'create_dual_stream_classifier'
]