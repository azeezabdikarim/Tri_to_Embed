"""
Utility functions for preparing MLP weights for embedding.

Handles extraction and preparation of weight matrices from the structured
MLP dictionaries created during preprocessing.
"""

import torch
from typing import Dict, List, Tuple, Optional
import warnings


def extract_layer_weights(mlp_dict: Dict, layer_idx: int, include_bias: bool = True) -> torch.Tensor:
    """
    Extract weight matrix (and optionally bias) for a specific layer.
    
    Args:
        mlp_dict: Structured MLP dictionary from preprocessing
        layer_idx: Layer index (0, 1, 2, ...)
        include_bias: Whether to concatenate bias as additional row
        
    Returns:
        weight_matrix: Tensor of shape (B, output_dim, input_dim) or (B, output_dim, input_dim + 1) with bias
    """
    layer_key = f'layer_{layer_idx}'
    
    if layer_key not in mlp_dict:
        raise KeyError(f"Layer {layer_idx} not found in MLP dict. Available layers: {list(mlp_dict.keys())}")
    
    layer_data = mlp_dict[layer_key]
    weight = layer_data['weight']  # Shape: (B, output_dim, input_dim) or (output_dim, input_dim)
    
    if include_bias:
        bias = layer_data['bias']  # Shape: (B, output_dim) or (output_dim,)
        
        # Handle both batched and non-batched cases
        if weight.dim() == 3:  # Batched: (B, output_dim, input_dim)
            batch_size, output_dim, input_dim = weight.shape
            
            # Reshape bias to match weight structure
            if bias.dim() == 3 and bias.shape[1] == 1:  # Shape: (B, 1, output_dim)
                bias = bias.squeeze(1)  # Shape: (B, output_dim)
            elif bias.dim() == 2:  # Shape: (B, output_dim) - already correct
                pass
            else:
                raise ValueError(f"Unexpected bias shape {bias.shape} for batched weight {weight.shape}")
            
            # Add bias as additional input dimension
            bias = bias.unsqueeze(-1)  # Shape: (B, output_dim, 1)
            weight = torch.cat([weight, bias], dim=-1)  # Shape: (B, output_dim, input_dim + 1)
            
        else:  # Non-batched: (output_dim, input_dim) 
            if bias.dim() == 1:
                bias = bias.unsqueeze(1)  # Shape: (output_dim, 1)
                weight = torch.cat([weight, bias], dim=1)  # Shape: (output_dim, input_dim + 1)
            else:
                raise ValueError(f"Unexpected bias shape {bias.shape} for non-batched weight {weight.shape}")
    
    return weight


def prepare_weight_matrices(mlp_dict: Dict, include_bias: bool = True) -> List[torch.Tensor]:
    """
    Extract all layer weight matrices from an MLP dictionary.
    
    Args:
        mlp_dict: Structured MLP dictionary from preprocessing
        include_bias: Whether to include bias terms
        
    Returns:
        weight_matrices: List of weight tensors, one per layer
    """
    if 'metadata' not in mlp_dict:
        warnings.warn("No metadata found in MLP dict, inferring layer count")
        
    # Find all layer keys
    layer_keys = [k for k in mlp_dict.keys() if k.startswith('layer_') and k != 'layer_count']
    layer_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort by layer index
    
    weight_matrices = []
    for layer_key in layer_keys:
        layer_idx = int(layer_key.split('_')[1])
        try:
            weights = extract_layer_weights(mlp_dict, layer_idx, include_bias)
            weight_matrices.append(weights)
        except KeyError as e:
            warnings.warn(f"Could not extract weights for {layer_key}: {e}")
            continue
            
    return weight_matrices


def get_layer_dimensions(mlp_dict: Dict) -> List[Tuple[int, int]]:
    """
    Get input and output dimensions for each layer.
    
    Args:
        mlp_dict: Structured MLP dictionary
        
    Returns:
        dimensions: List of (input_dim, output_dim) tuples
    """
    weight_matrices = prepare_weight_matrices(mlp_dict, include_bias=False)
    dimensions = []
    
    for weight in weight_matrices:
        output_dim, input_dim = weight.shape
        dimensions.append((input_dim, output_dim))
        
    return dimensions


def validate_mlp_structure(mlp_dict: Dict, expected_architecture: Optional[List[int]] = None) -> bool:
    """
    Validate that MLP structure matches expectations.
    
    Args:
        mlp_dict: Structured MLP dictionary
        expected_architecture: Expected layer sizes [input, hidden1, hidden2, ..., output]
        
    Returns:
        is_valid: Whether structure is valid
    """
    try:
        # Get layer keys
        layer_keys = [k for k in mlp_dict.keys() if k.startswith('layer_')]
        layer_keys.sort(key=lambda x: int(x.split('_')[1]))
        
        if expected_architecture is not None:
            if len(layer_keys) != len(expected_architecture) - 1:
                return False
            
            for i, layer_key in enumerate(layer_keys):
                layer_data = mlp_dict[layer_key]
                weight = layer_data['weight']
                
                # Handle both batched and non-batched tensors
                if weight.dim() == 3:  # Batched: (B, output_dim, input_dim)
                    _, output_dim, input_dim = weight.shape
                elif weight.dim() == 2:  # Non-batched: (output_dim, input_dim)
                    output_dim, input_dim = weight.shape
                else:
                    return False
                
                expected_input = expected_architecture[i]
                expected_output = expected_architecture[i + 1]
                
                if input_dim != expected_input or output_dim != expected_output:
                    return False
                    
        return True
        
    except Exception:
        return False


def handle_extra_params(mlp_dict: Dict) -> Optional[torch.Tensor]:
    """
    Handle extra parameters that don't fit the standard layer structure.
    
    Args:
        mlp_dict: Structured MLP dictionary
        
    Returns:
        extra_params: Extra parameter tensor if present, None otherwise
    """
    if 'extra_params' in mlp_dict:
        return mlp_dict['extra_params']
    return None


def get_mlp_info(mlp_dict: Dict) -> Dict:
    """
    Get comprehensive information about an MLP structure.
    
    Args:
        mlp_dict: Structured MLP dictionary
        
    Returns:
        info: Dictionary with MLP information
    """
    info = {
        'dimensions': get_layer_dimensions(mlp_dict),
        'num_layers': len([k for k in mlp_dict.keys() if k.startswith('layer_')]),
        'has_extra_params': 'extra_params' in mlp_dict,
        'metadata': mlp_dict.get('metadata', {}),
    }
    
    if info['has_extra_params']:
        extra_params = mlp_dict['extra_params']
        info['extra_params_count'] = len(extra_params) if hasattr(extra_params, '__len__') else 1
        
    return info


def debug_mlp_structure(mlp_dict: Dict, name: str = "MLP"):
    """
    Print detailed information about MLP structure for debugging.
    
    Args:
        mlp_dict: Structured MLP dictionary
        name: Name for the MLP (for logging)
    """
    print(f"\n=== {name} Structure Debug ===")
    print(f"Available keys: {list(mlp_dict.keys())}")
    
    # Check metadata
    if 'metadata' in mlp_dict:
        metadata = mlp_dict['metadata']
        print(f"Metadata: {metadata}")
    
    # Check layers
    layer_keys = [k for k in mlp_dict.keys() if k.startswith('layer_')]
    layer_keys.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"Found {len(layer_keys)} layers: {layer_keys}")
    
    for layer_key in layer_keys:
        layer_data = mlp_dict[layer_key]
        weight_shape = layer_data['weight'].shape
        bias_shape = layer_data['bias'].shape
        print(f"  {layer_key}: weight {weight_shape}, bias {bias_shape}")
    
    # Check extra params
    if 'extra_params' in mlp_dict:
        extra_shape = mlp_dict['extra_params'].shape
        print(f"Extra params: {extra_shape}")
    
    # Get dimensions
    try:
        dimensions = get_layer_dimensions(mlp_dict)
        print(f"Architecture: {' → '.join([str(d[0]) for d in dimensions] + [str(dimensions[-1][1])])}")
    except Exception as e:
        print(f"Could not determine architecture: {e}")
    
    print("=" * (len(name) + 25))