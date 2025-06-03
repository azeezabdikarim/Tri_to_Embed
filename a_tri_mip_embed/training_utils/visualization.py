import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Optional, List, Tuple

def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    save_path: Optional[Path] = None,
    method: str = 'tsne',
    title: str = 'Embedding Visualization'
) -> plt.Figure:
    """
    Visualize embeddings using t-SNE or PCA.
    
    Args:
        embeddings: Tensor of shape (N, D)
        labels: Tensor of shape (N,) with class labels
        save_path: Optional path to save figure
        method: 'tsne' or 'pca'
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'pca':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for each rotation class
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    rotation_names = ['x_180', 'y_180', 'z_120', 'z_240', 'compound']
    
    # Plot each class
    for i in range(5):
        mask = labels == i
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=rotation_names[i],
            alpha=0.6,
            s=50
        )
    
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_plane_importance(
    plane_weights: List[float],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot importance of each plane.
    
    Args:
        plane_weights: List of 3 weights for XY, XZ, YZ planes
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    planes = ['XY', 'XZ', 'YZ']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(planes, plane_weights, color=colors)
    
    # Add value labels on bars
    for bar, weight in zip(bars, plane_weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}',
                ha='center', va='bottom')
    
    ax.set_ylabel('Weight')
    ax.set_title('Plane Importance for Rotation Classification')
    ax.set_ylim(0, max(plane_weights) * 1.2)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def visualize_rotation_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 16,
    device: str = 'cuda'
) -> Tuple[plt.Figure, float]:
    """
    Visualize model predictions on sample data.
    
    Args:
        model: Trained model
        dataloader: Data loader
        num_samples: Number of samples to visualize
        device: Device to run on
        
    Returns:
        Figure and accuracy
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            base_planes = batch['base_planes'].to(device)
            aug_planes = batch['aug_planes'].to(device)
            labels = batch['rotation_label']
            
            logits = model(base_planes, aug_planes)
            preds = logits.argmax(dim=1).cpu()
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            
            if len(all_preds) >= num_samples:
                break
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    rotation_names = ['x_180', 'y_180', 'z_120', 'z_240', 'compound']
    
    for i in range(min(num_samples, len(all_preds))):
        ax = axes[i]
        
        true_label = rotation_names[all_labels[i]]
        pred_label = rotation_names[all_preds[i]]
        
        # Color based on correctness
        color = 'green' if all_labels[i] == all_preds[i] else 'red'
        
        ax.text(0.5, 0.5, f"True: {true_label}\nPred: {pred_label}",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(min(num_samples, len(all_preds)), 16):
        axes[i].axis('off')
    
    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    
    fig.suptitle(f'Rotation Predictions (Accuracy: {accuracy:.2%})', fontsize=16)
    
    return fig, accuracy