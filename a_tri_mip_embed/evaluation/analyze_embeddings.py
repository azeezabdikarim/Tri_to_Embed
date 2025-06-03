import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

from training_utils.visualization import visualize_embeddings, plot_plane_importance

@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    max_samples: int = 1000
) -> Dict[str, torch.Tensor]:
    """
    Extract embeddings from the model for analysis.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        max_samples: Maximum number of samples to extract
        
    Returns:
        Dictionary with embeddings and metadata
    """
    model.eval()
    
    all_base_embeddings = []
    all_aug_embeddings = []
    all_combined_embeddings = []
    all_labels = []
    all_object_ids = []
    all_rotation_names = []
    
    samples_processed = 0
    
    for batch in tqdm(dataloader, desc='Extracting embeddings'):
        if samples_processed >= max_samples:
            break
            
        # Move to device
        base_planes = batch['base_planes'].to(device)
        aug_planes = batch['aug_planes'].to(device)
        
        # Get embeddings
        base_embed = model.encode_nerf(base_planes)
        aug_embed = model.encode_nerf(aug_planes)
        combined_embed = model.combine_pair_embeddings(base_embed, aug_embed)
        
        # Store
        all_base_embeddings.append(base_embed.cpu())
        all_aug_embeddings.append(aug_embed.cpu())
        all_combined_embeddings.append(combined_embed.cpu())
        all_labels.append(batch['rotation_label'])
        all_object_ids.extend(batch['object_id'])
        all_rotation_names.extend(batch['rotation_name'])
        
        samples_processed += len(batch['rotation_label'])
    
    return {
        'base_embeddings': torch.cat(all_base_embeddings),
        'aug_embeddings': torch.cat(all_aug_embeddings),
        'combined_embeddings': torch.cat(all_combined_embeddings),
        'labels': torch.cat(all_labels),
        'object_ids': all_object_ids,
        'rotation_names': all_rotation_names
    }

def analyze_plane_importance(
    model: torch.nn.Module,
    save_dir: Path
) -> Dict[str, float]:
    """
    Analyze the importance of each plane for the model.
    
    Args:
        model: Trained model with SumFusion
        save_dir: Directory to save results
        
    Returns:
        Dictionary with plane weights
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model uses sum fusion
    if hasattr(model, 'fusion') and hasattr(model.fusion, 'weights'):
        weights = torch.softmax(model.fusion.weights, dim=0).cpu().numpy()
        
        # Create visualization
        fig = plot_plane_importance(weights.tolist(), save_dir / 'plane_importance.png')
        plt.close(fig)
        
        # Save weights
        plane_names = ['XY', 'XZ', 'YZ']
        results = {name: float(w) for name, w in zip(plane_names, weights)}
        
        print("\nPlane Importance:")
        for name, weight in results.items():
            print(f"  {name}: {weight:.3f}")
        
        return results
    else:
        print("Model does not use sum fusion, skipping plane importance analysis")
        return {}

def analyze_embedding_space(
    embeddings_dict: Dict[str, torch.Tensor],
    save_dir: Path,
    num_samples: int = 500
) -> Dict[str, plt.Figure]:
    """
    Analyze the embedding space with various visualizations.
    
    Args:
        embeddings_dict: Dictionary from extract_embeddings
        save_dir: Directory to save results
        num_samples: Number of samples to visualize
        
    Returns:
        Dictionary of figures
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Subsample if needed
    if len(embeddings_dict['labels']) > num_samples:
        indices = torch.randperm(len(embeddings_dict['labels']))[:num_samples]
        embeddings_dict = {
            k: v[indices] if isinstance(v, torch.Tensor) else [v[i] for i in indices]
            for k, v in embeddings_dict.items()
        }
    
    # 1. Combined embeddings (used for classification)
    fig = visualize_embeddings(
        embeddings_dict['combined_embeddings'],
        embeddings_dict['labels'],
        save_dir / 'combined_embeddings_tsne.png',
        method='tsne',
        title='Combined Embeddings (t-SNE)'
    )
    figures['combined_tsne'] = fig
    
    # 2. Base vs Augmented embeddings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA for base embeddings
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    base_pca = pca.fit_transform(embeddings_dict['base_embeddings'].numpy())
    aug_pca = pca.transform(embeddings_dict['aug_embeddings'].numpy())
    
    # Plot base embeddings (should all be similar)
    ax1.scatter(base_pca[:, 0], base_pca[:, 1], alpha=0.5, s=30)
    ax1.set_title('Base Model Embeddings (PCA)')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    
    # Plot augmented embeddings colored by rotation
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    rotation_names = ['x_180', 'y_180', 'z_120', 'z_240', 'compound']
    
    for i in range(5):
        mask = embeddings_dict['labels'] == i
        ax2.scatter(
            aug_pca[mask, 0],
            aug_pca[mask, 1],
            c=[colors[i]],
            label=rotation_names[i],
            alpha=0.6,
            s=50
        )
    
    ax2.set_title('Augmented Model Embeddings (PCA)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(save_dir / 'base_vs_aug_embeddings.png', dpi=150, bbox_inches='tight')
    figures['base_vs_aug'] = fig
    
    # 3. Embedding norm analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    base_norms = torch.norm(embeddings_dict['base_embeddings'], dim=1)
    aug_norms = torch.norm(embeddings_dict['aug_embeddings'], dim=1)
    combined_norms = torch.norm(embeddings_dict['combined_embeddings'], dim=1)
    
    data = [base_norms.numpy(), aug_norms.numpy(), combined_norms.numpy()]
    labels = ['Base', 'Augmented', 'Combined']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Embedding Norm')
    ax.set_title('Distribution of Embedding Norms')
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_dir / 'embedding_norms.png', dpi=150, bbox_inches='tight')
    figures['norms'] = fig
    
    return figures

def run_full_embedding_analysis(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    save_dir: Path,
    device: str = 'cuda'
):
    """
    Run complete embedding analysis and save results.
    
    Args:
        model: Trained model
        dataloader: Validation data loader
        save_dir: Directory to save all results
        device: Device to run on
    """
    save_dir = Path(save_dir)
    analysis_dir = save_dir / 'embedding_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting embeddings...")
    embeddings_dict = extract_embeddings(model, dataloader, device)
    
    print("Analyzing embedding space...")
    figures = analyze_embedding_space(embeddings_dict, analysis_dir)
    
    print("Analyzing plane importance...")
    plane_weights = analyze_plane_importance(model, analysis_dir)
    
    # Save embedding statistics
    stats = {
        'num_samples': len(embeddings_dict['labels']),
        'embedding_dims': {
            'base': embeddings_dict['base_embeddings'].shape[1],
            'augmented': embeddings_dict['aug_embeddings'].shape[1],
            'combined': embeddings_dict['combined_embeddings'].shape[1]
        },
        'plane_weights': plane_weights
    }
    
    import json
    with open(analysis_dir / 'embedding_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {analysis_dir}")
    
    # Close all figures
    for fig in figures.values():
        plt.close(fig)