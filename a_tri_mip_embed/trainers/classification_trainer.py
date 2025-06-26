import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
import json

from training_utils.logging import setup_tensorboard, log_metrics
from training_utils.checkpointing import save_checkpoint, load_checkpoint
from losses.classification import RotationClassificationLoss
from evaluation.metrics import compute_confusion_matrix, compute_per_class_accuracy

class ClassificationTrainer:
    """
    Trainer for rotation classification experiments.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=float(config['training']['weight_decay'])
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=1e-6
        )
        
        # Setup loss
        self.criterion = RotationClassificationLoss(
            num_classes=5,
            label_smoothing=config['training'].get('label_smoothing', 0.0)
        )
        
        # Setup logging with timestamped directory
        self.writer, self.timestamped_log_dir = setup_tensorboard(config['logging']['log_dir'])
        self.checkpoint_dir = self.timestamped_log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_accuracy = 0.0
    
    def _prepare_forward_kwargs(self, batch):
        """Prepare forward pass arguments based on model configuration and available data."""
        # Check what the model expects
        model_uses_mlp = getattr(self.model, 'use_mlp_features', False)
        model_uses_planes = getattr(self.model, 'use_planes', True)
        
        # Prepare arguments based on model configuration and available data
        forward_kwargs = {}
        
        if model_uses_planes:
            forward_kwargs['base_planes'] = batch['base_planes'].to(self.device)
            forward_kwargs['aug_planes'] = batch['aug_planes'].to(self.device)
        
        if model_uses_mlp:
            if 'base_mlp_base' in batch:
                forward_kwargs['base_mlp_base'] = batch['base_mlp_base']
                forward_kwargs['aug_mlp_base'] = batch['aug_mlp_base']
                forward_kwargs['base_mlp_head'] = batch['base_mlp_head']
                forward_kwargs['aug_mlp_head'] = batch['aug_mlp_head']
            else:
                raise ValueError(
                    f"Model is configured to use MLP features (use_mlp_features=True) "
                    f"but MLP features are not available in the dataset batch. "
                    f"Please ensure your dataset is configured with load_mlp_features=True "
                    f"or set use_mlp_features=False in your model config.\n"
                    f"Available batch keys: {list(batch.keys())}"
                )
        
        return forward_kwargs
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'accuracy': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Prepare forward arguments
            forward_kwargs = self._prepare_forward_kwargs(batch)
            labels = batch['rotation_label'].to(self.device)
            
            # Forward pass
            logits = self.model(**forward_kwargs)
            
            # Compute loss
            loss_dict = self.criterion(logits, labels)
            loss = loss_dict['loss']
            
            # if batch_idx == 35:
            #     print(f"\nBatch 35 DETAILED DEBUG:")
            #     print(f"  Input stats - Base: [{forward_kwargs['base_planes'].min():.4f}, {forward_kwargs['base_planes'].max():.4f}]")
            #     print(f"  Input stats - Aug: [{forward_kwargs['aug_planes'].min():.4f}, {forward_kwargs['aug_planes'].max():.4f}]")
            #     print(f"  Raw logits: {logits[0].tolist()}")  # First sample
            #     print(f"  Softmax probs: {torch.softmax(logits, dim=1)[0].tolist()}")  # First sample
            #     print(f"  Label smoothing: {self.config['training'].get('label_smoothing', 0.0)}")
            #     # Check intermediate values in the model
            #     with torch.no_grad():
            #         test_base_embed = self.model.encode_nerf(**{k: v[:1] for k, v in forward_kwargs.items() if k.startswith('base_')})
            #         test_aug_embed = self.model.encode_nerf(**{k.replace('base_', 'aug_'): v[:1] for k, v in forward_kwargs.items() if k.startswith('base_')})
            #         test_combined = self.model.combine_pair_embeddings(test_base_embed, test_aug_embed)
            #         print(f"  Embed norms - Base: {test_base_embed.norm():.4f}, Aug: {test_aug_embed.norm():.4f}, Combined: {test_combined.norm():.4f}")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['accuracy'] += loss_dict['accuracy'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{loss_dict['accuracy'].item():.4f}"
            })
            
            # Log to tensorboard
            if self.global_step % self.config['logging']['log_freq'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/accuracy', loss_dict['accuracy'].item(), self.global_step)
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        val_metrics = {
            'loss': 0.0,
            'accuracy': 0.0
        }
        
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Prepare forward arguments
            forward_kwargs = self._prepare_forward_kwargs(batch)
            labels = batch['rotation_label'].to(self.device)
            
            # Forward pass
            logits = self.model(**forward_kwargs)
            
            # Compute loss
            loss_dict = self.criterion(logits, labels)
            
            # Update metrics
            val_metrics['loss'] += loss_dict['loss'].item()
            val_metrics['accuracy'] += loss_dict['accuracy'].item()
            
            # Store predictions for confusion matrix
            predictions = logits.argmax(dim=1)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
        
        # Average metrics
        num_batches = len(self.val_loader)
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        
        # Compute confusion matrix
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        confusion_matrix = compute_confusion_matrix(all_predictions, all_labels, num_classes=5)
        
        # Compute per-class accuracy
        per_class_acc = compute_per_class_accuracy(confusion_matrix)
        
        # Log confusion matrix
        self.writer.add_figure(
            'val/confusion_matrix',
            self._plot_confusion_matrix(confusion_matrix),
            self.epoch
        )
        
        # Log per-class accuracy
        rotation_names = ['x_180', 'y_180', 'z_120', 'z_240', 'compound_90']
        for i, (name, acc) in enumerate(zip(rotation_names, per_class_acc)):
            self.writer.add_scalar(f'val/accuracy_{name}', acc, self.epoch)
        
        return val_metrics, confusion_matrix
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        print(f"All outputs will be saved to: {self.timestamped_log_dir}")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Log epoch metrics
            for name, value in train_metrics.items():
                self.writer.add_scalar(f'train_epoch/{name}', value, epoch)
            
            # Validate
            if epoch % self.config['evaluation']['eval_freq'] == 0:
                val_metrics, confusion_matrix = self.validate()
                
                # Log validation metrics
                for name, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{name}', value, epoch)
                
                # Save best model
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        val_metrics,
                        self.checkpoint_dir / 'best_model.pth'
                    )
                
                print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                      f"Train Acc={train_metrics['accuracy']:.4f}, "
                      f"Val Loss={val_metrics['loss']:.4f}, "
                      f"Val Acc={val_metrics['accuracy']:.4f}")
            
            # Save checkpoint
            if epoch % self.config['logging']['save_freq'] == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    train_metrics,
                    self.checkpoint_dir / f'epoch_{epoch}.pth'
                )
            
            # Update learning rate
            self.scheduler.step()
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], epoch)
        
        print(f"Training complete! Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"All outputs saved to: {self.timestamped_log_dir}")
        self.writer.close()
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Create confusion matrix plot."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        classes = ['x_180', 'y_180', 'z_120', 'z_240', 'compound']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes,
               yticklabels=classes,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig