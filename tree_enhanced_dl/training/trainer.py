"""
Training Loop and Trainer Class
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

from ..losses.combined_loss import CombinedLoss
from ..evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for tree-enhanced deep learning model
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: CombinedLoss,
        scheduler: Optional[_LRScheduler],
        config: Dict,
        device: torch.device,
    ):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            loss_fn: Combined loss function
            scheduler: Learning rate scheduler
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        self.training_config = config['training']
        self.early_stopping_config = self.training_config['early_stopping']
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(config)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = -float('inf') if self.early_stopping_config['mode'] == 'max' else float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
        }
        
        # Gradient clipping
        self.gradient_clip = self.training_config['regularization']['gradient_clip']
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.loss_fn.set_epoch(self.current_epoch)
        
        epoch_losses = {
            'total': [],
            'ce': [],
            'contrastive': [],
            'auc': [],
            'focal': [],
        }
        
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            labels = batch['label']
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(batch, return_embeddings=True)
            logits = output['logits']
            
            # Get features for contrastive loss
            features = output['embeddings']['fused']
            
            # Compute loss
            losses = self.loss_fn(logits, labels, features)
            total_loss = losses['total']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Record losses
            for key in epoch_losses.keys():
                epoch_losses[key].append(losses[key].item())
            
            # Record predictions
            probs = torch.softmax(logits, dim=1)
            all_predictions.append(probs.detach().cpu())
            all_labels.append(labels.cpu())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr'],
            })
        
        # Compute epoch metrics
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        metrics = self.metrics_calculator.compute_metrics(
            all_predictions,
            all_labels,
        )
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # Combine results
        results = {
            'loss': avg_losses['total'],
            **metrics,
            'losses': avg_losses,
        }
        
        return results
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        epoch_losses = {
            'total': [],
            'ce': [],
            'contrastive': [],
            'auc': [],
            'focal': [],
        }
        
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            labels = batch['label']
            
            # Forward pass
            output = self.model(batch, return_embeddings=True)
            logits = output['logits']
            features = output['embeddings']['fused']
            
            # Compute loss
            losses = self.loss_fn(logits, labels, features)
            
            # Record losses
            for key in epoch_losses.keys():
                epoch_losses[key].append(losses[key].item())
            
            # Record predictions
            probs = torch.softmax(logits, dim=1)
            all_predictions.append(probs.cpu())
            all_labels.append(labels.cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': losses['total'].item()})
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        metrics = self.metrics_calculator.compute_metrics(
            all_predictions,
            all_labels,
        )
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # Combine results
        results = {
            'loss': avg_losses['total'],
            **metrics,
            'losses': avg_losses,
        }
        
        return results
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (overrides config)
            
        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.training_config['num_epochs']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_results = self.train_epoch(train_loader)
            
            # Validate
            val_results = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
            
            # Log results
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_results['loss']:.4f}, "
                f"Val Loss={val_results['loss']:.4f}, "
                f"Val AUC={val_results.get('auc', 0):.4f}, "
                f"Val PR-AUC={val_results.get('pr_auc', 0):.4f}"
            )
            
            # Record history
            self.training_history['train_loss'].append(train_results['loss'])
            self.training_history['val_loss'].append(val_results['loss'])
            self.training_history['train_metrics'].append(train_results)
            self.training_history['val_metrics'].append(val_results)
            
            # Check for improvement
            metric_name = self.early_stopping_config['metric']
            current_metric = val_results.get(metric_name, val_results['loss'])
            
            improved = self._check_improvement(current_metric)
            
            if improved:
                self.best_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint('best')
                logger.info(f"New best {metric_name}: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{self.early_stopping_config['patience']}")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['system']['checkpoint']['save_frequency'] == 0:
                self._save_checkpoint(f'epoch_{epoch}')
            
            # Early stopping
            if self.early_stopping_config['enabled']:
                if self.patience_counter >= self.early_stopping_config['patience']:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Save final checkpoint
        self._save_checkpoint('last')
        
        logger.info("Training completed")
        
        return self.training_history
    
    def _check_improvement(self, current_metric: float) -> bool:
        """Check if current metric is better than best"""
        mode = self.early_stopping_config['mode']
        
        if mode == 'max':
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        save_dir = Path(self.config['system']['checkpoint']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = save_dir / f'checkpoint_{name}.pt'
        torch.save(checkpoint, save_path)
        
        logger.info(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")


if __name__ == "__main__":
    # Test trainer (requires full setup)
    pass
