"""
Combined Loss Function with Dynamic Weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

from .contrastive_loss import ContrastiveLossFactory
from .ranking_loss import RankingLossFactory

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
            
        Returns:
            Focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss with inner and outer objectives
    """
    
    def __init__(
        self,
        config: Dict,
        num_classes: int,
        feature_dim: int,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            config: Configuration dictionary
            num_classes: Number of classes
            feature_dim: Feature dimension for contrastive loss
            class_weights: Optional class weights for CE loss
        """
        super().__init__()
        
        self.config = config
        self.loss_config = config['loss']
        self.num_classes = num_classes
        
        # Inner losses
        self.ce_weight = self.loss_config['inner']['ce_weight']
        self.contrastive_weight = self.loss_config['inner']['contrastive_weight']
        
        # Outer losses
        self.auc_weight = self.loss_config['outer']['auc_weight']
        self.focal_weight = self.loss_config['outer']['focal_weight']
        
        # Loss components
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        self.contrastive_loss = ContrastiveLossFactory.create_loss(
            config, num_classes, feature_dim
        )
        
        self.ranking_loss = RankingLossFactory.create_loss(config)
        
        self.focal_loss = FocalLoss(
            alpha=self.loss_config['outer']['focal_alpha'],
            gamma=self.loss_config['outer']['focal_gamma'],
        )
        
        # Dynamic weighting
        self.use_dynamic_weighting = self.loss_config['dynamic_weighting']['enabled']
        self.dynamic_method = self.loss_config['dynamic_weighting']['method']
        
        if self.use_dynamic_weighting:
            if self.dynamic_method == 'uncertainty':
                # Learnable uncertainty parameters
                self.log_vars = nn.Parameter(torch.zeros(4))  # 4 loss components
            elif self.dynamic_method == 'gradnorm':
                # GradNorm parameters
                self.loss_weights = nn.Parameter(torch.ones(4))
                self.initial_losses = None
        
        # Warmup
        self.warmup_config = self.loss_config['warmup']
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Set current epoch for warmup strategy"""
        self.current_epoch = epoch
    
    def _get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights based on warmup and dynamic weighting"""
        weights = {
            'ce': self.ce_weight,
            'contrastive': self.contrastive_weight,
            'auc': self.auc_weight,
            'focal': self.focal_weight,
        }
        
        # Apply warmup
        if self.warmup_config['enabled']:
            warmup_epochs = self.warmup_config['warmup_epochs']
            contrastive_start = self.warmup_config['contrastive_start_epoch']
            auc_start = self.warmup_config['auc_start_epoch']
            
            if self.current_epoch < contrastive_start:
                weights['contrastive'] = 0.0
            elif self.current_epoch < warmup_epochs:
                # Linear warmup
                ratio = (self.current_epoch - contrastive_start) / (warmup_epochs - contrastive_start)
                weights['contrastive'] *= ratio
            
            if self.current_epoch < auc_start:
                weights['auc'] = 0.0
            elif self.current_epoch < warmup_epochs:
                ratio = (self.current_epoch - auc_start) / (warmup_epochs - auc_start)
                weights['auc'] *= ratio
        
        # Apply dynamic weighting
        if self.use_dynamic_weighting:
            if self.dynamic_method == 'uncertainty':
                # Uncertainty weighting: w_i = 1 / (2 * sigma_i^2)
                precisions = torch.exp(-self.log_vars)
                weights['ce'] *= precisions[0].item()
                weights['contrastive'] *= precisions[1].item()
                weights['auc'] *= precisions[2].item()
                weights['focal'] *= precisions[3].item()
            elif self.dynamic_method == 'gradnorm':
                # GradNorm weighting
                weights['ce'] *= self.loss_weights[0].item()
                weights['contrastive'] *= self.loss_weights[1].item()
                weights['auc'] *= self.loss_weights[2].item()
                weights['focal'] *= self.loss_weights[3].item()
        
        return weights
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
            features: (batch_size, feature_dim) - for contrastive loss
            
        Returns:
            Dictionary containing total loss and individual components
        """
        losses = {}
        
        # Cross-entropy loss
        ce_loss = self.ce_loss(logits, labels)
        losses['ce'] = ce_loss
        
        # Contrastive loss
        if features is not None and self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss(features, labels)
            losses['contrastive'] = contrastive_loss
        else:
            losses['contrastive'] = torch.tensor(0.0, device=logits.device)
        
        # AUC/Ranking loss
        if self.auc_weight > 0:
            auc_loss = self.ranking_loss(logits, labels)
            losses['auc'] = auc_loss
        else:
            losses['auc'] = torch.tensor(0.0, device=logits.device)
        
        # Focal loss
        if self.focal_weight > 0:
            focal_loss = self.focal_loss(logits, labels)
            losses['focal'] = focal_loss
        else:
            losses['focal'] = torch.tensor(0.0, device=logits.device)
        
        # Get loss weights
        weights = self._get_loss_weights()
        
        # Compute total loss
        total_loss = (
            weights['ce'] * losses['ce'] +
            weights['contrastive'] * losses['contrastive'] +
            weights['auc'] * losses['auc'] +
            weights['focal'] * losses['focal']
        )
        
        # Add uncertainty regularization if using uncertainty weighting
        if self.use_dynamic_weighting and self.dynamic_method == 'uncertainty':
            # Regularization term: sum of log variances
            total_loss += 0.5 * self.log_vars.sum()
        
        losses['total'] = total_loss
        losses['weights'] = weights
        
        return losses
    
    def update_gradnorm_weights(
        self,
        losses: Dict[str, torch.Tensor],
        shared_parameters: torch.nn.Parameter,
        alpha: float = 1.5,
    ):
        """
        Update loss weights using GradNorm
        
        Args:
            losses: Dictionary of individual losses
            shared_parameters: Shared model parameters
            alpha: GradNorm hyperparameter
        """
        if not self.use_dynamic_weighting or self.dynamic_method != 'gradnorm':
            return
        
        # Store initial losses
        if self.initial_losses is None:
            self.initial_losses = {
                'ce': losses['ce'].item(),
                'contrastive': losses['contrastive'].item(),
                'auc': losses['auc'].item(),
                'focal': losses['focal'].item(),
            }
        
        # Compute gradients for each loss
        grads = {}
        for key in ['ce', 'contrastive', 'auc', 'focal']:
            if losses[key].item() > 0:
                grad = torch.autograd.grad(
                    losses[key],
                    shared_parameters,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                grads[key] = grad.norm()
            else:
                grads[key] = torch.tensor(0.0, device=losses[key].device)
        
        # Compute inverse training rate
        loss_ratios = {}
        for key in ['ce', 'contrastive', 'auc', 'focal']:
            if self.initial_losses[key] > 0:
                loss_ratios[key] = losses[key].item() / self.initial_losses[key]
            else:
                loss_ratios[key] = 1.0
        
        # Average gradient norm
        avg_grad = sum(grads.values()) / len(grads)
        
        # Compute target gradients
        avg_loss_ratio = sum(loss_ratios.values()) / len(loss_ratios)
        
        # GradNorm loss
        gradnorm_loss = 0
        for i, key in enumerate(['ce', 'contrastive', 'auc', 'focal']):
            target_grad = avg_grad * (loss_ratios[key] / avg_loss_ratio) ** alpha
            gradnorm_loss += torch.abs(grads[key] - target_grad)
        
        # Update weights
        gradnorm_loss.backward()


class LossFactory:
    """
    Factory for creating combined loss
    """
    
    @staticmethod
    def create_loss(
        config: Dict,
        num_classes: int,
        feature_dim: int,
        class_weights: Optional[torch.Tensor] = None,
    ) -> CombinedLoss:
        """
        Create combined loss
        
        Args:
            config: Configuration dictionary
            num_classes: Number of classes
            feature_dim: Feature dimension
            class_weights: Optional class weights
            
        Returns:
            CombinedLoss instance
        """
        loss = CombinedLoss(
            config=config,
            num_classes=num_classes,
            feature_dim=feature_dim,
            class_weights=class_weights,
        )
        
        logger.info("Created combined loss with dynamic weighting")
        
        return loss


if __name__ == "__main__":
    # Test combined loss
    import yaml
    
    with open('../configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    batch_size = 32
    num_classes = 2
    feature_dim = 128
    
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))
    features = torch.randn(batch_size, feature_dim, requires_grad=True)
    
    # Create loss
    combined_loss = LossFactory.create_loss(config, num_classes, feature_dim)
    
    # Test different epochs
    for epoch in range(10):
        combined_loss.set_epoch(epoch)
        losses = combined_loss(logits, labels, features)
        
        print(f"Epoch {epoch}:")
        print(f"  Total: {losses['total'].item():.4f}")
        print(f"  CE: {losses['ce'].item():.4f}")
        print(f"  Contrastive: {losses['contrastive'].item():.4f}")
        print(f"  AUC: {losses['auc'].item():.4f}")
        print(f"  Focal: {losses['focal'].item():.4f}")
        print(f"  Weights: {losses['weights']}")
        print()
