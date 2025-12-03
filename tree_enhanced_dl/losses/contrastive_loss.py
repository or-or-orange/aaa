"""
Contrastive Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    Reference: https://arxiv.org/abs/2004.11362
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        """
        Args:
            temperature: Temperature parameter
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feature_dim) - normalized features
            labels: (batch_size,) - class labels
            mask: Optional mask for valid samples
            
        Returns:
            Contrastive loss
        """
        device = features.device
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)  # (batch, batch)
        
        # Create label mask: 1 if same class, 0 otherwise
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.ones_like(similarity_matrix)
        logits_mask.fill_diagonal_(0)
        
        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive samples
        # Mask out self-contrast cases
        label_mask = label_mask * logits_mask
        
        # Compute loss
        mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + 1e-12)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class CenterLoss(nn.Module):
    """
    Center Loss for deep feature learning
    Reference: https://ydwen.github.io/papers/WenECCV16.pdf
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        lambda_c: float = 0.01,
    ):
        """
        Args:
            num_classes: Number of classes
            feature_dim: Feature dimension
            lambda_c: Weight for center loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c
        
        # Centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feature_dim)
            labels: (batch_size,)
            
        Returns:
            Center loss
        """
        batch_size = features.size(0)
        
        # Get centers for each sample
        centers_batch = self.centers[labels]  # (batch, feature_dim)
        
        # Compute distance to centers
        loss = F.mse_loss(features, centers_batch)
        
        return self.lambda_c * loss


class TripletLoss(nn.Module):
    """
    Triplet Loss with hard negative mining
    """
    
    def __init__(
        self,
        margin: float = 0.3,
        mining: str = 'hard',
    ):
        """
        Args:
            margin: Margin for triplet loss
            mining: Mining strategy ('hard', 'semi-hard', 'all')
        """
        super().__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feature_dim)
            labels: (batch_size,)
            
        Returns:
            Triplet loss
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(features, features, p=2)
        
        # Create masks
        labels = labels.unsqueeze(1)
        same_label_mask = (labels == labels.T).float()
        diff_label_mask = (labels != labels.T).float()
        
        # Remove diagonal
        same_label_mask.fill_diagonal_(0)
        
        if self.mining == 'hard':
            # Hard positive: farthest same-class sample
            pos_dist = (dist_matrix * same_label_mask).max(dim=1)[0]
            
            # Hard negative: closest different-class sample
            neg_dist = (dist_matrix + 1e6 * (1 - diff_label_mask)).min(dim=1)[0]
            
        elif self.mining == 'semi-hard':
            # Semi-hard negative mining
            pos_dist = (dist_matrix * same_label_mask).max(dim=1)[0]
            
            # Negatives that are farther than positive but within margin
            semi_hard_mask = diff_label_mask * (dist_matrix > pos_dist.unsqueeze(1)).float()
            semi_hard_mask = semi_hard_mask * (dist_matrix < pos_dist.unsqueeze(1) + self.margin).float()
            
            if semi_hard_mask.sum() > 0:
                neg_dist = (dist_matrix * semi_hard_mask + 1e6 * (1 - semi_hard_mask)).min(dim=1)[0]
            else:
                # Fallback to hard negative
                neg_dist = (dist_matrix + 1e6 * (1 - diff_label_mask)).min(dim=1)[0]
        
        else:  # 'all'
            # Average over all pairs
            pos_dist = (dist_matrix * same_label_mask).sum(dim=1) / (same_label_mask.sum(dim=1) + 1e-12)
            neg_dist = (dist_matrix * diff_label_mask).sum(dim=1) / (diff_label_mask.sum(dim=1) + 1e-12)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        loss = loss.mean()
        
        return loss


class ContrastiveLossFactory:
    """
    Factory for creating contrastive losses
    """
    
    @staticmethod
    def create_loss(config: dict, num_classes: int, feature_dim: int) -> nn.Module:
        """
        Create contrastive loss based on config
        
        Args:
            config: Configuration dictionary
            num_classes: Number of classes
            feature_dim: Feature dimension
            
        Returns:
            Contrastive loss module
        """
        loss_config = config['loss']['inner']
        contrastive_type = loss_config['contrastive_type']
        
        if contrastive_type == 'supcon':
            loss = SupervisedContrastiveLoss(
                temperature=loss_config['temperature'],
            )
        elif contrastive_type == 'center':
            loss = CenterLoss(
                num_classes=num_classes,
                feature_dim=feature_dim,
            )
        elif contrastive_type == 'triplet':
            loss = TripletLoss(margin=0.3)
        else:
            raise ValueError(f"Unsupported contrastive loss: {contrastive_type}")
        
        logger.info(f"Created {contrastive_type} contrastive loss")
        
        return loss


if __name__ == "__main__":
    # Test contrastive losses
    batch_size = 32
    feature_dim = 128
    num_classes = 2
    
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test SupCon
    supcon_loss = SupervisedContrastiveLoss()
    loss = supcon_loss(features, labels)
    print(f"SupCon loss: {loss.item():.4f}")
    
    # Test Center loss
    center_loss = CenterLoss(num_classes, feature_dim)
    loss = center_loss(features, labels)
    print(f"Center loss: {loss.item():.4f}")
    
    # Test Triplet loss
    triplet_loss = TripletLoss()
    loss = triplet_loss(features, labels)
    print(f"Triplet loss: {loss.item():.4f}")
