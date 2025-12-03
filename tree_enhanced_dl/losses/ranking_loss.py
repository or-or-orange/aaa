"""
Ranking and AUC-based Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AUCLoss(nn.Module):
    """
    Differentiable AUC loss (pairwise ranking)
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean',
    ):
        """
        Args:
            margin: Margin for ranking
            reduction: Reduction method ('mean', 'sum')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes) or (batch_size,)
            labels: (batch_size,)
            
        Returns:
            AUC loss
        """
        # Get positive class scores
        if logits.dim() == 2:
            scores = logits[:, 1] if logits.size(1) == 2 else logits[:, 0]
        else:
            scores = logits
        
        # Separate positive and negative samples
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Compute pairwise differences
        # pos_scores: (n_pos,), neg_scores: (n_neg,)
        # diff: (n_pos, n_neg)
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        
        # Hinge loss: max(0, margin - diff)
        loss = F.relu(self.margin - diff)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss with configurable margin
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        use_sigmoid: bool = True,
    ):
        """
        Args:
            margin: Margin for ranking
            use_sigmoid: Whether to apply sigmoid to scores
        """
        super().__init__()
        self.margin = margin
        self.use_sigmoid = use_sigmoid
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch_size,) or (batch_size, num_classes)
            labels: (batch_size,)
            
        Returns:
            Pairwise ranking loss
        """
        # Get scores
        if logits.dim() == 2:
            scores = logits[:, 1] if logits.size(1) == 2 else logits.squeeze()
        else:
            scores = logits
        
        if self.use_sigmoid:
            scores = torch.sigmoid(scores)
        
        # Create all pairs
        batch_size = scores.size(0)
        
        # Expand for pairwise comparison
        scores_i = scores.unsqueeze(1).expand(batch_size, batch_size)
        scores_j = scores.unsqueeze(0).expand(batch_size, batch_size)
        
        labels_i = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_j = labels.unsqueeze(0).expand(batch_size, batch_size)
        
        # Mask for valid pairs (label_i > label_j)
        valid_pairs = (labels_i > labels_j).float()
        
        if valid_pairs.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Ranking loss: score_i should be higher than score_j
        diff = scores_i - scores_j
        loss = F.relu(self.margin - diff) * valid_pairs
        
        # Average over valid pairs
        loss = loss.sum() / (valid_pairs.sum() + 1e-12)
        
        return loss


class ListwiseRankingLoss(nn.Module):
    """
    Listwise ranking loss (ListNet)
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch_size,) or (batch_size, num_classes)
            labels: (batch_size,)
            
        Returns:
            Listwise ranking loss
        """
        # Get scores
        if logits.dim() == 2:
            scores = logits[:, 1] if logits.size(1) == 2 else logits.squeeze()
        else:
            scores = logits
        
        # Predicted distribution
        pred_dist = F.softmax(scores / self.temperature, dim=0)
        
        # Target distribution (based on labels)
        target_dist = F.softmax(labels.float() / self.temperature, dim=0)
        
        # KL divergence
        loss = F.kl_div(
            pred_dist.log(),
            target_dist,
            reduction='batchmean',
        )
        
        return loss


class PRLoss(nn.Module):
    """
    Precision-Recall optimized loss
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
    ):
        """
        Args:
            alpha: Weight for precision
            beta: Weight for recall (F-beta score)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
            threshold: Classification threshold
            
        Returns:
            PR loss
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)[:, 1]
        
        # Predictions
        preds = (probs > threshold).float()
        
        # True positives, false positives, false negatives
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        
        # Precision and recall
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        
        # F-beta score
        f_beta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + 1e-12)
        
        # Loss: 1 - F-beta
        loss = 1 - f_beta
        
        return loss


class RankingLossFactory:
    """
    Factory for creating ranking losses
    """
    
    @staticmethod
    def create_loss(config: dict) -> nn.Module:
        """
        Create ranking loss based on config
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Ranking loss module
        """
        # Default to AUC loss
        loss = AUCLoss(margin=1.0)
        
        logger.info("Created AUC ranking loss")
        
        return loss


if __name__ == "__main__":
    # Test ranking losses
    batch_size = 32
    num_classes = 2
    
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Test AUC loss
    auc_loss = AUCLoss()
    loss = auc_loss(logits, labels)
    print(f"AUC loss: {loss.item():.4f}")
    
    # Test pairwise ranking loss
    pairwise_loss = PairwiseRankingLoss()
    loss = pairwise_loss(logits, labels)
    print(f"Pairwise ranking loss: {loss.item():.4f}")
    
    # Test listwise ranking loss
    listwise_loss = ListwiseRankingLoss()
    loss = listwise_loss(logits, labels)
    print(f"Listwise ranking loss: {loss.item():.4f}")
    
    # Test PR loss
    pr_loss = PRLoss()
    loss = pr_loss(logits, labels)
    print(f"PR loss: {loss.item():.4f}")
