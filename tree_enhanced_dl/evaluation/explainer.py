"""
Model Interpretation and Explanation
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Explain model predictions at individual and global level
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict,
        device: torch.device,
    ):
        """
        Args:
            model: Trained model
            config: Configuration dictionary
            device: Device
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def explain_instance(
        self,
        batch: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        rule_metadata: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Explain prediction for a single instance
        
        Args:
            batch: Input batch
            sample_idx: Index of sample to explain
            rule_metadata: Metadata about cross feature rules
            
        Returns:
            Dictionary containing explanation
        """
        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        output = self.model(batch, return_embeddings=True)
        
        logits = output['logits'][sample_idx]
        probs = torch.softmax(logits, dim=0)
        
        explanation = {
            'prediction': probs.cpu().numpy(),
            'predicted_class': probs.argmax().item(),
            'confidence': probs.max().item(),
        }
        
        # Attention weights
        if 'attention_weights' in output:
            attn_weights = output['attention_weights']
            
            if 'fusion' in attn_weights and attn_weights['fusion'] is not None:
                # Fusion attention: [original, path, cross]
                fusion_attn = attn_weights['fusion'][sample_idx].cpu().numpy()
                explanation['fusion_attention'] = {
                    'original': float(fusion_attn[0, 0]),
                    'path': float(fusion_attn[0, 1]),
                    'cross': float(fusion_attn[0, 2]),
                }
            
            if 'gates' in attn_weights and attn_weights['gates'] is not None:
                # Gate values
                gates = attn_weights['gates'][sample_idx].cpu().numpy()
                explanation['gate_values'] = {
                    'original': float(gates[0]),
                    'path': float(gates[1]),
                    'cross': float(gates[2]),
                }
        
        # Active cross features (rules)
        if 'cross_features' in batch:
            cross_features = batch['cross_features'][sample_idx].cpu().numpy()
            active_rules = np.where(cross_features > 0)[0]
            
            explanation['active_rules'] = active_rules.tolist()
            explanation['num_active_rules'] = len(active_rules)
            
            if rule_metadata is not None:
                # Get rule details
                active_rule_details = []
                for rule_idx in active_rules:
                    if rule_idx < len(rule_metadata):
                        rule_info = rule_metadata.iloc[rule_idx].to_dict()
                        active_rule_details.append(rule_info)
                
                explanation['rule_details'] = active_rule_details
        
        # Path information
        if 'path_tokens' in batch:
            path_tokens = batch['path_tokens'][sample_idx].cpu().numpy()
            path_length = batch['path_length'][sample_idx].item()
            
            explanation['path_length'] = path_length
            explanation['path_tokens'] = path_tokens[:path_length].tolist()
        
        return explanation
    
    def compute_feature_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        method: str = 'gradient',
    ) -> Dict[str, np.ndarray]:
        """
        Compute global feature importance
        
        Args:
            dataloader: Data loader
            method: Importance computation method ('gradient', 'permutation')
            
        Returns:
            Dictionary of feature importances
        """
        if method == 'gradient':
            return self._compute_gradient_importance(dataloader)
        elif method == 'permutation':
            return self._compute_permutation_importance(dataloader)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _compute_gradient_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, np.ndarray]:
        """Compute gradient-based feature importance"""
        self.model.train()  # Enable gradients
        
        importance_scores = {
            'original': [],
            'path': [],
            'cross': [],
        }
        
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with gradients
            output = self.model(batch, return_embeddings=True)
            logits = output['logits']
            embeddings = output['embeddings']
            
            # Compute gradients w.r.t. embeddings
            for key in ['original', 'path', 'cross']:
                if key in embeddings:
                    emb = embeddings[key]
                    emb.requires_grad_(True)
                    
                    # Compute gradient
                    grad = torch.autograd.grad(
                        logits.sum(),
                        emb,
                        retain_graph=True,
                    )[0]
                    
                    # Importance = |embedding * gradient|
                    importance = torch.abs(emb * grad).sum(dim=-1)
                    importance_scores[key].append(importance.detach().cpu().numpy())
        
        self.model.eval()
        
        # Average over batches
        averaged_importance = {}
        for key, scores in importance_scores.items():
            if scores:
                averaged_importance[key] = np.concatenate(scores, axis=0).mean(axis=0)
        
        return averaged_importance
    
    def _compute_permutation_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, np.ndarray]:
        """Compute permutation-based feature importance"""
        # Get baseline performance
        baseline_loss = self._evaluate_loss(dataloader)
        
        importance_scores = {}
        
        # Test each feature type
        for feature_type in ['numerical_features', 'categorical_features', 'cross_features']:
            # Permute this feature type
            permuted_loss = self._evaluate_loss(dataloader, permute_feature=feature_type)
            
            # Importance = increase in loss
            importance = permuted_loss - baseline_loss
            importance_scores[feature_type] = importance
        
        return importance_scores
    
    @torch.no_grad()
    def _evaluate_loss(
        self,
        dataloader: torch.utils.data.DataLoader,
        permute_feature: Optional[str] = None,
    ) -> float:
        """Evaluate loss, optionally with permuted features"""
        total_loss = 0.0
        n_samples = 0
        
        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Permute if specified
            if permute_feature and permute_feature in batch:
                perm_idx = torch.randperm(batch[permute_feature].size(0))
                batch[permute_feature] = batch[permute_feature][perm_idx]
            
            # Forward pass
            output = self.model(batch)
            logits = output['logits']
            labels = batch['label']
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            total_loss += loss.item() * len(labels)
            n_samples += len(labels)
        
        return total_loss / n_samples
    
    def visualize_attention(
        self,
        explanation: Dict,
        save_path: Optional[str] = None,
    ):
        """
        Visualize attention weights
        
        Args:
            explanation: Explanation dictionary from explain_instance
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Fusion attention
        if 'fusion_attention' in explanation:
            attn = explanation['fusion_attention']
            
            axes[0].bar(attn.keys(), attn.values())
            axes[0].set_title('Fusion Attention Weights')
            axes[0].set_ylabel('Weight')
            axes[0].set_ylim([0, 1])
        
        # Gate values
        if 'gate_values' in explanation:
            gates = explanation['gate_values']
            
            axes[1].bar(gates.keys(), gates.values())
            axes[1].set_title('Gate Values')
            axes[1].set_ylabel('Value')
            axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Attention visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_rule_importance(
        self,
        rule_metadata: pd.DataFrame,
        top_k: int = 20,
        save_path: Optional[str] = None,
    ):
        """
        Visualize top-K important rules
        
        Args:
            rule_metadata: Rule metadata dataframe
            top_k: Number of top rules to show
            save_path: Path to save figure
        """
        # Sort by gain
        top_rules = rule_metadata.nlargest(top_k, 'gain')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gain
        axes[0].barh(range(top_k), top_rules['gain'].values)
        axes[0].set_yticks(range(top_k))
        axes[0].set_yticklabels(top_rules['rule_id'].values, fontsize=8)
        axes[0].set_xlabel('Gain')
        axes[0].set_title(f'Top {top_k} Rules by Gain')
        axes[0].invert_yaxis()
        
        # Coverage
        axes[1].barh(range(top_k), top_rules['coverage'].values)
        axes[1].set_yticks(range(top_k))
        axes[1].set_yticklabels(top_rules['rule_id'].values, fontsize=8)
        axes[1].set_xlabel('Coverage')
        axes[1].set_title(f'Top {top_k} Rules by Coverage')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Rule importance visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(
        self,
        explanations: List[Dict],
        save_path: str,
    ):
        """
        Generate comprehensive explanation report
        
        Args:
            explanations: List of explanations for multiple samples
            save_path: Path to save report
        """
        report = []
        
        report.append("=" * 80)
        report.append("MODEL EXPLANATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        
        avg_confidence = np.mean([exp['confidence'] for exp in explanations])
        report.append(f"Average Confidence: {avg_confidence:.4f}")
        
        if 'num_active_rules' in explanations[0]:
            avg_rules = np.mean([exp['num_active_rules'] for exp in explanations])
            report.append(f"Average Active Rules: {avg_rules:.2f}")
        
        report.append("")
        
        # Individual explanations
        report.append("INDIVIDUAL EXPLANATIONS")
        report.append("-" * 80)
        
        for i, exp in enumerate(explanations[:10]):  # Show first 10
            report.append(f"\nSample {i}:")
            report.append(f"  Predicted Class: {exp['predicted_class']}")
            report.append(f"  Confidence: {exp['confidence']:.4f}")
            
            if 'fusion_attention' in exp:
                report.append(f"  Fusion Attention: {exp['fusion_attention']}")
            
            if 'active_rules' in exp:
                report.append(f"  Active Rules: {exp['active_rules'][:5]}...")  # Show first 5
        
        # Write report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Explanation report saved to {save_path}")


if __name__ == "__main__":
    # Test explainer (requires trained model)
    pass
