"""
Complete Tree-Enhanced Deep Learning Model
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .embeddings import UnifiedEmbeddingLayer
from .encoders import SequenceEncoderFactory
from .fusion import FusionModuleFactory
from .heads import HeadFactory

logger = logging.getLogger(__name__)


class TreeEnhancedModel(nn.Module):
    """
    Complete tree-enhanced deep learning model
    Integrates embeddings, sequence encoding, fusion, and classification
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        
        # Embedding layer (initialized later with actual feature counts)
        self.embedding_layer = UnifiedEmbeddingLayer(config)
        
        # Sequence encoder for paths
        self.sequence_encoder = SequenceEncoderFactory.create_encoder(config)
        
        # Fusion module (initialized later)
        self.fusion_module = None
        
        # Classification head (initialized later)
        self.classification_head = None
        
        # Dimensions (set during initialization)
        self.original_feature_dim = 0
        self.path_feature_dim = config['model']['sequence_encoder']['hidden_dim']
        self.cross_feature_dim = config['model']['embedding']['cross_feature_dim']
        
        # For interpretation
        self.attention_weights = {}
        self.feature_contributions = {}
    
    def initialize_model(
        self,
        num_numerical: int,
        num_categorical: int,
        categorical_cardinalities: list,
        num_rules: int,
        path_vocab_size: int,
        num_trees: int,
        num_leaves_per_tree: int,
        num_classes: int = 2,
    ):
        """
        Initialize model components with actual feature dimensions
        
        Args:
            num_numerical: Number of numerical features
            num_categorical: Number of categorical features
            categorical_cardinalities: List of cardinalities for categorical features
            num_rules: Number of cross feature rules
            path_vocab_size: Size of path token vocabulary
            num_trees: Number of trees in ensemble
            num_leaves_per_tree: Maximum leaves per tree
            num_classes: Number of output classes
        """
        # Initialize embeddings
        self.embedding_layer.initialize_embeddings(
            num_numerical=num_numerical,
            num_categorical=num_categorical,
            categorical_cardinalities=categorical_cardinalities,
            num_rules=num_rules,
            path_vocab_size=path_vocab_size,
            num_trees=num_trees,
            num_leaves_per_tree=num_leaves_per_tree,
        )
        
        # Calculate original feature dimension
        num_emb_dim = self.config['model']['embedding']['numerical_dim']
        cat_emb_dim = self.config['model']['embedding']['categorical_dim']
        self.original_feature_dim = num_numerical * num_emb_dim + num_categorical * cat_emb_dim
        
        # Initialize fusion module
        self.fusion_module = FusionModuleFactory.create_fusion(
            config=self.config,
            original_dim=self.original_feature_dim,
            path_dim=self.path_feature_dim,
            cross_dim=self.cross_feature_dim,
        )
        
        # Get fusion output dimension
        if hasattr(self.fusion_module, 'hidden_dim'):
            fusion_output_dim = self.fusion_module.hidden_dim
        else:
            fusion_output_dim = max(
                self.original_feature_dim,
                self.path_feature_dim,
                self.cross_feature_dim
            )
        
        # Initialize classification head
        self.classification_head = HeadFactory.create_head(
            config=self.config,
            input_dim=fusion_output_dim,
            num_classes=num_classes,
        )
        
        logger.info(f"Model initialized: original_dim={self.original_feature_dim}, "
                   f"path_dim={self.path_feature_dim}, cross_dim={self.cross_feature_dim}, "
                   f"fusion_dim={fusion_output_dim}, num_classes={num_classes}")
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            batch: Dictionary containing:
                - numerical_features: (batch, num_numerical)
                - categorical_features: (batch, num_categorical)
                - cross_features: (batch, num_rules)
                - path_tokens: (batch, max_path_len)
                - path_length: (batch,)
                - leaf_indices: (batch, num_trees, 2)
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing:
                - logits: (batch, num_classes)
                - embeddings: (optional) intermediate embeddings
                - attention_weights: (optional) attention weights
        """
        # Get embeddings
        embeddings = self.embedding_layer(batch)
        
        # Process original features (numerical + categorical)
        original_features = []
        if 'numerical' in embeddings:
            # Pool numerical embeddings
            num_emb = embeddings['numerical']  # (batch, num_feat, emb_dim)
            num_pooled = num_emb.flatten(1)  # (batch, num_feat * emb_dim)
            original_features.append(num_pooled)
        
        if 'categorical' in embeddings:
            # Pool categorical embeddings
            cat_emb = embeddings['categorical']  # (batch, num_feat, emb_dim)
            cat_pooled = cat_emb.flatten(1)  # (batch, num_feat * emb_dim)
            original_features.append(cat_pooled)
        
        if original_features:
            original_features = torch.cat(original_features, dim=1)
        else:
            # Fallback if no original features
            batch_size = batch['cross_features'].size(0)
            original_features = torch.zeros(
                batch_size, self.original_feature_dim,
                device=batch['cross_features'].device
            )
        
        # Process path features
        if 'path' in embeddings:
            path_emb = embeddings['path']  # (batch, seq_len, emb_dim)
            path_lengths = batch.get('path_length')
            
            # Encode paths
            _, path_features = self.sequence_encoder(path_emb, path_lengths)
            # path_features: (batch, hidden_dim)
        else:
            batch_size = original_features.size(0)
            path_features = torch.zeros(
                batch_size, self.path_feature_dim,
                device=original_features.device
            )
        
        # Process cross features
        if 'cross' in embeddings:
            cross_features = embeddings['cross']  # (batch, cross_dim)
        else:
            batch_size = original_features.size(0)
            cross_features = torch.zeros(
                batch_size, self.cross_feature_dim,
                device=original_features.device
            )
        
        # Fusion
        fused_features = self.fusion_module(
            original_features,
            path_features,
            cross_features,
        )
        
        # Store attention weights if available
        if hasattr(self.fusion_module, 'get_attention_weights'):
            self.attention_weights['fusion'] = self.fusion_module.get_attention_weights()
        if hasattr(self.fusion_module, 'get_gate_values'):
            self.attention_weights['gates'] = self.fusion_module.get_gate_values()
        
        # Classification
        logits = self.classification_head(fused_features)
        
        # Prepare output
        output = {'logits': logits}
        
        if return_embeddings:
            output['embeddings'] = {
                'original': original_features,
                'path': path_features,
                'cross': cross_features,
                'fused': fused_features,
            }
            output['attention_weights'] = self.attention_weights
        
        return output
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get stored attention weights for interpretation"""
        return self.attention_weights
    
    def compute_feature_contributions(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature contributions for interpretation
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary of feature contributions
        """
        # Forward pass with embeddings
        output = self.forward(batch, return_embeddings=True)
        
        embeddings = output['embeddings']
        logits = output['logits']
        
        # Compute gradients w.r.t. embeddings
        contributions = {}
        
        for key in ['original', 'path', 'cross']:
            if key in embeddings:
                emb = embeddings[key]
                
                # Compute gradient
                if emb.requires_grad:
                    grad = torch.autograd.grad(
                        logits.sum(),
                        emb,
                        retain_graph=True,
                    )[0]
                    
                    # Contribution = embedding * gradient
                    contrib = (emb * grad).sum(dim=-1)
                    contributions[key] = contrib.detach()
        
        return contributions


class ModelFactory:
    """
    Factory for creating and initializing models
    """
    
    @staticmethod
    def create_model(
        config: Dict,
        num_numerical: int,
        num_categorical: int,
        categorical_cardinalities: list,
        num_rules: int,
        path_vocab_size: int,
        num_trees: int,
        num_leaves_per_tree: int,
        num_classes: int = 2,
    ) -> TreeEnhancedModel:
        """
        Create and initialize tree-enhanced model
        
        Args:
            config: Configuration dictionary
            num_numerical: Number of numerical features
            num_categorical: Number of categorical features
            categorical_cardinalities: Cardinalities of categorical features
            num_rules: Number of cross feature rules
            path_vocab_size: Path token vocabulary size
            num_trees: Number of trees
            num_leaves_per_tree: Max leaves per tree
            num_classes: Number of output classes
            
        Returns:
            Initialized TreeEnhancedModel
        """
        model = TreeEnhancedModel(config)
        
        model.initialize_model(
            num_numerical=num_numerical,
            num_categorical=num_categorical,
            categorical_cardinalities=categorical_cardinalities,
            num_rules=num_rules,
            path_vocab_size=path_vocab_size,
            num_trees=num_trees,
            num_leaves_per_tree=num_leaves_per_tree,
            num_classes=num_classes,
        )
        
        return model


if __name__ == "__main__":
    # Test model
    import yaml
    
    with open('../configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = ModelFactory.create_model(
        config=config,
        num_numerical=10,
        num_categorical=5,
        categorical_cardinalities=[10, 20, 5, 15, 8],
        num_rules=50,
        path_vocab_size=100,
        num_trees=10,
        num_leaves_per_tree=31,
        num_classes=2,
    )
    
    # Create dummy batch
    batch = {
        'numerical_features': torch.randn(32, 10),
        'categorical_features': torch.randint(0, 10, (32, 5)),
        'cross_features': torch.randint(0, 2, (32, 50)).float(),
        'path_tokens': torch.randint(0, 100, (32, 10)),
        'path_length': torch.randint(5, 10, (32,)),
        'leaf_indices': torch.randint(0, 10, (32, 10, 2)),
    }
    
    # Forward pass
    output = model(batch, return_embeddings=True)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Embeddings keys: {output['embeddings'].keys()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
