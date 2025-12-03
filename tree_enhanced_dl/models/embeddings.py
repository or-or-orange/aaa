"""
Embedding Layers for Features, Paths, and Tree Components
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NumericalEmbedding(nn.Module):
    """
    Embedding layer for numerical features
    Supports both linear projection and binning strategies
    """
    
    def __init__(
        self,
        num_features: int,
        embedding_dim: int,
        use_binning: bool = False,
        num_bins: int = 10,
    ):
        """
        Args:
            num_features: Number of numerical features
            embedding_dim: Embedding dimension
            use_binning: Whether to use binning before embedding
            num_bins: Number of bins if using binning
        """
        super().__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.use_binning = use_binning
        self.num_bins = num_bins
        
        if use_binning:
            # Bin + embed approach
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_bins, embedding_dim)
                for _ in range(num_features)
            ])
            self.register_buffer('bin_edges', torch.linspace(-3, 3, num_bins + 1))
        else:
            # Linear projection
            self.projection = nn.Linear(num_features, num_features * embedding_dim)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features)
            
        Returns:
            (batch_size, num_features, embedding_dim)
        """
        batch_size = x.size(0)
        
        if self.use_binning:
            # Digitize into bins
            embeddings = []
            for i in range(self.num_features):
                bins = torch.bucketize(x[:, i], self.bin_edges) - 1
                bins = bins.clamp(0, self.num_bins - 1)
                emb = self.embeddings[i](bins)
                embeddings.append(emb)
            
            output = torch.stack(embeddings, dim=1)  # (batch, num_features, emb_dim)
        else:
            # Linear projection
            projected = self.projection(x)  # (batch, num_features * emb_dim)
            output = projected.view(batch_size, self.num_features, self.embedding_dim)
        
        output = self.layer_norm(output)
        return output


class CategoricalEmbedding(nn.Module):
    """
    Embedding layer for categorical features
    """
    
    def __init__(
        self,
        cardinalities: List[int],
        embedding_dim: int,
        embedding_dims: Optional[List[int]] = None,
    ):
        """
        Args:
            cardinalities: List of cardinalities for each categorical feature
            embedding_dim: Default embedding dimension
            embedding_dims: Optional list of custom embedding dims per feature
        """
        super().__init__()
        
        self.num_features = len(cardinalities)
        self.cardinalities = cardinalities
        
        # Determine embedding dimension for each feature
        if embedding_dims is None:
            # Rule of thumb: min(50, (cardinality + 1) // 2)
            embedding_dims = [
                min(embedding_dim, (card + 1) // 2)
                for card in cardinalities
            ]
        
        self.embedding_dims = embedding_dims
        
        # Create embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, emb_dim, padding_idx=0)  # +1 for unknown
            for card, emb_dim in zip(cardinalities, embedding_dims)
        ])
        
        # Project to uniform dimension
        self.projections = nn.ModuleList([
            nn.Linear(emb_dim, embedding_dim)
            for emb_dim in embedding_dims
        ])
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features) - integer encoded categories
            
        Returns:
            (batch_size, num_features, embedding_dim)
        """
        embeddings = []
        
        for i in range(self.num_features):
            # Clamp to valid range
            indices = x[:, i].long().clamp(0, self.cardinalities[i])
            
            # Embed
            emb = self.embeddings[i](indices)
            
            # Project to uniform dimension
            emb = self.projections[i](emb)
            
            embeddings.append(emb)
        
        output = torch.stack(embeddings, dim=1)  # (batch, num_features, emb_dim)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class CrossFeatureEmbedding(nn.Module):
    """
    Embedding layer for tree-derived cross features
    """
    
    def __init__(
        self,
        num_rules: int,
        embedding_dim: int,
        use_binary: bool = True,
    ):
        """
        Args:
            num_rules: Number of cross feature rules
            embedding_dim: Embedding dimension
            use_binary: If True, treat as binary indicators; else as embeddings
        """
        super().__init__()
        
        self.num_rules = num_rules
        self.embedding_dim = embedding_dim
        self.use_binary = use_binary
        
        if use_binary:
            # Project binary indicators to embedding space
            self.projection = nn.Linear(num_rules, embedding_dim)
        else:
            # Treat each rule as a categorical feature
            self.embeddings = nn.Embedding(num_rules, embedding_dim)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_rules) - binary indicators or rule indices
            
        Returns:
            (batch_size, embedding_dim) or (batch_size, num_rules, embedding_dim)
        """
        if self.use_binary:
            # Pool binary indicators
            output = self.projection(x)  # (batch, emb_dim)
            output = self.layer_norm(output)
        else:
            # Embed each active rule
            active_rules = x.long()
            output = self.embeddings(active_rules)  # (batch, num_rules, emb_dim)
            output = self.layer_norm(output)
        
        return output


class PathTokenEmbedding(nn.Module):
    """
    Embedding layer for path token sequences
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_path_length: int,
        use_positional: bool = True,
    ):
        """
        Args:
            vocab_size: Size of token vocabulary
            embedding_dim: Embedding dimension
            max_path_length: Maximum path length
            use_positional: Whether to add positional encoding
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_path_length = max_path_length
        self.use_positional = use_positional
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )
        
        # Positional embeddings
        if use_positional:
            self.position_embeddings = nn.Embedding(
                max_path_length,
                embedding_dim,
            )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        path_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch_size, max_path_length)
            path_lengths: (batch_size,) - actual lengths
            
        Returns:
            (batch_size, max_path_length, embedding_dim)
        """
        batch_size, seq_len = token_ids.size()
        
        # Token embeddings
        token_emb = self.token_embeddings(token_ids)
        
        # Positional embeddings
        if self.use_positional:
            positions = torch.arange(seq_len, device=token_ids.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.position_embeddings(positions)
            
            embeddings = token_emb + pos_emb
        else:
            embeddings = token_emb
        
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class LeafIndexEmbedding(nn.Module):
    """
    Embedding layer for tree leaf indices
    """
    
    def __init__(
        self,
        num_trees: int,
        num_leaves_per_tree: int,
        embedding_dim: int,
    ):
        """
        Args:
            num_trees: Number of trees in ensemble
            num_leaves_per_tree: Maximum number of leaves per tree
            embedding_dim: Embedding dimension
        """
        super().__init__()
        
        self.num_trees = num_trees
        self.num_leaves_per_tree = num_leaves_per_tree
        self.embedding_dim = embedding_dim
        
        # Tree ID embeddings
        self.tree_embeddings = nn.Embedding(num_trees, embedding_dim)
        
        # Leaf ID embeddings (shared across trees)
        self.leaf_embeddings = nn.Embedding(num_leaves_per_tree, embedding_dim)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, leaf_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            leaf_indices: (batch_size, num_trees, 2) - [tree_id, leaf_id]
            
        Returns:
            (batch_size, num_trees, embedding_dim)
        """
        tree_ids = leaf_indices[:, :, 0].long()
        leaf_ids = leaf_indices[:, :, 1].long()
        
        # Clamp to valid range
        tree_ids = tree_ids.clamp(0, self.num_trees - 1)
        leaf_ids = leaf_ids.clamp(0, self.num_leaves_per_tree - 1)
        
        # Embed
        tree_emb = self.tree_embeddings(tree_ids)
        leaf_emb = self.leaf_embeddings(leaf_ids)
        
        # Combine
        embeddings = tree_emb + leaf_emb
        embeddings = self.layer_norm(embeddings)
        
        return embeddings


class UnifiedEmbeddingLayer(nn.Module):
    """
    Unified embedding layer that handles all feature types
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        emb_config = config['model']['embedding']
        
        # Feature counts (to be set during initialization)
        self.num_numerical = 0
        self.num_categorical = 0
        self.categorical_cardinalities = []
        
        # Embedding dimensions
        self.numerical_dim = emb_config['numerical_dim']
        self.categorical_dim = emb_config['categorical_dim']
        self.cross_feature_dim = emb_config['cross_feature_dim']
        self.path_token_dim = emb_config['path_token_dim']
        self.leaf_dim = emb_config['leaf_dim']
        
        # Embedding layers (initialized later)
        self.numerical_embedding = None
        self.categorical_embedding = None
        self.cross_feature_embedding = None
        self.path_token_embedding = None
        self.leaf_index_embedding = None
    
    def initialize_embeddings(
        self,
        num_numerical: int,
        num_categorical: int,
        categorical_cardinalities: List[int],
        num_rules: int,
        path_vocab_size: int,
        num_trees: int,
        num_leaves_per_tree: int,
    ):
        """
        Initialize embedding layers with actual feature counts
        """
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.categorical_cardinalities = categorical_cardinalities
        
        # Numerical features
        if num_numerical > 0:
            self.numerical_embedding = NumericalEmbedding(
                num_numerical,
                self.numerical_dim,
            )
        
        # Categorical features
        if num_categorical > 0:
            self.categorical_embedding = CategoricalEmbedding(
                categorical_cardinalities,
                self.categorical_dim,
            )
        
        # Cross features
        if num_rules > 0:
            self.cross_feature_embedding = CrossFeatureEmbedding(
                num_rules,
                self.cross_feature_dim,
            )
        
        # Path tokens
        if path_vocab_size > 0:
            self.path_token_embedding = PathTokenEmbedding(
                path_vocab_size,
                self.path_token_dim,
                self.config['tree']['max_path_length'],
            )
        
        # Leaf indices
        if num_trees > 0:
            self.leaf_index_embedding = LeafIndexEmbedding(
                num_trees,
                num_leaves_per_tree,
                self.leaf_dim,
            )
        
        logger.info(f"Initialized embeddings: num={num_numerical}, cat={num_categorical}, "
                   f"rules={num_rules}, path_vocab={path_vocab_size}, trees={num_trees}")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all embedding layers
        
        Args:
            batch: Dictionary containing:
                - numerical_features: (batch, num_numerical)
                - categorical_features: (batch, num_categorical)
                - cross_features: (batch, num_rules)
                - path_tokens: (batch, max_path_len)
                - path_length: (batch,)
                - leaf_indices: (batch, num_trees, 2)
        
        Returns:
            Dictionary of embeddings
        """
        embeddings = {}
        
        # Numerical features
        if self.numerical_embedding is not None and 'numerical_features' in batch:
            num_feat = batch['numerical_features']
            if num_feat.numel() > 0:
                embeddings['numerical'] = self.numerical_embedding(num_feat)
        
        # Categorical features
        if self.categorical_embedding is not None and 'categorical_features' in batch:
            cat_feat = batch['categorical_features']
            if cat_feat.numel() > 0:
                embeddings['categorical'] = self.categorical_embedding(cat_feat)
        
        # Cross features
        if self.cross_feature_embedding is not None and 'cross_features' in batch:
            embeddings['cross'] = self.cross_feature_embedding(batch['cross_features'])
        
        # Path tokens
        if self.path_token_embedding is not None and 'path_tokens' in batch:
            embeddings['path'] = self.path_token_embedding(
                batch['path_tokens'],
                batch.get('path_length'),
            )
        
        # Leaf indices
        if self.leaf_index_embedding is not None and 'leaf_indices' in batch:
            embeddings['leaf'] = self.leaf_index_embedding(batch['leaf_indices'])
        
        return embeddings


if __name__ == "__main__":
    # Test embeddings
    batch_size = 32
    
    # Test numerical embedding
    num_emb = NumericalEmbedding(10, 16)
    x_num = torch.randn(batch_size, 10)
    out_num = num_emb(x_num)
    print(f"Numerical embedding output: {out_num.shape}")
    
    # Test categorical embedding
    cat_emb = CategoricalEmbedding([10, 20, 5], 16)
    x_cat = torch.randint(0, 10, (batch_size, 3))
    out_cat = cat_emb(x_cat)
    print(f"Categorical embedding output: {out_cat.shape}")
    
    # Test path token embedding
    path_emb = PathTokenEmbedding(100, 32, 10)
    x_path = torch.randint(0, 100, (batch_size, 10))
    out_path = path_emb(x_path)
    print(f"Path embedding output: {out_path.shape}")
