"""
Tree Path Extraction and Encoding
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PathToken:
    """
    Represents a single token in a tree path
    """
    
    def __init__(
        self,
        feature: str,
        operator: str,
        threshold: float,
        token_id: int,
    ):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.token_id = token_id
    
    def __repr__(self) -> str:
        return f"Token({self.token_id}): {self.feature} {self.operator} {self.threshold:.3f}"


class PathEncoder:
    """
    Extract and encode tree paths as token sequences
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tree_config = config['tree']
        self.max_path_length = self.tree_config['max_path_length']
        self.token_type = self.tree_config['path_token_type']
        
        self.token_vocab = {}  # token_string -> token_id
        self.id_to_token = {}  # token_id -> PathToken
        self.next_token_id = 1  # 0 reserved for padding
        
        self.paths = []  # List of paths for each sample
        self.leaf_indices = []  # List of (tree_id, leaf_id) for each sample
    
    def extract_paths(
        self,
        tree_trainer,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract tree paths for all samples
        
        Args:
            tree_trainer: Trained TreeModelTrainer instance
            X: Feature dataframe
            
        Returns:
            Tuple of (path_tokens, path_lengths, leaf_indices)
            - path_tokens: (n_samples, max_path_len) token IDs
            - path_lengths: (n_samples,) actual path lengths
            - leaf_indices: (n_samples, n_trees, 2) [tree_id, leaf_id]
        """
        logger.info("Extracting tree paths...")
        
        framework = tree_trainer.framework
        n_samples = len(X)
        n_trees = tree_trainer.tree_config['n_estimators']
        
        # Initialize storage
        all_path_tokens = []
        all_path_lengths = []
        all_leaf_indices = np.zeros((n_samples, n_trees, 2), dtype=np.int32)
        
        # Get predictions to trace paths
        if framework == 'lightgbm':
            paths_data = self._extract_paths_lightgbm(tree_trainer, X)
        elif framework == 'xgboost':
            paths_data = self._extract_paths_xgboost(tree_trainer, X)
        elif framework == 'catboost':
            paths_data = self._extract_paths_catboost(tree_trainer, X)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        # Process paths
        for sample_idx in range(n_samples):
            sample_paths = paths_data[sample_idx]
            
            # Concatenate paths from all trees
            all_tokens = []
            for tree_id, (path, leaf_id) in enumerate(sample_paths):
                # Tokenize path
                tokens = self._tokenize_path(path)
                all_tokens.extend(tokens)
                
                # Store leaf index
                all_leaf_indices[sample_idx, tree_id] = [tree_id, leaf_id]
            
            # Truncate or pad
            path_length = min(len(all_tokens), self.max_path_length)
            padded_tokens = all_tokens[:self.max_path_length]
            padded_tokens += [0] * (self.max_path_length - len(padded_tokens))
            
            all_path_tokens.append(padded_tokens)
            all_path_lengths.append(path_length)
        
        path_tokens = np.array(all_path_tokens, dtype=np.int32)
        path_lengths = np.array(all_path_lengths, dtype=np.int32)
        
        logger.info(f"Extracted paths: tokens shape={path_tokens.shape}, vocab size={len(self.token_vocab)}")
        
        return path_tokens, path_lengths, all_leaf_indices
    
    def _extract_paths_lightgbm(
        self,
        tree_trainer,
        X: pd.DataFrame,
    ) -> List[List[Tuple[List, int]]]:
        """Extract paths from LightGBM model"""
        import lightgbm as lgb
        
        booster = tree_trainer.get_booster()
        n_samples = len(X)
        n_trees = tree_trainer.tree_config['n_estimators']
        
        # Get tree predictions (leaf indices)
        leaf_preds = booster.predict(X, pred_leaf=True)  # (n_samples, n_trees)
        
        paths_data = []
        
        for sample_idx in range(n_samples):
            sample_paths = []
            
            for tree_id in range(n_trees):
                leaf_id = leaf_preds[sample_idx, tree_id]
                
                # Trace path from root to leaf
                tree_dict = booster.dump_model()['tree_info'][tree_id]
                path = self._trace_path_lightgbm(
                    tree_dict['tree_structure'],
                    X.iloc[sample_idx],
                    X.columns,
                )
                
                sample_paths.append((path, leaf_id))
            
            paths_data.append(sample_paths)
        
        return paths_data
    
    def _trace_path_lightgbm(
        self,
        node: Dict,
        sample: pd.Series,
        feature_names: pd.Index,
    ) -> List[Tuple[str, str, float]]:
        """Trace path through LightGBM tree for a single sample"""
        path = []
        
        while 'split_feature' in node:
            feature_idx = node['split_feature']
            feature_name = feature_names[feature_idx]
            threshold = node['threshold']
            
            feature_value = sample[feature_name]
            
            if feature_value <= threshold:
                operator = '<='
                node = node['left_child']
            else:
                operator = '>'
                node = node['right_child']
            
            path.append((feature_name, operator, threshold))
        
        return path
    
    def _extract_paths_xgboost(
        self,
        tree_trainer,
        X: pd.DataFrame,
    ) -> List[List[Tuple[List, int]]]:
        """Extract paths from XGBoost model"""
        # Similar to LightGBM but using XGBoost API
        # Placeholder implementation
        return []
    
    def _extract_paths_catboost(
        self,
        tree_trainer,
        X: pd.DataFrame,
    ) -> List[List[Tuple[List, int]]]:
        """Extract paths from CatBoost model"""
        # Placeholder implementation
        return []
    
    def _tokenize_path(
        self,
        path: List[Tuple[str, str, float]],
    ) -> List[int]:
        """
        Convert path conditions to token IDs
        
        Args:
            path: List of (feature, operator, threshold) tuples
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        for feature, operator, threshold in path:
            if self.token_type == 'condition_value':
                # Include threshold value in token
                # Discretize threshold for vocabulary
                threshold_bucket = int(threshold * 10) / 10  # Round to 1 decimal
                token_str = f"{feature}_{operator}_{threshold_bucket}"
            else:
                # Only condition type
                token_str = f"{feature}_{operator}"
            
            # Get or create token ID
            if token_str not in self.token_vocab:
                self.token_vocab[token_str] = self.next_token_id
                self.id_to_token[self.next_token_id] = PathToken(
                    feature, operator, threshold, self.next_token_id
                )
                self.next_token_id += 1
            
            tokens.append(self.token_vocab[token_str])
        
        return tokens
    
    def decode_path(self, token_ids: List[int]) -> str:
        """Decode token IDs back to human-readable path"""
        conditions = []
        for token_id in token_ids:
            if token_id == 0:  # Padding
                continue
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                conditions.append(f"{token.feature} {token.operator} {token.threshold:.3f}")
        
        return " -> ".join(conditions)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size (including padding token)"""
        return len(self.token_vocab) + 1
    
    def save(self, path: str):
        """Save path encoder to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'token_vocab': self.token_vocab,
                'id_to_token': self.id_to_token,
                'next_token_id': self.next_token_id,
                'config': self.config,
            }, f)
        logger.info(f"Path encoder saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PathEncoder':
        """Load path encoder from disk"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        encoder = cls(data['config'])
        encoder.token_vocab = data['token_vocab']
        encoder.id_to_token = data['id_to_token']
        encoder.next_token_id = data['next_token_id']
        
        logger.info(f"Path encoder loaded from {path}")
        return encoder


if __name__ == "__main__":
    # Example usage
    import yaml
    from tree_trainer import TreeModelTrainer
    
    with open('../configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy data
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f'f{i}' for i in range(10)])
    y = np.random.randint(0, 2, 100)
    
    # Train tree
    trainer = TreeModelTrainer(config)
    trainer.fit(X, y)
    
    # Extract paths
    encoder = PathEncoder(config)
    path_tokens, path_lengths, leaf_indices = encoder.extract_paths(trainer, X)
    
    print(f"Path tokens shape: {path_tokens.shape}")
    print(f"Vocab size: {encoder.get_vocab_size()}")
    print(f"Sample path: {encoder.decode_path(path_tokens[0])}")
