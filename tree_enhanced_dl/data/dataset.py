"""
PyTorch Dataset and DataLoader utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TreeEnhancedDataset(Dataset):
    """
    PyTorch Dataset for tree-enhanced deep learning
    Handles original features, tree-derived features, paths, and labels
    """
    
    def __init__(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        tree_features: Optional[Dict[str, np.ndarray]] = None,
        numerical_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            features: Preprocessed feature dataframe
            labels: Encoded labels
            tree_features: Dictionary containing tree-derived features:
                - 'cross_features': Cross feature indicators (n_samples, n_rules)
                - 'path_tokens': Path token sequences (n_samples, max_path_len)
                - 'path_lengths': Actual path lengths (n_samples,)
                - 'leaf_indices': Leaf indices (n_samples, n_trees, 2) [tree_id, leaf_id]
            numerical_indices: Indices of numerical features
            categorical_indices: Indices of categorical features
        """
        self.features = torch.FloatTensor(features.values)
        self.labels = torch.LongTensor(labels)
        
        # Feature type indices
        self.numerical_indices = numerical_indices or []
        self.categorical_indices = categorical_indices or []
        
        # Tree-derived features
        self.has_tree_features = tree_features is not None
        if self.has_tree_features:
            self.cross_features = torch.FloatTensor(tree_features['cross_features'])
            self.path_tokens = torch.LongTensor(tree_features['path_tokens'])
            self.path_lengths = torch.LongTensor(tree_features['path_lengths'])
            self.leaf_indices = torch.LongTensor(tree_features['leaf_indices'])
        else:
            self.cross_features = None
            self.path_tokens = None
            self.path_lengths = None
            self.leaf_indices = None
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing all features for a sample
        """
        item = {
            'features': self.features[idx],
            'label': self.labels[idx],
            'numerical_features': self.features[idx, self.numerical_indices] if self.numerical_indices else torch.tensor([]),
            'categorical_features': self.features[idx, self.categorical_indices] if self.categorical_indices else torch.tensor([]),
        }
        
        if self.has_tree_features:
            item.update({
                'cross_features': self.cross_features[idx],
                'path_tokens': self.path_tokens[idx],
                'path_length': self.path_lengths[idx],
                'leaf_indices': self.leaf_indices[idx],
            })
        
        return item


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences
    """
    collated = {}
    
    # Stack fixed-size tensors
    for key in ['features', 'label', 'numerical_features', 'categorical_features']:
        if key in batch[0] and batch[0][key].numel() > 0:
            collated[key] = torch.stack([item[key] for item in batch])
    
    # Handle tree features if present
    if 'cross_features' in batch[0]:
        collated['cross_features'] = torch.stack([item['cross_features'] for item in batch])
        collated['path_tokens'] = torch.stack([item['path_tokens'] for item in batch])
        collated['path_length'] = torch.stack([item['path_length'] for item in batch])
        collated['leaf_indices'] = torch.stack([item['leaf_indices'] for item in batch])
    
    return collated


def create_dataloaders(
    train_dataset: TreeEnhancedDataset,
    val_dataset: TreeEnhancedDataset,
    test_dataset: Optional[TreeEnhancedDataset],
    config: Dict[str, Any],
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        config: Configuration dictionary
        sampler: Custom sampler for training data (optional)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = config['training']['batch_size']
    num_workers = config['system']['num_workers']
    pin_memory = config['system']['pin_memory']
    
    # Training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    # Test loader
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
    
    logger.info(f"Created dataloaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader) if test_loader else 0}")
    
    return train_loader, val_loader, test_loader


class InferenceDataset(Dataset):
    """
    Lightweight dataset for inference
    """
    
    def __init__(
        self,
        features: pd.DataFrame,
        tree_features: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            features: Preprocessed feature dataframe
            tree_features: Tree-derived features (optional)
        """
        self.features = torch.FloatTensor(features.values)
        
        self.has_tree_features = tree_features is not None
        if self.has_tree_features:
            self.cross_features = torch.FloatTensor(tree_features['cross_features'])
            self.path_tokens = torch.LongTensor(tree_features['path_tokens'])
            self.path_lengths = torch.LongTensor(tree_features['path_lengths'])
            self.leaf_indices = torch.LongTensor(tree_features['leaf_indices'])
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {'features': self.features[idx]}
        
        if self.has_tree_features:
            item.update({
                'cross_features': self.cross_features[idx],
                'path_tokens': self.path_tokens[idx],
                'path_length': self.path_lengths[idx],
                'leaf_indices': self.leaf_indices[idx],
            })
        
        return item


if __name__ == "__main__":
    # Example usage
    n_samples = 1000
    n_features = 20
    n_rules = 50
    n_trees = 10
    max_path_len = 10
    
    # Create dummy data
    features = pd.DataFrame(np.random.randn(n_samples, n_features))
    labels = np.random.randint(0, 2, n_samples)
    
    tree_features = {
        'cross_features': np.random.randint(0, 2, (n_samples, n_rules)),
        'path_tokens': np.random.randint(0, 100, (n_samples, max_path_len)),
        'path_lengths': np.random.randint(1, max_path_len + 1, n_samples),
        'leaf_indices': np.random.randint(0, 10, (n_samples, n_trees, 2)),
    }
    
    # Create dataset
    dataset = TreeEnhancedDataset(
        features=features,
        labels=labels,
        tree_features=tree_features,
        numerical_indices=list(range(10)),
        categorical_indices=list(range(10, 20)),
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample item keys: {dataset[0].keys()}")
    print(f"Feature shape: {dataset[0]['features'].shape}")
