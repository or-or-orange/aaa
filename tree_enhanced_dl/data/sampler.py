"""
Custom samplers for imbalanced data
"""

import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class ClassBalancedSampler(Sampler):
    """
    Sampler that balances classes by oversampling minority classes
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        num_samples: int = None,
        replacement: bool = True,
    ):
        """
        Args:
            labels: Array of class labels
            num_samples: Number of samples to draw per epoch
            replacement: Whether to sample with replacement
        """
        self.labels = labels
        self.replacement = replacement
        
        # Compute class weights
        class_counts = Counter(labels)
        self.num_classes = len(class_counts)
        
        # Weight inversely proportional to class frequency
        weights = np.zeros(len(labels))
        for cls, count in class_counts.items():
            cls_weight = 1.0 / count
            weights[labels == cls] = cls_weight
        
        # Normalize weights
        self.weights = weights / weights.sum()
        
        # Number of samples per epoch
        if num_samples is None:
            self.num_samples = len(labels)
        else:
            self.num_samples = num_samples
        
        logger.info(f"ClassBalancedSampler: {self.num_classes} classes, {self.num_samples} samples/epoch")
        for cls, count in sorted(class_counts.items()):
            logger.info(f"  Class {cls}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            torch.from_numpy(self.weights).float(),
            self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.num_samples


class StratifiedBatchSampler(Sampler):
    """
    Sampler that ensures each batch has balanced class distribution
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        drop_last: bool = True,
    ):
        """
        Args:
            labels: Array of class labels
            batch_size: Size of each batch
            drop_last: Whether to drop the last incomplete batch
        """
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by class
        self.class_indices = {}
        for cls in np.unique(labels):
            self.class_indices[cls] = np.where(labels == cls)[0]
        
        self.num_classes = len(self.class_indices)
        self.samples_per_class = batch_size // self.num_classes
        
        # Compute number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_size // self.samples_per_class
        
        if not self.drop_last:
            self.num_batches += 1
        
        logger.info(f"StratifiedBatchSampler: {self.num_classes} classes, {self.samples_per_class} samples/class/batch")
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each class
        shuffled_indices = {
            cls: np.random.permutation(indices)
            for cls, indices in self.class_indices.items()
        }
        
        for batch_idx in range(self.num_batches):
            batch = []
            
            for cls in sorted(shuffled_indices.keys()):
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                
                cls_indices = shuffled_indices[cls]
                
                # Handle last batch
                if end_idx > len(cls_indices):
                    if self.drop_last:
                        break
                    else:
                        end_idx = len(cls_indices)
                
                batch.extend(cls_indices[start_idx:end_idx].tolist())
            
            if len(batch) > 0:
                # Shuffle within batch
                np.random.shuffle(batch)
                yield batch
    
    def __len__(self) -> int:
        return self.num_batches


class MinorityOversamplingWrapper:
    """
    Wrapper that oversamples minority class to match majority class
    """
    
    @staticmethod
    def oversample(
        features: np.ndarray,
        labels: np.ndarray,
        target_ratio: float = 1.0,
    ) -> tuple:
        """
        Oversample minority class
        
        Args:
            features: Feature array
            labels: Label array
            target_ratio: Target ratio of minority/majority (1.0 = balanced)
        
        Returns:
            Tuple of (oversampled_features, oversampled_labels)
        """
        from sklearn.utils import resample
        
        # Identify majority and minority classes
        class_counts = Counter(labels)
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        majority_count = class_counts[majority_class]
        minority_count = class_counts[minority_class]
        
        # Calculate target minority count
        target_minority_count = int(majority_count * target_ratio)
        
        # Separate majority and minority
        majority_mask = labels == majority_class
        minority_mask = labels == minority_class
        
        majority_features = features[majority_mask]
        majority_labels = labels[majority_mask]
        
        minority_features = features[minority_mask]
        minority_labels = labels[minority_mask]
        
        # Oversample minority
        minority_features_resampled, minority_labels_resampled = resample(
            minority_features,
            minority_labels,
            n_samples=target_minority_count,
            replace=True,
            random_state=42,
        )
        
        # Combine
        features_balanced = np.vstack([majority_features, minority_features_resampled])
        labels_balanced = np.hstack([majority_labels, minority_labels_resampled])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(labels_balanced))
        features_balanced = features_balanced[shuffle_idx]
        labels_balanced = labels_balanced[shuffle_idx]
        
        logger.info(f"Oversampling: {minority_count} -> {target_minority_count} (ratio: {minority_count/majority_count:.2f} -> {target_ratio:.2f})")
        
        return features_balanced, labels_balanced


if __name__ == "__main__":
    # Example usage
    n_samples = 1000
    labels = np.array([0] * 900 + [1] * 100)  # Imbalanced
    
    # Class-balanced sampler
    sampler = ClassBalancedSampler(labels, num_samples=1000)
    sampled_indices = list(sampler)
    sampled_labels = labels[sampled_indices]
    print(f"Original distribution: {Counter(labels)}")
    print(f"Sampled distribution: {Counter(sampled_labels)}")
    
    # Stratified batch sampler
    batch_sampler = StratifiedBatchSampler(labels, batch_size=32)
    for i, batch in enumerate(batch_sampler):
        if i == 0:
            batch_labels = labels[batch]
            print(f"First batch distribution: {Counter(batch_labels)}")
        if i >= 2:
            break
