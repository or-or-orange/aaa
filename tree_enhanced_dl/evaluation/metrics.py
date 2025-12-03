"""
Evaluation Metrics
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate various evaluation metrics
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metric_names = config['evaluation']['metrics']
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all configured metrics
        
        Args:
            predictions: (n_samples, n_classes) probability predictions
            labels: (n_samples,) true labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Get positive class probabilities
        if predictions.ndim == 2:
            probs = predictions[:, 1]
        else:
            probs = predictions
        
        # Get predicted classes
        pred_classes = (probs >= threshold).astype(int)
        
        # Compute each metric
        for metric_name in self.metric_names:
            try:
                if metric_name == 'auc':
                    metrics['auc'] = self._compute_auc(probs, labels)
                
                elif metric_name == 'pr_auc':
                    metrics['pr_auc'] = self._compute_pr_auc(probs, labels)
                
                elif metric_name == 'f1':
                    metrics['f1'] = self._compute_f1(pred_classes, labels)
                
                elif metric_name == 'ks':
                    metrics['ks'] = self._compute_ks(probs, labels)
                
                elif metric_name == 'ece':
                    metrics['ece'] = self._compute_ece(probs, labels)
                
                elif metric_name == 'accuracy':
                    metrics['accuracy'] = accuracy_score(labels, pred_classes)
                
                elif metric_name == 'precision':
                    metrics['precision'] = precision_score(labels, pred_classes, zero_division=0)
                
                elif metric_name == 'recall':
                    metrics['recall'] = recall_score(labels, pred_classes, zero_division=0)
            
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        return metrics
    
    def _compute_auc(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute ROC AUC"""
        try:
            return roc_auc_score(labels, probs)
        except:
            return 0.0
    
    def _compute_pr_auc(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute Precision-Recall AUC"""
        try:
            return average_precision_score(labels, probs)
        except:
            return 0.0
    
    def _compute_f1(self, pred_classes: np.ndarray, labels: np.ndarray) -> float:
        """Compute F1 score"""
        return f1_score(labels, pred_classes, zero_division=0)
    
    def _compute_ks(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Kolmogorov-Smirnov statistic
        Measures separation between positive and negative distributions
        """
        pos_probs = probs[labels == 1]
        neg_probs = probs[labels == 0]
        
        if len(pos_probs) == 0 or len(neg_probs) == 0:
            return 0.0
        
        # Sort probabilities
        all_probs = np.sort(np.concatenate([pos_probs, neg_probs]))
        
        # Compute cumulative distributions
        pos_cdf = np.array([np.mean(pos_probs <= threshold) for threshold in all_probs])
        neg_cdf = np.array([np.mean(neg_probs <= threshold) for threshold in all_probs])
        
        # KS statistic is maximum difference
        ks = np.max(np.abs(pos_cdf - neg_cdf))
        
        return ks
    
    def _compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error
        Measures calibration of probability predictions
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Average confidence in bin
                avg_confidence = np.mean(probs[in_bin])
                
                # Average accuracy in bin
                avg_accuracy = np.mean(labels[in_bin])
                
                # Add weighted difference to ECE
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return ece
    
    def compute_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Compute confusion matrix"""
        if predictions.ndim == 2:
            probs = predictions[:, 1]
        else:
            probs = predictions
        
        pred_classes = (probs >= threshold).astype(int)
        
        return confusion_matrix(labels, pred_classes)
    
    def compute_classification_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> str:
        """Generate classification report"""
        if predictions.ndim == 2:
            probs = predictions[:, 1]
        else:
            probs = predictions
        
        pred_classes = (probs >= threshold).astype(int)
        
        return classification_report(labels, pred_classes)
    
    def compute_metrics_by_group(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each group
        
        Args:
            predictions: Probability predictions
            labels: True labels
            groups: Group assignments
            threshold: Classification threshold
            
        Returns:
            Dictionary mapping group -> metrics
        """
        group_metrics = {}
        
        for group in np.unique(groups):
            mask = groups == group
            group_preds = predictions[mask]
            group_labels = labels[mask]
            
            metrics = self.compute_metrics(group_preds, group_labels, threshold)
            group_metrics[str(group)] = metrics
        
        return group_metrics


class ThresholdOptimizer:
    """
    Optimize classification threshold
    """
    
    @staticmethod
    def optimize_f1(
        probs: np.ndarray,
        labels: np.ndarray,
        n_thresholds: int = 100,
    ) -> Tuple[float, float]:
        """
        Find threshold that maximizes F1 score
        
        Args:
            probs: Probability predictions
            labels: True labels
            n_thresholds: Number of thresholds to try
            
        Returns:
            Tuple of (best_threshold, best_f1)
        """
        thresholds = np.linspace(0, 1, n_thresholds)
        best_f1 = 0.0
        best_threshold = 0.5
        
        for threshold in thresholds:
            pred_classes = (probs >= threshold).astype(int)
            f1 = f1_score(labels, pred_classes, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1
    
    @staticmethod
    def optimize_precision_recall(
        probs: np.ndarray,
        labels: np.ndarray,
        target_precision: Optional[float] = None,
        target_recall: Optional[float] = None,
        n_thresholds: int = 100,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find threshold that achieves target precision or recall
        
        Args:
            probs: Probability predictions
            labels: True labels
            target_precision: Target precision (if specified)
            target_recall: Target recall (if specified)
            n_thresholds: Number of thresholds to try
            
        Returns:
            Tuple of (best_threshold, metrics)
        """
        thresholds = np.linspace(0, 1, n_thresholds)
        best_threshold = 0.5
        best_metrics = {}
        
        if target_precision is not None:
            # Find threshold that achieves target precision with max recall
            best_recall = 0.0
            
            for threshold in thresholds:
                pred_classes = (probs >= threshold).astype(int)
                precision = precision_score(labels, pred_classes, zero_division=0)
                recall = recall_score(labels, pred_classes, zero_division=0)
                
                if precision >= target_precision and recall > best_recall:
                    best_recall = recall
                    best_threshold = threshold
                    best_metrics = {'precision': precision, 'recall': recall}
        
        elif target_recall is not None:
            # Find threshold that achieves target recall with max precision
            best_precision = 0.0
            
            for threshold in thresholds:
                pred_classes = (probs >= threshold).astype(int)
                precision = precision_score(labels, pred_classes, zero_division=0)
                recall = recall_score(labels, pred_classes, zero_division=0)
                
                if recall >= target_recall and precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
                    best_metrics = {'precision': precision, 'recall': recall}
        
        return best_threshold, best_metrics


if __name__ == "__main__":
    # Test metrics
    import yaml
    
    with open('../configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate dummy predictions
    n_samples = 1000
    labels = np.random.randint(0, 2, n_samples)
    probs = np.random.rand(n_samples)
    predictions = np.stack([1 - probs, probs], axis=1)
    
    # Compute metrics
    calculator = MetricsCalculator(config)
    metrics = calculator.compute_metrics(predictions, labels)
    
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Confusion matrix
    cm = calculator.compute_confusion_matrix(predictions, labels)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Optimize threshold
    best_threshold, best_f1 = ThresholdOptimizer.optimize_f1(probs, labels)
    print(f"\nBest threshold: {best_threshold:.3f}, F1: {best_f1:.4f}")
