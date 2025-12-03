"""
Rule Extraction and Cross Feature Generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Rule:
    """
    Represents a single decision rule from tree
    """
    
    def __init__(
        self,
        conditions: List[Tuple[str, str, float]],
        gain: float,
        coverage: float,
        tree_id: int,
        node_id: int,
    ):
        """
        Args:
            conditions: List of (feature, operator, threshold) tuples
            gain: Information gain of this rule
            coverage: Fraction of samples covered by this rule
            tree_id: ID of source tree
            node_id: ID of node in tree
        """
        self.conditions = conditions
        self.gain = gain
        self.coverage = coverage
        self.tree_id = tree_id
        self.node_id = node_id
        self.rule_id = f"tree{tree_id}_node{node_id}"
    
    def __repr__(self) -> str:
        cond_str = " AND ".join([f"{f} {op} {v:.3f}" for f, op, v in self.conditions])
        return f"Rule({self.rule_id}): {cond_str} [gain={self.gain:.3f}, cov={self.coverage:.3f}]"
    
    def evaluate(self, X: pd.DataFrame) -> np.ndarray:
        """
        Evaluate rule on data
        
        Args:
            X: Feature dataframe
            
        Returns:
            Boolean array indicating which samples satisfy the rule
        """
        mask = np.ones(len(X), dtype=bool)
        
        for feature, operator, threshold in self.conditions:
            if feature not in X.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
            
            if operator == '<=':
                mask &= (X[feature] <= threshold).values
            elif operator == '>':
                mask &= (X[feature] > threshold).values
            elif operator == '==':
                mask &= (X[feature] == threshold).values
            elif operator == '!=':
                mask &= (X[feature] != threshold).values
        
        return mask


class RuleExtractor:
    """
    Extract decision rules and cross features from tree models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tree_config = config['tree']
        self.rule_config = self.tree_config['rule_selection']
        
        self.rules = []
        self.selected_rules = []
        self.rule_importance = {}
    
    def extract_rules_from_trees(
        self,
        tree_trainer,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[Rule]:
        """
        Extract all rules from trained tree model
        
        Args:
            tree_trainer: Trained TreeModelTrainer instance
            X: Feature dataframe
            y: Labels
            
        Returns:
            List of Rule objects
        """
        logger.info("Extracting rules from trees...")
        
        framework = tree_trainer.framework
        trees = tree_trainer.get_trees()
        
        all_rules = []
        
        for tree_id, tree in enumerate(trees):
            if framework == 'lightgbm':
                rules = self._extract_rules_lightgbm(tree, tree_id, X, y)
            elif framework == 'xgboost':
                rules = self._extract_rules_xgboost(tree, tree_id, X, y)
            elif framework == 'catboost':
                rules = self._extract_rules_catboost(tree, tree_id, X, y)
            else:
                raise ValueError(f"Unsupported framework: {framework}")
            
            all_rules.extend(rules)
        
        self.rules = all_rules
        logger.info(f"Extracted {len(all_rules)} rules from {len(trees)} trees")
        
        return all_rules
    
    def _extract_rules_lightgbm(
        self,
        tree_dict: Dict,
        tree_id: int,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[Rule]:
        """Extract rules from LightGBM tree"""
        rules = []
        
        def traverse(node, conditions):
            if 'split_feature' not in node:
                # Leaf node - create rule
                gain = node.get('split_gain', 0)
                
                # Evaluate coverage
                if conditions:
                    rule = Rule(conditions, gain, 0, tree_id, node.get('node_index', 0))
                    mask = rule.evaluate(X)
                    coverage = mask.sum() / len(X)
                    rule.coverage = coverage
                    rules.append(rule)
                return
            
            # Internal node
            feature_idx = node['split_feature']
            feature_name = X.columns[feature_idx]
            threshold = node['threshold']
            gain = node.get('split_gain', 0)
            
            # Left child (<=)
            left_conditions = conditions + [(feature_name, '<=', threshold)]
            if 'left_child' in node:
                traverse(node['left_child'], left_conditions)
            
            # Right child (>)
            right_conditions = conditions + [(feature_name, '>', threshold)]
            if 'right_child' in node:
                traverse(node['right_child'], right_conditions)
        
        traverse(tree_dict['tree_structure'], [])
        return rules
    
    def _extract_rules_xgboost(
        self,
        tree_df: pd.DataFrame,
        tree_id: int,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[Rule]:
        """Extract rules from XGBoost tree"""
        rules = []
        
        # Get leaf nodes
        leaf_nodes = tree_df[tree_df['Feature'] == 'Leaf']
        
        for _, leaf in leaf_nodes.iterrows():
            node_id = leaf['Node']
            conditions = []
            
            # Trace path from root to leaf
            current_node = node_id
            while current_node != 0:
                # Find parent
                parent_row = tree_df[
                    (tree_df['Yes'] == current_node) | (tree_df['No'] == current_node)
                ]
                
                if len(parent_row) == 0:
                    break
                
                parent = parent_row.iloc[0]
                feature = parent['Feature']
                threshold = parent['Split']
                gain = parent['Gain']
                
                # Determine direction
                if parent['Yes'] == current_node:
                    operator = '<='
                else:
                    operator = '>'
                
                conditions.insert(0, (feature, operator, threshold))
                current_node = parent['Node']
            
            if conditions:
                rule = Rule(conditions, gain, 0, tree_id, node_id)
                mask = rule.evaluate(X)
                coverage = mask.sum() / len(X)
                rule.coverage = coverage
                rules.append(rule)
        
        return rules
    
    def _extract_rules_catboost(
        self,
        tree_dict: Dict,
        tree_id: int,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[Rule]:
        """Extract rules from CatBoost tree"""
        # CatBoost tree structure is different, simplified extraction
        rules = []
        # Implementation depends on CatBoost version
        # This is a placeholder
        return rules
    
    def select_top_rules(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[Rule]:
        """
        Select top-K rules based on gain, coverage, and redundancy
        
        Args:
            X: Feature dataframe
            y: Labels
            
        Returns:
            List of selected Rule objects
        """
        logger.info("Selecting top rules...")
        
        top_k = self.rule_config['top_k']
        min_gain = self.rule_config['min_gain']
        min_coverage = self.rule_config['min_coverage']
        max_redundancy = self.rule_config['max_redundancy']
        
        # Filter by gain and coverage
        candidate_rules = [
            rule for rule in self.rules
            if rule.gain >= min_gain and rule.coverage >= min_coverage
        ]
        
        logger.info(f"Candidates after filtering: {len(candidate_rules)}")
        
        # Sort by gain
        candidate_rules.sort(key=lambda r: r.gain, reverse=True)
        
        # Select non-redundant rules
        selected = []
        rule_masks = {}
        
        for rule in candidate_rules:
            if len(selected) >= top_k:
                break
            
            # Evaluate rule
            mask = rule.evaluate(X)
            rule_masks[rule.rule_id] = mask
            
            # Check redundancy with already selected rules
            is_redundant = False
            for selected_rule in selected:
                selected_mask = rule_masks[selected_rule.rule_id]
                
                # Compute mutual information / overlap
                overlap = (mask & selected_mask).sum()
                union = (mask | selected_mask).sum()
                
                if union > 0:
                    jaccard = overlap / union
                    if jaccard > max_redundancy:
                        is_redundant = True
                        break
            
            if not is_redundant:
                selected.append(rule)
        
        self.selected_rules = selected
        logger.info(f"Selected {len(selected)} non-redundant rules")
        
        return selected
    
    def generate_cross_features(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate binary cross features from selected rules
        
        Args:
            X: Feature dataframe
            
        Returns:
            Binary array of shape (n_samples, n_rules)
        """
        if not self.selected_rules:
            raise ValueError("Must select rules before generating cross features")
        
        logger.info(f"Generating cross features from {len(self.selected_rules)} rules...")
        
        cross_features = np.zeros((len(X), len(self.selected_rules)), dtype=np.float32)
        
        for i, rule in enumerate(self.selected_rules):
            mask = rule.evaluate(X)
            cross_features[:, i] = mask.astype(np.float32)
        
        logger.info(f"Generated cross features shape: {cross_features.shape}")
        
        return cross_features
    
    def get_rule_metadata(self) -> pd.DataFrame:
        """Get metadata for selected rules"""
        metadata = []
        
        for i, rule in enumerate(self.selected_rules):
            metadata.append({
                'rule_idx': i,
                'rule_id': rule.rule_id,
                'n_conditions': len(rule.conditions),
                'gain': rule.gain,
                'coverage': rule.coverage,
                'conditions': str(rule.conditions),
            })
        
        return pd.DataFrame(metadata)
    
    def save(self, path: str):
        """Save rules to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'rules': self.rules,
                'selected_rules': self.selected_rules,
                'config': self.config,
            }, f)
        logger.info(f"Rules saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'RuleExtractor':
        """Load rules from disk"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        extractor = cls(data['config'])
        extractor.rules = data['rules']
        extractor.selected_rules = data['selected_rules']
        
        logger.info(f"Rules loaded from {path}")
        return extractor


if __name__ == "__main__":
    # Example usage
    import yaml
    from tree_trainer import TreeModelTrainer
    
    with open('../configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy data
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'f{i}' for i in range(10)])
    y = np.random.randint(0, 2, 1000)
    
    # Train tree
    trainer = TreeModelTrainer(config)
    trainer.fit(X, y)
    
    # Extract rules
    extractor = RuleExtractor(config)
    rules = extractor.extract_rules_from_trees(trainer, X, y)
    print(f"Extracted {len(rules)} rules")
    
    # Select top rules
    selected = extractor.select_top_rules(X, y)
    print(f"Selected {len(selected)} rules")
    
    # Generate cross features
    cross_features = extractor.generate_cross_features(X)
    print(f"Cross features shape: {cross_features.shape}")
