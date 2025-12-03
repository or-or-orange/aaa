"""
Tree Model Training Module
Supports LightGBM, XGBoost, and CatBoost
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import pickle
import logging

logger = logging.getLogger(__name__)


class TreeModelTrainer:
    """
    Unified interface for training gradient boosting tree models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tree_config = config['tree']
        self.framework = self.tree_config['framework']
        
        self.model = None
        self.feature_importance = None
        self.is_fitted = False
        
        # Initialize framework-specific model
        self._init_model()
    
    def _init_model(self):
        """Initialize tree model based on framework"""
        if self.framework == 'lightgbm':
            import lightgbm as lgb
            self.model_class = lgb.LGBMClassifier
            self.params = {
                'n_estimators': self.tree_config['n_estimators'],
                'max_depth': self.tree_config['max_depth'],
                'learning_rate': self.tree_config['learning_rate'],
                'num_leaves': self.tree_config['num_leaves'],
                'min_child_samples': self.tree_config['min_child_samples'],
                'objective': 'binary',
                'metric': 'auc',
                'verbose': -1,
                'random_state': self.config['system']['seed'],
            }
            
        elif self.framework == 'xgboost':
            import xgboost as xgb
            self.model_class = xgb.XGBClassifier
            self.params = {
                'n_estimators': self.tree_config['n_estimators'],
                'max_depth': self.tree_config['max_depth'],
                'learning_rate': self.tree_config['learning_rate'],
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'random_state': self.config['system']['seed'],
            }
            
        elif self.framework == 'catboost':
            from catboost import CatBoostClassifier
            self.model_class = CatBoostClassifier
            self.params = {
                'iterations': self.tree_config['n_estimators'],
                'depth': self.tree_config['max_depth'],
                'learning_rate': self.tree_config['learning_rate'],
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'verbose': False,
                'random_state': self.config['system']['seed'],
            }
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        
        logger.info(f"Initialized {self.framework} model with params: {self.params}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        categorical_features: Optional[list] = None,
    ) -> 'TreeModelTrainer':
        """
        Train tree model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            categorical_features: List of categorical feature names
        
        Returns:
            self
        """
        logger.info(f"Training {self.framework} model...")
        
        # Initialize model
        self.model = self.model_class(**self.params)
        
        # Prepare evaluation set
        eval_set = None
        if X_val is not None and y_val is not None:
            if self.framework == 'lightgbm':
                eval_set = [(X_val, y_val)]
            elif self.framework == 'xgboost':
                eval_set = [(X_train, y_train), (X_val, y_val)]
            elif self.framework == 'catboost':
                from catboost import Pool
                eval_set = Pool(X_val, y_val, cat_features=categorical_features)
        
        # Framework-specific fitting
        if self.framework == 'lightgbm':
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                categorical_feature=categorical_features or 'auto',
                callbacks=[
                    __import__('lightgbm').early_stopping(50),
                    __import__('lightgbm').log_evaluation(50),
                ] if eval_set else None,
            )
            
        elif self.framework == 'xgboost':
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50 if eval_set else None,
                verbose=False,
            )
            
        elif self.framework == 'catboost':
            if eval_set is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    cat_features=categorical_features,
                    early_stopping_rounds=50,
                )
            else:
                self.model.fit(
                    X_train, y_train,
                    cat_features=categorical_features,
                )
        
        # Extract feature importance
        self._extract_feature_importance(X_train.columns)
        
        self.is_fitted = True
        logger.info("Tree model training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def _extract_feature_importance(self, feature_names: list):
        """Extract and store feature importance"""
        if self.framework in ['lightgbm', 'xgboost']:
            importance = self.model.feature_importances_
        elif self.framework == 'catboost':
            importance = self.model.get_feature_importance()
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 important features:\n{self.feature_importance.head(10)}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance dataframe"""
        return self.feature_importance
    
    def get_booster(self):
        """Get underlying booster object"""
        if self.framework == 'lightgbm':
            return self.model.booster_
        elif self.framework == 'xgboost':
            return self.model.get_booster()
        elif self.framework == 'catboost':
            return self.model
        return None

    def get_trees(self) -> list:
        """Get list of trees from the model

        Returns:
            List of tree structures (format depends on framework)
        """
        booster = self.get_booster()

        if self.framework == 'lightgbm':
            import lightgbm as lgb

            # Get the actual tree info from the model
            model_dump = booster.dump_model()
            tree_info = model_dump['tree_info']

            # The actual number of trees (may be less than n_estimators due to early stopping)
            actual_n_trees = len(tree_info)

            logger.info(
                f"LightGBM: Extracting {actual_n_trees} trees (early stopping may have reduced from {self.tree_config['n_estimators']})")

            trees = []
            for i in range(actual_n_trees):
                trees.append(tree_info[i])

            return trees

        elif self.framework == 'xgboost':
            trees_df = booster.trees_to_dataframe()

            # Group by tree
            trees = []
            unique_tree_ids = sorted(trees_df['Tree'].unique())

            logger.info(f"XGBoost: Extracting {len(unique_tree_ids)} trees")

            for tree_id in unique_tree_ids:
                tree_data = trees_df[trees_df['Tree'] == tree_id]
                trees.append(tree_data)

            return trees

        elif self.framework == 'catboost':
            # CatBoost tree structure
            n_trees = self.model.tree_count_

            logger.info(f"CatBoost: Extracting {n_trees} trees")

            trees = []
            for i in range(n_trees):
                try:
                    tree_dict = self.model.get_tree_splits(i)
                    trees.append(tree_dict)
                except Exception as e:
                    logger.warning(f"Failed to extract tree {i}: {e}")

            return trees

        logger.warning(f"Unknown framework: {self.framework}")
        return []

    def save(self, path: str):
        """Save model to disk

        Args:
            path: Path to save model (can be str or Path object)
        """
        # Convert to string to handle both str and Path inputs
        path = str(path)

        # Save model based on framework
        if self.framework == 'lightgbm':
            self.model.booster_.save_model(path)
        elif self.framework == 'xgboost':
            self.model.save_model(path)
        elif self.framework == 'catboost':
            self.model.save_model(path)

        # Save metadata
        metadata_path = path + '.meta'
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'feature_importance': self.feature_importance,
                'is_fitted': self.is_fitted,
            }, f)

        logger.info(f"Model saved to {path}")
        logger.info(f"Metadata saved to {metadata_path}")

    @classmethod
    def load(cls, path: str, config: Dict[str, Any]) -> 'TreeModelTrainer':
        """Load model from disk

        Args:
            path: Path to load model from (can be str or Path object)
            config: Configuration dictionary

        Returns:
            Loaded TreeModelTrainer instance
        """
        # Convert to string to handle both str and Path inputs
        path = str(path)

        trainer = cls(config)

        # Load model based on framework
        if trainer.framework == 'lightgbm':
            import lightgbm as lgb
            trainer.model = lgb.Booster(model_file=path)
        elif trainer.framework == 'xgboost':
            import xgboost as xgb
            trainer.model = xgb.Booster()
            trainer.model.load_model(path)
        elif trainer.framework == 'catboost':
            from catboost import CatBoostClassifier
            trainer.model = CatBoostClassifier()
            trainer.model.load_model(path)

        # Load metadata
        metadata_path = path + '.meta'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            trainer.feature_importance = metadata['feature_importance']
            trainer.is_fitted = metadata['is_fitted']

        logger.info(f"Model loaded from {path}")
        logger.info(f"Metadata loaded from {metadata_path}")
        return trainer


class TreeEnsemble:
    """
    Wrapper for managing multiple tree models
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = []
    
    def add_model(self, model: TreeModelTrainer):
        """Add a trained model to ensemble"""
        self.models.append(model)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Average predictions from all models"""
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)
    
    def get_aggregated_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance across models"""
        all_importance = []
        for model in self.models:
            importance = model.get_feature_importance()
            all_importance.append(importance)
        
        # Average importance
        combined = pd.concat(all_importance).groupby('feature').mean()
        return combined.sort_values('importance', ascending=False)


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open('../configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy data
    X_train = pd.DataFrame(np.random.randn(1000, 20))
    y_train = np.random.randint(0, 2, 1000)
    X_val = pd.DataFrame(np.random.randn(200, 20))
    y_val = np.random.randint(0, 2, 200)
    
    # Train model
    trainer = TreeModelTrainer(config)
    trainer.fit(X_train, y_train, X_val, y_val)
    
    # Predict
    proba = trainer.predict_proba(X_val)
    print(f"Prediction shape: {proba.shape}")
    
    # Feature importance
    print(trainer.get_feature_importance().head())
