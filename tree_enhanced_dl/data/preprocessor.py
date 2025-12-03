"""
Data Preprocessing Module
Handles missing values, encoding, normalization, and feature engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Unified data preprocessor for numerical and categorical features
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.numerical_features = config['data']['numerical_features']
        self.categorical_features = config['data']['categorical_features']
        self.target_column = config['data']['target_column']
        
        # Preprocessing components
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.numerical_scaler = None
        self.categorical_encoder = None
        self.label_encoder = None
        
        # Feature statistics
        self.feature_stats = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessing components on training data
        
        Args:
            df: Training dataframe
            
        Returns:
            self
        """
        logger.info("Fitting data preprocessor...")
        
        # Separate features and target
        # print(self.target_column)
        # print(df.columns)
        # exit()
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Fit numerical preprocessing
        if self.numerical_features:
            self._fit_numerical(X[self.numerical_features])
            
        # Fit categorical preprocessing
        if self.categorical_features:
            self._fit_categorical(X[self.categorical_features])
            
        # Fit label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        
        # Store feature statistics
        self._compute_feature_stats(X, y)
        
        self.is_fitted = True
        logger.info("Data preprocessor fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Transform data using fitted preprocessors
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (transformed features, encoded labels)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Transform numerical features
        X_transformed = X.copy()
        if self.numerical_features:
            X_transformed[self.numerical_features] = self._transform_numerical(
                X[self.numerical_features]
            )
            
        # Transform categorical features
        if self.categorical_features:
            X_transformed[self.categorical_features] = self._transform_categorical(
                X[self.categorical_features]
            )
            
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        return X_transformed, y_encoded
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit and transform in one step
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (transformed features, encoded labels)
        """
        self.fit(df)
        return self.transform(df)
    
    def _fit_numerical(self, X_num: pd.DataFrame) -> None:
        """Fit numerical feature preprocessing"""
        # Missing value imputation
        strategy = self.config['data']['missing_value_strategy']
        if strategy in ['mean', 'median', 'most_frequent', 'constant']:
            self.numerical_imputer = SimpleImputer(strategy=strategy)
            self.numerical_imputer.fit(X_num)
        
        # Normalization
        X_imputed = self.numerical_imputer.transform(X_num) if self.numerical_imputer else X_num
        
        norm_type = self.config['data']['numerical_normalization']
        if norm_type == 'standard':
            self.numerical_scaler = StandardScaler()
        elif norm_type == 'minmax':
            self.numerical_scaler = MinMaxScaler()
        elif norm_type == 'robust':
            self.numerical_scaler = RobustScaler()
        
        if self.numerical_scaler:
            self.numerical_scaler.fit(X_imputed)
            
        logger.info(f"Fitted numerical preprocessing: {len(self.numerical_features)} features")
    
    def _transform_numerical(self, X_num: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical features"""
        X_transformed = X_num.copy()
        
        # Impute
        if self.numerical_imputer:
            X_transformed = pd.DataFrame(
                self.numerical_imputer.transform(X_transformed),
                columns=X_num.columns,
                index=X_num.index
            )
        
        # Scale
        if self.numerical_scaler:
            X_transformed = pd.DataFrame(
                self.numerical_scaler.transform(X_transformed),
                columns=X_num.columns,
                index=X_num.index
            )
        
        return X_transformed
    
    def _fit_categorical(self, X_cat: pd.DataFrame) -> None:
        """Fit categorical feature preprocessing"""
        # Missing value imputation
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        X_imputed = self.categorical_imputer.fit_transform(X_cat)
        
        # Encoding
        encoding_type = self.config['data']['categorical_encoding']
        if encoding_type == 'ordinal':
            self.categorical_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
        elif encoding_type == 'onehot':
            self.categorical_encoder = OneHotEncoder(
                handle_unknown='ignore',
                sparse=False
            )
        
        self.categorical_encoder.fit(X_imputed)
        
        logger.info(f"Fitted categorical preprocessing: {len(self.categorical_features)} features")
    
    def _transform_categorical(self, X_cat: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features"""
        # Impute
        X_imputed = self.categorical_imputer.transform(X_cat)
        
        # Encode
        X_encoded = self.categorical_encoder.transform(X_imputed)
        
        # Create dataframe
        if self.config['data']['categorical_encoding'] == 'onehot':
            # OneHot creates multiple columns
            feature_names = self.categorical_encoder.get_feature_names_out(X_cat.columns)
            X_transformed = pd.DataFrame(
                X_encoded,
                columns=feature_names,
                index=X_cat.index
            )
        else:
            # Ordinal keeps same columns
            X_transformed = pd.DataFrame(
                X_encoded,
                columns=X_cat.columns,
                index=X_cat.index
            )
        
        return X_transformed
    
    def _compute_feature_stats(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute and store feature statistics"""
        self.feature_stats = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'n_numerical': len(self.numerical_features),
            'n_categorical': len(self.categorical_features),
            'class_distribution': y.value_counts().to_dict(),
            'class_weights': self._compute_class_weights(y),
        }
        
        # Numerical feature stats
        if self.numerical_features:
            self.feature_stats['numerical_stats'] = {
                'mean': X[self.numerical_features].mean().to_dict(),
                'std': X[self.numerical_features].std().to_dict(),
                'missing_rate': X[self.numerical_features].isnull().mean().to_dict(),
            }
        
        # Categorical feature stats
        if self.categorical_features:
            self.feature_stats['categorical_stats'] = {
                feat: {
                    'n_unique': X[feat].nunique(),
                    'top_values': X[feat].value_counts().head(10).to_dict(),
                    'missing_rate': X[feat].isnull().mean(),
                }
                for feat in self.categorical_features
            }
    
    def _compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Compute class weights for imbalanced data"""
        class_counts = y.value_counts()
        n_samples = len(y)
        n_classes = len(class_counts)
        
        weights = {}
        for cls, count in class_counts.items():
            weights[cls] = n_samples / (n_classes * count)
        
        return weights
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names after transformation"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        feature_names = []
        
        # Numerical features
        feature_names.extend(self.numerical_features)
        
        # Categorical features
        if self.config['data']['categorical_encoding'] == 'onehot':
            feature_names.extend(
                self.categorical_encoder.get_feature_names_out(self.categorical_features)
            )
        else:
            feature_names.extend(self.categorical_features)
        
        return feature_names
    
    def save(self, path: str) -> None:
        """Save preprocessor to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'numerical_imputer': self.numerical_imputer,
                'categorical_imputer': self.categorical_imputer,
                'numerical_scaler': self.numerical_scaler,
                'categorical_encoder': self.categorical_encoder,
                'label_encoder': self.label_encoder,
                'feature_stats': self.feature_stats,
                'is_fitted': self.is_fitted,
            }, f)
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load preprocessor from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(data['config'])
        preprocessor.numerical_imputer = data['numerical_imputer']
        preprocessor.categorical_imputer = data['categorical_imputer']
        preprocessor.numerical_scaler = data['numerical_scaler']
        preprocessor.categorical_encoder = data['categorical_encoder']
        preprocessor.label_encoder = data['label_encoder']
        preprocessor.feature_stats = data['feature_stats']
        preprocessor.is_fitted = data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor


class FeatureEngineer:
    """
    Additional feature engineering utilities
    """
    
    @staticmethod
    def create_interaction_features(
        df: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """Create interaction features from feature pairs"""
        df_new = df.copy()
        
        for feat1, feat2 in feature_pairs:
            interaction_name = f"{feat1}_x_{feat2}"
            df_new[interaction_name] = df[feat1] * df[feat2]
        
        return df_new
    
    @staticmethod
    def create_polynomial_features(
        df: pd.DataFrame,
        features: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """Create polynomial features"""
        df_new = df.copy()
        
        for feat in features:
            for d in range(2, degree + 1):
                poly_name = f"{feat}_pow{d}"
                df_new[poly_name] = df[feat] ** d
        
        return df_new
    
    @staticmethod
    def create_binned_features(
        df: pd.DataFrame,
        features: List[str],
        n_bins: int = 10,
        strategy: str = 'quantile'
    ) -> pd.DataFrame:
        """Create binned versions of numerical features"""
        from sklearn.preprocessing import KBinsDiscretizer
        
        df_new = df.copy()
        
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode='ordinal',
            strategy=strategy
        )
        
        for feat in features:
            binned_name = f"{feat}_binned"
            df_new[binned_name] = discretizer.fit_transform(
                df[[feat]]
            ).astype(int)
        
        return df_new


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load config
    with open('../configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample data
    df = pd.DataFrame({
        'num1': np.random.randn(1000),
        'num2': np.random.randn(1000),
        'cat1': np.random.choice(['A', 'B', 'C'], 1000),
        'cat2': np.random.choice(['X', 'Y'], 1000),
        'label': np.random.choice([0, 1], 1000),
    })
    
    # Update config
    config['data']['numerical_features'] = ['num1', 'num2']
    config['data']['categorical_features'] = ['cat1', 'cat2']
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    X_transformed, y_encoded = preprocessor.fit_transform(df)
    
    print("Transformed shape:", X_transformed.shape)
    print("Feature stats:", preprocessor.feature_stats)
