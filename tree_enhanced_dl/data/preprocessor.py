"""
Data Preprocessing Module
Handles missing values, encoding, normalization, and feature engineering
"""

import numpy as np
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessor with Label Encoding for categorical features
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']

        # Encoders
        self.label_encoders = {}  # {feature_name: LabelEncoder}
        self.target_encoder = None
        self.scaler = StandardScaler()

        # Feature information
        self.numerical_features = []
        self.categorical_features = []
        self.original_categorical_features = []  # 保存原始类别特征名
        self.feature_names = []
        self.categorical_cardinalities = []  # 每个类别特征的唯一值数量

        # Statistics
        self.feature_stats = {}

    def fit_transform(
            self,
            df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data

        Args:
            df: Input dataframe

        Returns:
            X: Transformed features (numpy array)
            y: Transformed labels (numpy array)
        """
        logger.info("Fitting preprocessor...")

        # Separate features and target
        target_col = self.data_config['target_column']
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        # Identify feature types
        self.numerical_features = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        self.categorical_features = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        self.original_categorical_features = self.categorical_features.copy()

        logger.info(f"Found {len(self.numerical_features)} numerical features")
        logger.info(f"Found {len(self.categorical_features)} categorical features")

        # ✅ 核心修改：Label Encoding for categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            # 处理缺失值
            X[col] = X[col].fillna('missing')
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

            # 记录类别数量（用于后续embedding）
            self.categorical_cardinalities.append(len(le.classes_))

            logger.info(f"Encoded categorical feature '{col}': {len(le.classes_)} categories")

        # 处理数值特征的缺失值
        for col in self.numerical_features:
            X[col] = X[col].fillna(X[col].median())

        # 标准化数值特征
        if len(self.numerical_features) > 0:
            X[self.numerical_features] = self.scaler.fit_transform(
                X[self.numerical_features]
            )

        # ✅ 重要：更新特征列表（现在所有特征都是数值型）
        # 保持原始顺序：数值特征 + 编码后的类别特征
        self.feature_names = self.numerical_features + self.original_categorical_features

        # 编码目标变量
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y.astype(str))
            logger.info(f"Encoded target: {self.target_encoder.classes_}")
        else:
            y = y.values

        # 计算统计信息
        self._compute_statistics(X, y)

        # 转换为numpy数组（按正确的列顺序）
        X = X[self.feature_names].values

        logger.info(f"Preprocessing completed: X.shape={X.shape}, y.shape={y.shape}")

        return X, y

    def transform(
            self,
            df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new data using fitted preprocessor

        Args:
            df: Input dataframe

        Returns:
            X: Transformed features
            y: Transformed labels
        """
        target_col = self.data_config['target_column']
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        # Transform categorical features
        for col in self.original_categorical_features:
            X[col] = X[col].fillna('missing')
            # ✅ 处理未见过的类别
            X[col] = X[col].astype(str).apply(
                lambda x: x if x in self.label_encoders[col].classes_ else 'missing'
            )
            X[col] = self.label_encoders[col].transform(X[col])

        # Transform numerical features
        for col in self.numerical_features:
            X[col] = X[col].fillna(X[col].median())

        if len(self.numerical_features) > 0:
            X[self.numerical_features] = self.scaler.transform(
                X[self.numerical_features]
            )

        # Transform target
        if self.target_encoder is not None:
            y = y.astype(str).apply(
                lambda x: x if x in self.target_encoder.classes_ else self.target_encoder.classes_[0]
            )
            y = self.target_encoder.transform(y)
        else:
            y = y.values

        # Convert to numpy array
        X = X[self.feature_names].values

        return X, y

    def _compute_statistics(self, X: pd.DataFrame, y: np.ndarray):
        """Compute feature statistics"""
        self.feature_stats = {
            'n_features': len(self.feature_names),
            'n_numerical': len(self.numerical_features),
            'n_categorical': len(self.original_categorical_features),
            'categorical_cardinalities': self.categorical_cardinalities,
            'feature_names': self.feature_names,
        }

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        if self.target_encoder is not None:
            class_names = self.target_encoder.inverse_transform(unique)
            self.feature_stats['class_distribution'] = dict(zip(class_names, counts))
        else:
            self.feature_stats['class_distribution'] = dict(zip(unique, counts))

    def get_feature_names(self) -> list:
        """Get feature names"""
        return self.feature_names

    def get_categorical_cardinalities(self) -> list:
        """Get cardinalities of categorical features"""
        return self.categorical_cardinalities

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform target labels"""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(y)
        return y

    def save(self, path: str):
        """Save preprocessor"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str) -> 'DataPreprocessor':
        """Load preprocessor"""
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
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
