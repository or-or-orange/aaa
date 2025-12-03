"""
Inference Script
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import pickle

sys.path.append(str(Path(__file__).parent))

from utils.config_parser import ConfigParser
from utils.logger import setup_logger
from data.preprocessor import DataPreprocessor
from data.dataset import InferenceDataset
from tree_module.tree_trainer import TreeModelTrainer
from tree_module.rule_extractor import RuleExtractor
from tree_module.path_encoder import PathEncoder
from models.tree_enhanced_model import TreeEnhancedModel
from evaluation.explainer import ModelExplainer


class InferencePipeline:
    """
    End-to-end inference pipeline
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        device: str = 'cuda',
    ):
        """
        Args:
            checkpoint_dir: Directory containing saved models and artifacts
            device: Device for inference
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load artifacts
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all saved artifacts"""
        print("Loading artifacts...")
        
        # Load config
        config_path = self.checkpoint_dir / 'checkpoint_best.pt'
        checkpoint = torch.load(config_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor.load(
            self.checkpoint_dir / 'preprocessor.pkl'
        )
        
        # Load tree model
        self.tree_trainer = TreeModelTrainer.load(
            str(self.checkpoint_dir / 'tree_model.txt'),
            self.config
        )
        
        # Load rule extractor
        if (self.checkpoint_dir / 'rule_extractor.pkl').exists():
            self.rule_extractor = RuleExtractor.load(
                str(self.checkpoint_dir / 'rule_extractor.pkl')
            )
        else:
            self.rule_extractor = None
        
        # Load path encoder
        if (self.checkpoint_dir / 'path_encoder.pkl').exists():
            self.path_encoder = PathEncoder.load(
                str(self.checkpoint_dir / 'path_encoder.pkl')
            )
        else:
            self.path_encoder = None
        
        # Load deep model
        self.model = self._load_model(checkpoint)
        self.model.eval()
        
        print("Artifacts loaded successfully")
    
    def _load_model(self, checkpoint):
        """Load deep learning model"""
        # Reconstruct model architecture
        from models.tree_enhanced_model import ModelFactory
        
        # Get dimensions from checkpoint or config
        num_numerical = len(self.preprocessor.numerical_features)
        num_categorical = len(self.preprocessor.categorical_features)
        
        categorical_cardinalities = []
        for feat in self.preprocessor.categorical_features:
            if feat in self.preprocessor.feature_stats.get('categorical_stats', {}):
                n_unique = self.preprocessor.feature_stats['categorical_stats'][feat]['n_unique']
                categorical_cardinalities.append(n_unique)
            else:
                categorical_cardinalities.append(10)
        
        num_rules = len(self.rule_extractor.selected_rules) if self.rule_extractor else 0
        path_vocab_size = self.path_encoder.get_vocab_size() if self.path_encoder else 1
        num_trees = self.config['tree']['n_estimators']
        num_leaves_per_tree = self.config['tree']['num_leaves']
        
        model = ModelFactory.create_model(
            config=self.config,
            num_numerical=num_numerical,
            num_categorical=num_categorical,
            categorical_cardinalities=categorical_cardinalities,
            num_rules=num_rules,
            path_vocab_size=path_vocab_size,
            num_trees=num_trees,
            num_leaves_per_tree=num_leaves_per_tree,
            num_classes=2,
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        data: pd.DataFrame,
        return_proba: bool = True,
        return_explanations: bool = False,
        batch_size: int = 512,
    ):
        """
        Make predictions on new data
        
        Args:
            data: Input dataframe
            return_proba: Whether to return probabilities
            return_explanations: Whether to return explanations
            batch_size: Batch size for inference
            
        Returns:
            Predictions (and optionally explanations)
        """
        print(f"Making predictions on {len(data)} samples...")
        
        # Preprocess
        # Add dummy target column if not present
        if self.config['data']['target_column'] not in data.columns:
            data[self.config['data']['target_column']] = 0
        
        X, _ = self.preprocessor.transform(data)
        
        # Extract tree features
        tree_features = self._extract_tree_features(X)
        
        # Create dataset
        dataset = InferenceDataset(X, tree_features)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Predict
        all_predictions = []
        all_explanations = []
        
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            output = self.model(batch, return_embeddings=return_explanations)
            logits = output['logits']
            
            if return_proba:
                probs = torch.softmax(logits, dim=1)
                all_predictions.append(probs.cpu().numpy())
            else:
                preds = logits.argmax(dim=1)
                all_predictions.append(preds.cpu().numpy())
            
            # Extract explanations if requested
            if return_explanations:
                explainer = ModelExplainer(self.model, self.config, self.device)
                for i in range(len(logits)):
                    explanation = explainer.explain_instance(batch, i)
                    all_explanations.append(explanation)
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        
        print(f"Predictions completed: {predictions.shape}")
        
        if return_explanations:
            return predictions, all_explanations
        else:
            return predictions
    
    def _extract_tree_features(self, X: pd.DataFrame):
        """Extract tree-derived features"""
        tree_features = {}
        
        # Cross features
        if self.rule_extractor:
            cross_features = self.rule_extractor.generate_cross_features(X)
            tree_features['cross_features'] = cross_features
        else:
            tree_features['cross_features'] = np.zeros((len(X), 0))
        
        # Path features
        if self.path_encoder:
            path_tokens, path_lengths, leaf_indices = \
                self.path_encoder.extract_paths(self.tree_trainer, X)
            
            tree_features['path_tokens'] = path_tokens
            tree_features['path_lengths'] = path_lengths
            tree_features['leaf_indices'] = leaf_indices
        else:
            max_path_len = self.config['tree']['max_path_length']
            n_trees = self.config['tree']['n_estimators']
            
            tree_features['path_tokens'] = np.zeros((len(X), max_path_len), dtype=np.int32)
            tree_features['path_lengths'] = np.zeros(len(X), dtype=np.int32)
            tree_features['leaf_indices'] = np.zeros((len(X), n_trees, 2), dtype=np.int32)
        
        return tree_features
    
    def predict_file(
        self,
        input_path: str,
        output_path: str,
        return_proba: bool = True,
    ):
        """
        Predict on data from file and save results
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save predictions
            return_proba: Whether to return probabilities
        """
        # Load data
        data = pd.read_csv(input_path)
        
        # Predict
        predictions = self.predict(data, return_proba=return_proba)
        
        # Save results
        results = data.copy()
        
        if return_proba:
            results['prob_class_0'] = predictions[:, 0]
            results['prob_class_1'] = predictions[:, 1]
            results['predicted_class'] = predictions.argmax(axis=1)
        else:
            results['predicted_class'] = predictions
        
        results.to_csv(output_path, index=False)
        
        print(f"Predictions saved to {output_path}")


def main(args):
    """Main inference function"""
    # Setup logging
    logger = setup_logger(console=True, file=False)
    
    logger.info("=" * 80)
    logger.info("TREE-ENHANCED DEEP LEARNING INFERENCE")
    logger.info("=" * 80)
    
    # Create pipeline
    pipeline = InferencePipeline(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    
    # Run inference
    if args.input_file:
        pipeline.predict_file(
            input_path=args.input_file,
            output_path=args.output_file,
            return_proba=args.return_proba,
        )
    else:
        logger.error("No input file specified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='Directory containing model checkpoints'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='predictions.csv',
        help='Path to save predictions'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--return-proba',
        action='store_true',
        help='Return probabilities instead of class labels'
    )
    
    args = parser.parse_args()
    main(args)
