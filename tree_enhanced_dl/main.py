"""
Main Training Script
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.config_parser import ConfigParser
from utils.logger import setup_logger, TensorBoardLogger, WandBLogger
from data.preprocessor import DataPreprocessor
from data.dataset import TreeEnhancedDataset, create_dataloaders
from data.sampler import ClassBalancedSampler
from tree_module.tree_trainer import TreeModelTrainer
from tree_module.rule_extractor import RuleExtractor
from tree_module.path_encoder import PathEncoder
from models.tree_enhanced_model import ModelFactory
from losses.combined_loss import LossFactory
from training.trainer import Trainer
from training.scheduler import SchedulerFactory
from evaluation.metrics import MetricsCalculator
from evaluation.explainer import ModelExplainer


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(config: dict, logger):
    """Load and preprocess data"""
    logger.info("Loading data...")
    
    # Load dataframes
    train_df = pd.read_csv(config['data']['train_path'],
                           sep=";",  # 关键：指定分隔符为分号
                           encoding="latin-1"  # 这个数据集的编码通常是latin-1
                           )
    val_df = pd.read_csv(config['data']['val_path'],
                         sep=";",  # 关键：指定分隔符为分号
                         encoding="latin-1"  # 这个数据集的编码通常是latin-1
                         )
    
    test_df = None
    if config['data'].get('test_path'):
        test_df = pd.read_csv(config['data']['test_path'],
                              sep=";",  # 关键：指定分隔符为分号
                              encoding="latin-1"  # 这个数据集的编码通常是latin-1
                              )
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    
    X_test, y_test = None, None
    if test_df is not None:
        X_test, y_test = preprocessor.transform(test_df)
    
    # Save preprocessor
    save_dir = Path(config['system']['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(save_dir / 'preprocessor.pkl')
    
    logger.info(f"Preprocessed features: {X_train.shape[1]}")
    logger.info(f"Class distribution: {preprocessor.feature_stats['class_distribution']}")
    
    return preprocessor, (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_tree_model(config: dict, preprocessor, X_train, y_train, X_val, y_val, logger):
    """Train tree model and extract features"""
    logger.info("Training tree model...")

    # Train tree
    tree_trainer = TreeModelTrainer(config)
    tree_trainer.fit(
        X_train, y_train,
        X_val, y_val,
        categorical_features=preprocessor.categorical_features,
    )

    # Save tree model
    save_dir = Path(config['system']['checkpoint']['save_dir'])
    tree_trainer.save(str(save_dir / 'tree_model.txt'))  # ✅ 已修复

    logger.info("Tree model training completed")

    # Extract rules
    rule_extractor = None
    rule_metadata = None

    if config['tree']['extract_rules']:
        logger.info("Extracting cross features...")

        rule_extractor = RuleExtractor(config)

        # ✅ 必须先提取规则
        all_rules = rule_extractor.extract_rules_from_trees(tree_trainer, X_train, y_train)
        logger.info(f"Extracted {len(all_rules)} rules from trees")

        # ✅ 然后选择top规则
        selected_rules = rule_extractor.select_top_rules(X_train, y_train)
        logger.info(f"Selected {len(selected_rules)} high-quality rules")

        # ✅ 最后生成交叉特征
        cross_train = rule_extractor.generate_cross_features(X_train)
        cross_val = rule_extractor.generate_cross_features(X_val)

        # Save rule extractor
        rule_extractor.save(str(save_dir / 'rule_extractor.pkl'))

        # Get rule metadata
        rule_metadata = rule_extractor.get_rule_metadata()
        rule_metadata.to_csv(save_dir / 'rule_metadata.csv', index=False)

        logger.info(f"Generated {cross_train.shape[1]} cross features")
    else:
        cross_train = np.zeros((len(X_train), 0))
        cross_val = np.zeros((len(X_val), 0))

    # Extract paths
    if config['tree']['extract_paths']:
        logger.info("Extracting tree paths...")

        path_encoder = PathEncoder(config)

        path_tokens_train, path_lengths_train, leaf_indices_train = \
            path_encoder.extract_paths(tree_trainer, X_train)

        path_tokens_val, path_lengths_val, leaf_indices_val = \
            path_encoder.extract_paths(tree_trainer, X_val)

        # Save path encoder
        path_encoder.save(str(save_dir / 'path_encoder.pkl'))

        logger.info(f"Path vocabulary size: {path_encoder.get_vocab_size()}")
    else:
        max_path_len = config['tree']['max_path_length']
        n_trees = config['tree']['n_estimators']

        path_tokens_train = np.zeros((len(X_train), max_path_len), dtype=np.int32)
        path_lengths_train = np.zeros(len(X_train), dtype=np.int32)
        leaf_indices_train = np.zeros((len(X_train), n_trees, 2), dtype=np.int32)

        path_tokens_val = np.zeros((len(X_val), max_path_len), dtype=np.int32)
        path_lengths_val = np.zeros(len(X_val), dtype=np.int32)
        leaf_indices_val = np.zeros((len(X_val), n_trees, 2), dtype=np.int32)

        path_encoder = None

    # Package tree features
    tree_features_train = {
        'cross_features': cross_train,
        'path_tokens': path_tokens_train,
        'path_lengths': path_lengths_train,
        'leaf_indices': leaf_indices_train,
    }

    tree_features_val = {
        'cross_features': cross_val,
        'path_tokens': path_tokens_val,
        'path_lengths': path_lengths_val,
        'leaf_indices': leaf_indices_val,
    }

    return tree_trainer, path_encoder, rule_metadata, tree_features_train, tree_features_val


def create_datasets(
    preprocessor,
    X_train, y_train, tree_features_train,
    X_val, y_val, tree_features_val,
    X_test, y_test,
    config,
    path_encoder,
    logger,
):
    """Create PyTorch datasets"""
    logger.info("Creating datasets...")
    
    # Get feature indices
    num_features = preprocessor.numerical_features
    cat_features = preprocessor.categorical_features
    
    all_features = preprocessor.get_feature_names()
    numerical_indices = [i for i, f in enumerate(all_features) if f in num_features]
    categorical_indices = [i for i, f in enumerate(all_features) if f in cat_features]
    
    # Create datasets
    train_dataset = TreeEnhancedDataset(
        X_train, y_train, tree_features_train,
        numerical_indices, categorical_indices,
    )
    
    val_dataset = TreeEnhancedDataset(
        X_val, y_val, tree_features_val,
        numerical_indices, categorical_indices,
    )
    
    test_dataset = None
    if X_test is not None:
        # Extract tree features for test set
        if path_encoder is not None:
            # This would require the tree model - simplified here
            tree_features_test = {
                'cross_features': np.zeros((len(X_test), tree_features_train['cross_features'].shape[1])),
                'path_tokens': np.zeros((len(X_test), tree_features_train['path_tokens'].shape[1]), dtype=np.int32),
                'path_lengths': np.zeros(len(X_test), dtype=np.int32),
                'leaf_indices': np.zeros((len(X_test), tree_features_train['leaf_indices'].shape[1], 2), dtype=np.int32),
            }
        else:
            tree_features_test = None
        
        test_dataset = TreeEnhancedDataset(
            X_test, y_test, tree_features_test,
            numerical_indices, categorical_indices,
        )
    
    # Create sampler if needed
    sampler = None
    if config['data']['sampling_strategy'] == 'class_balanced':
        sampler = ClassBalancedSampler(y_train)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        config, sampler,
    )
    
    logger.info(f"Created dataloaders: train={len(train_loader)}, val={len(val_loader)}")
    
    return train_loader, val_loader, test_loader, numerical_indices, categorical_indices


def create_model(config, preprocessor, tree_features, path_encoder, device, logger):
    """Create and initialize model"""
    logger.info("Creating model...")
    
    # Get dimensions
    num_numerical = len(preprocessor.numerical_features)
    num_categorical = len(preprocessor.categorical_features)
    
    # Get categorical cardinalities
    categorical_cardinalities = []
    if num_categorical > 0:
        for feat in preprocessor.categorical_features:
            if feat in preprocessor.feature_stats.get('categorical_stats', {}):
                n_unique = preprocessor.feature_stats['categorical_stats'][feat]['n_unique']
                categorical_cardinalities.append(n_unique)
            else:
                categorical_cardinalities.append(10)  # Default
    
    # Get tree dimensions
    num_rules = tree_features['cross_features'].shape[1]
    path_vocab_size = path_encoder.get_vocab_size() if path_encoder else 1
    num_trees = config['tree']['n_estimators']
    num_leaves_per_tree = config['tree']['num_leaves']
    
    # Create model
    model = ModelFactory.create_model(
        config=config,
        num_numerical=num_numerical,
        num_categorical=num_categorical,
        categorical_cardinalities=categorical_cardinalities,
        num_rules=num_rules,
        path_vocab_size=path_vocab_size,
        num_trees=num_trees,
        num_leaves_per_tree=num_leaves_per_tree,
        num_classes=2,
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created: {total_params:,} total params, {trainable_params:,} trainable")
    
    return model


def create_optimizer_and_scheduler(model, config, steps_per_epoch, logger):
    """Create optimizer and learning rate scheduler"""
    logger.info("Creating optimizer and scheduler...")
    
    optimizer_config = config['training']['optimizer']
    optimizer_type = optimizer_config['type']
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'],
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'],
            betas=optimizer_config['betas'],
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=0.9,
            weight_decay=optimizer_config['weight_decay'],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    # Create scheduler
    scheduler = SchedulerFactory.create_scheduler(
        optimizer, config, steps_per_epoch
    )
    
    logger.info(f"Created {optimizer_type} optimizer and {config['training']['scheduler']['type']} scheduler")
    
    return optimizer, scheduler


def main(args):
    """Main training function"""
    # Load config
    config = ConfigParser.load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['optimizer']['lr'] = args.lr
    
    # Setup logging
    log_dir = Path(config['system']['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        log_dir=str(log_dir),
        log_level=config['system']['logging']['level'],
    )
    
    logger.info("=" * 80)
    logger.info("TREE-ENHANCED DEEP LEARNING TRAINING")
    logger.info("=" * 80)
    
    # Set seed
    set_seed(config['system']['seed'])
    logger.info(f"Random seed: {config['system']['seed']}")
    
    # Setup device
    device = torch.device(config['system']['device'])
    logger.info(f"Using device: {device}")
    
    # Setup TensorBoard
    tb_logger = None
    if config['system']['logging']['tensorboard']:
        tb_logger = TensorBoardLogger(log_dir / 'tensorboard')
    
    # Setup W&B
    wandb_logger = None
    if config['system']['logging']['wandb']['enabled']:
        wandb_logger = WandBLogger(config)
    
    try:
        # Load and preprocess data
        preprocessor, train_data, val_data, test_data = load_data(config, logger)
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Train tree model and extract features
        tree_trainer, path_encoder, rule_metadata, tree_features_train, tree_features_val = \
            train_tree_model(config, preprocessor, X_train, y_train, X_val, y_val, logger)
        
        # Create datasets and dataloaders
        train_loader, val_loader, test_loader, numerical_indices, categorical_indices = \
            create_datasets(
                preprocessor, X_train, y_train, tree_features_train,
                X_val, y_val, tree_features_val,
                X_test, y_test,
                config, path_encoder, logger
            )
        
        # Create model
        model = create_model(
            config, preprocessor, tree_features_train, path_encoder, device, logger
        )
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, config, len(train_loader), logger
        )
        
        # Create loss function
        class_weights = None
        if config['data']['class_weights'] == 'balanced':
            class_counts = np.bincount(y_train)
            class_weights = torch.FloatTensor(
                [len(y_train) / (len(class_counts) * c) for c in class_counts]
            ).to(device)
        
        loss_fn = LossFactory.create_loss(
            config=config,
            num_classes=2,
            feature_dim=config['model']['sequence_encoder']['hidden_dim'],
            class_weights=class_weights,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            config=config,
            device=device,
        )
        
        # Train
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        if test_loader is not None:
            logger.info("Evaluating on test set...")
            test_results = trainer.validate(test_loader)
            
            logger.info("Test Results:")
            for key, value in test_results.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
        
        # Generate explanations
        if config['evaluation']['interpretation']['enabled']:
            logger.info("Generating explanations...")
            
            explainer = ModelExplainer(model, config, device)
            
            # Explain a few samples
            sample_batch = next(iter(val_loader))
            explanations = []
            
            for i in range(min(10, len(sample_batch['label']))):
                explanation = explainer.explain_instance(
                    sample_batch, i, rule_metadata
                )
                explanations.append(explanation)
            
            # Save explanations
            save_dir = Path(config['system']['checkpoint']['save_dir'])
            explainer.generate_report(
                explanations,
                save_dir / 'explanation_report.txt'
            )
            
            # Visualize
            if config['evaluation']['interpretation']['generate_plots']:
                explainer.visualize_attention(
                    explanations[0],
                    save_dir / 'attention_visualization.png'
                )
                
                if rule_metadata is not None:
                    explainer.visualize_rule_importance(
                        rule_metadata,
                        save_dir=save_dir / 'rule_importance.png'
                    )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        if tb_logger:
            tb_logger.close()
        if wandb_logger:
            wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Tree-Enhanced Deep Learning Model')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    args = parser.parse_args()
    main(args)
