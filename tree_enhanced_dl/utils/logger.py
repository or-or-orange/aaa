"""
Logging Utilities
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = 'tree_enhanced_dl',
    log_dir: Optional[str] = None,
    log_level: str = 'INFO',
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level
        console: Whether to log to console
        file: Whether to log to file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


class TensorBoardLogger:
    """
    TensorBoard logging wrapper
    """
    
    def __init__(self, log_dir: str, enabled: bool = True):
        """
        Args:
            log_dir: TensorBoard log directory
            enabled: Whether TensorBoard logging is enabled
        """
        self.enabled = enabled
        
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
                logging.info(f"TensorBoard logging enabled: {log_dir}")
            except ImportError:
                logging.warning("TensorBoard not available, logging disabled")
                self.enabled = False
        else:
            self.writer = None
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: dict, step: int):
        """Log multiple scalar values"""
        if self.enabled:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log image"""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def close(self):
        """Close writer"""
        if self.enabled:
            self.writer.close()


class WandBLogger:
    """
    Weights & Biases logging wrapper
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        wandb_config = config['system']['logging']['wandb']
        self.enabled = wandb_config['enabled']
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                
                self.wandb.init(
                    project=wandb_config['project'],
                    entity=wandb_config['entity'],
                    config=config,
                )
                
                logging.info("W&B logging enabled")
            except ImportError:
                logging.warning("wandb not available, logging disabled")
                self.enabled = False
    
    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics"""
        if self.enabled:
            self.wandb.log(metrics, step=step)
    
    def finish(self):
        """Finish logging"""
        if self.enabled:
            self.wandb.finish()


if __name__ == "__main__":
    # Test logger
    logger = setup_logger(log_dir='../logs')
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
