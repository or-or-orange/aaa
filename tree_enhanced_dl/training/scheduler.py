"""
Learning Rate Schedulers
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer
            first_cycle_steps: Number of steps in first cycle
            cycle_mult: Cycle length multiplier
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps
            gamma: Learning rate decay factor
            last_epoch: Last epoch index
        """
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # Initialize learning rate
        self.init_lr()
    
    def init_lr(self):
        """Initialize learning rates"""
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        """Calculate learning rate"""
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # Warmup
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            return [
                base_lr + (self.max_lr - base_lr) * 
                (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                             (self.cur_cycle_steps - self.warmup_steps))) / 2
                for base_lr in self.base_lrs
            ]
    
    def step(self, epoch=None):
        """Step scheduler"""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                ) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log(
                        (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                        self.cycle_mult
                    ))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by constant or decay
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate"""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * (1 - progress)
                for base_lr in self.base_lrs
            ]


class SchedulerFactory:
    """
    Factory for creating learning rate schedulers
    """
    
    @staticmethod
    def create_scheduler(
        optimizer: Optimizer,
        config: Dict,
        steps_per_epoch: int,
    ) -> _LRScheduler:
        """
        Create scheduler based on config
        
        Args:
            optimizer: Optimizer
            config: Configuration dictionary
            steps_per_epoch: Number of steps per epoch
            
        Returns:
            Learning rate scheduler
        """
        scheduler_config = config['training']['scheduler']
        scheduler_type = scheduler_config['type']
        
        num_epochs = config['training']['num_epochs']
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = scheduler_config['warmup_steps']
        
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer=optimizer,
                first_cycle_steps=total_steps,
                max_lr=config['training']['optimizer']['lr'],
                min_lr=scheduler_config['min_lr'],
                warmup_steps=warmup_steps,
            )
        
        elif scheduler_type == 'linear':
            scheduler = LinearWarmupScheduler(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=scheduler_config['min_lr'],
            )
        
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1,
            )
        
        elif scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=scheduler_config['min_lr'],
            )
        
        elif scheduler_type == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['training']['optimizer']['lr'],
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
            )
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        logger.info(f"Created {scheduler_type} scheduler")
        
        return scheduler


if __name__ == "__main__":
    # Test schedulers
    import matplotlib.pyplot as plt
    
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test cosine annealing with warmup
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=1000,
        max_lr=0.001,
        min_lr=1e-6,
        warmup_steps=100,
    )
    
    lrs = []
    for step in range(1000):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing with Warmup')
    plt.grid(True)
    plt.savefig('scheduler_test.png')
    print("Scheduler test plot saved")
