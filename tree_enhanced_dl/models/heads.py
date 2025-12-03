"""
Classification Heads and Output Layers
"""

import torch
import torch.nn as nn
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class MLPHead(nn.Module):
    """
    Multi-layer perceptron classification head
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        activation: str = 'relu',
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            activation: Activation function ('relu', 'gelu', 'silu')
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_residual = use_residual
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_classes)
        
        # Residual projection if needed
        if use_residual and input_dim != hidden_dims[-1]:
            self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            logits: (batch_size, num_classes)
        """
        identity = x
        
        # Hidden layers
        hidden = self.hidden_layers(x)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            hidden = hidden + identity
        
        # Output
        logits = self.output_layer(hidden)
        
        return logits


class AttentionPoolingHead(nn.Module):
    """
    Classification head with attention pooling
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Attention pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Query vector for pooling
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            logits: (batch_size, num_classes)
        """
        # Ensure 3D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        batch_size = x.size(0)
        
        # Expand query
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, input_dim)
        
        # Attention pooling
        pooled, _ = self.attention(query, x, x)  # (batch, 1, input_dim)
        pooled = pooled.squeeze(1)  # (batch, input_dim)
        
        # MLP
        logits = self.mlp(pooled)
        
        return logits


class MultiTaskHead(nn.Module):
    """
    Multi-task classification head
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes_list: List[int],
        task_names: List[str],
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_classes_list: List of number of classes for each task
            task_names: List of task names
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.task_names = task_names
        self.num_tasks = len(task_names)
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, num_classes)
            for name, num_classes in zip(task_names, num_classes_list)
        })
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            Dictionary of task logits
        """
        # Shared representation
        shared = self.shared_layers(x)
        
        # Task-specific outputs
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared)
        
        return outputs


class HeadFactory:
    """
    Factory for creating classification heads
    """
    
    @staticmethod
    def create_head(
        config: dict,
        input_dim: int,
        num_classes: int = 2,
    ) -> nn.Module:
        """
        Create classification head based on config
        
        Args:
            config: Configuration dictionary
            input_dim: Input feature dimension
            num_classes: Number of output classes
            
        Returns:
            Classification head module
        """
        head_config = config['model']['mlp']
        
        head = MLPHead(
            input_dim=input_dim,
            hidden_dims=head_config['hidden_dims'],
            num_classes=num_classes,
            activation=head_config['activation'],
            dropout=head_config['dropout'],
            use_batch_norm=head_config['use_batch_norm'],
            use_residual=head_config['use_residual'],
        )
        
        logger.info(f"Created MLP head with hidden_dims={head_config['hidden_dims']}")
        
        return head


if __name__ == "__main__":
    # Test heads
    batch_size = 32
    input_dim = 128
    num_classes = 2
    
    # Test MLP head
    mlp_head = MLPHead(input_dim, [64, 32], num_classes)
    x = torch.randn(batch_size, input_dim)
    logits = mlp_head(x)
    print(f"MLP head output: {logits.shape}")
    
    # Test attention pooling head
    attn_head = AttentionPoolingHead(input_dim, 64, num_classes)
    x_seq = torch.randn(batch_size, 10, input_dim)
    logits = attn_head(x_seq)
    print(f"Attention pooling head output: {logits.shape}")
    
    # Test multi-task head
    multi_head = MultiTaskHead(input_dim, 64, [2, 3], ['task1', 'task2'])
    x = torch.randn(batch_size, input_dim)
    outputs = multi_head(x)
    print(f"Multi-task head outputs: {[(k, v.shape) for k, v in outputs.items()]}")
