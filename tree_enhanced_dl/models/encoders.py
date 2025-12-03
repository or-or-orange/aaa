"""
Sequence Encoders for Path Representations
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for path sequences
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,  # Divide by 2 for bidirectional
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) - actual sequence lengths
            
        Returns:
            Tuple of (sequence_output, pooled_output)
            - sequence_output: (batch_size, seq_len, hidden_dim)
            - pooled_output: (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Pack padded sequences if lengths provided
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            output, (hidden, cell) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=seq_len
            )
        else:
            output, (hidden, cell) = self.lstm(x)
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        # Pool: concatenate final hidden states from both directions
        # hidden: (num_layers * 2, batch, hidden_dim // 2)
        hidden_fwd = hidden[-2, :, :]  # Forward direction, last layer
        hidden_bwd = hidden[-1, :, :]  # Backward direction, last layer
        pooled = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (batch, hidden_dim)
        
        return output, pooled


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for path sequences
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            feedforward_dim: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Project input to hidden dimension if needed
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = nn.Identity()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) - actual sequence lengths
            
        Returns:
            Tuple of (sequence_output, pooled_output)
            - sequence_output: (batch_size, seq_len, hidden_dim)
            - pooled_output: (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input
        x = self.input_projection(x)
        
        # Create padding mask
        if lengths is not None:
            mask = self._create_padding_mask(seq_len, lengths)
        else:
            mask = None
        
        # Transformer encoding
        output = self.transformer(x, src_key_padding_mask=mask)
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        # Pool: mean pooling over non-padded positions
        if lengths is not None:
            # Mask padded positions
            mask_expanded = (~mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (output * mask_expanded).sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled = output.mean(dim=1)
        
        return output, pooled
    
    def _create_padding_mask(
        self,
        seq_len: int,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create padding mask for transformer
        
        Args:
            seq_len: Sequence length
            lengths: (batch_size,) actual lengths
            
        Returns:
            (batch_size, seq_len) boolean mask (True for padded positions)
        """
        batch_size = lengths.size(0)
        mask = torch.arange(seq_len, device=lengths.device).expand(batch_size, seq_len)
        mask = mask >= lengths.unsqueeze(1)
        return mask


class GRUEncoder(nn.Module):
    """
    Bidirectional GRU encoder for path sequences
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim,
            hidden_dim // 2,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) - actual sequence lengths
            
        Returns:
            Tuple of (sequence_output, pooled_output)
        """
        batch_size, seq_len, _ = x.size()
        
        # Pack if lengths provided
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            output, hidden = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=seq_len
            )
        else:
            output, hidden = self.gru(x)
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        # Pool: concatenate final hidden states
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        pooled = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        return output, pooled


class SequenceEncoderFactory:
    """
    Factory for creating sequence encoders
    """
    
    @staticmethod
    def create_encoder(config: dict) -> nn.Module:
        """
        Create sequence encoder based on config
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Sequence encoder module
        """
        encoder_config = config['model']['sequence_encoder']
        encoder_type = encoder_config['type']
        
        input_dim = config['model']['embedding']['path_token_dim']
        hidden_dim = encoder_config['hidden_dim']
        num_layers = encoder_config['num_layers']
        dropout = encoder_config['dropout']
        
        if encoder_type == 'bilstm':
            encoder = BiLSTMEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif encoder_type == 'transformer':
            encoder = TransformerEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=encoder_config['num_heads'],
                num_layers=num_layers,
                feedforward_dim=encoder_config['feedforward_dim'],
                dropout=dropout,
            )
        elif encoder_type == 'gru':
            encoder = GRUEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        logger.info(f"Created {encoder_type} encoder with hidden_dim={hidden_dim}")
        
        return encoder


if __name__ == "__main__":
    # Test encoders
    batch_size = 32
    seq_len = 10
    input_dim = 32
    hidden_dim = 64
    
    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.randint(5, seq_len + 1, (batch_size,))
    
    # Test BiLSTM
    lstm_encoder = BiLSTMEncoder(input_dim, hidden_dim)
    seq_out, pooled_out = lstm_encoder(x, lengths)
    print(f"BiLSTM - Sequence: {seq_out.shape}, Pooled: {pooled_out.shape}")
    
    # Test Transformer
    transformer_encoder = TransformerEncoder(input_dim, hidden_dim)
    seq_out, pooled_out = transformer_encoder(x, lengths)
    print(f"Transformer - Sequence: {seq_out.shape}, Pooled: {pooled_out.shape}")
    
    # Test GRU
    gru_encoder = GRUEncoder(input_dim, hidden_dim)
    seq_out, pooled_out = gru_encoder(x, lengths)
    print(f"GRU - Sequence: {seq_out.shape}, Pooled: {pooled_out.shape}")
