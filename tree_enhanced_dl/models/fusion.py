"""
Fusion Modules for Combining Multiple Feature Representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttentionFusion(nn.Module):
    """
    Multi-head attention-based fusion of original, path, and cross features
    """
    
    def __init__(
        self,
        original_dim: int,
        path_dim: int,
        cross_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            original_dim: Dimension of original features
            path_dim: Dimension of path representations
            cross_dim: Dimension of cross features
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.original_dim = original_dim
        self.path_dim = path_dim
        self.cross_dim = cross_dim
        self.num_heads = num_heads
        
        # Project all to same dimension
        self.hidden_dim = max(original_dim, path_dim, cross_dim)
        
        self.original_proj = nn.Linear(original_dim, self.hidden_dim)
        self.path_proj = nn.Linear(path_dim, self.hidden_dim)
        self.cross_proj = nn.Linear(cross_dim, self.hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for interpretation
        self.attention_weights = None
    
    def forward(
        self,
        original_features: torch.Tensor,
        path_features: torch.Tensor,
        cross_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            original_features: (batch, original_dim)
            path_features: (batch, path_dim)
            cross_features: (batch, cross_dim)
            
        Returns:
            Fused features: (batch, hidden_dim)
        """
        batch_size = original_features.size(0)
        
        # Project to same dimension
        orig_proj = self.original_proj(original_features)  # (batch, hidden_dim)
        path_proj = self.path_proj(path_features)
        cross_proj = self.cross_proj(cross_features)
        
        # Stack as sequence: [original, path, cross]
        features = torch.stack([orig_proj, path_proj, cross_proj], dim=1)  # (batch, 3, hidden_dim)
        
        # Self-attention
        attn_out, attn_weights = self.attention(
            features, features, features,
            need_weights=True,
        )
        
        # Store attention weights for interpretation
        self.attention_weights = attn_weights.detach()
        
        # Flatten and project
        attn_out_flat = attn_out.view(batch_size, -1)  # (batch, 3 * hidden_dim)
        output = self.output_proj(attn_out_flat)
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get stored attention weights for interpretation"""
        return self.attention_weights


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism with learnable gates
    """
    
    def __init__(
        self,
        original_dim: int,
        path_dim: int,
        cross_dim: int,
        hidden_dim: int,
    ):
        """
        Args:
            original_dim: Dimension of original features
            path_dim: Dimension of path representations
            cross_dim: Dimension of cross features
            hidden_dim: Output hidden dimension
        """
        super().__init__()
        
        self.original_dim = original_dim
        self.path_dim = path_dim
        self.cross_dim = cross_dim
        self.hidden_dim = hidden_dim
        
        # Project to hidden dimension
        self.original_proj = nn.Linear(original_dim, hidden_dim)
        self.path_proj = nn.Linear(path_dim, hidden_dim)
        self.cross_proj = nn.Linear(cross_dim, hidden_dim)
        
        # Gate networks
        gate_input_dim = original_dim + path_dim + cross_dim
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 gates for 3 inputs
            nn.Softmax(dim=1),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Store gate values for interpretation
        self.gate_values = None
    
    def forward(
        self,
        original_features: torch.Tensor,
        path_features: torch.Tensor,
        cross_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            original_features: (batch, original_dim)
            path_features: (batch, path_dim)
            cross_features: (batch, cross_dim)
            
        Returns:
            Fused features: (batch, hidden_dim)
        """
        # Project features
        orig_proj = self.original_proj(original_features)
        path_proj = self.path_proj(path_features)
        cross_proj = self.cross_proj(cross_features)
        
        # Compute gates
        concat_features = torch.cat([original_features, path_features, cross_features], dim=1)
        gates = self.gate_network(concat_features)  # (batch, 3)
        
        # Store gate values
        self.gate_values = gates.detach()
        
        # Apply gates
        gate_orig = gates[:, 0:1]  # (batch, 1)
        gate_path = gates[:, 1:2]
        gate_cross = gates[:, 2:3]
        
        fused = gate_orig * orig_proj + gate_path * path_proj + gate_cross * cross_proj
        
        fused = self.layer_norm(fused)
        
        return fused
    
    def get_gate_values(self) -> Optional[torch.Tensor]:
        """Get stored gate values for interpretation"""
        return self.gate_values


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion
    """
    
    def __init__(
        self,
        original_dim: int,
        path_dim: int,
        cross_dim: int,
        hidden_dim: int,
        use_projection: bool = True,
    ):
        """
        Args:
            original_dim: Dimension of original features
            path_dim: Dimension of path representations
            cross_dim: Dimension of cross features
            hidden_dim: Output hidden dimension
            use_projection: Whether to project concatenated features
        """
        super().__init__()
        
        self.original_dim = original_dim
        self.path_dim = path_dim
        self.cross_dim = cross_dim
        self.hidden_dim = hidden_dim
        self.use_projection = use_projection
        
        concat_dim = original_dim + path_dim + cross_dim
        
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(concat_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
            )
        else:
            self.projection = nn.Identity()
            self.hidden_dim = concat_dim
    
    def forward(
        self,
        original_features: torch.Tensor,
        path_features: torch.Tensor,
        cross_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            original_features: (batch, original_dim)
            path_features: (batch, path_dim)
            cross_features: (batch, cross_dim)
            
        Returns:
            Fused features: (batch, hidden_dim)
        """
        # Concatenate
        fused = torch.cat([original_features, path_features, cross_features], dim=1)
        
        # Project
        if self.use_projection:
            fused = self.projection(fused)
        
        return fused


class FusionModuleFactory:
    """
    Factory for creating fusion modules
    """
    
    @staticmethod
    def create_fusion(
        config: dict,
        original_dim: int,
        path_dim: int,
        cross_dim: int,
    ) -> nn.Module:
        """
        Create fusion module based on config
        
        Args:
            config: Configuration dictionary
            original_dim: Dimension of original features
            path_dim: Dimension of path features
            cross_dim: Dimension of cross features
            
        Returns:
            Fusion module
        """
        fusion_config = config['model']['fusion']
        fusion_type = fusion_config['type']
        
        if fusion_type == 'multi_head_attention':
            fusion = MultiHeadAttentionFusion(
                original_dim=original_dim,
                path_dim=path_dim,
                cross_dim=cross_dim,
                num_heads=fusion_config['attention_heads'],
                dropout=fusion_config['attention_dropout'],
            )
        elif fusion_type == 'gated':
            fusion = GatedFusion(
                original_dim=original_dim,
                path_dim=path_dim,
                cross_dim=cross_dim,
                hidden_dim=max(original_dim, path_dim, cross_dim),
            )
        elif fusion_type == 'concat':
            fusion = ConcatFusion(
                original_dim=original_dim,
                path_dim=path_dim,
                cross_dim=cross_dim,
                hidden_dim=max(original_dim, path_dim, cross_dim),
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        logger.info(f"Created {fusion_type} fusion module")
        
        return fusion


if __name__ == "__main__":
    # Test fusion modules
    batch_size = 32
    original_dim = 64
    path_dim = 64
    cross_dim = 32
    
    orig_feat = torch.randn(batch_size, original_dim)
    path_feat = torch.randn(batch_size, path_dim)
    cross_feat = torch.randn(batch_size, cross_dim)
    
    # Test attention fusion
    attn_fusion = MultiHeadAttentionFusion(original_dim, path_dim, cross_dim)
    fused = attn_fusion(orig_feat, path_feat, cross_feat)
    print(f"Attention fusion output: {fused.shape}")
    print(f"Attention weights: {attn_fusion.get_attention_weights().shape}")
    
    # Test gated fusion
    gated_fusion = GatedFusion(original_dim, path_dim, cross_dim, 64)
    fused = gated_fusion(orig_feat, path_feat, cross_feat)
    print(f"Gated fusion output: {fused.shape}")
    print(f"Gate values: {gated_fusion.get_gate_values().shape}")
    
    # Test concat fusion
    concat_fusion = ConcatFusion(original_dim, path_dim, cross_dim, 64)
    fused = concat_fusion(orig_feat, path_feat, cross_feat)
    print(f"Concat fusion output: {fused.shape}")
