"""
MTF Fusion Strategies

Different approaches for fusing multi-timeframe features.
"""

import torch
import torch.nn as nn
from typing import List, Dict


class BaselineFusion(nn.Module):
    """
    Simple concatenation-based fusion.
    """
    
    def __init__(self, input_dims: Dict[str, int]):
        """
        Parameters
        ----------
        input_dims : Dict[str, int]
            Dictionary mapping timeframe to feature dimension
        """
        super().__init__()
        self.input_dims = input_dims
        self.total_dim = sum(input_dims.values())
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate features from all timeframes.
        
        Parameters
        ----------
        features : Dict[str, torch.Tensor]
            Dictionary of features per timeframe
            
        Returns
        -------
        torch.Tensor
            Concatenated features
        """
        # Concatenate along feature dimension
        feat_list = [features[tf] for tf in sorted(features.keys())]
        return torch.cat(feat_list, dim=-1)


class DualTowerFusion(nn.Module):
    """
    Dual-tower architecture: separate processing for LTF and HTF,
    then fusion.
    """
    
    def __init__(
        self,
        ltf_dim: int,
        htf_dim: int,
        hidden_dim: int = 128,
        fusion_method: str = 'concat'
    ):
        """
        Parameters
        ----------
        ltf_dim : int
            LTF feature dimension
        htf_dim : int
            HTF feature dimension
        hidden_dim : int
            Hidden dimension for towers
        fusion_method : str
            Fusion method: 'concat', 'add', or 'mul'
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # LTF tower
        self.ltf_tower = nn.Sequential(
            nn.Linear(ltf_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # HTF tower
        self.htf_tower = nn.Sequential(
            nn.Linear(htf_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layer
        if fusion_method == 'concat':
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fusion = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        ltf_features: torch.Tensor,
        htf_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Process and fuse LTF and HTF features.
        
        Parameters
        ----------
        ltf_features : torch.Tensor
            LTF features
        htf_features : torch.Tensor
            HTF features
            
        Returns
        -------
        torch.Tensor
            Fused features
        """
        ltf_out = self.ltf_tower(ltf_features)
        htf_out = self.htf_tower(htf_features)
        
        if self.fusion_method == 'concat':
            fused = torch.cat([ltf_out, htf_out], dim=-1)
        elif self.fusion_method == 'add':
            fused = ltf_out + htf_out
        elif self.fusion_method == 'mul':
            fused = ltf_out * htf_out
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return self.fusion(fused)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of multi-timeframe features.
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        attention_dim: int = 64
    ):
        """
        Parameters
        ----------
        feature_dims : List[int]
            List of feature dimensions for each timeframe
        attention_dim : int
            Attention hidden dimension
        """
        super().__init__()
        
        self.num_timeframes = len(feature_dims)
        self.feature_dims = feature_dims
        
        # Project each timeframe to common dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, attention_dim) for dim in feature_dims
        ])
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply attention-based fusion.
        
        Parameters
        ----------
        features : List[torch.Tensor]
            List of feature tensors from different timeframes
            
        Returns
        -------
        torch.Tensor
            Fused features with attention weighting
        """
        # Project all features to common dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        # Stack features
        stacked = torch.stack(projected, dim=1)  # (batch, num_tf, attention_dim)
        
        # Compute attention weights
        attn_scores = self.attention(stacked)  # (batch, num_tf, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Apply attention
        fused = (stacked * attn_weights).sum(dim=1)  # (batch, attention_dim)
        
        return fused


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion: progressively integrate HTF into LTF.
    """
    
    def __init__(
        self,
        ltf_dim: int,
        mtf_dims: List[int],
        hidden_dim: int = 128
    ):
        """
        Parameters
        ----------
        ltf_dim : int
            LTF feature dimension
        mtf_dims : List[int]
            List of MTF dimensions (1H, 4H, 1D, etc.)
        hidden_dim : int
            Hidden dimension for fusion
        """
        super().__init__()
        
        self.fusion_layers = nn.ModuleList()
        
        current_dim = ltf_dim
        for mtf_dim in mtf_dims:
            fusion_layer = nn.Sequential(
                nn.Linear(current_dim + mtf_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.fusion_layers.append(fusion_layer)
            current_dim = hidden_dim
    
    def forward(
        self,
        ltf_features: torch.Tensor,
        mtf_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Hierarchically fuse features.
        
        Parameters
        ----------
        ltf_features : torch.Tensor
            LTF features
        mtf_features : List[torch.Tensor]
            List of MTF features in order of increasing timeframe
            
        Returns
        -------
        torch.Tensor
            Hierarchically fused features
        """
        current = ltf_features
        
        for mtf_feat, fusion_layer in zip(mtf_features, self.fusion_layers):
            # Concatenate current representation with next HTF
            combined = torch.cat([current, mtf_feat], dim=-1)
            # Fuse
            current = fusion_layer(combined)
        
        return current
