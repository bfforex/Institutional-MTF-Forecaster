"""
FiLM (Feature-wise Linear Modulation) Conditioning Layers

Allows HTF context to modulate LTF feature representations.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Applies affine transformation to features based on conditioning vector:
    output = gamma * x + beta
    
    where gamma and beta are learned from HTF context.
    """
    
    def __init__(self, feature_dim: int, condition_dim: int):
        """
        Parameters
        ----------
        feature_dim : int
            Dimension of features to modulate
        condition_dim : int
            Dimension of conditioning vector (HTF context)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        
        # Networks to generate scale (gamma) and shift (beta)
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.
        
        Parameters
        ----------
        x : torch.Tensor
            Features to modulate, shape (batch, ..., feature_dim)
        condition : torch.Tensor
            Conditioning vector, shape (batch, condition_dim)
            
        Returns
        -------
        torch.Tensor
            Modulated features, same shape as x
        """
        # Generate scale and shift parameters
        gamma = self.gamma_net(condition)  # (batch, feature_dim)
        beta = self.beta_net(condition)    # (batch, feature_dim)
        
        # Reshape for broadcasting if needed
        if x.dim() == 3:  # (batch, seq, features)
            gamma = gamma.unsqueeze(1)  # (batch, 1, feature_dim)
            beta = beta.unsqueeze(1)
        
        # Apply modulation
        return gamma * x + beta


class MultiScaleFiLM(nn.Module):
    """
    Multi-scale FiLM conditioning for different feature hierarchies.
    """
    
    def __init__(
        self,
        feature_dims: list,
        condition_dim: int
    ):
        """
        Parameters
        ----------
        feature_dims : list
            List of feature dimensions at different scales
        condition_dim : int
            Dimension of conditioning vector
        """
        super().__init__()
        
        self.film_layers = nn.ModuleList([
            FiLMLayer(dim, condition_dim) for dim in feature_dims
        ])
    
    def forward(
        self,
        features: list,
        condition: torch.Tensor
    ) -> list:
        """
        Apply FiLM to multiple feature hierarchies.
        
        Parameters
        ----------
        features : list
            List of feature tensors at different scales
        condition : torch.Tensor
            Conditioning vector
            
        Returns
        -------
        list
            List of modulated features
        """
        modulated = []
        for feat, film_layer in zip(features, self.film_layers):
            modulated.append(film_layer(feat, condition))
        return modulated


class ConditionalBatchNorm(nn.Module):
    """
    Conditional Batch Normalization with FiLM-like conditioning.
    """
    
    def __init__(
        self,
        num_features: int,
        condition_dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        """
        Parameters
        ----------
        num_features : int
            Number of features/channels
        condition_dim : int
            Dimension of conditioning vector
        eps : float
            BatchNorm epsilon
        momentum : float
            BatchNorm momentum
        """
        super().__init__()
        
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        
        # Conditioning networks
        self.gamma_net = nn.Linear(condition_dim, num_features)
        self.beta_net = nn.Linear(condition_dim, num_features)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply conditional batch normalization.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        condition : torch.Tensor
            Conditioning vector
            
        Returns
        -------
        torch.Tensor
            Normalized and modulated features
        """
        # Apply standard batch norm
        normalized = self.bn(x)
        
        # Generate conditional scale and shift
        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)
        
        # Reshape for broadcasting
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        # Apply conditioning
        return gamma * normalized + beta
