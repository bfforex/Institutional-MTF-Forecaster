"""
Hybrid CNN-BiLSTM-Attention Model

Main architecture combining:
- Temporal CNN blocks for local pattern extraction
- Bi-LSTM for sequence modeling
- Additive Attention for focus
- FiLM conditioning from HTF context
- Classification head for 3-class prediction
"""

import torch
import torch.nn as nn
from typing import Optional, List

from .film_conditioning import FiLMLayer
from .fusion import BaselineFusion


class TemporalConvBlock(nn.Module):
    """
    Temporal convolutional block with residual connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, sequence_len)
            
        Returns
        -------
        torch.Tensor
            Output tensor of same shape
        """
        residual = self.residual(x)
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out + residual


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        
        self.W1 = nn.Linear(hidden_dim, attention_dim)
        self.W2 = nn.Linear(hidden_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
    
    def forward(
        self,
        query: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Apply additive attention.
        
        Parameters
        ----------
        query : torch.Tensor
            Query tensor (batch, hidden_dim)
        values : torch.Tensor
            Value tensor (batch, seq_len, hidden_dim)
        mask : torch.Tensor, optional
            Mask for attention scores
            
        Returns
        -------
        tuple
            (context_vector, attention_weights)
        """
        # Expand query to match sequence length
        query_expanded = query.unsqueeze(1).expand(-1, values.size(1), -1)
        
        # Compute attention scores
        score = self.V(torch.tanh(self.W1(query_expanded) + self.W2(values)))
        score = score.squeeze(-1)  # (batch, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = torch.softmax(score, dim=-1)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), values)
        context = context.squeeze(1)  # (batch, hidden_dim)
        
        return context, attention_weights


class HybridForexModel(nn.Module):
    """
    Hybrid CNN → Bi-LSTM → Additive Attention → Dense architecture
    with optional FiLM conditioning from HTF context.
    """
    
    def __init__(
        self,
        input_dim: int,
        window: int = 256,
        conv_channels: List[int] = [32, 64],
        kernel_size: int = 5,
        lstm_hidden: int = 192,
        lstm_layers: int = 2,
        attention_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2,
        film_conditioning: bool = True,
        htf_context_dim: int = 20
    ):
        """
        Parameters
        ----------
        input_dim : int
            Total input features after MTF fusion
        window : int
            Sequence length (number of timesteps)
        conv_channels : List[int]
            List of channel sizes for CNN blocks
        kernel_size : int
            Kernel size for convolutions
        lstm_hidden : int
            LSTM hidden dimension
        lstm_layers : int
            Number of LSTM layers
        attention_dim : int
            Attention mechanism dimension
        num_classes : int
            Number of output classes (3 for Up/Neutral/Down)
        dropout : float
            Dropout rate
        film_conditioning : bool
            Whether to use FiLM conditioning from HTF
        htf_context_dim : int
            Dimension of HTF context for FiLM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.window = window
        self.film_conditioning = film_conditioning
        self.lstm_hidden = lstm_hidden
        
        # Temporal CNN blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = input_dim
        
        for out_channels in conv_channels:
            self.conv_blocks.append(
                TemporalConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dropout
                )
            )
            in_channels = out_channels
        
        # FiLM conditioning (optional)
        if film_conditioning:
            self.film_layer = FiLMLayer(
                feature_dim=conv_channels[-1],
                condition_dim=htf_context_dim
            )
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Additive Attention
        self.attention = AdditiveAttention(
            hidden_dim=lstm_hidden * 2,  # *2 for bidirectional
            attention_dim=attention_dim
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Confidence head (optional)
        self.confidence_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        htf_context: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, window, input_dim)
        htf_context : torch.Tensor, optional
            HTF context for FiLM conditioning (batch, htf_context_dim)
            
        Returns
        -------
        tuple
            (logits, confidence_scores, attention_weights)
        """
        batch_size = x.size(0)
        
        # Transpose for Conv1d: (batch, input_dim, window)
        x = x.transpose(1, 2)
        
        # Apply CNN blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Transpose back for LSTM: (batch, window, channels)
        x = x.transpose(1, 2)
        
        # Apply FiLM conditioning if enabled
        if self.film_conditioning and htf_context is not None:
            x = self.film_layer(x, htf_context)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, window, lstm_hidden * 2)
        
        # Use last hidden state as query for attention
        # h_n: (num_layers * 2, batch, lstm_hidden)
        # Take last layer, concatenate forward and backward
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        query = torch.cat([h_forward, h_backward], dim=-1)
        
        # Apply attention
        context, attention_weights = self.attention(query, lstm_out)
        
        # Classification
        logits = self.classifier(context)
        
        # Confidence scores
        confidence = self.confidence_head(context)
        
        return logits, confidence, attention_weights
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        htf_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract feature embeddings (for analysis/visualization).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        htf_context : torch.Tensor, optional
            HTF context
            
        Returns
        -------
        torch.Tensor
            Feature embeddings before classification
        """
        batch_size = x.size(0)
        
        # Transpose for Conv1d
        x = x.transpose(1, 2)
        
        # Apply CNN blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Transpose back
        x = x.transpose(1, 2)
        
        # Apply FiLM if enabled
        if self.film_conditioning and htf_context is not None:
            x = self.film_layer(x, htf_context)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Get query
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        query = torch.cat([h_forward, h_backward], dim=-1)
        
        # Apply attention
        context, _ = self.attention(query, lstm_out)
        
        return context


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
        
    Returns
    -------
    int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
