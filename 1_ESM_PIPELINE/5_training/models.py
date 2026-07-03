"""
Model Architectures for Drug Resistance Prediction

Two architectures available:
1. ResNet - Deeper residual CNN with adaptive pooling (handles variable length)
2. SimpleCNN - Original shallow CNN with fixed flattening (requires seq_len)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetCNN(nn.Module):
    """
    ResNet-style CNN with residual blocks and adaptive pooling.
    
    Advantages:
    - Deeper network (13 conv layers) captures complex patterns
    - Adaptive pooling handles variable sequence lengths
    - Residual connections prevent vanishing gradients
    
    Input: One-hot encoded sequences (B, 20, L)
    Output: Logit for binary classification (B, 1)
    """
    
    def __init__(
        self,
        input_channels: int = 20,
        stem_channels: int = 64,
    ):
        super().__init__()
        
        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(stem_channels, 64, num_blocks=2)
        self.layer2 = self._make_layer(64, 128, num_blocks=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2)
        
        # Global pooling (handles any sequence length)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.head = nn.Linear(256, 1)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """Create a layer with residual blocks."""
        layers = []
        
        # First block with stride=2 for downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (B, 20, L) - one-hot encoded sequences
        
        Returns:
            logits: (B, 1)
        """
        x = self.stem(x)       # (B, 64, L/2)
        x = self.layer1(x)     # (B, 64, L/4)
        x = self.layer2(x)     # (B, 128, L/8)
        x = self.layer3(x)     # (B, 256, L/16)
        x = self.global_pool(x)  # (B, 256, 1)
        x = x.squeeze(-1)      # (B, 256)
        logits = self.head(x)  # (B, 1)
        
        return logits


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class SimpleCNN(nn.Module):
    """
    Simple/Original CNN architecture.
    
    Advantages:
    - Fewer parameters (~200K vs ~800K for ResNet)
    - Less prone to overfitting on small datasets
    - Faster training
    
    Requires seq_len to be specified (for flattening layer).
    
    Input: One-hot encoded sequences (B, 20, L)
    Output: Logit for binary classification (B, 1)
    """
    
    def __init__(
        self,
        seq_len: int,
        input_channels: int = 20,
        stem_channels: int = 64,
    ):
        super().__init__()
        
        self.stem = nn.Conv1d(input_channels, stem_channels, kernel_size=1)
        self.conv1 = nn.Conv1d(stem_channels, 64, kernel_size=12, padding=6)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(3)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, seq_len)
            flat_size = self._forward_features(dummy).numel()
        
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)
    
    def _forward_features(self, x):
        x = self.stem(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        return x
    
    def forward(self, x):
        """
        Args:
            x: (B, 20, L) - one-hot encoded sequences
        
        Returns:
            logits: (B, 1)
        """
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)
        
        return logits


def create_model(
    model_type: str,
    seq_len: int = None,
    stem_channels: int = 64,
) -> nn.Module:
    """
    Factory function to create model.
    
    Args:
        model_type: 'resnet' or 'simple'
        seq_len: Required for 'simple' model
        stem_channels: Number of channels in stem layer
    
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'resnet':
        return ResNetCNN(stem_channels=stem_channels)
    elif model_type == 'simple':
        if seq_len is None:
            raise ValueError("seq_len is required for SimpleCNN")
        return SimpleCNN(seq_len=seq_len, stem_channels=stem_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'resnet' or 'simple'")