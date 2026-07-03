"""
Loss Functions for Drug Resistance Prediction
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with optional positive class weighting.
    
    Handles class imbalance by weighting positive samples more heavily.
    """
    
    def __init__(self, pos_weight: float = None):
        """
        Args:
            pos_weight: Weight for positive class. If None, uses equal weighting.
        """
        super().__init__()
        
        if pos_weight is not None:
            self.pos_weight = torch.tensor([pos_weight])
        else:
            self.pos_weight = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) - raw logits from model
            targets: (B, 1) - target labels in [0, 1]
        
        Returns:
            loss: scalar tensor
        """
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        
        return loss


def compute_pos_weight(dataloader: DataLoader, device: torch.device) -> float:
    """
    Compute positive class weight from dataloader.
    
    pos_weight = num_negative / num_positive
    
    This makes the loss treat each class equally despite imbalance.
    """
    n_pos = 0
    n_neg = 0
    
    for _, targets in dataloader:
        targets = targets.to(device)
        # Binarize at 0.5
        binary = (targets >= 0.5).float()
        n_pos += binary.sum().item()
        n_neg += (1 - binary).sum().item()
    
    if n_pos == 0:
        print("Warning: No positive samples found, using pos_weight=1.0")
        return 1.0
    
    pos_weight = n_neg / n_pos
    print(f"\nClass distribution: pos={int(n_pos)}, neg={int(n_neg)}")
    print(f"Computed pos_weight: {pos_weight:.4f}")
    
    return pos_weight