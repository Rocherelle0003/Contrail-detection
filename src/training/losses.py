"""Loss d'entraînement : BCE + Dice (soft), comme recommandé pour la baseline."""

import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        soft_dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        dice_loss = 1.0 - soft_dice.mean()

        return (1.0 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss
