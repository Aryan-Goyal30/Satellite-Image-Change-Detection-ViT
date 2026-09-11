"""Loss for heavily imbalanced binary change detection.

LEVIR-CD is roughly 95% unchanged pixels. Plain BCE converges to "predict
nothing changed", which scores ~95% pixel accuracy and 0.0 F1. Two corrections:

  * BCE with pos_weight  -> raises the cost of missing a change pixel
  * Soft Dice            -> optimises overlap directly, which is what F1/IoU
                            actually measure, and is insensitive to the vast
                            background

Combined as  L = w_bce * BCE + w_dice * (1 - Dice).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, w_bce=0.5, w_dice=0.5, pos_weight=None, smooth=1.0):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.smooth = smooth
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        else:
            self.pos_weight = None

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight)

        probs = torch.sigmoid(logits)
        p = probs.reshape(probs.size(0), -1)
        t = target.reshape(target.size(0), -1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.w_bce * bce + self.w_dice * dice_loss
