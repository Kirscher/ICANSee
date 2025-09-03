import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Binary Focal Loss with optional per-class alpha.

        Args:
            alpha (float or tuple): If float, interpreted as weight for positive class (1).
                                    If tuple/list of two floats, interpreted as (alpha_neg, alpha_pos).
            gamma (float): Focusing parameter.
            reduction (str): 'none', 'mean', or 'sum'.
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (tuple, list)):
            assert len(alpha) == 2, "alpha as tuple must be (alpha_neg, alpha_pos)"
            self.alpha_neg = float(alpha[0])
            self.alpha_pos = float(alpha[1])
        else:
            self.alpha_neg = 1.0 - float(alpha)
            self.alpha_pos = float(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N, 1) or (N,), raw scores; targets: (N, 1) or (N,), float {0,1}
        logits = logits.view(-1)
        targets = targets.view(-1)

        prob_pos = torch.sigmoid(logits)
        prob_neg = 1.0 - prob_pos

        # p_t and alpha_t selection without branching
        p_t = prob_pos * targets + prob_neg * (1.0 - targets)
        alpha_t = self.alpha_pos * targets + self.alpha_neg * (1.0 - targets)

        # BCE with logits per-sample
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
