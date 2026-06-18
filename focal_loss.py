import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights (same as CrossEntropy weight parameter)
        gamma: Focusing parameter (higher = more focus on hard examples)
               gamma=0 → same as CrossEntropy
               gamma=2 → standard Focal Loss
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights tensor
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) - raw model outputs
            targets: (B,) - class labels
        Returns:
            loss: scalar
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of correct class
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        p_t = (probs * targets_one_hot).sum(dim=1)  # (B,)
        
        # Focal weight: (1 - p_t)^gamma
        # Easy examples (p_t → 1): weight → 0
        # Hard examples (p_t → 0): weight → 1
        focal_weight = (1 - p_t) ** self.gamma
        
        # Cross-entropy: -log(p_t)
        ce = -torch.log(p_t + 1e-8)
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # (B,)
            loss = alpha_t * focal_weight * ce
        else:
            loss = focal_weight * ce
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss