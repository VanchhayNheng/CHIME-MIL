# import torch
# import torch.nn as nn

# class CausalLoss(nn.Module):
#     # Update __init__ to accept 'weight'
#     def __init__(self, weight=None):
#         super(CausalLoss, self).__init__()
#         # Pass the weights to the internal CrossEntropyLoss
#         self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        
#     def forward(self, logits, logits_c, logits_r, label):
#         # 1. Standard Classification Loss (Weighted)
#         cls_loss = self.ce_loss(logits, label)
        
#         # 2. Causal Loss (Intervention)
#         probs_c = torch.softmax(logits_c, dim=1)
#         probs_r = torch.softmax(logits_r, dim=1)
        
#         # Minimizing probability of tumor in counterfactual
#         zeros = torch.zeros_like(label)
#         causal_term = self.ce_loss(logits_c, zeros)
        
#         # Robustness term
#         robust_term = self.ce_loss(logits_r, label)
        
#         aux_loss = causal_term + robust_term
        
#         return aux_loss, cls_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from focal_loss import FocalLoss  # ← Import the new Focal Loss

class CausalLoss(nn.Module):
    """
    Causal Loss with optional Focal Loss for classification.
    
    Total Loss = Classification Loss + Causal Loss
    
    Args:
        weight: Class weights for imbalanced data
        use_focal: If True, use Focal Loss; if False, use CrossEntropy
        gamma: Focal Loss focusing parameter (only used if use_focal=True)
    """
    def __init__(self, weight=None, use_focal=True, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.use_focal = use_focal
        
        # Choose loss function
        if use_focal:
            print(f"Using Focal Loss (gamma={gamma}) for classification")
            self.ce_loss = FocalLoss(alpha=weight, gamma=gamma)
        else:
            print("Using CrossEntropy Loss for classification")
            self.ce_loss = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, logits_orig, logits_causal, logits_random, labels):
        """
        Compute causal loss.
        
        Args:
            logits_orig: Original predictions (B, C)
            logits_causal: Predictions after removing causal regions (B, C)
            logits_random: Predictions after removing random regions (B, C)
            labels: Ground truth (B,)
        
        Returns:
            total_loss, loss_dict
        """
        # 1. Classification loss (now using Focal Loss)
        cls_loss = self.ce_loss(logits_orig, labels)
        
        # 2. Causal term: Prediction SHOULD change when causal regions removed
        # KL divergence between original and causal predictions
        p_orig = F.softmax(logits_orig, dim=1)
        p_causal = F.log_softmax(logits_causal, dim=1)
        causal_term = F.kl_div(p_causal, p_orig, reduction='batchmean')
        
        # 3. Robustness term: Prediction should NOT change when random regions removed
        p_random = F.log_softmax(logits_random, dim=1)
        robustness_term = F.kl_div(p_random, p_orig, reduction='batchmean')
        
        # Total auxiliary loss
        # ReLU hinge: penalize only when robustness_term >= causal_term (wrong direction)
        # When causal_term > robustness_term (correct state), gradient = 0 — stable training
        aux_loss = (robustness_term - causal_term + 0.1).clamp(min=0)
        
        # Loss breakdown for monitoring
        loss_dict = {
            'cls_loss': cls_loss.item(),
            'causal_term': causal_term.item(),
            'robustness_term': robustness_term.item(),
            'aux_loss': aux_loss.item(),
            'hinge_active': int(aux_loss.item() > 0)
        }
        
        return aux_loss, loss_dict



