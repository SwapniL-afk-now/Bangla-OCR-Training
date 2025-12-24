import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1, reduction='mean', ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        mask = targets != self.ignore_index
        valid_targets = targets[mask]
        valid_logits = logits[mask]
        
        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        num_classes = valid_logits.size(-1)
        smoothed_targets = (1 - self.label_smoothing) * F.one_hot(valid_targets, num_classes).float() + \
                          self.label_smoothing / num_classes
        
        log_probs = F.log_softmax(valid_logits, dim=-1)
        ce_loss = -torch.sum(smoothed_targets * log_probs, dim=-1)
        
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
