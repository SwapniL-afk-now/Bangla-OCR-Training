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

class ConfusionWeightedLoss(nn.Module):
    def __init__(self, tokenizer, confusion_chars, weight_multiplier=2.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.confusion_token_ids = set()
        self.weight_multiplier = weight_multiplier
        
        for char in confusion_chars:
            token_ids = tokenizer.encode(char, add_special_tokens=False)
            self.confusion_token_ids.update(token_ids)
    
    def forward(self, logits, labels):
        weights = torch.ones_like(labels, dtype=torch.float)
        
        for token_id in self.confusion_token_ids:
            weights[labels == token_id] *= self.weight_multiplier
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
            ignore_index=-100
        )
        
        weighted_loss = (loss * shift_weights.view(-1)).sum() / shift_weights.view(-1).sum().clamp(min=1.0)
        
        return weighted_loss

class CombinedLoss(nn.Module):
    def __init__(self, processor, focal_alpha=0.25, focal_gamma=2.0, label_smoothing=0.1, 
                 confusion_chars=None, confusion_weight=2.0):
        super().__init__()
        self.focal_loss = FocalLossWithSmoothing(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
        
        self.confusion_loss = None
        if confusion_chars and len(confusion_chars) > 0:
            self.confusion_loss = ConfusionWeightedLoss(processor.tokenizer, confusion_chars, confusion_weight)
    
    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        focal_loss = self.focal_loss(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        if self.confusion_loss is not None:
            confusion_loss = self.confusion_loss(logits, labels)
            total_loss = 0.7 * focal_loss + 0.3 * confusion_loss
            return total_loss, {
                'focal_loss': focal_loss.item(), 
                'confusion_loss': confusion_loss.item(),
                'total_loss': total_loss.item()
            }
        
        return focal_loss, {'focal_loss': focal_loss.item(), 'total_loss': focal_loss.item()}
