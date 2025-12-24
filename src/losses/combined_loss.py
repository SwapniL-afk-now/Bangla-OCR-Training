import torch.nn as nn
from .focal_loss import FocalLossWithSmoothing
from .confusion_loss import ConfusionWeightedLoss

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
