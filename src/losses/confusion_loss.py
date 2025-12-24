import torch
import torch.nn as nn
import torch.nn.functional as F

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
