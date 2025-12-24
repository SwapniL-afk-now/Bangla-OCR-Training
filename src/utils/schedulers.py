import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class DampenedCosineScheduler(_LRScheduler):
    """
    Implements a Cosine Annealing scheduler with a dampening factor that reduces 
    the maximum learning rate over time or ensures a graceful decay to a minimum.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, dampen_factor=1.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.dampen_factor = dampen_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            return [base_lr * float(step) / float(max(1, self.warmup_steps)) for base_lr in self.base_lrs]
        
        if step > self.total_steps:
            return [base_lr * self.min_lr_ratio for base_lr in self.base_lrs]
        
        # Cosine decay with dampening
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Dampening effect: The "envelope" of the cosine can be reduced exponentially if dampen_factor != 1.0
        # For now, we'll implement a standard dampened approach where we decay towards min_lr
        
        return [
            base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay * (self.dampen_factor ** progress))
            for base_lr in self.base_lrs
        ]
