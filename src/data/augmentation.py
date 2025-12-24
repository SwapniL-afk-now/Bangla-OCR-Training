import torch
import torch.nn as nn
import random
import kornia.augmentation as K

class StandardAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.aug = K.AugmentationSequential(
            K.RandomBrightness(brightness=(0.85, 1.15), p=0.5),
            K.RandomContrast(contrast=(0.85, 1.15), p=0.5),
            K.RandomRotation(degrees=3.0, p=0.4),
            K.RandomGaussianNoise(mean=0., std=0.02, p=0.3),
            data_keys=["input"],
            random_apply=(1, 3),
        )
    
    def forward(self, images):
        if not self.training:
            return images
        
        # Skip if images too small
        if images.shape[-2] < 5 or images.shape[-1] < 5:
            return images
        
        try:
            if random.random() < 0.4:
                images = self.aug(images)
        except Exception:
            pass  # Skip augmentation on error
        
        return images
