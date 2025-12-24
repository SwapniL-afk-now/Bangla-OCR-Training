import torch
from torch.utils.data import Dataset
from PIL import Image

class BanglaOCRDataset(Dataset):
    def __init__(self, dataset, max_width=2048, max_height=192, is_training=False):
        self.dataset = dataset
        self.is_training = is_training
        self.max_width = max_width
        self.max_height = max_height
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        image.thumbnail((self.max_width, self.max_height), Image.BILINEAR)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        text = item['text'].strip()
        
        return {"image": image, "text": text}
