import gc
from PIL import Image
from tqdm.auto import tqdm

def scan_dataset_dimensions(dataset, config, accelerator, sample_size=1000):
    """Memory-efficient dimension scanning"""
    accelerator.print(f"Scanning {min(sample_size, len(dataset))} samples...")
    
    max_w, max_h = 0, 0
    samples_to_check = min(sample_size, len(dataset))
    
    for i in tqdm(range(samples_to_check), desc="Scanning", disable=not accelerator.is_main_process):
        item = dataset[i]
        img = item['image']
        
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        
        w, h = img.size
        max_w = max(max_w, w)
        max_h = max(max_h, h)
        
        # Clear memory
        del img
        if i % 100 == 0:
            gc.collect()
    
    max_w = int(max_w * config.padding_buffer)
    max_h = int(max_h * config.padding_buffer)
    
    accelerator.print(f"✓ Detected: {max_w} x {max_h}")
    
    return max_w, max_h
