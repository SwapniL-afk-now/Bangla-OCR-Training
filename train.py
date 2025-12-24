import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
import bitsandbytes as bnb
from transformers import get_scheduler

from src.config import config
from src.data.dataset import BanglaOCRDataset
from src.data.collator import OptimizedOCRDataCollator
from src.data.augmentation import StandardAugmentation
from src.models.model_factory import setup_model_and_processor
from src.losses import CombinedLoss
from src.utils.confusion import CharacterConfusionMatrix
from src.utils.data_utils import scan_dataset_dimensions
from src.utils.checkpoints import load_checkpoint
from src.trainer import Trainer

def main():
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if config.fp16 else "no",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Setup Model & Processor
    model, processor = setup_model_and_processor(config, accelerator)
    
    # 2. Prepare Data
    accelerator.print("Loading dataset...")
    dataset = load_dataset(config.dataset_name)
    train_raw = dataset['train']
    
    if 'all' not in [c.lower() for c in config.dataset_classes]:
        train_raw = train_raw.filter(lambda x: x['class'] in config.dataset_classes)
        
    if config.scan_dataset_dimensions:
        config.max_width, config.max_height = scan_dataset_dimensions(train_raw, config, accelerator, config.dimension_scan_samples)
    else:
        config.max_width, config.max_height = config.manual_max_width, config.manual_max_height
        
    # Split
    total_samples = len(train_raw)
    test_size = int(total_samples * (config.val_ratio + config.test_ratio))
    split_dataset = train_raw.train_test_split(test_size=test_size, seed=42)
    
    val_test_split = split_dataset['test'].train_test_split(test_size=0.5, seed=42)
    
    train_ds = BanglaOCRDataset(split_dataset['train'], config.max_width, config.max_height, True)
    val_ds = BanglaOCRDataset(val_test_split['train'], config.max_width, config.max_height, False)
    
    data_collator = OptimizedOCRDataCollator(processor)
    
    train_dataloader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=data_collator,
        num_workers=config.num_workers, pin_memory=config.pin_memory, 
        prefetch_factor=config.prefetch_factor, persistent_workers=config.persistent_workers
    )
    
    val_dataloader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=data_collator,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    
    # 3. Setup Training
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    max_steps = config.max_steps if config.max_steps > 0 else int(config.num_train_epochs * len(train_dataloader))
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=max_steps)
    
    loss_fn = CombinedLoss(processor=processor, focal_alpha=config.focal_alpha, focal_gamma=config.focal_gamma, label_smoothing=config.label_smoothing)
    
    gpu_augmentor = StandardAugmentation().to(accelerator.device)
    confusion_matrix = CharacterConfusionMatrix(max_size=config.confusion_map_max_size)
    
    # Prepare with Accelerator
    optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # 4. Resume Checkpoint
    start_step, start_epoch = 0, 0
    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        start_step, start_epoch = load_checkpoint(model, optimizer, lr_scheduler, config.resume_from_checkpoint, config, accelerator)

    # 5. Start Training
    trainer = Trainer(
        model, processor, train_dataloader, val_dataloader, 
        optimizer, lr_scheduler, loss_fn, gpu_augmentor, 
        confusion_matrix, config, accelerator
    )
    
    accelerator.print("\n" + "="*70)
    accelerator.print("BANGLA OCR TRAINING - MODULAR VERSION")
    accelerator.print("="*70)
    
    trainer.train(start_step, start_epoch, max_steps)
    
    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model.save_pretrained(config.output_dir)
        processor.save_pretrained(config.output_dir)
        accelerator.print("\n" + "="*70 + "\nTRAINING COMPLETED\n" + "="*70)

if __name__ == "__main__":
    main()
