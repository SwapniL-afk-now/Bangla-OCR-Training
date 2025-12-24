import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
import bitsandbytes as bnb
from transformers import get_scheduler

import argparse
from src.config import config
from src.data.dataset import BanglaOCRDataset
from src.data.collator import OptimizedOCRDataCollator
from src.data.augmentation import StandardAugmentation
from src.models.model_factory import setup_model_and_processor
from src.losses import CombinedLoss
from src.utils.confusion import CharacterConfusionMatrix
from src.utils.data_utils import scan_dataset_dimensions
from src.utils.checkpoints import load_checkpoint
from src.utils.schedulers import DampenedCosineScheduler
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Bangla OCR Model")
    
    # Model Config
    parser.add_argument("--model_name", type=str, default=config.model_name)
    parser.add_argument("--lora_r", type=int, default=config.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=config.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=config.lora_dropout)
    
    # Training Config
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate)
    parser.add_argument("--num_train_epochs", type=int, default=config.num_train_epochs)
    parser.add_argument("--max_steps", type=int, default=config.max_steps)
    parser.add_argument("--warmup_steps", type=int, default=config.warmup_steps)
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay)
    parser.add_argument("--max_grad_norm", type=float, default=config.max_grad_norm)
    parser.add_argument("--min_lr_ratio", type=float, default=config.min_lr_ratio)
    parser.add_argument("--dampen_factor", type=float, default=config.dampen_factor)
    parser.add_argument("--bf16", action="store_true", default=config.bf16)
    parser.add_argument("--fp16", action="store_true", default=config.fp16)
    
    # Dataset Config
    parser.add_argument("--dataset_name", type=str, default=config.dataset_name)
    parser.add_argument("--dataset_classes", type=str, default=",".join(config.dataset_classes), help="Comma-separated classes")
    parser.add_argument("--samples_per_class", type=str, default=str(config.samples_per_class))
    parser.add_argument("--scan_dataset_dimensions", type=bool, default=config.scan_dataset_dimensions)
    parser.add_argument("--dimension_scan_samples", type=int, default=config.dimension_scan_samples)
    parser.add_argument("--padding_buffer", type=float, default=config.padding_buffer)
    
    # Loss Config
    parser.add_argument("--focal_alpha", type=float, default=config.focal_alpha)
    parser.add_argument("--focal_gamma", type=float, default=config.focal_gamma)
    parser.add_argument("--label_smoothing", type=float, default=config.label_smoothing)
    
    # Logging & Eval
    parser.add_argument("--logging_steps", type=int, default=config.logging_steps)
    parser.add_argument("--eval_steps", type=int, default=config.eval_steps)
    parser.add_argument("--save_steps", type=int, default=config.save_steps)
    parser.add_argument("--compute_train_metrics", type=bool, default=config.compute_train_metrics)
    
    # IO
    parser.add_argument("--output_dir", type=str, default=config.output_dir)
    parser.add_argument("--resume_from_checkpoint", type=str, default=config.resume_from_checkpoint)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Update config with CLI arguments
    for key, value in vars(args).items():
        if key == "dataset_classes" and isinstance(value, str):
            setattr(config, key, [c.strip() for c in value.split(",")])
        elif key == "samples_per_class" and value != 'all':
            try:
                setattr(config, key, int(value))
            except:
                setattr(config, key, value)
        else:
            setattr(config, key, value)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16" if config.bf16 else ("fp16" if config.fp16 else "no"),
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
    lr_scheduler = DampenedCosineScheduler(
        optimizer=optimizer, 
        warmup_steps=config.warmup_steps, 
        total_steps=max_steps,
        min_lr_ratio=config.min_lr_ratio,
        dampen_factor=config.dampen_factor
    )
    
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
