import os
import glob
import shutil
import torch
import safetensors.torch
from peft import set_peft_model_state_dict

def cleanup_old_checkpoints(output_dir, keep_last_n=0):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if keep_last_n == 0:
        for checkpoint_path in checkpoints:
            shutil.rmtree(checkpoint_path)
    elif len(checkpoints) > keep_last_n:
        checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
        for checkpoint_path in checkpoints_sorted[:-keep_last_n]:
            shutil.rmtree(checkpoint_path)

def save_checkpoint(model, optimizer, lr_scheduler, global_step, epoch, checkpoint_dir, accelerator):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Model uses device_map, save directly
    if hasattr(model, 'module'):
        model.module.save_pretrained(checkpoint_dir)
    else:
        model.save_pretrained(checkpoint_dir)
    
    # Unwrap optimizer/scheduler from accelerator
    training_state = {
        "optimizer": accelerator.unwrap_model(optimizer).state_dict() if hasattr(optimizer, 'module') else optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
    }
    accelerator.save(training_state, os.path.join(checkpoint_dir, "training_states.pt"))

def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, config, accelerator):
    """Load model and training states from checkpoint"""
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
    accelerator.print(f"{'='*70}")
    
    # Load LoRA adapter weights
    if config.use_lora:
        adapter_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            accelerator.print("Loading LoRA adapter weights...")
            state_dict = safetensors.torch.load_file(adapter_file)
            set_peft_model_state_dict(model, state_dict)
            accelerator.print("✓ LoRA adapter loaded")
        else:
            accelerator.print(f"Warning: {adapter_file} not found")
    
    # Load training states
    training_states_path = os.path.join(checkpoint_path, "training_states.pt")
    if os.path.exists(training_states_path):
        accelerator.print("Loading training states...")
        training_states = torch.load(training_states_path, map_location='cpu')
        
        optimizer.load_state_dict(training_states['optimizer'])
        lr_scheduler.load_state_dict(training_states['scheduler'])
        
        global_step = training_states['global_step']
        epoch = training_states['epoch']
        
        accelerator.print(f"✓ Optimizer and scheduler loaded")
        accelerator.print(f"✓ Resuming from step {global_step}, epoch {epoch}")
        accelerator.print(f"{'='*70}\n")
        
        return global_step, epoch
    else:
        accelerator.print(f"Warning: training_states.pt not found")
        return 0, 0
