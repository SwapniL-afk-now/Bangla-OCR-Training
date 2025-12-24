import os
from typing import Dict, List, Optional

class Config:
    # Model
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    use_lora = True
    lora_r = 512
    lora_alpha = 1024
    lora_dropout = 0.1
    
    # Resume training from checkpoint
    resume_from_checkpoint = None  # Default changed for clean repo
    
    # Training
    batch_size = 16
    gradient_accumulation_steps = 6
    learning_rate = 1e-4
    num_train_epochs = 3
    max_steps = -1
    warmup_steps = 100
    weight_decay = 0.01
    max_grad_norm = 1.0
    min_lr_ratio = 0.1
    dampen_factor = 1.0
    
    # Dataset
    dataset_name = "swapnillo/BN-HTR-Handwritten-Dataset"
    dataset_classes = ['word']
    samples_per_class = 'all'
    
    train_ratio = 0.9
    val_ratio = 0.03
    test_ratio = 0.07
    
    # Image dimensions - AUTO-DETECTED
    scan_dataset_dimensions = True
    dimension_scan_samples = 50000
    padding_buffer = 1.1
    manual_max_width = 1500
    manual_max_height = 128
    
    max_width = None
    max_height = None
    
    # Logging & Evaluation
    logging_steps = 5
    eval_steps = 1000
    save_steps = 200
    compute_train_metrics = True
    
    # Loss
    focal_alpha = 0.25
    focal_gamma = 2.0
    label_smoothing = 0.1
    
    fp16 = True
    bf16 = False
    
    output_dir = "./outputs/qwen2vl-bangla-ocr"
    
    # DataLoader
    num_workers = 8
    pin_memory = True
    prefetch_factor = 2
    persistent_workers = True
    
    empty_cache_frequency = 20
    
    # Augmentation
    use_gpu_augmentation = True
    augmentation_probability = 0.4
    
    # Confusion tracking
    auto_update_confusion_chars = True
    auto_update_after_step = 1000
    confusion_update_threshold = 10
    confusion_max_additions = 5
    confusion_map_max_size = 50
    show_confusion_matrix_steps = 1000
    
    confusion_char_loss_weight = 2.0
    adaptive_sampling_update_freq = 1000
    
    enable_timing = True
    max_new_tokens = 512
    num_beams = 1
    
    CONFUSION_CHARS = {}

config = Config()
