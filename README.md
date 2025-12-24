# Optimized Bangla OCR Training

A modular, multi-GPU optimized training pipeline for handwritten Bangla OCR using **Qwen-VL-2B**.

## 🚀 Key Features
- **Multi-GPU Scaling**: Seamlessly integrated with `accelerate`.
- **Memory Efficient**: 8-bit optimization and gradient checkpointing.
- **Dynamic Loss Weighting**: Automatically identifies and penalizes character-level confusion.
- **Robust Metrics**: Real-time CER/WER tracking during training.
- **GPU Augmentation**: High-speed image augmentations using `kornia`.

## 📁 Project Structure
```text
.
├── src/
│   ├── data/          # Dataset, Collator, Augmentation
│   ├── models/        # Model factory
│   ├── utils/         # Metrics, Checkpoints, Confusion Matrix
│   ├── config.py      # Training configuration
│   ├── losses.py      # Focal and Confusion losses
│   └── trainer.py     # Core training logic
├── train.py           # Main entry point
├── requirements.txt   # Dependencies
└── README.md
```

## 🛠️ Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Accelerate**:
```bash
accelerate config
```

## 📈 Training

### Standard (Interactive)
To start training with multi-GPU support:
```bash
accelerate launch train.py
```

### Kaggle / Notebooks (Non-interactive)
On Kaggle with **Dual T4 GPUs**, use the following comprehensive command to pass all hyperparameters (including **Rank** and **Lora Alpha**) directly to the script:

```bash
!accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    --gradient_accumulation_steps=6 \
    train.py \
    --lora_r 512 \
    --lora_alpha 1024 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --output_dir "./outputs/qwen2vl-bangla-ocr"
```

> [!TIP]
> This command explicitly sets hardware, precision, and performance hyperparameters. You can override any setting defined in `src/config.py` by adding more flags here.

Settings can be adjusted in `src/config.py`.

## 🧠 Model
The pipeline uses **Qwen/Qwen3-VL-2B-Instruct** as the base model with **LoRA** fine-tuning.

## 📊 Loss Function
A combination of **Focal Loss** and **Confusion-Weighted Cross Entropy**:
- **Focal Loss**: Focuses on hard examples.
- **Confusion Weighting**: Dynamically increases loss for characters the model frequently misidentifies.

## 📜 License
MIT
