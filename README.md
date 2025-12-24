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
On Kaggle with **Dual T4 GPUs**, use the following command to ensure all internal configurations (like mixed precision and accumulation steps) are respected:

```bash
!accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 train.py
```

> [!NOTE]
> The script automatically reads `gradient_accumulation_steps` and `mixed_precision` from `src/config.py`. The launch flags ensure the environment is correctly initialized for two GPUs.

Settings can be adjusted in `src/config.py`.

## 🧠 Model
The pipeline uses **Qwen/Qwen3-VL-2B-Instruct** as the base model with **LoRA** fine-tuning.

## 📊 Loss Function
A combination of **Focal Loss** and **Confusion-Weighted Cross Entropy**:
- **Focal Loss**: Focuses on hard examples.
- **Confusion Weighting**: Dynamically increases loss for characters the model frequently misidentifies.

## 📜 License
MIT
