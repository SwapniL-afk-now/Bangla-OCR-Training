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
On Kaggle with **Dual T4 GPUs**, use this comprehensive command to pass all hyperparameters directly. This is the recommended way to use both GPUs with `device_map="auto"` in a single-process environment:

```bash
!python train.py \
    --lora_r 256 \
    --lora_alpha 512 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --fp16 \
    --output_dir "./outputs/qwen2vl-bangla-ocr" \
    --dataset_name "swapnillo/BN-HTR-Handwritten-Dataset" \
    --dataset_classes "word" \
    --samples_per_class "all" \
    --scan_dataset_dimensions True \
    --dimension_scan_samples 50000 \
    --padding_buffer 1.1 \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --label_smoothing 0.1 \
    --logging_steps 5 \
    --eval_steps 1000 \
    --save_steps 200 \
    --compute_train_metrics True
```

> [!TIP]
> Using `python train.py` allows the model's internal `device_map="auto"` to intelligently split layers across GPUs without the overhead or contention of multiple processes.

Settings can be adjusted in `src/config.py`.

## 🧠 Model
The pipeline uses **Qwen/Qwen3-VL-2B-Instruct** as the base model with **LoRA** fine-tuning.

## 📊 Loss Function
A combination of **Focal Loss** and **Confusion-Weighted Cross Entropy**:
- **Focal Loss**: Focuses on hard examples.
- **Confusion Weighting**: Dynamically increases loss for characters the model frequently misidentifies.

## 📜 License
MIT
