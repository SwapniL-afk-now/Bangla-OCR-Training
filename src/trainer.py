import os
import json
import torch
from tqdm.auto import tqdm
from jiwer import cer, wer
from src.utils.metrics import calculate_batch_metrics_from_logits
from src.utils.checkpoints import save_checkpoint, load_checkpoint, cleanup_old_checkpoints
from src.losses import CombinedLoss

def validate(model, processor, val_dataloader, loss_fn, accelerator):
    """Run validation - using logits for metrics"""
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating", leave=False, disable=not accelerator.is_main_process):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch['image_grid_thw'],
                labels=batch['labels']
            )
            
            loss, _ = loss_fn(outputs.logits, batch['labels'])
            total_loss += loss.item()
            
            # Calculate metrics from logits (not generation)
            cer_score, wer_score, preds, refs = calculate_batch_metrics_from_logits(
                outputs.logits, batch['labels'], processor
            )
            all_predictions.extend(preds)
            all_references.extend(refs)
    
    model.train()
    
    avg_loss = total_loss / len(val_dataloader)
    
    # Calculate overall metrics
    if all_predictions and all_references:
        try:
            overall_cer = cer(all_references, all_predictions) * 100
            overall_wer = wer(all_references, all_predictions) * 100
        except:
            overall_cer = 0.0
            overall_wer = 0.0
    else:
        overall_cer = 0.0
        overall_wer = 0.0
    
    return {
        'val_loss': avg_loss,
        'val_cer': overall_cer,
        'val_wer': overall_wer
    }

class Trainer:
    def __init__(self, model, processor, train_dataloader, val_dataloader, 
                 optimizer, lr_scheduler, loss_fn, gpu_augmentor, 
                 confusion_matrix, config, accelerator):
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.gpu_augmentor = gpu_augmentor
        self.confusion_matrix = confusion_matrix
        self.config = config
        self.accelerator = accelerator
        
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_cer = float('inf')

    def train(self, start_step=0, start_epoch=0, max_steps=-1):
        self.global_step = start_step
        self.start_epoch = start_epoch
        self.model.train()
        
        for epoch in range(self.start_epoch, self.config.num_train_epochs):
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}", disable=not self.accelerator.is_main_process)
            
            for step, batch in enumerate(progress_bar):
                # Skip steps if resuming
                if self.global_step < start_step:
                    self.global_step += 1
                    continue
                    
                with self.accelerator.accumulate(self.model):
                    # Augmentation
                    if batch['pixel_values'] is not None:
                        batch['pixel_values'] = self.gpu_augmentor(batch['pixel_values'])
                    
                    # Forward
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        pixel_values=batch['pixel_values'],
                        image_grid_thw=batch['image_grid_thw'],
                        labels=batch['labels']
                    )
                    
                    custom_loss, loss_components = self.loss_fn(outputs.logits, batch['labels'])
                    
                    # Backward
                    self.accelerator.backward(custom_loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Logging
                if self.global_step % self.config.logging_steps == 0 and self.accelerator.is_main_process:
                    self._log_progress(custom_loss, outputs, batch, progress_bar)
                
                # Validation
                if self.global_step % self.config.eval_steps == 0 and self.global_step > 0:
                    self._run_validation(epoch)
                
                # Confusion matrix summary
                if self.config.compute_train_metrics and self.global_step % self.config.show_confusion_matrix_steps == 0 and self.global_step > 0:
                    if self.accelerator.is_main_process:
                        self.confusion_matrix.print_summary(self.global_step, self.config.confusion_update_threshold)
                
                # Adaptive updates
                self._check_adaptive_updates()
                
                # Checkpoint
                if self.global_step % self.config.save_steps == 0 and self.global_step > 0:
                    self._save_checkpoint(epoch)
                
                if self.global_step % self.config.empty_cache_frequency == 0:
                    torch.cuda.empty_cache()
                
                self.global_step += 1
                if max_steps > 0 and self.global_step >= max_steps:
                    break
            
            if max_steps > 0 and self.global_step >= max_steps:
                break

    def _log_progress(self, loss, outputs, batch, progress_bar):
        log_info = f"Step {self.global_step} | Loss: {loss.item():.4f}"
        
        if self.config.compute_train_metrics:
            batch_cer, batch_wer, predictions, ground_truths = calculate_batch_metrics_from_logits(
                outputs.logits.detach(), batch['labels'], self.processor
            )
            
            if self.global_step <= 20 and predictions and ground_truths:
                self.accelerator.print(f"\nDEBUG Step {self.global_step}:")
                self.accelerator.print(f"  Ground Truth: '{ground_truths[0][:50]}'")
                self.accelerator.print(f"  Prediction:   '{predictions[0][:50]}'")
            
            if predictions and ground_truths:
                self.confusion_matrix.update(predictions, ground_truths)
            
            log_info += f" | CER: {batch_cer:.2f}% | WER: {batch_wer:.2f}%"
        
        self.accelerator.print(log_info)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    def _run_validation(self, epoch):
        self.accelerator.print(f"\n{'='*70}\nVALIDATION - Step {self.global_step}\n{'='*70}")
        val_metrics = validate(self.model, self.processor, self.val_dataloader, self.loss_fn, self.accelerator)
        
        if self.accelerator.is_main_process:
            self.accelerator.print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            self.accelerator.print(f"Val CER:  {val_metrics['val_cer']:.2f}%")
            self.accelerator.print(f"Val WER:  {val_metrics['val_wer']:.2f}%")
            
            if val_metrics['val_cer'] < self.best_val_cer:
                self.best_val_cer = val_metrics['val_cer']
                best_dir = os.path.join(self.config.output_dir, "best_model")
                save_checkpoint(self.model, self.optimizer, self.lr_scheduler, self.global_step, epoch, best_dir, self.accelerator)
                self.accelerator.print(f"✓ Best model saved (CER: {self.best_val_cer:.2f}%)\n")

    def _check_adaptive_updates(self):
        if (self.config.auto_update_confusion_chars and self.config.compute_train_metrics and
            self.global_step > self.config.auto_update_after_step and 
            self.global_step % self.config.adaptive_sampling_update_freq == 0):
            
            problematic = self.confusion_matrix.get_problematic_chars(self.config.confusion_update_threshold)
            
            if problematic and self.accelerator.is_main_process:
                old_count = len(self.config.CONFUSION_CHARS)
                new_chars = list(problematic.keys())[:self.config.confusion_max_additions]
                for char in new_chars:
                    if char not in self.config.CONFUSION_CHARS and len(self.config.CONFUSION_CHARS) < self.config.confusion_map_max_size:
                        self.config.CONFUSION_CHARS[char] = problematic[char]
                
                if len(self.config.CONFUSION_CHARS) > old_count:
                    self.accelerator.print(f"\n🔄 Confusion tracking: {len(self.config.CONFUSION_CHARS)} chars")
                    self.loss_fn = CombinedLoss(
                        processor=self.processor,
                        focal_alpha=self.config.focal_alpha,
                        focal_gamma=self.config.focal_gamma,
                        label_smoothing=self.config.label_smoothing,
                        confusion_chars=self.config.CONFUSION_CHARS,
                        confusion_weight=self.config.confusion_char_loss_weight
                    )

    def _save_checkpoint(self, epoch):
        if self.accelerator.is_main_process:
            cleanup_old_checkpoints(self.config.output_dir, keep_last_n=0)
            
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        save_checkpoint(self.model, self.optimizer, self.lr_scheduler, self.global_step, epoch, checkpoint_dir, self.accelerator)
        
        if self.config.compute_train_metrics and self.accelerator.is_main_process:
            confusion_path = os.path.join(self.config.output_dir, f"confusion_step_{self.global_step}.json")
            with open(confusion_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'confusion_counts': {k: dict(v) for k, v in self.confusion_matrix.confusion_counts.items()},
                    'total_errors': self.confusion_matrix.total_errors,
                    'tracked_chars': list(self.config.CONFUSION_CHARS.keys()),
                    'step': self.global_step
                }, f, ensure_ascii=False, indent=2)
