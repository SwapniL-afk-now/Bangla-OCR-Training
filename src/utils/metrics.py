import torch
import re
from jiwer import cer, wer

def compute_metrics(predictions, references):
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    
    if not valid_pairs:
        return {"cer": 0.0, "wer": 0.0}
    
    preds, refs = zip(*valid_pairs)
    
    try:
        cer_score = cer(refs, preds) * 100
        wer_score = wer(refs, preds) * 100
    except:
        cer_score = 0.0
        wer_score = 0.0
    
    return {"cer": cer_score, "wer": wer_score}

def calculate_batch_metrics_from_logits(logits, labels, processor):
    """Calculate CER/WER directly from logits (training time metrics)"""
    try:
        # Align predictions and labels like in loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        predicted_ids = torch.argmax(shift_logits, dim=-1)
        
        # Prepare for decoding
        answer_mask = shift_labels != -100
        
        all_pred_texts = []
        all_label_texts = []
        
        for j in range(len(shift_labels)):
            pred_ids_j = predicted_ids[j][answer_mask[j]]
            label_ids_j = shift_labels[j][answer_mask[j]]
            
            pred_text = processor.tokenizer.decode(pred_ids_j, skip_special_tokens=True)
            label_text = processor.tokenizer.decode(label_ids_j, skip_special_tokens=True)
            
            # Clean up artifacts
            pred_text = re.sub(r'</?tool_call>|</?tool_code>', '', pred_text).strip()
            
            all_pred_texts.append(pred_text)
            all_label_texts.append(label_text)
        
        if not all_label_texts or not all_pred_texts:
            return 0.0, 0.0, all_pred_texts, all_label_texts
        
        cer_score = cer(all_label_texts, all_pred_texts) * 100
        wer_score = wer(all_label_texts, all_pred_texts) * 100
        
        return cer_score, wer_score, all_pred_texts, all_label_texts
    except Exception as e:
        print(f"Warning: Could not calculate batch metrics: {e}")
        return 0.0, 0.0, [], []
