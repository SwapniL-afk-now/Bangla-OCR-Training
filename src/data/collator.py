import torch
from typing import List, Dict

class OptimizedOCRDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self._prompt_template = "Transcribe the handwritten Bangla text from the image. Respond only with the transcribed text."
        self._prompt_length = None
        self._cache_prompt_tokens()
    
    def _cache_prompt_tokens(self):
        dummy_message = [
            [
                {"role": "user", "content": [{"type": "text", "text": self._prompt_template}, {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]}
            ]
        ]
        prompt_text = self.processor.apply_chat_template(
            dummy_message, 
            tokenize=False, 
            add_generation_prompt=False
        )
        prompt_tokens = self.processor.tokenizer(prompt_text, add_special_tokens=False).input_ids
        self._prompt_length = len(prompt_tokens) - 1
    
    def __call__(self, batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]

        messages_batch = [
            [
                {"role": "user", "content": [{"type": "text", "text": self._prompt_template}, {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            ]
            for text in texts
        ]
        
        text_prompts = self.processor.apply_chat_template(
            messages_batch, 
            tokenize=False, 
            add_generation_prompt=False
        )

        processed = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        
        labels = processed['input_ids'].clone()
        
        response_tokens_batch = self.processor.tokenizer(
            texts, 
            add_special_tokens=False,
            padding=False
        ).input_ids
        
        for i, response_tokens in enumerate(response_tokens_batch):
            search_start = max(0, self._prompt_length - 10)
            full_tokens = labels[i]
            
            response_len = len(response_tokens)
            found = False
            
            for start_idx in range(search_start, len(full_tokens) - response_len + 1):
                if torch.all(full_tokens[start_idx:start_idx + response_len] == torch.tensor(response_tokens)):
                    labels[i, :start_idx] = -100
                    found = True
                    break
            
            if not found:
                labels[i, :] = -100
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        processed['labels'] = labels
        processed['ground_truth_texts'] = texts
        
        return processed
