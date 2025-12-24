import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

def setup_model_and_processor(config, accelerator):
    accelerator.print("Loading model and processor...")
    
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'
    
    # Keep device_map="auto" for multi-GPU model split
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(processor.tokenizer))
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.gradient_checkpointing_enable()
    
    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.enable_input_require_grads()
    
    return model, processor
