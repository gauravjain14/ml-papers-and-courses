from unsloth import FastLanguageModel
import torch
from prepare_dataset import get_gsm8k_questions
from grpo_reward_func import int_reward_func, \
    strict_format_reward_func, \
    xmlcount_reward_func, \
    soft_format_reward_func, \
    correctness_reward_func
from trl import GRPOTrainer, GRPOConfig

max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    lora_rank=lora_rank,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    fast_inference=True,
    device="cuda",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=42
)

dataset = get_gsm8k_questions()

max_prompt_length = 256
training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    num_generations=6,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        int_reward_func,
        strict_format_reward_func,
        xmlcount_reward_func,
        soft_format_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_lora("grpo_saved_lora")