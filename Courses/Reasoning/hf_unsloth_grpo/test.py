# load model from safetensors in grpo_saved_lora

import json
from unsloth import FastLanguageModel
from vllm import SamplingParams
import torch
from prepare_dataset import SYSTEM_PROMPT


max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    fast_inference=True,
    device="cuda",
)

model.load_adapter("grpo_saved_lora")
model.eval()

text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Calculate partial derivative of x^2 + y^2 with respect to x."},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = (
    model.fast_generate(
        text,
        sampling_params=sampling_params
    )[0]
    .outputs[0]
    .text
)

print(output)