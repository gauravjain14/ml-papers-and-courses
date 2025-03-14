# Prepare the dataset

import re
from datasets import load_dataset, Dataset


# Sample from the dataset
# question:
# James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
# answer:
# He writes each friend 3*2=<<3*2=6>>6 pages a week
# So he writes 6*2=<<6*2=12>>12 pages every week
# That means he writes 12*52=<<12*52=624>>624 pages a year
# #### 624
#
# In the system format - 
# {'question': 'Natalia sold clips to 48 of her friends in April, 
# and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 
# 'answer': '72', 
# 'prompt': [{'content': '\nRespond in the following format:\n<reasoning>\n...\n</reasoning>
# \n<answer>\n...\n</answer>\n', 'role': 'system'}, 
# {'content': 'Natalia sold clips to 48 of her friends in April, 
# and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 
# 'role': 'user'}]}



### Data Preparation
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text):
    return text.split("<answer>")[-1].split("</answer>")[0].strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]}
            ],
            "answer": extract_hash_answer(x["answer"])
        }
    )
    return data