"""A local instruction-tuned language model behind the Generator interface.

Kept separate from rag_for_pandas.generation so that prompt building and citation
checks can be imported and tested without loading PyTorch models.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 250


class LocalGenerator:
    """Greedy decoding, so the same prompt gives the same answer."""

    def __init__(self, model_name: str = LOCAL_MODEL, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        # bfloat16 halves memory on GPU: the 1.5B model peaks at about 3.1 GiB there.
        on_gpu = torch.cuda.is_available()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if on_gpu else torch.float32,
            device_map="cuda" if on_gpu else "cpu",
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        prompt_length = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
