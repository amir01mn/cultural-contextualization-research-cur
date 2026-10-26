"""Hugging Face next-token A/B/C/D log-probability scorer for Qwen3.

Runs the model unmodified and always disables Qwen3 thinking in the chat
template. Fails if candidate choices are not single tokens for this tokenizer.
"""
from __future__ import annotations

import math

from .kazculture_phase1 import question_text
from ..loaders.kazculture_loader import CHOICES, KazCultureQuestion


class HFQwenLetterScorer:
    def __init__(self, model_name: str = "Qwen/Qwen3-32B",
                 *, model_revision: str | None = None,
                 dtype: str = "bfloat16", device_map: str = "auto",
                 max_context_tokens: int = 4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.model_revision = model_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=model_revision, torch_dtype=getattr(torch, dtype),
            device_map=device_map, trust_remote_code=False,
            low_cpu_mem_usage=True).eval()
        self.max_context_tokens = max_context_tokens
        self.token_ids = {}
        # With the assistant suffix 'Жауап:' these are the allowed next tokens.
        # Tokenizer boundaries are explicit; do not silently score a different task.
        for label in CHOICES:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Choice {label!r} is not one token: {ids}")
            self.token_ids[label] = ids[0]
        if len(set(self.token_ids.values())) != 4:
            raise ValueError("Candidate token IDs must be distinct")

    def _inputs(self, q: KazCultureQuestion):
        messages = [{"role": "user", "content": question_text(q)}]
        # Qwen3 non-thinking inference. Do not append a written assistant prefix
        # with additional free text that would change the scored next token.
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        # Force a stable boundary for candidate next-token scoring.
        rendered += "Жауап: "
        inputs = self.tokenizer(rendered, return_tensors="pt",
                                add_special_tokens=False)
        if inputs["input_ids"].shape[-1] > self.max_context_tokens:
            raise ValueError(f"Input exceeds context budget ({self.max_context_tokens})")
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in inputs.items()}

    def score(self, questions):
        results = []
        with self.torch.inference_mode():
            for q in questions:
                logits = self.model(**self._inputs(q)).logits[0, -1].float()
                logprobs = self.torch.log_softmax(logits, dim=-1)
                results.append({k: float(logprobs[t].item())
                                for k, t in self.token_ids.items()})
        return results
