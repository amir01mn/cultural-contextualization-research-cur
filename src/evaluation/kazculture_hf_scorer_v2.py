"""Qwen3-32B non-thinking MCQ scoring with an explicit assistant answer prefix.

V2 fixes the V1 scoring position: the V1 code selected bare letter tokens
immediately after the assistant chat header. Here, the assistant is explicitly
prefilled with 'Answer:' and we score context-valid ' A'...' D' tokens.
This is an independent protocol, not an assertion of exact paper reproduction.
"""
from __future__ import annotations

import hashlib

from ..loaders.kazculture_loader import CHOICES, KazCultureQuestion
from .kazculture_phase1 import question_text

PROTOCOL_VERSION = "kazculture-qwen3-assistant-answer-prefix-v2"


def user_question(q: KazCultureQuestion) -> str:
    """Move V1's user-side Answer: delimiter to the assistant continuation."""
    full = question_text(q)
    suffix = "\n\nAnswer:"
    if not full.endswith(suffix):
        raise ValueError("Unexpected Phase 1 question template")
    return full[:-len(suffix)]


def render_prompt(tokenizer, q: KazCultureQuestion) -> str:
    messages = [{"role": "user", "content": user_question(q)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    ) + "Answer:"


def prompt_sha256(q: KazCultureQuestion) -> str:
    return hashlib.sha256(
        (PROTOCOL_VERSION + "\n" + user_question(q) + "\nassistant:Answer:").encode("utf-8")
    ).hexdigest()


def candidate_token_ids(tokenizer, rendered: str) -> dict[str, int]:
    """Verify *in context* each candidate is one next token.

    Tokenizers can encode 'A' differently from ' A'. This checks full
    contextual tokenization, not merely tokenizing isolated letters.
    """
    base = tokenizer.encode(rendered, add_special_tokens=False)
    result: dict[str, int] = {}
    for letter in CHOICES:
        joint = tokenizer.encode(rendered + " " + letter, add_special_tokens=False)
        continuation = joint[len(base):]
        if joint[:len(base)] != base or len(continuation) != 1:
            raise ValueError(
                f"Label {letter!r}: cannot use one-token next-token scoring "
                "with this exact assistant prefix; use full-sequence scoring "
                "instead. Refusing to silently score the wrong token."
            )
        result[letter] = continuation[0]
    if len(set(result.values())) != 4:
        raise ValueError("A/B/C/D must have distinct candidate token IDs")
    return result


def gold_continuation_ids(tokenizer, prompt: str, gold: str) -> tuple[list[int], int]:
    """Token IDs of ``prompt`` followed by `` <gold>``, and the prompt length.

    The prompt (ending in the assistant ``Answer:`` prefix) and the gold answer
    are tokenized separately and concatenated, so the scored span is exactly
    the gold answer's tokens regardless of BPE merges at the boundary.
    """
    gold = " ".join(str(gold).split())
    if not gold:
        raise ValueError("Empty gold answer cannot be scored")
    base = list(tokenizer.encode(prompt, add_special_tokens=False))
    cont = list(tokenizer.encode(" " + gold, add_special_tokens=False))
    if not base or not cont:
        raise ValueError("Prompt and gold answer must both be non-empty in tokens")
    return base + cont, len(base)


class HFQwenLetterScorerV2:
    def __init__(self, model_name: str = "Qwen/Qwen3-32B",
                 *, model_revision: str | None = None,
                 max_context_tokens: int = 4096):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision, trust_remote_code=False
        )
        self.max_context_tokens = max_context_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=model_revision, dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True, trust_remote_code=False
        ).eval()

    def score_gold_loglik(self, prompts: list[str], golds: list[str]) -> list[dict]:
        """Mean per-token log-likelihood of each free-text gold answer.

        ``prompts`` are fully rendered chat prompts ending in the assistant
        ``Answer:`` prefix; the gold is scored as the continuation `` <gold>``
        (teacher forcing, no EOS). Used only for the train-dev gate, whose
        train questions have free-text answers (preregistration §4).
        """
        if len(prompts) != len(golds):
            raise ValueError("prompts and golds must have equal length")
        torch = self.torch
        device = next(self.model.parameters()).device
        results = []
        with torch.inference_mode():
            for prompt, gold in zip(prompts, golds):
                ids, n_prompt = gold_continuation_ids(self.tokenizer, prompt, gold)
                if len(ids) > self.max_context_tokens:
                    raise ValueError(f"Input exceeds {self.max_context_tokens} tokens")
                input_ids = torch.tensor([ids], device=device)
                logits = self.model(input_ids=input_ids).logits[0].float()
                # Position t predicts token t+1: rows n_prompt-1 .. len-2.
                logprobs = torch.log_softmax(logits[n_prompt - 1:-1], dim=-1)
                targets = input_ids[0, n_prompt:]
                tok = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
                results.append({
                    "mean_loglik": float(tok.mean().item()),
                    "sum_loglik": float(tok.sum().item()),
                    "n_gold_tokens": int(targets.shape[0]),
                    "n_prompt_tokens": n_prompt,
                })
        return results

    def score(self, questions):
        results = []
        with self.torch.inference_mode():
            for q in questions:
                rendered = render_prompt(self.tokenizer, q)
                labels = candidate_token_ids(self.tokenizer, rendered)
                inputs = self.tokenizer(
                    rendered, return_tensors="pt", add_special_tokens=False
                )
                if inputs["input_ids"].shape[-1] > self.max_context_tokens:
                    raise ValueError(f"Input exceeds {self.max_context_tokens} tokens")
                inputs = {k: v.to(next(self.model.parameters()).device)
                          for k, v in inputs.items()}
                next_logits = self.model(**inputs).logits[0, -1].float()
                logprobs = self.torch.log_softmax(next_logits, dim=-1)
                results.append({
                    k: float(logprobs[tid].item())
                    for k, tid in labels.items()
                })
        return results
