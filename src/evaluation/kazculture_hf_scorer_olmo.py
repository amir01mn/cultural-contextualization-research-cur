"""OLMo-2-1124-7B-Instruct replication scorer for KazCulture (criterion 5).

Same protocol as the Qwen3 V2 scorer: the assistant turn is prefilled with
``Answer:`` and the next-token log-probabilities of `` A``..`` D`` are compared.
Each candidate is verified *in context* to be a single token
(``candidate_token_ids``); for this tokenizer they are 362/426/356/423. OLMo-2
has no thinking mode, so no ``enable_thinking`` kwarg is passed to its chat
template. Context is capped at 4,096 tokens; the evaluator truncates evidence
by whole blocks to fit and records every truncation.
"""
from __future__ import annotations

from .kazculture_rag import V2EvidenceScorer

OLMO_MODEL = "allenai/OLMo-2-1124-7B-Instruct"
OLMO_REVISION = "470b1fba1ae01581f270116362ee4aa1b97f4c84"
OLMO_MAX_CONTEXT_TOKENS = 4096
PROTOCOL_VERSION = "kazculture-olmo2-assistant-answer-prefix-v1"


class HFOlmoLetterScorer(V2EvidenceScorer):
    template_kwargs = {}

    def __init__(self, model_name: str = OLMO_MODEL, *,
                 model_revision: str | None = OLMO_REVISION,
                 max_context_tokens: int = OLMO_MAX_CONTEXT_TOKENS):
        super().__init__(model_name, model_revision=model_revision,
                         max_context_tokens=max_context_tokens)
