"""Lossless, token-budgeted splitting for KazCulture vLLM extraction.

All source characters remain represented in overlapping segments. Quotes are
validated against the individual segment that the model actually saw, never
merely against the entire original passage. Frame-level provenance stays at
the full passage; chunk-specific interpretation stays on each triple.
"""
from __future__ import annotations

from ..construction.kazculture_kg import EXTRACTION_SYSTEM, extract_frame, MAX_TRIPLES

CHUNKING_VERSION = "token_budgeted_original_text_overlapping_v1"
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_SAFETY_TOKENS = 64
DEFAULT_OVERLAP_CHARS = 160


def render_extraction_prompt(tokenizer, passage: dict, *, system: str = EXTRACTION_SYSTEM) -> str:
    from ..construction.kazculture_kg import extraction_user
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": extraction_user(passage)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def prompt_tokens(tokenizer, passage: dict, *, system: str = EXTRACTION_SYSTEM) -> int:
    # Match the actual string sent to vLLM. An extra reserve handles any
    # minor tokenizer/runtime differences before model.generate() sees it.
    text = render_extraction_prompt(tokenizer, passage, system=system)
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_passage(
    tokenizer, passage: dict, *, max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_tokens: int = 512, safety_tokens: int = DEFAULT_SAFETY_TOKENS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS, max_chunks: int = 128,
    system: str = EXTRACTION_SYSTEM,
) -> list[dict]:
    """Return verbatim original substrings, each guaranteed below prompt budget."""
    if not isinstance(passage.get("text"), str) or not passage["text"]:
        raise ValueError("Missing source passage text")
    budget = max_model_len - max_tokens - safety_tokens
    if budget <= 0:
        raise ValueError("No input token budget after reserving generation and safety")
    original = passage["text"]
    if prompt_tokens(tokenizer, passage, system=system) <= budget:
        return [{**passage, "_chunk_start": 0, "_chunk_end": len(original)}]

    chunks = []
    start = 0
    while start < len(original):
        if len(chunks) >= max_chunks:
            raise ValueError(
                f"{passage.get('passage_id')}: exceeds {max_chunks} chunks; "
                "increase the limit with explicit provenance review"
            )
        low, high, best = start + 1, len(original), -1
        while low <= high:
            mid = (low + high) // 2
            probe = {**passage, "text": original[start:mid]}
            if prompt_tokens(tokenizer, probe, system=system) <= budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best <= start:
            raise ValueError(
                f"{passage.get('passage_id')}: source header alone exceeds token budget"
            )
        # Prefer a word/line boundary while preserving a full exact substring.
        # Never drop characters or normalize the source. 160-char overlaps
        # let sentences near boundaries be extracted with shared context.
        end = best
        if best < len(original):
            search = original.rfind(" ", max(start + 1, best - 160), best)
            if search > start + max(32, (best - start) // 2):
                end = search + 1
        chunk = {**passage, "text": original[start:end],
                 "_chunk_start": start, "_chunk_end": end}
        if prompt_tokens(tokenizer, chunk, system=system) > budget:
            raise AssertionError("Chunk exceeds verified token budget")
        chunks.append(chunk)
        if end == len(original):
            break
        # Boundary overlap must progress and cover all original characters.
        start = max(start + 1, end - overlap_chars)
    if chunks[0]["_chunk_start"] != 0 or chunks[-1]["_chunk_end"] != len(original):
        raise AssertionError("Source passage was not fully covered")
    for previous, current in zip(chunks, chunks[1:]):
        if current["_chunk_start"] > previous["_chunk_end"]:
            raise AssertionError("Gap between source chunks")
    return chunks


def aggregate_chunks(passage: dict, chunks: list[dict],
                     responses: list[str], *, model: str) -> dict:
    """Merge candidate relations without erasing their local interpretation."""
    if len(chunks) != len(responses) or not chunks:
        raise ValueError("Each chunk must have exactly one response")
    if any(chunk["passage_id"] != passage["passage_id"] for chunk in chunks):
        raise ValueError("Cannot join different source passages")
    frames = [extract_frame(chunk, raw, model=model)
              for chunk, raw in zip(chunks, responses)]
    triples = []
    seen = set()
    segment_labels = []
    for idx, (chunk, frame) in enumerate(zip(chunks, frames)):
        segment_labels.append({
            "chunk_index": idx, "start_char": chunk["_chunk_start"],
            "end_char": chunk["_chunk_end"],
            "norm": frame["norm"], "norm_type": frame["norm_type"],
            "values": frame["values"],
            "label_status": frame["label_status"],
        })
        for triple in frame["triples"]:
            # Verified quote must truly be inside this model-visible chunk.
            assert triple["evidence"] in chunk["text"]
            if triple["triple_id"] in seen:
                continue
            seen.add(triple["triple_id"])
            triples.append({
                **triple, "source_chunk_index": idx,
                "source_start_char": chunk["_chunk_start"],
                "source_end_char": chunk["_chunk_end"],
                "chunk_norm": frame["norm"],
                "chunk_norm_type": frame["norm_type"],
                "chunk_values": frame["values"],
            })

    # If different segments inferred different norms, do not present a
    # fabricated passage-level consensus as a universal norm.
    def unanimous(field):
        vals = [fr[field] for fr in frames if fr["triples"] and fr[field]]
        return vals[0] if vals and all(v == vals[0] for v in vals) else (
            vals[0] if len(vals) == 1 else ""
        )

    norm = unanimous("norm")
    norm_type = unanimous("norm_type")
    values = []
    labeled = [tuple(fr["values"]) for fr in frames if fr["triples"] and fr["values"]]
    if labeled and all(v == labeled[0] for v in labeled):
        values = list(labeled[0])
    return {
        **passage,
        "triples": triples,
        "norm": norm, "norm_type": norm_type, "values": values,
        "label_status": "candidate" if triples else "failed",
        "interpretation_status": "provisional",
        "extraction_model": model,
        "chunking_policy": CHUNKING_VERSION,
        "chunk_count": len(chunks),
        "chunk_interpretations": segment_labels,
    }
