"""Multilingual indexed evidence for KazCulture S1-S4R and combined KG.

Only training-passage-derived text can enter the index. A model-specific dense
index is frozen on disk and reused unchanged for evaluation. Any CultureBank
evidence is explicitly marked INDIRECT, with its original frame provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_NAME = "BAAI/bge-m3"
# Frozen retriever revision (docs/KAZCULTURE_PREREGISTRATION.md §2). Loading is
# pinned so an updated hub snapshot can never silently change the embeddings.
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
SYSTEMS = ("S1", "S2", "S3", "S4F", "S4R")


def make_units(passages: list[dict], frames: list[dict] | None = None,
               culturebank_frames: list[dict] | None = None) -> list[dict]:
    """Create provenance-preserving retrieval units for all baseline systems."""
    by_pid = {f["passage_id"]: f for f in (frames or [])}
    units = []
    for p in passages:
        base = {
            "passage_id": p["passage_id"], "dataset": "KazCulture",
            "source": p["sources"], "culture": "Kazakh",
            "category": p["categories"], "values": [],
            "indirect": False,
        }
        units.append({**base, "unit_id": f"S1:{p['passage_id']}",
                      "system": "S1", "text": p["text"]})
        frame = by_pid.get(p["passage_id"])
        if not frame or frame.get("label_status") != "candidate":
            continue
        is_v2 = frame.get("schema_version") in {"kazculture-kg-factual-normative-v2", "kazculture-kg-factual-normative-v2.1", "kazculture-kg-nested-norms-v2.2"}
        norm_parts = []
        for t in frame["triples"]:
            # Facts remain retrievable as S2 even when no social-norm layer is
            # justified. Only explicitly proposed v2 pathways can enter
            # layered S3/S4R systems. The v1 representation is unchanged.
            explicit_path = t.get("pathway") if is_v2 else None
            triple_values = (explicit_path.get("values", []) if explicit_path
                             else [] if is_v2 else
                             t.get("chunk_values", frame["values"]))
            triple_norm = (explicit_path.get("norm", "") if explicit_path
                           else "" if is_v2 else
                           t.get("chunk_norm", frame["norm"]))
            triple_norm_type = (explicit_path.get("norm_type", "") if explicit_path
                                else "" if is_v2 else
                                t.get("chunk_norm_type", frame["norm_type"]))
            fact = f"{t['subject']} — {t['predicate']} — {t['object']}. Quote: {t['evidence']}"
            units.append({
                **base, "unit_id": "S2:" + t["triple_id"],
                "system": "S2", "text": fact,
                "triple_id": t["triple_id"], "values": triple_values,
            })
            if is_v2 and not explicit_path:
                continue
            layers = (
                f"L3 (provisional value): {', '.join(triple_values) or '(unclassified)'}\n"
                f"L2 (provisional norm type): {triple_norm_type or '(unclassified)'}\n"
                f"L2 (provisional norm): {triple_norm or '(unclassified)'}\n"
                f"L1 (source-quoted practice): {fact}"
            )
            units.append({
                **base, "unit_id": "S3:" + t["triple_id"],
                "system": "S3", "text": layers,
                "triple_id": t["triple_id"], "values": triple_values,
            })
            units.append({
                **base, "unit_id": "S4R:" + t["triple_id"],
                "system": "S4R",
                "text": (
                    f"Value (provisional): {', '.join(triple_values)}. "
                    f"Norm type: {triple_norm_type}. "
                    f"Culture-specific norm: {triple_norm}. "
                    f"Source-quoted practice: {t['evidence']}. "
                    f"Relation: {t['subject']} — {t['predicate']} — {t['object']}."
                ),
                "values": triple_values, "triple_id": t["triple_id"],
            })
            norm_parts.append((triple_norm, t["evidence"], triple_values))
        if (norm_parts if is_v2 else frame["triples"]):
            evidence = (" | ".join(
                f"Norm: {norm}. Evidence: {quote}" for norm, quote, _ in norm_parts
                ) if is_v2 else
                f"Kazakh cultural frame, provisional norm: {frame['norm']}. Source: {p['text']}")
            frame_values = (sorted({val for _, _, vals in norm_parts for val in vals})
                            if is_v2 else frame["values"])
            units.append({
                **base, "unit_id": "S4F:" + p["passage_id"],
                "system": "S4F",
                "text": evidence,
                "values": frame_values,
            })

    # Only source-preserving CultureBank KG v3 frames are eligible to align.
    for frame in culturebank_frames or []:
        if frame.get("label_status") != "ok" or not frame.get("text"):
            continue
        frame_id = str(frame["frame_id"])
        culture = str(frame.get("culture_label") or frame.get("culture") or "Unknown")
        units.append({
            "unit_id": "CB:S4R:" + frame_id,
            "system": "S4R", "dataset": "CultureBank",
            "passage_id": "cb_" + frame_id,
            "culture": culture, "category": [frame.get("topic") or "Unknown"],
            "source": [f"CultureBank v3 frame {frame_id}"],
            "values": list(frame.get("values") or []),
            "indirect": True,
            "text": (
                f"INDIRECT comparison; {culture}, not necessarily Kazakh. "
                f"Value (provisional): {', '.join(frame.get('values') or [])}. "
                f"Norm type: {frame.get('norm_type_label') or ''}. "
                f"Norm: {frame.get('norm') or ''}. "
                f"Source description: {frame['text']}"
            ),
        })
    return units


class MultilingualIndex:
    """Fixed BGE-M3 cosine index; refuses model changes and silent TF-IDF fallback."""

    def __init__(self, index_dir: str | Path, *, device: str = "cpu"):
        self.index_dir = Path(index_dir)
        meta = json.loads((self.index_dir / "metadata.json").read_text(encoding="utf-8"))
        if meta["embedding_model"] != MODEL_NAME:
            raise ValueError("Unexpected embedding model; rebuild index explicitly")
        # Indexes built before the pin carry no revision; the cached snapshot
        # they were built from is this same revision (the only one in hf_cache).
        recorded = meta.get("embedding_model_revision")
        if recorded is not None and recorded != MODEL_REVISION:
            raise ValueError(f"Index built with BGE-M3 revision {recorded}; "
                             f"expected {MODEL_REVISION}")
        self.recorded_model_revision = recorded
        # Read JSONL by physical newline records. str.splitlines() also splits on
        # Unicode separators such as U+2028/U+2029, which may legitimately
        # occur inside source text and would corrupt an otherwise valid JSON row.
        with (self.index_dir / "units.jsonl").open(encoding="utf-8") as stream:
            self.units = [json.loads(line) for line in stream if line.strip()]
        self.emb = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        if len(self.units) != len(self.emb):
            raise ValueError("Inconsistent unit/embedding counts")
        self._model = None
        self.device = device

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                MODEL_NAME, revision=MODEL_REVISION, device=self.device)
        return self._model

    @staticmethod
    def create(out: str | Path, units: list[dict], *, device: str = "cpu",
               batch_size: int = 16, chunk_size: int = 256) -> dict:
        """Checkpointed BGE-M3 encoding for 3-hour Narval CPU jobs.

        Each independently persisted shard is validated on resume. A half-built
        index cannot be used for retrieval until metadata.json is present.
        """
        import hashlib
        import os
        if not units:
            raise ValueError("Cannot build an empty evidence index")
        from sentence_transformers import SentenceTransformer
        out = Path(out)
        if (out / "metadata.json").exists():
            raise FileExistsError(f"Index already complete: {out}. Use a new output directory.")
        out.mkdir(parents=True, exist_ok=True)
        units_path = out / "units.jsonl"
        unit_text = "".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n"
                            for u in units)
        unit_digest = hashlib.sha256(unit_text.encode("utf-8")).hexdigest()
        if units_path.exists():
            old_digest = hashlib.sha256(units_path.read_bytes()).hexdigest()
            if old_digest != unit_digest:
                raise ValueError("Resumed embedding job has different source units")
        else:
            units_path.write_text(unit_text, encoding="utf-8")

        plan_path = out / "embedding_plan.json"
        plan = {"units_sha256": unit_digest, "n_units": len(units),
                "model": MODEL_NAME, "model_revision": MODEL_REVISION,
                "chunk_size": chunk_size}
        if plan_path.exists():
            old_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            # Plans written before the revision pin lack the key; the only
            # cached snapshot is the pinned one, so they remain resumable.
            old_plan.setdefault("model_revision", MODEL_REVISION)
            if old_plan != plan:
                raise ValueError("Embedding checkpoint plan changed")
        else:
            plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

        chunk_dir = out / "embedding_shards"
        chunk_dir.mkdir(exist_ok=True)
        model = None
        docs = [u["text"] for u in units]
        shards = []
        for start in range(0, len(docs), chunk_size):
            stop = min(start + chunk_size, len(docs))
            shard = chunk_dir / f"{start:07d}_{stop:07d}.npy"
            if shard.exists():
                arr = np.load(shard, mmap_mode="r")
                if arr.shape[0] != stop - start or arr.ndim != 2:
                    raise ValueError(f"Invalid embedding shard: {shard}")
            else:
                if model is None:
                    model = SentenceTransformer(
                        MODEL_NAME, revision=MODEL_REVISION, device=device)
                arr = model.encode(
                    docs[start:stop], batch_size=batch_size, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True
                ).astype(np.float32)
                if arr.shape[0] != stop - start or not np.isfinite(arr).all():
                    raise ValueError(f"Embedding failed for records {start}:{stop}")
                tmp_path = shard.with_suffix(".tmp")
                with tmp_path.open("wb") as fh:
                    np.save(fh, arr)
                os.replace(tmp_path, shard)
                print(f"Index encoded {stop}/{len(docs)} units", flush=True)
            shards.append(shard)

        emb = np.concatenate([np.load(p) for p in shards], axis=0)
        if len(emb) != len(units) or not np.isfinite(emb).all():
            raise ValueError("Embedding shards failed final validation")
        tmp_emb = out / "embeddings.pending"
        with tmp_emb.open("wb") as fh:
            np.save(fh, emb)
        os.replace(tmp_emb, out / "embeddings.npy")
        meta = {
            "embedding_model": MODEL_NAME, "embedding_model_revision": MODEL_REVISION,
            "n_units": len(units),
            "dimension": emb.shape[1], "units_sha256": unit_digest,
            "system_counts": {s: sum(u["system"] == s for u in units) for s in SYSTEMS},
            "culturebank_units": sum(u["dataset"] == "CultureBank" for u in units),
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return meta

    def search(self, query: str, *, system: str, k: int = 5,
               min_similarity: float = 0.0, exclude_passage_ids: set[str] | None = None,
               combined: bool = False) -> list[dict]:
        if system not in SYSTEMS:
            raise ValueError(f"Unsupported retrieval system: {system}")
        if k < 1:
            raise ValueError("top_k must be positive")
        vec = np.asarray(self._load_model().encode(
            [query], normalize_embeddings=True, convert_to_numpy=True)[0], dtype=np.float32)
        scores = np.asarray(self.emb @ vec, dtype=np.float32)
        exclude = exclude_passage_ids or set()
        valid = [
            i for i, unit in enumerate(self.units)
            if unit["system"] == system and unit["passage_id"] not in exclude
            and (combined or unit["dataset"] == "KazCulture")
            and np.isfinite(scores[i]) and float(scores[i]) >= min_similarity
        ]
        valid.sort(key=lambda i: (-float(scores[i]), self.units[i]["unit_id"]))
        results = []
        seen_passages = set()
        # Passage diversity is fixed across all modes so top_k is comparable.
        for i in valid:
            unit = self.units[i]
            if unit["passage_id"] in seen_passages:
                continue
            if unit["indirect"] and not combined:
                continue
            seen_passages.add(unit["passage_id"])
            results.append({**unit, "similarity": round(float(scores[i]), 6)})
            if len(results) >= k:
                break
        return results


def pack_numbered_blocks(
    items: list, make_block, *, max_chars: int, skip_oversize: bool = False,
    start: int = 1, prefix_chars: int | None = None,
) -> tuple[list[str], list]:
    """Pack numbered blocks into a character budget.

    ``make_block(number, item)`` renders one block; numbers are continuous over
    the blocks actually kept. The joined text is ``"\n\n".join(blocks)``.

    Frozen behaviour (``skip_oversize=False``, used by every published S1 run)
    stops at the first block that does not fit, even if a later, shorter block
    would. ``skip_oversize=True`` skips such a block and keeps trying later
    ones. ``prefix_chars`` is the length of already-rendered text these blocks
    are appended to (a ``"\n\n"`` separator is then counted too).
    """
    blocks: list[str] = []
    used: list = []
    for item in items:
        block = make_block(start + len(used), item)
        body = len("\n\n".join(blocks + [block]))
        total = body if prefix_chars is None else prefix_chars + 2 + body
        if total > max_chars:
            if skip_oversize:
                continue
            break
        blocks.append(block)
        used.append(item)
    return blocks, used


def s1_block(number: int, unit: dict) -> str:
    label = ("INDIRECT COMPARISON" if unit["indirect"] else "KAZAKH TRAINING SOURCE")
    return (
        f"[{number}] {label} | dataset={unit['dataset']} | source={'; '.join(unit['source'])}"
        f" | id={unit['passage_id']}\n{unit['text']}"
    )


def render_s1_blocks(
    hits: list[dict], *, max_chars: int = 5000, skip_oversize: bool = False,
    annotate=None,
) -> tuple[list[str], list[dict]]:
    """Numbered S1 blocks that fit, and their hits (see ``render_evidence_with_hits``)."""
    def make(number, unit):
        block = s1_block(number, unit)
        extra = annotate(unit) if annotate else None
        return block + "\n" + extra if extra else block

    return pack_numbered_blocks(
        hits, make, max_chars=max_chars, skip_oversize=skip_oversize)


def render_evidence_with_hits(
    hits: list[dict], *, max_chars: int = 5000, skip_oversize: bool = False,
    annotate=None,
) -> tuple[str, list[dict]]:
    """Render only evidence that actually fits and return the corresponding hits.

    Retrieval metrics and prediction metadata must describe what the answering
    model saw, not additional candidates that were silently dropped by the
    prompt character budget.

    The default reproduces the frozen S1 runs exactly (stop at the first block
    that overflows the budget). ``skip_oversize`` is opt-in; ``annotate(unit)``
    may return extra text appended to that unit's block (S1+L labels).
    """
    blocks, used = render_s1_blocks(hits, max_chars=max_chars,
                                    skip_oversize=skip_oversize, annotate=annotate)
    return "\n\n".join(blocks), used


def oversize_break_affected(hits: list[dict], *, max_chars: int = 5000) -> bool:
    """True when the frozen first-overflow ``break`` hid a later hit that fits."""
    frozen = render_evidence_with_hits(hits, max_chars=max_chars)[1]
    skipping = render_evidence_with_hits(hits, max_chars=max_chars, skip_oversize=True)[1]
    return len(skipping) > len(frozen)


def render_evidence(hits: list[dict], *, max_chars: int = 5000) -> str:
    """Backward-compatible text-only evidence renderer."""
    return render_evidence_with_hits(hits, max_chars=max_chars)[0]
