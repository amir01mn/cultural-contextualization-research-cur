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
        for t in frame["triples"]:
            fact = f"{t['subject']} — {t['predicate']} — {t['object']}. Quote: {t['evidence']}"
            units.append({
                **base, "unit_id": "S2:" + t["triple_id"],
                "system": "S2", "text": fact,
                "triple_id": t["triple_id"], "values": frame["values"],
            })
            layers = (
                f"L3 (provisional value): {', '.join(frame['values']) or '(unclassified)'}\n"
                f"L2 (provisional norm type): {frame['norm_type'] or '(unclassified)'}\n"
                f"L2 (provisional norm): {frame['norm'] or '(unclassified)'}\n"
                f"L1 (source-quoted practice): {fact}"
            )
            units.append({
                **base, "unit_id": "S3:" + t["triple_id"],
                "system": "S3", "text": layers,
                "triple_id": t["triple_id"], "values": frame["values"],
            })
        if frame["triples"]:
            units.append({
                **base, "unit_id": "S4F:" + p["passage_id"],
                "system": "S4F",
                "text": (f"Kazakh cultural frame, provisional norm: {frame['norm']}. "
                         f"Source: {p['text']}"),
                "values": frame["values"],
            })
            for t in frame["triples"]:
                units.append({
                    **base, "unit_id": "S4R:" + t["triple_id"],
                    "system": "S4R",
                    "text": (
                        f"Value (provisional): {', '.join(frame['values'])}. "
                        f"Norm type: {frame['norm_type']}. "
                        f"Culture-specific norm: {frame['norm']}. "
                        f"Source-quoted practice: {t['evidence']}. "
                        f"Relation: {t['subject']} — {t['predicate']} — {t['object']}."
                    ),
                    "values": frame["values"], "triple_id": t["triple_id"],
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
        self.units = [
            json.loads(line) for line in (self.index_dir / "units.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.emb = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        if len(self.units) != len(self.emb):
            raise ValueError("Inconsistent unit/embedding counts")
        self._model = None
        self.device = device

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME, device=self.device)
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
                "model": MODEL_NAME, "chunk_size": chunk_size}
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
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
                    model = SentenceTransformer(MODEL_NAME, device=device)
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
            "embedding_model": MODEL_NAME, "n_units": len(units),
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


def render_evidence(hits: list[dict], *, max_chars: int = 5000) -> str:
    blocks = []
    for i, unit in enumerate(hits, 1):
        label = ("INDIRECT COMPARISON" if unit["indirect"] else "KAZAKH TRAINING SOURCE")
        block = (
            f"[{i}] {label} | dataset={unit['dataset']} | source={'; '.join(unit['source'])}"
            f" | id={unit['passage_id']}\n{unit['text']}"
        )
        if len("\n\n".join(blocks + [block])) > max_chars:
            break
        blocks.append(block)
    return "\n\n".join(blocks)
