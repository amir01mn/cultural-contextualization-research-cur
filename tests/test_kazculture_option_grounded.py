"""Option-grounded retrieval (v2 dev): synthetic passages, no model loads."""
from types import SimpleNamespace

import numpy as np

from src.retrieval.kazculture_option_grounded import (
    OptionGroundedRetriever, grounding, norm_abstraction, render_option_grounded, stems)

UNITS = [
    {"passage_id": "p0", "unit_id": "S1:p0", "text": "Шыбыртқы — көлікті ат үстінен айдайтын қамшы."},
    {"passage_id": "p1", "unit_id": "S1:p1", "text": "Арқан жүннен есіледі, малды байлайды."},
    {"passage_id": "p2", "unit_id": "S1:p2", "text": "Домбыра екі ішекті аспап."},
    {"passage_id": "p3", "unit_id": "S1:p3", "text": "Таяқпен қой қайырады."},
]
OPTS = {"A": "Арқанмен айдайды.", "B": "Таяқпен айдайды.",
        "C": "Шыбыртқымен айдайды.", "D": "Қолмен айдайды."}


def retriever():
    return OptionGroundedRetriever(UNITS, lambda q: np.zeros(len(UNITS)))


def test_distinctive_stems():
    from src.retrieval.kazculture_option_grounded import distinctive_stems
    d = distinctive_stems("Көлікті немен айдайды?", OPTS)
    assert d["C"] == ["шыбыр"] and d["B"] == ["таяқп"]


def test_stems_and_grounding():
    assert stems("Шыбыртқымен айдайды") == ["шыбыр", "айдай"]
    assert grounding("Шыбыртқымен айдайды", set(stems(UNITS[0]["text"]))) == 1.0
    assert grounding("Қолмен", set(stems(UNITS[0]["text"]))) == 0.0


def test_pooled_excludes_and_ranks():
    r = retriever()
    hits = r.pooled("Көлікті немен айдайды?", OPTS, k=4, exclude={"p0"})
    assert "p0" not in [h["passage_id"] for h in hits[:3]]
    assert hits[-1]["similarity"] == float("-inf")


def test_per_option_grounds_each_option():
    per = retriever().per_option("Көлікті немен айдайды?", OPTS, k=2)
    assert per["C"][0]["passage_id"] == "p0" and per["C"][0]["grounding"] == 1.0
    assert per["B"][0]["passage_id"] == "p3"
    inter = OptionGroundedRetriever.interleave(per, 3)
    assert len({h["passage_id"] for h in inter}) == 3


def test_render_budget_dedupe_and_abstraction():
    per = retriever().per_option("Көлікті немен айдайды?", OPTS, k=2)
    fr = {u["passage_id"]: SimpleNamespace(values=["tradition"], norm_type_label="tools",
                                            norm_type_id="nt1", norm=f"norm {u['passage_id']}")
          for u in UNITS}
    blocks, meta = render_option_grounded(OPTS, per, fr, budget_chars=5000, per_option=2)
    text = "\n".join(blocks)
    assert len(text) <= 5000 and text.count("Шыбыртқы — көлікті") == 1
    assert len(blocks) == len(meta)
    ab = norm_abstraction(fr, {"nt1": ["p0", "p1", "p2", "p3"]}, exclude={"p1"})
    per = retriever().per_option("Көлікті немен айдайды?", OPTS, k=2, exclude={"p1"})
    blocks, meta = render_option_grounded(OPTS, per, fr, budget_chars=5000, per_option=2,
                                          tau_ground=0.99, abstraction=ab)
    assert any(m["layer"] == "abstract" for m in meta)
    assert "norm p1" not in "\n".join(blocks)
    small, _ = render_option_grounded(OPTS, per, fr, budget_chars=200, per_option=2)
    assert len("\n".join(small)) <= 200
