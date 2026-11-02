#!/usr/bin/env python3
"""Build a passage-only KazCulture BGE-M3 index for full and strict evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

from src.loaders.kazculture_corpus import read_rows, validate_no_test_content
from src.retrieval.kazculture_index import MultilingualIndex, make_units


def fingerprint(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--passages",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--batch_size",type=int,default=16)
    p.add_argument("--strict_passage_disjoint",action="store_true")
    p.add_argument("--overlap_file",type=Path)
    a=p.parse_args()

    passages=read_rows(a.passages)
    if not passages:
        raise ValueError("No passages")
    if any(x.get("split")!="train" or x.get("dataset")!="issai/KazCulture" for x in passages):
        raise ValueError("S1 index requires KazCulture train passages only")

    excluded=set()
    if a.strict_passage_disjoint:
        if a.overlap_file is None:
            raise ValueError("Strict mode requires --overlap_file")
        overlap=json.loads(a.overlap_file.read_text(encoding="utf-8"))
        excluded=set(overlap["exclude_train_passage_ids"])
        if not excluded <= {x["passage_id"] for x in passages}:
            raise ValueError("Overlap manifest does not match this corpus")
        passages=[x for x in passages if x["passage_id"] not in excluded]
        validate_no_test_content(passages,excluded)
    elif a.overlap_file is not None:
        raise ValueError("--overlap_file is only valid with --strict_passage_disjoint")

    units=make_units(passages,frames=None,culturebank_frames=None)
    if not units or any(u["system"]!="S1" for u in units):
        raise ValueError("Passage-only index unexpectedly contains non-S1 units")

    a.out.mkdir(parents=True,exist_ok=True)
    metadata=a.out/"metadata.json"
    if metadata.exists():
        expected=hashlib.sha256("".join(
            json.dumps(u,ensure_ascii=False,sort_keys=True)+"\n" for u in units
        ).encode("utf-8")).hexdigest()
        existing=json.loads(metadata.read_text(encoding="utf-8"))
        if existing.get("units_sha256")!=expected:
            raise ValueError("Completed S1 index belongs to different inputs")
        stats=existing
    else:
        stats=MultilingualIndex.create(
            a.out,units,device=a.device,batch_size=a.batch_size
        )

    manifest={
        **stats,
        "status":"passage_only_s1_index",
        "passages":len(passages),
        "strict_passage_disjoint":a.strict_passage_disjoint,
        "excluded_test_overlap_passages":len(excluded),
        "culturebank_frame_count":0,
        "source_input_sha256":fingerprint(a.passages),
        "overlap_manifest":str(a.overlap_file.resolve()) if a.overlap_file else None,
        "systems":["S1"],
        "note":"Raw train passages only; no generated KG evidence.",
    }
    m=a.out/"build_manifest.json"
    if m.exists():
        if json.loads(m.read_text(encoding="utf-8"))!=manifest:
            raise ValueError("Existing S1 manifest differs")
    else:
        m.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
