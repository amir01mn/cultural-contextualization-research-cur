#!/usr/bin/env python3
"""Build a strictly exploratory BGE-M3 index on 150 selected TRAIN sources.

This does NOT bypass the manually validated full KG build gate. Candidate
graphs remain explicitly unreviewed, with train QA separated from evidence.
Produces no evaluation/test metadata; not for final benchmark reporting.
"""
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


def verify_sources(passages, frames, method):
    if len(passages)!=150 or len(frames)!=150:
        raise ValueError("Pilot requires exactly 150 sources and 150 frames")
    pids={p["passage_id"] for p in passages}
    fids={f["passage_id"] for f in frames}
    if len(pids)!=150 or len(fids)!=150 or pids!=fids:
        raise ValueError("Source IDs must be unique and identical")
    expected={
        "v21":"kazculture-kg-factual-normative-v2.1",
        "v22":"kazculture-kg-nested-norms-v2.2",
    }[method]
    by_pid={p["passage_id"]:p for p in passages}
    validate_no_test_content(passages, set())
    for f in frames:
        src=by_pid[f["passage_id"]]
        if f.get("schema_version")!=expected or f.get("text")!=src["text"]:
            raise ValueError("Extraction schema/source text mismatch")
        if f.get("split")!="train" or f.get("dataset")!="issai/KazCulture":
            raise ValueError("Non-train frame prohibited")
        if any(x in f for x in ("question","answer","answer_label","item_description")):
            raise ValueError("Question/answer content in KG evidence")


def fingerprint(path):
    sha=hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda:fd.read(1048576),b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--passages",required=True,type=Path)
    p.add_argument("--frames",required=True,type=Path)
    p.add_argument("--method",choices=["v21","v22"],required=True)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--device",default="cpu")
    p.add_argument("--batch_size",type=int,default=16)
    args=p.parse_args()
    if args.batch_size<=0:
        raise ValueError("batch_size must be positive")
    if args.out.name != f"{args.method}_diagnostic":
        raise ValueError("Use a uniquely named method-specific diagnostic index directory")
    passages=read_rows(args.passages)
    frames=read_rows(args.frames)
    verify_sources(passages,frames,args.method)
    units=make_units(passages,frames)
    if not units or any(u.get("dataset")!="KazCulture" or u["indirect"] for u in units):
        raise ValueError("Only direct source-derived KazCulture evidence permitted")
    diag={
        "status":"exploratory_unreviewed_do_not_report_as_human_validated",
        "method":args.method,
        "source_input_sha256":fingerprint(args.passages),
        "frame_input_sha256":fingerprint(args.frames),
        "source_count":len(passages),
        "unit_counts":{s:sum(u["system"]==s for u in units)
                       for s in ("S1","S2","S3","S4F","S4R")},
        "note":"Train-QA withheld from embeddings. Self-source QA retrieval only.",
    }
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=args.out/"diagnostic_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))!=diag:
        raise ValueError("Existing manifest differs; refuse mixing experiments")
    if not (args.out/"metadata.json").exists():
        MultilingualIndex.create(args.out,units,device=args.device,batch_size=args.batch_size)
    else:
        expected=hashlib.sha256("".join(
            json.dumps(u,ensure_ascii=False,sort_keys=True)+"\n" for u in units
        ).encode("utf-8")).hexdigest()
        existing=json.loads((args.out/"metadata.json").read_text(encoding="utf-8"))
        if existing.get("units_sha256")!=expected:
            raise ValueError("Completed index units do not match exact inputs")
    if not manifest.exists():
        manifest.write_text(json.dumps(diag,ensure_ascii=False,indent=2)+"\n",
                            encoding="utf-8")
    print(json.dumps(diag,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
