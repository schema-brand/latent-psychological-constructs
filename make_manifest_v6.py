"""Freeze materials v6 for the formal runs and the preregistration: sha256 and counts of every materials file -> materials_v6/MANIFEST_v6.json.
Usage: python make_manifest_v6.py"""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"
FILES = ["spec_v6.json", "pairs_v6_final.jsonl", "pairs_v6.jsonl", "exclusions_v6.json", "variants_v6.jsonl", "agents_v6.json", "judgment_items_v6.json",
         "exp_questions_v00.json", "ratings_v6.csv", "coding_v6/state_fact_final.json", "coding_v6/definition_similarity.csv", "QC_REPORT_v6.md",
         "PRERUN_ADJUSTMENTS_v6.md"] + sorted(str(p.relative_to(M6)).replace("\\", "/") for p in (M6 / "experiments").glob("*.json"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return sum(1 for _ in open(p, encoding="utf-8")) if p.suffix in (".jsonl", ".csv") else None
M = {"frozen": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "files": {}}
for f in FILES:
    p = M6 / f
    if p.exists(): M["files"][f] = {"sha256": sha(p), "bytes": p.stat().st_size, **({"lines": lines(p)} if lines(p) is not None else {})}
    else: M["files"][f] = {"missing": True}
C = json.loads((M6 / "spec_v6.json").read_text(encoding="utf-8"))["constructs"]
M["counts"] = {"constructs": len(C), "by_role": {r: sum(c["role"] == r for c in C.values()) for r in ("mediator", "rival", "comparison")},
               "pairs_final": M["files"]["pairs_v6_final.jsonl"].get("lines"), "rewordings": M["files"]["variants_v6.jsonl"].get("lines"),
               "experiments": len(list((M6 / "experiments").glob("*.json")))}
(M6 / "MANIFEST_v6.json").write_text(json.dumps(M, indent=1), encoding="utf-8"); print(json.dumps(M["counts"]), "|", sum("missing" in v for v in M["files"].values()), "missing")
