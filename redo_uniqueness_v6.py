"""Materials v6 pre-run fix: replace the uniqueness specification (materials_v6/fix/uniqueness_v2.json) and write it again with all six
writers (16 pairs each: 8 with the L1 low pole, 8 with L4), then rate the new texts with the same raters. Runs locally (or_runner/local_run.py).
Usage: python redo_uniqueness_v6.py spec      # install the revised spec into spec_v6.json (old one kept as uniqueness_v1)
       python redo_uniqueness_v6.py write     # writing jobs (OpenRouter) + Claude prompt files
       python redo_uniqueness_v6.py rate      # rating jobs for the new uniqueness texts (after collecting the pairs)"""
import json, sys, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"; J = M6 / "jobs_local"; J.mkdir(exist_ok=True)
_j = importlib.util.spec_from_file_location("j6", HERE / "make_or_jobs_v6.py"); j6 = importlib.util.module_from_spec(_j); _j.loader.exec_module(j6)
mode = sys.argv[1]
S = json.loads((M6 / "spec_v6.json").read_text(encoding="utf-8"))
if mode == "spec":
    new = json.loads((M6 / "fix" / "uniqueness_v2.json").read_text(encoding="utf-8"))
    old = S["constructs"]["uniqueness"]; new["group"] = old["group"]
    new["banned"] = [w for w in new.get("banned", []) if w.lower() not in j6.GENERIC_BANNED]
    (M6 / "fix" / "uniqueness_v1.json").write_text(json.dumps(dict(old, superseded_by="uniqueness v2 (19 Oct 21:15 UTC)"), indent=1, ensure_ascii=False), encoding="utf-8")
    S["constructs"]["uniqueness"] = dict(new, prompt_version="v6.1")
    (M6 / "spec_v6.json").write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding="utf-8"); print("installed uniqueness v2")
elif mode == "write":
    c = S["constructs"]["uniqueness"]; reqs = []; claude = []
    for w in j6.WRITERS:
        for i, (low, j) in enumerate((("L1", 0), ("L4", 1))):
            call = dict(writer=w, construct="uniqueness", low=low, n=8, j=j, idx=i, call=f"{w}__uniqueness__{low}_u{i}")
            if w == "claude": (M6 / "claude_prompts" / f"{call['call']}.txt").write_text(j6.prompt_v6(call, c), encoding="utf-8"); claude.append(call["call"]); continue
            r = j6.request(call, c); r["meta"]["prompt_version"] = "v6.1"
            if w in j6.REASONING_OFF: r["params"]["reasoning"] = {"enabled": False}
            reqs.append(r)
    (J / "U1_uniqueness.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reqs) + "\n", encoding="utf-8")
    (M6 / "claude_calls_uniqueness.json").write_text(json.dumps(claude), encoding="utf-8"); print(len(reqs), "OpenRouter calls;", claude)
