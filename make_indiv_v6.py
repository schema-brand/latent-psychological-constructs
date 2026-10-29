"""Individuality materials (MVCA Study 4 counterpart; PI-approved 20 Oct 2026): 7 person characteristics (materials_v6/indiv/trait_spec_v2.json)
written by the six v6 writers (8 pairs with the L1 low pole and 8 with L4 each), checked with the v6 pair checks, rated blind by the v6
raters on the 7 characteristic scales and 3 confound scales (wealth, pleasantness, sociability). OpenRouter calls run locally
(or_runner/local_run.py; budget from OR_BUDGET). Usage:
  python make_indiv_v6.py write     # indiv/jobs/I1_write.jsonl (OpenRouter writers) + indiv/claude_prompts/*.txt (local Claude writer)
  python make_indiv_v6.py collect   # checks every output -> indiv/pairs_indiv_raw.jsonl, indiv/qc_indiv.json
  python make_indiv_v6.py rate      # indiv/jobs/I2_rate.jsonl for the pairs that pass the checks
  python make_indiv_v6.py check     # ratings -> indiv/rating_check_indiv.json + indiv/pairs_indiv_final.jsonl (own-scale high > low)
  python make_indiv_v6.py bundle    # materials_v6/upload_v6/indiv_v6.json for the GPU step (pairs, MVCA characteristic statements, tests)"""
import json, sys, glob, collections, importlib.util, csv
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"; IND = M6 / "indiv"
JOBS, CP, CO = IND / "jobs", IND / "claude_prompts", IND / "claude_out"
for d in (JOBS, CP, CO): d.mkdir(parents=True, exist_ok=True)
def _mod(name, f):
    s = importlib.util.spec_from_file_location(name, HERE / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
j6 = _mod("j6", "make_or_jobs_v6.py"); r6 = _mod("r6", "make_or_rate_v6.py")
L = {"OR_ROOT": str(HERE / "or_runner" / "local_root"), "OR_KEY": "x"}
exec(compile((HERE / "or_runner" / "or_runner_lib.py").read_text(encoding="utf-8"), "or_runner_lib.py", "exec"), L)
SPEC = json.loads((IND / "trait_spec_v2.json").read_text(encoding="utf-8")); T = SPEC["traits"]
SPEC3 = json.loads((IND / "trait_spec_v3.json").read_text(encoding="utf-8")); ROUND2 = ["materialism", "collective_self_esteem"]   # rewritten after the rating check
def spec_of(k, rnd): return (SPEC3 if rnd == 1 else SPEC)["traits"][k]
RULE7 = "No gender, age, family members, job or city for 'you', so the text fits any adult."
SOFT = {"ending_differs", "number_rule"}
OUT = HERE / "or_runner" / "local_root" / "out"

import re as _re
_PAIR = _re.compile(r'"domain"\s*:\s*"(.*?)"\s*,\s*"high"\s*:\s*"(.*?)"\s*,\s*"low"\s*:\s*"(.*?)"\s*}', _re.S)
def repair_pairs(txt):
    """Writers sometimes leave unescaped quotes inside the JSON strings; recover the pairs field by field (as collect_v6.py)."""
    P = [{"domain": d, "high": h.replace('\\"', '"'), "low": l.replace('\\"', '"')} for d, h, l in _PAIR.findall(txt or "")]
    return {"pairs": P} if P else None

def construct(k, rnd=0):
    c = dict(spec_of(k, rnd)); c["construct"] = k; c["role"] = "trait"; c["number_exempt"] = False; return c

def prompt(call, k):
    c = construct(k, call.get("round", 0)); p = j6.prompt_v6(call, c)
    p = p.replace(f"Everything else stays the same in both versions, including {j6.COMPARISON_POINTS}.",
                  f"Everything else stays the same in both versions, including {j6.COMPARISON_POINTS}. Also keep the same in both versions: {c['hold_constant']}.")
    if k in SPEC["rule_overrides"]:
        assert RULE7 in p, "rule text changed"; p = p.replace(RULE7, SPEC["rule_overrides"][k])
    return p

def calls():
    out = []
    for rnd, ks in ((0, list(T)), (1, ROUND2)):
        for w in j6.WRITERS:
            for k in ks:
                for i, (low, j) in enumerate((("L1", 0), ("L4", 1))):
                    out.append(dict(writer=w, construct=k, low=low, n=SPEC["pairs_per_writer"][low], j=j + 2 * rnd, idx=i, round=rnd, call=f"{w}__{k}__{low}_t{i + 2 * rnd}"))
    return out

def meta_of(call):
    return {"writer": call["writer"], "construct": call["construct"], "role": "trait", "low": call["low"], "n": call["n"], "banned": T[call["construct"]]["banned"],
            "domains": j6.DOMAINS[call["writer"]], "number_exempt": False, "prompt_version": "indiv_v3" if call.get("round") else "indiv_v2", "round": call.get("round", 0), "call": call["call"]}

def results():
    R = {}
    for f in glob.glob(str(OUT / "I1_*" / "part-*.jsonl")) + glob.glob(str(OUT / "I3_write" / "part-*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in R or r["t"] >= R[r["id"]]["t"]: R[r["id"]] = r
    C = {c["call"]: c for c in calls()}
    for f in CO.glob("claude__*.json"):
        cid = f.stem
        try: obj = L["parse_json"](f.read_text(encoding="utf-8")); st = "ok"
        except Exception as e: obj, st = None, f"parse_error {e}"
        if cid in C: R[cid] = {"id": cid, "status": st, "parsed": obj, "meta": meta_of(C[cid]), "t": "local"}
    return R

mode = sys.argv[1]
if mode in ("write", "write2"):
    reqs = []; rnd = 1 if mode == "write2" else 0
    for c in [c for c in calls() if c["round"] == rnd]:
        if c["writer"] == "claude": (CP / f"{c['call']}.txt").write_text(prompt(c, c["construct"]), encoding="utf-8"); continue
        r = j6.request(c, construct(c["construct"])); r["prompt"] = prompt(c, c["construct"]); r["meta"] = meta_of(c)
        if c["writer"] in j6.REASONING_OFF: r["params"]["reasoning"] = {"enabled": False}
        reqs.append(r)
    (JOBS / ("I3_write.jsonl" if rnd else "I1_write.jsonl")).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reqs) + "\n", encoding="utf-8")
    print(len(reqs), "OpenRouter calls;", len(list(CP.glob("*.txt"))), "Claude prompts")
elif mode == "collect":
    rows, Q = [], collections.defaultdict(collections.Counter)
    for cid, r in results().items():
        meta = r.get("meta") or {}; obj = r.get("parsed")
        if obj is None and r.get("content"): obj = repair_pairs(r["content"])
        qc = L["qc_pairs"](obj, meta) if obj else {"items": [], "n": 0, "ok": 0}
        key = f"{meta['writer']}|{meta['construct']}"; Q[key]["calls"] += 1; Q[key]["pairs"] += qc.get("n", 0)
        for j, (p, it) in enumerate(zip((obj or {}).get("pairs", []), qc["items"])):
            hard = [f for f in it["flags"] if f in L["HARD_FLAGS"] and f not in SOFT]
            Q[key]["ok"] += not hard
            rows.append({"pid": f"{cid}__{j}", "writer": meta["writer"], "construct": meta["construct"], "low_type": meta["low"], "round": meta.get("round", 0), "domain": p.get("domain"),
                         "high": p.get("high"), "low": p.get("low"), "flags": it["flags"], "diff": it.get("diff"), "hard_fail": bool(hard), "call": cid})
    with open(IND / "pairs_indiv_raw.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({k: dict(v) for k, v in sorted(Q.items())}, open(IND / "qc_indiv.json", "w"), indent=1)
    by = collections.Counter((r["writer"], r["hard_fail"]) for r in rows); fl = collections.Counter(f for r in rows for f in r["flags"])
    print(len(rows), "pairs; pass by writer:", {w: f"{by[(w, False)]}/{by[(w, False)] + by[(w, True)]}" for w in j6.WRITERS}, "| flags", dict(fl.most_common(8)))
    print("pass by trait:", {k: sum(1 for r in rows if r["construct"] == k and not r["hard_fail"]) for k in T})
elif mode == "rate":
    TAG = "I2_rate" if not list(JOBS.glob("I2_rate.jsonl")) else "I2_rate_r2"
    scales = list(T) + list(SPEC["confound_scales"])
    defs = {**{k: v["definition"] for k, v in T.items()}, **{k: v["definition"] for k, v in SPEC["confound_scales"].items()}}
    sc = "\n".join(f"- {k}: {defs[k]}" for k in scales); keys = ", ".join(f'"{k}": <1-7>' for k in scales)
    TX = []; done_ids = set()
    for f in JOBS.glob("I2_rate*.jsonl"):
        for l in open(f, encoding="utf-8"): done_ids |= set(json.loads(l)["meta"]["ids"])
    for l in open(IND / "pairs_indiv_raw.jsonl", encoding="utf-8"):
        p = json.loads(l)
        if p["hard_fail"] or f"{p['pid']}|high" in done_ids: continue
        for side in ("high", "low"): TX.append(dict(id=f"{p['pid']}|{side}", unit=p["pid"], writer=p["writer"], text=p[side]))
    reqs = []
    for rater in r6.RATERS:
        mine = [t for t in TX if (t["writer"] == "deepseek") == (rater == "kimi")] if rater != "glm" else list(TX)
        for i, b in enumerate(r6.batches(list(mine))):
            reqs.append({"id": f"{TAG}__{rater}__{i:04d}", "model": r6.RATERS[rater], "prompt": r6.PROMPT.format(scales=sc, texts="\n".join(f"[{t['id']}] {t['text']}" for t in b), keys=keys),
                         "params": {"max_tokens": 16000, "temperature": 0.2, "reasoning": {"effort": "low"}}, "check": "ratings",
                         "meta": {"rater": rater, "ids": [t["id"] for t in b], "scales": scales}})
    (JOBS / f"{TAG}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reqs) + "\n", encoding="utf-8")
    print(len(TX), "texts;", collections.Counter(r["meta"]["rater"] for r in reqs), "calls")
elif mode == "check":
    R = {}
    for f in glob.glob(str(OUT / "I2_rate*" / "part-*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in R or r["t"] >= R[r["id"]]["t"]: R[r["id"]] = r
    V = collections.defaultdict(list)   # (text id, scale) -> ratings
    for r in R.values():
        o = r.get("parsed") or {}; want = r["meta"]["ids"]; by_unit = {w.split("|")[0]: w for w in want}
        for e in o.get("ratings", []) if isinstance(o, dict) else []:
            tid = str(e.get("id")); tid = tid if tid in want else by_unit.get(tid.split("|")[0], tid)
            for s in r["meta"]["scales"]:
                v = e.get(s)
                if isinstance(v, int) and 1 <= v <= 7: V[(tid, s)].append(v)
    P = [json.loads(l) for l in open(IND / "pairs_indiv_raw.jsonl", encoding="utf-8")]; P = [p for p in P if not p["hard_fail"]]
    conf = list(SPEC["confound_scales"]); keep, per = [], collections.defaultdict(list)
    for p in P:
        g = lambda side, s: np.mean(V[(f"{p['pid']}|{side}", s)]) if V[(f"{p['pid']}|{side}", s)] else np.nan
        own = g("high", p["construct"]) - g("low", p["construct"]); cd = {s: g("high", s) - g("low", s) for s in conf}
        oth = {k: g("high", k) - g("low", k) for k in T if k != p["construct"]}
        p.update(own_diff=None if np.isnan(own) else round(float(own), 2), confound_diff={s: (None if np.isnan(v) else round(float(v), 2)) for s, v in cd.items()},
                 max_other_diff=round(float(np.nanmax(np.abs(list(oth.values())))), 2) if oth else None)
        per[f"{p['construct']}|r{p.get('round', 0)}"].append(p)
        if p["own_diff"] is not None and p["own_diff"] > 0: keep.append(p)
    S = {}
    for k, ps in per.items():
        od = [p["own_diff"] for p in ps if p["own_diff"] is not None]
        S[k] = dict(n_pairs=len(ps), n_kept=sum(1 for p in keep if f"{p['construct']}|r{p.get('round', 0)}" == k), own_diff_mean=round(float(np.mean(od)), 2) if od else None,
                    own_moves_most=round(float(np.mean([p["own_diff"] > (p["max_other_diff"] or 0) for p in ps if p["own_diff"] is not None])), 2) if od else None,
                    confound_diff_mean={s: round(float(np.nanmean([p["confound_diff"][s] for p in ps if p["confound_diff"][s] is not None])), 2) for s in conf},
                    by_writer={w: sum(1 for p in keep if f"{p['construct']}|r{p.get('round', 0)}" == k and p["writer"] == w) for w in j6.WRITERS})
    json.dump(S, open(IND / "rating_check_indiv.json", "w"), indent=1)
    with open(IND / "pairs_indiv_final.jsonl", "w", encoding="utf-8") as f:
        for p in keep: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps(S, indent=1)); print(len(keep), "pairs kept of", len(P))
elif mode == "bundle":
    REL = HERE / "external" / "agent-psychonomicus"   # REPOSITORY: was HERE.parent / "release_2026-10-15" / ...
    B = json.loads((REL / "materials" / "study4" / "CP4_BELIEF_FILE_V5b.json").read_text(encoding="utf-8"))
    tests = list(csv.DictReader(open(REL / "data" / "study4_tests.csv", encoding="utf-8")))
    trait_of = {"mogilner_materialism": "materialism", "whiteargo_cse": "collective_self_esteem", "kivetz_guilt": "luxury_guilt", "thaler_costs_justify": "costs_justify_prices",
                "freedman_consistency": "preference_for_consistency", "reed_importance": "family_identity_importance", "thaler_selfmonitoring_null": "self_monitoring",
                "thaler_selfmonitoring_null2": "self_monitoring"}
    use = lambda p: p.get("round", 0) == (1 if p["construct"] in ROUND2 else 0)          # rewritten characteristics: only the rewrite round
    pairs = [{k: p.get(k) for k in ("pid", "writer", "construct", "low_type", "round", "high", "low", "own_diff", "confound_diff")}
             for p in map(json.loads, open(IND / "pairs_indiv_final.jsonl", encoding="utf-8")) if use(p)]
    out = {"version": "indiv_v2", "traits": {k: {f: v[f] for f in ("definition", "mvca_test")} for k, v in T.items()}, "pairs": pairs,
           "statements": [{"experiment": e, "test": b["test"], "level": b["level"], "text": b["text"], "trait": trait_of[b["test"]]} for e, L_ in B.items() for b in L_],
           "tests": [dict(t, trait=trait_of[t["test"]]) for t in tests]}
    (M6 / "upload_v6" / "indiv_v6.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(pairs), "pairs;", len(out["statements"]), "statements;", len(out["tests"]), "tests")
