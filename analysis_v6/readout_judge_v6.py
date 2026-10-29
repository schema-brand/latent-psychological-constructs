"""Study 1, content of the representation (PAPER_OUTLINE_v1 3.5 item 4): blind identification of the words read out from each construct's
representation. Items = model x construct x readout layer, from the logit lens (V1 end reading, 50-100% depth) and, for Qwen3.6 and Gemma 4
31B, the J-lens (V2). Each item shows only the top-30 word pieces (opaque id) and 10 candidate definitions in random order: the construct's
own and 9 random other v6 constructs. Two raters from other families (DeepSeek V4.1 Flash, GLM 5.3) pick one definition and a top 3.
Chance: 10% (pick), 30% (top 3). Runs through or_runner/local_run.py (check "readout": no QC).
Usage: python analysis_v6/readout_judge_v6.py build JOB...   -> analysis_v6/jobs/R1_readout.jsonl
       python analysis_v6/readout_judge_v6.py score          -> analysis_v6/readout_judge_v6.json (+ csv)
Extension (21 Oct, more models and OLMo 3 7B checkpoints): R_TAG=R2 R_MID=1 ... build JOB... -> jobs/R2_readout.jsonl, logit lens only, without the
first and last readout layer of each model (R_MID=1); score with R_TAG=R2 -> readout_judge_v6_R2.json / .csv. Same prompt, raters and settings."""
import json, sys, random, collections, re
from pathlib import Path
import pandas as pd, os
TAG = os.environ.get("R_TAG", "R1"); MIDONLY = os.environ.get("R_MID") == "1"; SFX = "" if TAG == "R1" else "_" + TAG
HERE = Path(__file__).resolve().parent; BASE = HERE.parent; RV = BASE / "results_v6"; JD = HERE / "jobs"; JD.mkdir(exist_ok=True)
SPEC = json.loads((BASE / "materials_v6" / "upload_v6" / "spec_v6.json").read_text(encoding="utf-8"))["constructs"]; CONS = sorted(SPEC)
RATERS = {"dsnr": "deepseek/deepseek-v4.1-flash-1017", "glm": "z-ai/glm-5.3"}   # DeepSeek without reasoning (pilot: low effort ran to 35k reasoning tokens)
PARAMS = {"dsnr": {"max_tokens": 3000, "temperature": 0.2, "reasoning": {"enabled": False}}, "glm": {"max_tokens": 8000, "temperature": 0.2, "reasoning": {"effort": "low"}}}
PER_CALL = 10
PROMPT = """Below are lists of word pieces read out from the internal state of a language model while it read short everyday situations. Each list
comes from the difference between situations that are high and low on ONE psychological construct. The lists are noisy: they mix
languages, fragments and unrelated tokens. Each list has its own 10 candidate constructs, labelled A to J. For each list, decide which ONE
of its candidates the meaningful words in the list most plausibly express, and give your top 3.

{items}

OUTPUT: only a JSON object, no other text: {{"answers": [{{"id": "<id>", "pick": "<letter>", "top3": ["<letter>", "<letter>", "<letter>"]}}]}}, one entry per list."""

def items(jobs):
    out = []
    for job in jobs:
        lg = json.loads((RV / job / "V1_read" / "readout_logitlens.json").read_text(encoding="utf-8"))
        for c, by in lg.items():
            Ls = sorted(by, key=int); Ls = Ls[1:-1] if MIDONLY and len(Ls) > 2 else Ls
            for l in Ls: out.append(dict(model=job, construct=c, source="logitlens", layer=int(l), tokens=[t for t, _ in by[l][:30]]))
        jf = RV / job / "V2_jlens" / "summary.json"
        if jf.exists() and not MIDONLY:
            top = json.loads(jf.read_text(encoding="utf-8")).get("top") or {}
            for c, by in top.items():
                for l, toks in by.items(): out.append(dict(model=job, construct=c, source="jlens", layer=int(l), tokens=[t for t, _ in toks[:30]]))
    rng = random.Random(2026 if TAG == "R1" else 2027)
    for i, x in enumerate(out):
        x["item"] = f"{'r' if TAG == 'R1' else TAG.lower()}{i:05d}"; cand = [x["construct"]] + rng.sample([c for c in CONS if c != x["construct"]], 9); rng.shuffle(cand)
        x["cands"] = cand; x["key"] = "ABCDEFGHIJ"[cand.index(x["construct"])]
    return out

def block(x):
    defs = "\n".join(f"  {'ABCDEFGHIJ'[k]}. {SPEC[c]['definition']}" for k, c in enumerate(x["cands"]))
    return f"LIST [{x['item']}]: {', '.join(repr(t) for t in x['tokens'])}\nCandidates:\n{defs}\n"

if sys.argv[1] == "build":
    X = items(sys.argv[2:]); rng = random.Random(7); order = list(range(len(X))); rng.shuffle(order); reqs = []
    for rater, model in RATERS.items():
        for b in range(0, len(order), PER_CALL):
            bi = [X[k] for k in order[b:b + PER_CALL]]
            reqs.append({"id": f"{TAG}__{rater}__{b // PER_CALL:04d}", "model": model, "prompt": PROMPT.format(items="\n".join(block(x) for x in bi)),
                         "params": PARAMS[rater], "check": "readout",
                         "meta": {"rater": rater, "ids": [x["item"] for x in bi]}})
    (JD / f"{TAG}_readout.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reqs) + "\n", encoding="utf-8")
    json.dump(X, open(JD / f"{TAG}_items.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(len(X), "items;", collections.Counter((x["model"], x["source"]) for x in X), ";", len(reqs), "calls")
elif sys.argv[1] == "score":
    X = {x["item"]: x for x in json.load(open(JD / f"{TAG}_items.json", encoding="utf-8"))}; R = {}
    for f in (BASE / "or_runner" / "local_root" / "out" / f"{TAG}_readout").glob("part-*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in R or r["t"] >= R[r["id"]]["t"]: R[r["id"]] = r
    rows = []
    for r in R.values():
        if r["meta"]["rater"] not in RATERS or r.get("status") != "ok": continue
        try: ans = json.loads(re.sub(r"^```(json)?|```$", "", (r.get("content") or "").strip()))["answers"]
        except Exception: continue
        for a in ans:
            x = X.get(str(a.get("id")))
            if not x: continue
            top3 = [str(t).strip()[:1].upper() for t in (a.get("top3") or [])]
            rows.append(dict(item=x["item"], rater=r["meta"]["rater"], model=x["model"], construct=x["construct"], source=x["source"], layer=x["layer"],
                             hit=str(a.get("pick", "")).strip()[:1].upper() == x["key"], hit3=x["key"] in top3))
    D = pd.DataFrame(rows); D.to_csv(HERE / f"readout_judge_v6{SFX}.csv", index=False)
    S = {"n_answers": len(D), "chance": {"pick": 0.1, "top3": 0.3},
         "by_model_source": D.groupby(["model", "source"])[["hit", "hit3"]].mean().round(3).reset_index().to_dict("records"),
         "by_model_source_layer": D.groupby(["model", "source", "layer"])[["hit", "hit3"]].mean().round(3).reset_index().to_dict("records"),
         "rater_agreement_on_hit": round(float(D.pivot_table(index="item", columns="rater", values="hit").dropna().corr().iloc[0, 1]), 3) if D.rater.nunique() > 1 else None,
         "constructs_hit_by_both_raters_any_layer": D[D.hit].groupby(["model", "construct"]).rater.nunique().ge(2).groupby(level=0).sum().to_dict()}
    json.dump(S, open(HERE / f"readout_judge_v6{SFX}.json", "w"), indent=1, default=str); print(json.dumps(S["by_model_source"], indent=1)); print("agreement", S["rater_agreement_on_hit"])
