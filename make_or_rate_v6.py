"""Materials v6, ratings (OPENROUTER_PLAN_2026-10-19.md, phase D): every contrast-pair text (and every experiment variant text) is rated
alone, blind to its pair, construct and writer, on all constructs of its MVCA home process plus the three MVCA comparison scales.
8 texts per call, shuffled so that the two versions of a pair never share a call. Raters: DeepSeek V4.1 Flash (not on DeepSeek texts),
GLM 5.3 (all), Kimi K3 (DeepSeek texts only). Usage: python make_or_rate_v6.py pairs|exp <tag>   -> materials_v6/jobs/D_<tag>_<rater>.jsonl"""
import json, sys, random, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"; JOBS = M6 / "jobs"
RATERS = {"deepseek": "deepseek/deepseek-v4.1-flash-1017", "glm": "z-ai/glm-5.3", "kimi": "moonshotai/kimi-k3"}
COMPARISONS = ["favorable_evaluation", "pleasantness", "setting_appeal"]
PROMPT = """You rate short descriptions of everyday situations. Each is written in the second person about "you". For each description,
rate, on each scale below, how strongly the situation AS DESCRIBED makes the statement true of "you" (the person in it):
1 = the situation makes clear that it is not so at all; 4 = the situation does not bear on it (not mentioned, neutral); 7 = the situation
makes it strongly so. Use the whole scale, rate each description on its own, and judge only what the text describes.

SCALES
{scales}

DESCRIPTIONS
{texts}

OUTPUT: only a JSON object, no other text: {{"ratings": [{{"id": "<id>", {keys}}}]}} with one entry per description and an integer 1-7 for every scale."""

def panels(C):
    """Scales per home-process group: the group's constructs + comparisons; rivals and reused constructs go to their experiments' group."""
    E = {e["experiment"]: e["home_process"] for e in json.loads((M6 / "mvca_constructs_raw.json").read_text(encoding="utf-8"))}
    grp = {}
    for k, c in C.items():
        if k in COMPARISONS: continue
        g = c["group"] if c["group"] != "comparison" else E.get(c["sources"][0]["experiment"], "comparison")
        grp[k] = g
    P = collections.defaultdict(list)
    for k, g in grp.items(): P[g].append(k)
    return grp, {g: sorted(v) + COMPARISONS for g, v in P.items()}

def texts_pairs():
    T = []
    for l in open(M6 / "pairs_v6.jsonl", encoding="utf-8"):
        p = json.loads(l)
        for side in ("high", "low"): T.append(dict(id=f"{p['pid']}|{side}", unit=p["pid"], writer=p["writer"], construct=p["construct"], text=p[side]))
    return T

def texts_exp():
    T = []
    for l in open(M6 / "variants_v6.jsonl", encoding="utf-8"):
        v = json.loads(l)
        for side, key in (("T", "body_treatment"), ("C", "body_control"), ("TB", "body_treatment_B"), ("CB", "body_control_B")):
            if v.get(key): T.append(dict(id=f"{v['uid']}|{side}", unit=v["uid"], writer=v["writer"], construct=v["construct"], text=v[key]))
    return T

def batches(T, size=8, seed=11):
    random.Random(seed).shuffle(T); out = []
    while T:
        b, rest = [], []
        for t in T:
            if len(b) < size and all(t["unit"] != x["unit"] for x in b): b.append(t)
            else: rest.append(t)
        out.append(b); T = rest
    return out

if __name__ == "__main__":
    what, tag = sys.argv[1], sys.argv[2]
    C = json.loads((M6 / "spec_v6.json").read_text(encoding="utf-8"))["constructs"]; grp, P = panels(C)
    T = texts_pairs() if what == "pairs" else texts_exp()
    if len(sys.argv) > 3: T = [t for t in T if t["writer"] in sys.argv[3].split(",")]
    seen = set()   # texts already in earlier rating jobs (incremental tags)
    for f in JOBS.glob("D_*.jsonl"):
        for l in open(f, encoding="utf-8"): seen |= set(json.loads(l)["meta"]["ids"])
    T = [t for t in T if t["id"] not in seen]; print(len(T), "texts to rate")
    jobs = collections.defaultdict(list)
    for g, scales in P.items():
        Tg = [t for t in T if grp.get(t["construct"], "comparison") == g or (t["construct"] in COMPARISONS and g == "comparison")]
        if g == "comparison": continue
        sc = "\n".join(f"- {k}: {C[k]['definition']}" for k in scales); keys = ", ".join(f'"{k}": <1-7>' for k in scales)
        for rater in RATERS:
            mine = [t for t in Tg if (t["writer"] == "deepseek") == (rater == "kimi")] if rater != "glm" else list(Tg)
            for i, b in enumerate(batches(mine)):
                texts = "\n".join(f"[{t['id']}] {t['text']}" for t in b)
                jobs[rater].append({"id": f"{tag}__{rater}__{g.replace(' ', '_')}__{i:04d}", "model": RATERS[rater], "prompt": PROMPT.format(scales=sc, texts=texts, keys=keys),
                                    "params": {"max_tokens": 16000, "temperature": 0.2, "reasoning": {"effort": "low"}}, "check": "ratings",
                                    "meta": {"rater": rater, "group": g, "ids": [t["id"] for t in b], "scales": scales}})
    # comparison-construct texts are rated on the panel of a randomly assigned group (all panels contain the comparisons)
    comp = [t for t in T if t["construct"] in COMPARISONS]; groups = sorted(P)
    for t in comp: t["g"] = groups[hash(t["unit"]) % len(groups)] if False else groups[sum(map(ord, t["unit"])) % len(groups)]
    for g in groups:
        Tg = [t for t in comp if t["g"] == g]
        if not Tg: continue
        scales = P[g]; sc = "\n".join(f"- {k}: {C[k]['definition']}" for k in scales); keys = ", ".join(f'"{k}": <1-7>' for k in scales)
        for rater in RATERS:
            mine = [t for t in Tg if (t["writer"] == "deepseek") == (rater == "kimi")] if rater != "glm" else list(Tg)
            for i, b in enumerate(batches(mine)):
                texts = "\n".join(f"[{t['id']}] {t['text']}" for t in b)
                jobs[rater].append({"id": f"{tag}__{rater}__cmp_{g.replace(' ', '_')}__{i:04d}", "model": RATERS[rater], "prompt": PROMPT.format(scales=sc, texts=texts, keys=keys),
                                    "params": {"max_tokens": 16000, "temperature": 0.2, "reasoning": {"effort": "low"}}, "check": "ratings",
                                    "meta": {"rater": rater, "group": g, "ids": [t["id"] for t in b], "scales": scales}})
    for r, xs in jobs.items():
        (JOBS / f"D_{tag}_{r}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in xs) + "\n", encoding="utf-8")
        print(r, len(xs), "calls", sum(len(x["meta"]["ids"]) for x in xs), "texts")
