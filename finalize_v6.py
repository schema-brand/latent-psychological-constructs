"""Materials v6: select the pairs and variants that pass the quality checks and write the final tables.
Policy (DEVLOG 19:50 UTC): hard flags = missing text, words out of range, length mismatch (>15%), banned word, stated mental state, study frame,
diff too small or too large; soft flags (kept, recorded) = ending differs, number rule.
  pairs: per writer x construct x low type keep up to the target (12 for mediators, 8 for rivals and comparisons), in call order.
  variants: every passing rewording, attached to its experiment (after the frozen v00).
Usage: python finalize_v6.py [topup]   # topup: also write top-up calls (B3) for writer x construct x low cells below target"""
import json, sys, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"
SOFT = {"ending_differs", "number_rule"}
_j = __import__("importlib.util").util.spec_from_file_location("j6", HERE / "make_or_jobs_v6.py"); j6 = __import__("importlib.util").util.module_from_spec(_j); _j.loader.exec_module(j6)
spec = json.loads((M6 / "spec_v6.json").read_text(encoding="utf-8")); C = spec["constructs"]

def target(c, low):
    if C[c].get("v4_format"): return 16
    return 12 if C[c]["role"] == "mediator" else 8

def pairs():
    R = [json.loads(l) for l in open(M6 / "pairs_v6_raw.jsonl", encoding="utf-8")]
    R.sort(key=lambda r: r["pid"]); keep, cell = [], collections.defaultdict(list)
    for r in R:
        if [f for f in r["flags"] if f not in SOFT]: continue
        if r["construct"] == "uniqueness" and r.get("prompt_version") != "v6.1": continue   # spec v1 superseded (DEVLOG 21:15 UTC)
        k = (r["writer"], r["construct"], r["low_type"])
        if len(cell[k]) < target(r["construct"], r["low_type"]): cell[k].append(r); keep.append(r)
    with open(M6 / "pairs_v6.jsonl", "w", encoding="utf-8") as f:
        for r in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    short = {}
    for w in j6.WRITERS:
        for c, cc in C.items():
            lows = ["L4"] if cc.get("v4_format") else (["L1", "L4"])
            for low in lows:
                need = target(c, low); have = len(cell[(w, c, low)])
                if have < need: short[(w, c, low)] = need - have
    print(len(keep), "pairs kept;", "per writer:", dict(collections.Counter(r["writer"] for r in keep)))
    print("cells short:", len(short), "missing pairs:", sum(short.values()))
    return short

def topup(short):
    """Budget-aware top-up (DEVLOG 20:13 UTC): mediator cells missing >= 3 of 12; rival/comparison cells missing > 3; one call per cell,
    asking for a few more pairs than missing to absorb failures; settings start at the writer's 4th domain to vary them."""
    calls = collections.defaultdict(list)
    for (w, c, low), miss in short.items():
        cc = C[c]
        if (cc["role"] == "mediator" and miss < 3) or (cc["role"] != "mediator" and miss <= 3): continue
        n = min(8, miss + 2)
        call = dict(writer=w, construct=c, low=low, n=n, j=0.5, idx=9, call=f"{w}__{c}__{low}_t{miss}")
        if w == "claude": (M6 / "claude_prompts" / f"{call['call']}.txt").write_text(j6.prompt_v6(call, cc), encoding="utf-8"); calls[w].append(call["call"])
        else:
            r = j6.request(call, cc)
            if w in getattr(j6, "REASONING_OFF", set()): r["params"]["reasoning"] = {"enabled": False}
            calls[w].append(r)
    for w, xs in calls.items():
        if w == "claude": (M6 / "claude_calls_topup.json").write_text(json.dumps(xs, indent=0), encoding="utf-8")
        else: (M6 / "jobs" / f"B3_{w}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in xs) + "\n", encoding="utf-8")
        print("topup", w, len(xs))

def variants():
    E2C = {s["experiment"]: k for k, c in C.items() if c["role"] == "mediator" for s in c["sources"]}
    R = [json.loads(l) for l in open(M6 / "variants_v6_raw.jsonl", encoding="utf-8")]
    keep = []
    for r in R:
        if r["hard_fail"]: continue
        r["uid"] = f"{r['writer']}__{r['experiment']}__{r.get('vid')}"; r["construct"] = E2C[r["experiment"]]; keep.append(r)
    with open(M6 / "variants_v6.jsonl", "w", encoding="utf-8") as f:
        for r in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = collections.defaultdict(list)
    for r in keep: by[r["experiment"]].append(r)
    for p in (M6 / "experiments").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8")); d["construct"] = E2C[d["id"]]
        d["variants"] = [d["variants"][0]] + [{k: v for k, v in r.items() if k not in ("experiment", "hard_fail")} for r in by[d["id"]]]
        p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    n = collections.Counter(len(v) for v in by.values())
    print(len(keep), "variants kept; experiments by count of rewordings:", dict(sorted(n.items())), "| experiments with none:", 60 - len(by))

def exclude():
    """Preregistrable exclusion rules applied after the ratings (DEVLOG 21:05 UTC):
    R1 the high version is not rated higher than the low version on its own construct (mean over raters; pairs with no rating are kept);
    R2 near-duplicate texts within writer x construct (tf-idf cosine of the high versions > .80): the later pair is dropped.
    Writes materials_v6/pairs_v6_final.jsonl and materials_v6/exclusions_v6.json."""
    import csv, numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    P = [json.loads(l) for l in open(M6 / "pairs_v6.jsonl", encoding="utf-8")]
    V = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(M6 / "ratings_v6.csv", encoding="utf-8")): V[r["text_id"]][r["scale"]].append(int(r["rating"]))
    def own(p):
        h, l = V[p["pid"] + "|high"].get(p["construct"]), V[p["pid"] + "|low"].get(p["construct"])
        return (np.mean(h) - np.mean(l)) if h and l else None
    drop = {}
    for p in P:
        d = own(p)
        if d is not None and d <= 0: drop[p["pid"]] = f"R1 own-scale high-low = {d:.2f}"
    X = TfidfVectorizer(sublinear_tf=True).fit_transform([p["high"] for p in P]); S = (X @ X.T).tocoo()
    for i, j, v in zip(S.row, S.col, S.data):
        if i < j and v > .8 and P[i]["writer"] == P[j]["writer"] and P[i]["construct"] == P[j]["construct"] and P[i]["pid"] not in drop:
            drop.setdefault(P[j]["pid"], f"R2 near-duplicate of {P[i]['pid']} (cos {v:.2f})")
    keep = [p for p in P if p["pid"] not in drop]
    with open(M6 / "pairs_v6_final.jsonl", "w", encoding="utf-8") as f:
        for p in keep: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    json.dump(drop, open(M6 / "exclusions_v6.json", "w"), indent=1)
    rule = collections.Counter(v[:2] for v in drop.values()); per = collections.Counter(p["construct"] for p in keep)
    print(len(P), "->", len(keep), "pairs; dropped", dict(rule), "| fewest per construct:", sorted((n, c) for c, n in per.items())[:6])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "exclude": exclude(); sys.exit()
    s = pairs(); variants()
    if len(sys.argv) > 1 and sys.argv[1] == "topup": topup(s)
