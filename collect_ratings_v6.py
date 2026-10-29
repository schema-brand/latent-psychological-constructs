"""Materials v6 ratings: collect the D_* rating results from the packs, and check the contrast pairs.
Per pair: the high-minus-low difference on its own construct's scale (target) and on every other scale of the panel (specificity).
Per construct and writer: mean target difference, mean largest off-target difference, share of pairs whose own scale moves most.
Writes materials_v6/ratings_v6.csv (one row per text x rater x scale) and materials_v6/rating_check_v6.json. Usage: python collect_ratings_v6.py"""
import json, zipfile, collections, csv
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"; PK = M6 / "packs"
R = {}
for z in sorted(PK.glob("*.zip")):
    with zipfile.ZipFile(z) as Z:
        for n in Z.namelist():
            if not n.split("/")[0].startswith("D_"): continue
            for line in Z.read(n).decode("utf-8").splitlines():
                r = json.loads(line)
                if r["id"] not in R or r["t"] >= R[r["id"]]["t"]: R[r["id"]] = r
rows = []
for r in R.values():
    o = r.get("parsed") or {}; rater = r["meta"]["rater"]
    want = r["meta"]["ids"]; by_unit = {w.split("|")[0]: w for w in want}   # a batch never holds two texts of one unit
    for e in o.get("ratings", []) if isinstance(o, dict) else []:
        tid = str(e.get("id"))
        if tid not in want: tid = by_unit.get(tid.split("|")[0], tid)   # some raters drop the |high / |low suffix
        for s in r["meta"]["scales"]:
            v = e.get(s)
            if isinstance(v, int) and 1 <= v <= 7: rows.append((tid, rater, s, v))
with open(M6 / "ratings_v6.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["text_id", "rater", "scale", "rating"]); w.writerows(rows)
V = collections.defaultdict(dict)
for tid, rater, s, v in rows: V[(tid, rater)][s] = v
P = {json.loads(l)["pid"]: json.loads(l) for l in open(M6 / "pairs_v6.jsonl", encoding="utf-8")}
out = collections.defaultdict(lambda: {"n": 0, "target": [], "off_max": [], "own_top": []}); agree = []
for pid, p in P.items():
    for rater in ("deepseek", "glm", "kimi"):
        h, l = V.get((f"{pid}|high", rater)), V.get((f"{pid}|low", rater))
        if not h or not l or p["construct"] not in h: continue
        d = {s: h[s] - l[s] for s in h if s in l}; t = d.pop(p["construct"]); off = max(d.values(), key=abs) if d else 0
        k = f"{p['construct']}|{p['writer']}"; X = out[k]; X["n"] += 1; X["target"].append(t); X["off_max"].append(abs(off)); X["own_top"].append(t > max(map(abs, d.values()), default=0))
    a, b = V.get((f"{pid}|high", "glm")), V.get((f"{pid}|high", "deepseek")) or V.get((f"{pid}|high", "kimi"))
    if a and b and p["construct"] in a and p["construct"] in b: agree.append((a[p["construct"]], b[p["construct"]]))
summ = {k: {"n": v["n"], "target_diff": round(float(np.mean(v["target"])), 2), "max_off_target": round(float(np.mean(v["off_max"])), 2),
            "own_scale_moves_most": round(float(np.mean(v["own_top"])), 2)} for k, v in out.items() if v["n"]}
byc = collections.defaultdict(list)
for k, v in summ.items(): byc[k.split("|")[0]].append(v)
cons = {c: {"target_diff": round(float(np.mean([x["target_diff"] for x in L])), 2), "own_top": round(float(np.mean([x["own_scale_moves_most"] for x in L])), 2)} for c, L in byc.items()}
json.dump({"by_construct_writer": summ, "by_construct": cons, "rater_r": float(np.corrcoef(np.array(agree).T)[0, 1]) if len(agree) > 10 else None}, open(M6 / "rating_check_v6.json", "w"), indent=1)
print(len(rows), "ratings;", len(R), "calls;", "pairs with ratings:", sum(v["n"] for v in out.values()), "| rater agreement r (high texts, own scale):", round(float(np.corrcoef(np.array(agree).T)[0, 1]), 2) if len(agree) > 10 else None)
w = sorted(cons.items(), key=lambda x: x[1]["target_diff"])
print("weakest constructs (mean high-low on own scale):", [(c, v["target_diff"], v["own_top"]) for c, v in w[:8]])
print("strongest:", [(c, v["target_diff"]) for c, v in w[-5:]])
