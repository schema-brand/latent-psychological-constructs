"""Study 1, criterion validity: does the internal reading of a pair predict how the model's own first-person judgments move between the pair's
high and low situation? Data: V4a natural rows (each of ~395 pairs read high and low, judged on its own construct's item and ~10 other items;
expected value of the 1-9 answer), and V1's writer-held-out cosine of each pair's state difference with each of the 66 construct centroids
(lowo_cos_v6.pt, band 1). For every pair x judged item j: x = cosine with construct j's centroid, y = natural shift of judgment j
(high - low). Criterion correlation = Pearson r after removing pair means and item means (two-way centring); also the share of pairs whose
largest judgment shift falls on the item whose centroid is closest. Permutation null: items shuffled within pair (2,000).
Usage: python analysis_v6/criterion_v6.py -> analysis_v6/criterion_v6.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; BASE = HERE.parent; RV = BASE / "results_v6"
JOBS = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
OUT = {}; rng = np.random.default_rng(3)
for name, job in JOBS.items():
    D = pd.read_csv(RV / job / "V4a_causal/causal_v4a.csv"); N = D[(D.part == "natural") & (D.mass >= .5)]
    w = N.pivot_table(index=["pid", "agent", "j"], columns="side", values="ev").dropna(); w["y"] = w.high - w.low
    Y = w.y.groupby(["pid", "j"]).mean().reset_index()
    L = torch.load(RV / job / "V1_read/lowo_cos_v6.pt", weights_only=False); cons = L["constructs"]; li = L["layers"].index("B1")
    meta = pd.read_csv(RV / job / "V1_read/pairs_meta.csv"); row = {p: i for i, p in enumerate(meta.pid)}
    Y = Y[Y.pid.isin(row) & Y.j.isin(cons)].copy(); Y["x"] = [float(L["proj"][row[p], li, cons.index(j)]) for p, j in zip(Y.pid, Y.j)]
    Y["own"] = [meta.construct.iloc[row[p]] == j for p, j in zip(Y.pid, Y.j)]
    def centre(s, df): return s - df.groupby("pid")[s.name].transform("mean") - df.groupby("j")[s.name].transform("mean") + s.mean()
    xc, yc = centre(Y.x, Y), centre(Y.y, Y); r = float(np.corrcoef(xc, yc)[0, 1])
    null = []
    for _ in range(2000):
        ys = Y.groupby("pid").y.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
        null.append(np.corrcoef(xc, centre(ys.rename("y"), Y.assign(y=ys)))[0, 1])
    top = Y.loc[Y.groupby("pid").x.idxmax()].set_index("pid").j; big = Y.loc[Y.groupby("pid").y.idxmax()].set_index("pid").j
    n_items = Y.groupby("pid").size().mean()
    OUT[name] = {"n_pairs": int(Y.pid.nunique()), "items_per_pair": round(float(n_items), 1), "criterion_r": round(r, 3), "perm_p": round(float((np.array(null) >= r).mean()), 4),
                 "null_95": round(float(np.quantile(null, .95)), 3), "closest_centroid_is_largest_shift": round(float((top == big.reindex(top.index)).mean()), 3),
                 "chance": round(1 / n_items, 3), "own_item_shift_mean": round(float(Y[Y.own].y.mean()), 3), "other_items_shift_mean": round(float(Y[~Y.own].y.mean()), 3)}
    print(name, OUT[name])
json.dump(OUT, open(HERE / "criterion_v6.json", "w"), indent=1)
