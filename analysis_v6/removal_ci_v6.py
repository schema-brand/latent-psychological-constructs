"""Study 4 removal: median share of the manipulation's effect left after removal, with 95% bootstrap CIs over experiments
(same rules as removal_v6.py: items with digit mass < .5 excluded; experiments with |unedited effect| > .3). Psych Science version.
Writes analysis_v6/removal_ci_v6.json."""
import json, numpy as np, pandas as pd
from pathlib import Path
A = Path(__file__).resolve().parent; RV = A.parent / "results_v6"
def grp(c):
    w = ("__" + c.split("__")[1]) if "__" in c else ""
    for p in ("near_spec_k10", "rand_spec_k10", "unif"):
        if c.startswith(p): return p + w
    return c
rng = np.random.default_rng(3); out = {}
for name, job in (("Qwen3.6 27B", "V610_qwen36"), ("Gemma 4 31B", "V611_gemma4_31b"), ("OLMo 3.1 32B", "V612_olmo32")):
    f = RV / job / "V4c_remove" / "remove_v4c.csv"
    if not f.exists(): continue
    D = pd.read_csv(f, usecols=["exp", "cond", "vid", "agent", "arm", "ev", "mass", "x_to_y_sign"]); k = ["exp", "cond", "vid", "agent"]
    W = D.pivot_table(index=k, columns="arm", values=["ev", "mass"]).reset_index(); W.columns = ["_".join(c).strip("_") for c in W.columns]
    sg = D.groupby("exp").x_to_y_sign.first(); W["eff"] = (W.ev_treatment - W.ev_control) * W.exp.map(sg)
    W["ok"] = (W.mass_treatment >= .5) & (W.mass_control >= .5)
    E = W[W.ok].groupby(["exp", "cond"]).eff.mean().reset_index(); E["grp"] = E.cond.map(grp); E = E.groupby(["exp", "grp"]).eff.mean().unstack()
    big = E[E["none"].abs() > .3]; S = big.div(big["none"], axis=0); res = {"n_experiments": int(len(S))}
    for col in S.columns:
        if col == "none": continue
        v = S[col].dropna().values
        if len(v) < 5: continue
        bs = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(5000)]
        res[col] = {"median": round(float(np.median(v)), 3), "ci": [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)], "n": int(len(v))}
    out[name] = res
json.dump(out, open(A / "removal_ci_v6.json", "w"), indent=1)
for m, r in out.items(): print(m, r["n_experiments"], {k: (v["median"], v["ci"]) for k, v in r.items() if isinstance(v, dict) and k.endswith("mean")})
