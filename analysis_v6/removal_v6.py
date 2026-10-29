"""4c removal summary with the fixed analysis rules: items whose digit mass < .5 (either arm, that condition) are excluded; experiments
with |unedited effect| > .3 are summarized. Share left = effect under the condition / unedited effect (experiment level, MVCA sign).
Usage: python analysis_v6/removal_v6.py V610_qwen36 V611_gemma4_31b ..."""
import sys, pandas as pd, numpy as np
from pathlib import Path
RV = Path(__file__).resolve().parent.parent / "results_v6"
def grp(c):
    w = ("__" + c.split("__")[1]) if "__" in c else ""
    for p in ("near_spec_k10", "rand_spec_k10", "unif"):
        if c.startswith(p): return p + w
    return c
for job in sys.argv[1:]:
    D = pd.read_csv(RV / job / "V4c_remove" / "remove_v4c.csv", usecols=["exp", "cond", "vid", "agent", "arm", "ev", "mass", "x_to_y_sign"])
    k = ["exp", "cond", "vid", "agent"]
    W = D.pivot_table(index=k, columns="arm", values=["ev", "mass"]).reset_index()
    W.columns = ["_".join(c).strip("_") for c in W.columns]
    sg = D.groupby("exp").x_to_y_sign.first(); W["eff"] = (W.ev_treatment - W.ev_control) * W.exp.map(sg)
    W["ok"] = (W.mass_treatment >= .5) & (W.mass_control >= .5); W["grp"] = W.cond.map(grp)
    excl = W.groupby("grp").ok.apply(lambda x: 1 - x.mean())
    E = W[W.ok].groupby(["exp", "grp"]).eff.mean().unstack()           # several near/rand/unif sets -> mean of their item means
    E = W[W.ok].groupby(["exp", "cond"]).eff.mean().reset_index(); E["grp"] = E.cond.map(grp); E = E.groupby(["exp", "grp"]).eff.mean().unstack()
    base = E["none"]; big = E[base.abs() > .3]; S = big.div(big["none"], axis=0)
    print(f"== {job}: {len(big)}/{len(E)} experiments with |effect| > .3")
    out = pd.DataFrame({"median_share_left": S.median().round(2), "items_excluded_mass<.5": excl.round(3)})
    print(out.to_string())
    for form in ("zero", "mean"):
        o = S.get(f"own_spec_k10__{form}")
        if o is None: continue
        ctl = S[[c for c in S.columns if c.endswith("__" + form) and c.split("__")[0] in ("near_spec_k10", "rand_spec_k10", "unif")]].min(axis=1)
        print(f"  own-specific removes more than every control ({form}): {int((o < ctl).sum())}/{int(o.notna().sum())}")
        for c in ("near_spec_k10", "rand_spec_k10", "unif"):
            if f"{c}__{form}" in S: print(f"    ... more than {c} ({form}): {int((o < S[f'{c}__{form}']).sum())}/{int((o.notna() & S[f'{c}__{form}'].notna()).sum())}")
    if "own_spec_k10__sitzero" in S:
        print(f"  situation-only: own-specific below uniform in {int((S['own_spec_k10__sitzero'] < S['unif__sitzero']).sum())}/{len(S)}")
