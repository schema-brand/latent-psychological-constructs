"""Refined ceiling/floor check for V3 (fixed rule): 'blocked' = share of items whose comparison arm sits at the end of the 1-9 scale that
the predicted effect moves toward (expected value >= 8.5 when the treatment should raise the answer, <= 1.5 when it should lower it);
'same_end' = both arms at that end. A model x experiment with blocked > .30 is excluded from response-based tests (reported with and
without). Writes results_v6/<job>/V3_exp/ceiling_check.csv. Usage: python analysis_v6/ceiling_v6.py JOB ..."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent; RV = HERE / "results_v6"
E = json.load(open(HERE / "materials_v6/upload_v6/experiments_v6.json", encoding="utf-8"))
sg = lambda s: int(float(str(s).replace("+", ""))) if str(s).strip() not in ("", "None") else 0
for job in sys.argv[1:]:
    D = pd.read_csv(RV / job / "V3_exp/items.csv"); D = D[D.mass >= .5]
    rows = []
    for eid, g in D.groupby("exp"):
        s = sg(E[eid]["x_to_y_sign"]) or 1; k = ["vid", "level", "agent"]
        t = g[g.arm == "treatment"].set_index(k).ev; c = g[g.arm == "control"].set_index(k).ev; t, c = t.align(c, join="inner")
        if len(t) == 0: continue
        end_c = (c >= 8.5) if s > 0 else (c <= 1.5); end_t = (t >= 8.5) if s > 0 else (t <= 1.5); d = (t - c) * s
        rows.append(dict(exp=eid, n=len(t), effect=round(d.mean(), 2), dz=round(d.mean() / d.std(), 2) if d.std() > 0 else np.nan,
                         blocked=round(end_c.mean(), 3), same_end=round((end_c & end_t).mean(), 3), ctrl_mean=round(c.mean(), 2), trt_mean=round(t.mean(), 2)))
    R = pd.DataFrame(rows).sort_values("blocked", ascending=False); R.to_csv(RV / job / "V3_exp/ceiling_check_v2.csv", index=False)
    print(f"== {job}: blocked > .15 in {int((R.blocked > .15).sum())}, > .30 in {int((R.blocked > .30).sum())}; effect in MVCA's direction {int((R.effect > 0).sum())}/{len(R)}"
          f" (without blocked > .30: {int((R[R.blocked <= .30].effect > 0).sum())}/{int((R.blocked <= .30).sum())})")
    print(R[R.blocked > .15][["exp", "blocked", "same_end", "effect", "ctrl_mean", "trt_mean"]].to_string(index=False))
