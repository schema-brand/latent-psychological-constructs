"""4d moderation with MVCA's sign definition. MVCA's study3_interaction_sign = sign of (X->Y effect at level A - at level B), i.e. the
effect is predicted to be stronger at A (the level of the Study 1 comparison). For a push along the mediator's high pole the expected
direction on the answer is s_my = x_to_m_sign * x_to_y_sign, so the prediction is aligned = (push_A - push_B) * s_my > 0.
Check against MVCA's own rule (release data/study3_tests.csv: interaction = interaction_sign x (statement effect at larger_level -
at other_level); A = larger_level in all 7; the statement states the low pole for White & Argo): interaction_sign x statement pole =
x_to_m x x_to_y for all 7, so both definitions give the same prediction. Items with digit mass < .5 (pushed or unpushed) are excluded (fixed rule). Usage: python analysis_v6/moderation_v6.py JOB ..."""
import sys, json, pandas as pd, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent; RV = HERE / "results_v6"
E = json.load(open(HERE / "materials_v6/upload_v6/experiments_v6.json", encoding="utf-8"))
sg = lambda s: int(float(str(s).replace("+", ""))) if str(s).strip() not in ("", "None") else 0
def dz(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; m, s = x.mean(), x.std(ddof=1)
    return m, m / s, m / (s / np.sqrt(len(x))), len(x)
for job in sys.argv[1:]:
    D = pd.read_csv(RV / job / "V4d_moderation" / "moderation_v4d.csv", usecols=["exp", "kind", "dir", "dose", "vid", "agent", "level", "ev", "ev0", "d", "mass"])
    D["p0"] = 1.0; mass_ok = D.mass >= .5; print(f"== {job}: rows {len(D)}, excluded mass<.5: {1 - mass_ok.mean():.3f}"); D = D[mass_ok]
    k = ["exp", "kind", "dir", "level", "vid", "agent"]
    a = D[D.dose == 2].groupby(k).d.mean(); b = D[D.dose == -2].groupby(k).d.mean(); P = ((a - b.reindex(a.index)) / 2).rename("push").reset_index()
    P = P.groupby(["exp", "kind", "level", "vid", "agent"]).push.mean().unstack("level").dropna()
    rows = []
    for (eid, kind), g in P.groupby(level=[0, 1]):
        e = E[eid]; s = sg(e["x_to_m_sign"]) * sg(e["x_to_y_sign"])
        m, d, t, n = dz((g["A"] - g["B"]) * s)
        rows.append(dict(exp=eid, kind=kind, s_my=s, push_A=round(g["A"].mean() * s, 3), push_B=round(g["B"].mean() * s, 3), aligned=round(m, 3), dz=round(d, 2), t=round(t, 1), n=n))
    R = pd.DataFrame(rows); print(R.pivot(index="exp", columns="kind", values="t").round(1).to_string())
    o = R[R.kind == "own"].set_index("exp")
    print(o[["push_A", "push_B", "aligned", "dz", "t", "n"]].to_string())
    for kind in ("own", "near", "span"):
        r = R[R.kind == kind]; print(f"  {kind}: aligned > 0 in {int((r.aligned > 0).sum())}/{len(r)}; t > 2 in {int((r.t > 2).sum())}, t < -2 in {int((r.t < -2).sum())}")
    R.to_csv(RV / job / "V4d_moderation" / "moderation_aligned.csv", index=False)
