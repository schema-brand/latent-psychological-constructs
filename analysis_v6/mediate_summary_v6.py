"""Individuality mediation by removal (V7m, run 4): unpack single-piece packs saved by the Drive connector into results_v6/<job>, then
summarize per LLM x test x place (stmt / after) x kind: interaction mean and d_z (as V7i: ((T - C) at the larger level - (T - C) at the other
level) x x_to_y_sign, items with digit mass >= .5), the share of the as-is interaction left, and the share of items with broken answers.
Usage: python analysis_v6/mediate_summary_v6.py -> analysis_v6/mediate_summary_v6.csv"""
import json, glob, base64, zipfile, io, os, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; RV = HERE.parent / "results_v6"
TR = os.environ.get("TOOL_RESULTS", "tool-results")   # REPOSITORY: folder of Drive downloads (was a local session folder)
for f in glob.glob(TR + "/mcp-739c3c66-*download_file_content*.txt"):
    try: d = json.load(open(f, encoding="utf-8"))
    except Exception: continue
    t = d.get("title", "")
    if re.match(r"V84\d_mediate.*__c000\.bin$", t):
        job = t.split("__c000")[0]; out = RV / job; out.mkdir(exist_ok=True)
        zipfile.ZipFile(io.BytesIO(base64.b64decode(d["content"]))).extractall(out); os.remove(f); print("unpacked", job)
rows = []
for f in sorted(RV.glob("V84*_mediate_*/V7m_mediate/mediate.csv")):
    job = f.parts[-3]; D = pd.read_csv(f)
    for test, a in D.groupby("test", sort=False):
        sx = int(a.x_to_y.iloc[0]); big = a.larger.iloc[0]; oth = "low" if big == "high" else "high"
        def inter(g):
            ok = g[g.mass >= .5]; w = ok.pivot_table(index=["vid", "agent", "level"], columns="arm", values="ev")
            if not {"treatment", "control"} <= set(w.columns): return np.nan, np.nan, 0
            e = (w.treatment - w.control).unstack("level")
            if big not in e or oth not in e: return np.nan, np.nan, 0
            d = ((e[big] - e[oth]) * sx).dropna()
            return (d.mean(), d.mean() / d.std(ddof=1) if len(d) > 2 and d.std() > 0 else np.nan, len(d))
        base = inter(a[a.kind == "asis"])
        for (kind, loc), g in a[a.level != "none"].groupby(["kind", "loc"]):
            m, dz, n = inter(g)
            rows.append(dict(job=job, test=test, kind=kind, loc=loc, mean=m, dz=dz, n=n, share_left=m / base[0] if base[0] and base[0] == base[0] else np.nan,
                             broken=round(float((g.mass < .5).mean()), 3)))
S = pd.DataFrame(rows); S.to_csv(HERE / "mediate_summary_v6.csv", index=False)
for job, g in S.groupby("job"):
    print("==", job); P = g[g.test.isin(["thaler_costs_justify", "freedman_consistency", "whiteargo_cse"])].copy(); P["k"] = P.kind.str.replace(r"\d", "", regex=True)
    print(P.groupby(["test", "loc", "k"]).agg(mean=("mean", "mean"), share=("share_left", "mean"), broken=("broken", "mean")).round(3).to_string())
