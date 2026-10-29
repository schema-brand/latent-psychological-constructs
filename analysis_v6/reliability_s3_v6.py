"""Study 3: reliability of the two per-experiment effects across wordings, and the correlation between them corrected for attenuation
(reviewer request). Same estimators as code_v6/V3_exp.py: level-A items; effect on the mediator's representation = treatment - control in
(own projection - mean of the other 65), situation tokens ('body'), band B1, standardized projections, times x_to_m_sign, averaged over agent
profiles per wording; effect on the answer = treatment - control expected answer (items with digit mass >= .5), times x_to_y_sign, averaged
over agent profiles per wording. Split-half reliability: wordings split into two halves (all 35 splits of 7 into 3 + 4), per-experiment means
correlated across experiments (Pearson), Spearman-Brown corrected, averaged over splits. Corrected r = r_xy / sqrt(rel_x * rel_y).
Usage: python analysis_v6/reliability_s3_v6.py -> analysis_v6/reliability_s3_v6.json"""
import json, itertools
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
HERE = Path(__file__).resolve().parent; BASE = HERE.parent; RV = BASE / "results_v6"
E = json.load(open(BASE / "materials_v6/upload_v6/experiments_v6.json", encoding="utf-8"))
def sg(s):
    try: return int(float(str(s).replace("+", ""))) if str(s).strip() not in ("", "None") else 0
    except ValueError: return 0
JOBS = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
OUT = {}
for name, job in JOBS.items():
    D = pd.read_csv(RV / job / "V3_exp/items.csv"); P = torch.load(RV / job / "V3_exp/proj_v3.pt", weights_only=False)
    MK, GN, CONS = P["marks"], P["groups"], P["constructs"]; PS = P["std"].float()[:, MK.index("body"), GN.index("B1")].numpy()
    rows = []
    for eid, g in D.groupby("exp"):
        e = E[eid]; A = g[g.level == "A"]; oc = CONS.index(e["construct"]); pj = PS[A.index.values]
        om = pj[:, oc] - (pj.sum(1) - pj[:, oc]) / (len(CONS) - 1); A = A.assign(om=om)
        k = ["vid", "agent"]; t, c = A[A.arm == "treatment"].set_index(k), A[A.arm == "control"].set_index(k)
        sh = ((t.om - c.om.reindex(t.index)) * (sg(e["x_to_m_sign"]) or 1)).groupby(level=0).mean()
        ok_t, ok_c = t[t.mass >= .5].ev, c[c.mass >= .5].ev
        an = ((ok_t - ok_c.reindex(ok_t.index)) * (sg(e["x_to_y_sign"]) or 1)).groupby(level=0).mean()
        for v in sh.index: rows.append(dict(exp=eid, vid=v, shift=sh[v], ans=an.get(v, np.nan)))
    R = pd.DataFrame(rows); W = sorted(R.vid.unique()); full = R.groupby("exp")[["shift", "ans"]].mean()
    rel = {"shift": [], "ans": []}
    for half in itertools.combinations(W, 3):
        a = R[R.vid.isin(half)].groupby("exp")[["shift", "ans"]].mean(); b = R[~R.vid.isin(half)].groupby("exp")[["shift", "ans"]].mean()
        for col in rel:
            x = pd.concat([a[col], b[col]], axis=1).dropna(); r = np.corrcoef(x.iloc[:, 0], x.iloc[:, 1])[0, 1]; rel[col].append(2 * r / (1 + r))
    rs, ra = float(np.mean(rel["shift"])), float(np.mean(rel["ans"])); f = full.dropna()
    rxy = float(np.corrcoef(f["shift"], f["ans"])[0, 1]); rho = float(stats.spearmanr(f["shift"], f["ans"]).statistic)
    OUT[name] = {"n_exp": len(f), "n_wordings": len(W), "reliability_shift": round(rs, 3), "reliability_answer": round(ra, 3),
                 "r_shift_answer": round(rxy, 3), "rho_shift_answer": round(rho, 3), "r_corrected": round(rxy / np.sqrt(rs * ra), 3)}
    print(name, OUT[name], flush=True)
json.dump(OUT, open(HERE / "reliability_s3_v6.json", "w"), indent=1)
