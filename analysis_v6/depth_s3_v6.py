"""Study 3 at other depths and positions (BR4, Kimi M1): does the manipulation's effect on the mediator's representation predict its effect
on the answer when the representation is measured deeper (band 2, about 60% to 75% of the depth) or at the question / decision token?
Recomputes, from V3_exp/proj_v3.pt of the three main LLMs, the per-experiment effect on the mediator's representation exactly as
V3_exp.py does for (situation, band 1) -- standardized projections, own minus the mean of the other 65, treatment minus control per
wording label and agent, mean over agents, signed by x_to_m, d_z over wording labels -- for every position x band, checks the
(situation, band 1) value against the digest, and correlates each with the answer d_z of v3_all_models.csv across the 60 experiments."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

HERE = Path(__file__).resolve().parent; LAT = HERE.parent; RV = LAT / "results_v6"
E = json.load(open(LAT / "materials_v6" / "upload_v6" / "experiments_v6.json", encoding="utf-8"))
def sgn(s):
    try: return int(float(str(s).replace("+", ""))) if str(s).strip() not in ("", "None") else 0
    except ValueError: return 0
def dz(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean() / x.std(ddof=1)) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan
JOBS = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
ANS = pd.read_csv(HERE / "v3_all_models.csv")
out = {}
for name, job in JOBS.items():
    P = torch.load(RV / job / "V3_exp" / "proj_v3.pt", map_location="cpu", weights_only=False)
    PS = P["std"].float().numpy(); MK, GN, CONS = P["marks"], P["groups"], P["constructs"]
    D = pd.read_csv(RV / job / "V3_exp" / "items.csv")
    dig = {m.group(1): float(m.group(3)) for m in re.finditer(r"^\| (\w+) \| (-?[\d.]+|nan|None) \| (-?[\d.]+|nan|None) \|", (RV / job / "V3_exp" / "digest.md").read_text(encoding="utf-8"), re.M) if m.group(3) not in ("nan", "None")}
    res = {}
    for m in ("x", "body", "q", "dec"):
        for g in GN:
            sh = {}
            for eid, g_ in D.groupby("exp"):
                e = E[eid]; A_ = g_[g_.level == "A"]; oc = CONS.index(e["construct"])
                pj = PS[A_.index.values][:, MK.index(m), GN.index(g)]
                om = pj[:, oc] - (pj.sum(1) - pj[:, oc]) / (len(CONS) - 1)
                A2 = A_.assign(om=om); ts = A2[A2.arm == "treatment"].set_index(["vid", "agent"]).om; cs = A2[A2.arm == "control"].set_index(["vid", "agent"]).om
                s = (ts - cs.reindex(ts.index)).groupby(level=0).mean() * (sgn(e["x_to_m_sign"]) or 1)
                sh[eid] = dz(s.values)
            sh = pd.Series(sh)
            a = ANS[ANS.model == name].set_index("exp").dz
            x, y = sh.align(a, join="inner"); ok = ~(x.isna() | y.isna())
            rho, p = stats.spearmanr(x[ok], y[ok])
            res[f"{m}_{g}"] = dict(n=int(ok.sum()), pos=int((sh > 0).sum()), rho=round(float(rho), 3), p=round(float(p), 3))
            if m == "body" and g == "B1":
                common = [k for k in sh.index if k in dig]
                res["check_body_B1_vs_digest_maxabs"] = round(float(np.nanmax(np.abs(sh[common].values - np.array([dig[k] for k in common])))), 3)
    out[name] = res
    print(name, json.dumps(res))
json.dump(out, open(HERE / "depth_s3_v6.json", "w"), indent=1)
