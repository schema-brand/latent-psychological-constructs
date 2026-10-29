"""Specificity of the Study 3 representation-answer correlation by position (follow-up to depth_s3_v6.py).
For each position x band, computes the effect of the manipulation on every construct's representation (own minus the mean of the other
65, d_z over wording labels, signed by the experiment's x_to_m), and compares the cross-experiment Spearman correlation with the answer
d_z for the experiment's own mediator against (a) 2,000 random reassignments of other experiments' mediators (derangements that never
assign an experiment its own construct) and (b) the experiment's own rival/comparison set is not used. Also reports the share of
experiments in which the own mediator ranks first among the 66 at each position."""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

HERE = Path(__file__).resolve().parent; LAT = HERE.parent; RV = LAT / "results_v6"
E = json.load(open(LAT / "materials_v6" / "upload_v6" / "experiments_v6.json", encoding="utf-8"))
def sgn(s):
    try: return int(float(str(s).replace("+", ""))) if str(s).strip() not in ("", "None") else 0
    except ValueError: return 0
JOBS = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
ANS = pd.read_csv(HERE / "v3_all_models.csv")
rng = np.random.default_rng(20261021); NPERM = 2000
out = {}
for name, job in JOBS.items():
    P = torch.load(RV / job / "V3_exp" / "proj_v3.pt", map_location="cpu", weights_only=False)
    PS = P["std"].float().numpy(); MK, GN, CONS = P["marks"], P["groups"], P["constructs"]; K = len(CONS)
    D = pd.read_csv(RV / job / "V3_exp" / "items.csv")
    a = ANS[ANS.model == name].set_index("exp").dz
    exps = [x for x in D.exp.unique() if x in a.index and not np.isnan(a[x])]
    own = np.array([CONS.index(E[x]["construct"]) for x in exps]); y = a[exps].values
    res = {}
    for m in ("x", "body", "q", "dec"):
        for g in ("B1", "B2"):
            Z = np.full((len(exps), K), np.nan)                       # d_z of the shift for every construct
            for i, eid in enumerate(exps):
                e = E[eid]; g_ = D[D.exp == eid]; A_ = g_[g_.level == "A"]
                pj = PS[A_.index.values][:, MK.index(m), GN.index(g)]          # [n, 66]
                om = pj - (pj.sum(1, keepdims=True) - pj) / (K - 1)            # own minus mean of the other 65, for every construct
                t = A_.arm.values == "treatment"
                key = list(zip(A_.vid, A_.agent))
                T = pd.DataFrame(om[t], index=pd.MultiIndex.from_tuples([k for k, f in zip(key, t) if f]))
                C = pd.DataFrame(om[~t], index=pd.MultiIndex.from_tuples([k for k, f in zip(key, t) if not f]))
                d = (T - C.reindex(T.index)).groupby(level=0).mean() * (sgn(e["x_to_m_sign"]) or 1)
                Z[i] = (d.mean(0) / d.std(0, ddof=1)).values
            zo = Z[np.arange(len(exps)), own]
            rho = stats.spearmanr(zo, y)[0]
            null = []
            for _ in range(NPERM):
                while True:
                    perm = rng.permutation(len(exps))
                    if not np.any(own[perm] == own): break
                null.append(stats.spearmanr(Z[np.arange(len(exps)), own[perm]], y)[0])
            null = np.array(null)
            rank1 = int(np.sum(np.argmax(Z, 1) == own))
            res[f"{m}_{g}"] = dict(rho_own=round(float(rho), 3), null_mean=round(float(null.mean()), 3), null_95=round(float(np.quantile(null, .95)), 3),
                                   p_perm=round(float((np.sum(null >= rho) + 1) / (NPERM + 1)), 4), own_pos=int(np.sum(zo > 0)), own_first=rank1)
            print(name, m, g, res[f"{m}_{g}"], flush=True)
    out[name] = res
json.dump(out, open(HERE / "depth_s3_spec_v6.json", "w"), indent=1)
