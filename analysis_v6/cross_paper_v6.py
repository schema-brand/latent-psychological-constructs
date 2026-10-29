"""Study 3 rank and cross-paper links with MVCA (release_2026-10-15 data), for the run-1 models (local V3 projections and V4b digests) and
the Study 3 shift counts of all models with digests.
(1) Rank of the experiment's own mediator among the 66 constructs: per experiment, treatment - control shift of the situation state (mean over
    the body tokens, band 1, standardized dimensions) projected on each construct direction, paired by variant x level x agent; d_z per
    construct; the shift common to all experiments (mean d_z of each construct over the 60 experiments) is removed before ranking
    ("centred"); raw ranks also kept. MVCA: own mediator first in 24/60 (fig6C).
(2) Per-experiment Spearman correlations: latent own-minus-other shift d_z (digest) vs MVCA Study 1 rating d_z (fig5A); latent silicon-
    sample response d_z vs MVCA Study 1; latent V4b own push (L*) and V6b own push (mid depth) vs MVCA Study 2 induced-mediator d_z (fig5B);
    latent shift vs the share of answers in which a separate LLM identified the experiment from scenario + question (identification_test).
Usage: python analysis_v6/cross_paper_v6.py -> analysis_v6/cross_paper_v6.json, study3_rank_v6.csv"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
HERE = Path(__file__).resolve().parent; BASE = HERE.parent; RV = BASE / "results_v6"
REL = BASE / "external" / "agent-psychonomicus" / "data"   # REPOSITORY: was BASE.parent / "release_2026-10-15" / ...
E = json.load(open(BASE / "materials_v6/upload_v6/experiments_v6.json", encoding="utf-8"))
RUN1 = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
OUT = {"rank": {}, "corr": {}}; RK = []
for name, job in RUN1.items():
    I = pd.read_csv(RV / job / "V3_exp/items.csv"); P = torch.load(RV / job / "V3_exp/proj_v3.pt", weights_only=False)
    cons = P["constructs"]; mi = P["marks"].index("body"); gi = 0; X = P["std"][:, mi, gi, :].float().numpy()
    ok = (I.mass >= .5).values; I = I.assign(r=np.arange(len(I)))[ok]
    shifts = {}
    for eid, g in I.groupby("exp"):
        k = ["vid", "level", "agent"]; t = g[g.arm == "treatment"].set_index(k).r; c = g[g.arm == "control"].set_index(k).r; t, c = t.align(c, join="inner")
        if len(t) < 3: continue
        d = X[t.values] - X[c.values]; shifts[eid] = d.mean(0) / (d.std(0, ddof=1) + 1e-9)
    S = pd.DataFrame(shifts, index=cons).T; Sc = S - S.mean(0)
    for eid in S.index:
        own = E[eid]["construct"] if E[eid].get("construct") in cons else I[I.exp == eid].construct.iloc[0]
        RK.append(dict(model=name, exp=eid, own=own, rank_raw=int((S.loc[eid] >= S.loc[eid, own]).sum()), rank_centred=int((Sc.loc[eid] >= Sc.loc[eid, own]).sum())))
    R = pd.DataFrame([r for r in RK if r["model"] == name])
    OUT["rank"][name] = {"n": len(R), "first_centred": int((R.rank_centred == 1).sum()), "top5_centred": int((R.rank_centred <= 5).sum()), "median_centred": float(R.rank_centred.median()),
                         "first_raw": int((R.rank_raw == 1).sum()), "top5_raw": int((R.rank_raw <= 5).sum()), "median_raw": float(R.rank_raw.median()), "chance_first": round(len(R) / 66, 2)}
pd.DataFrame(RK).to_csv(HERE / "study3_rank_v6.csv", index=False)
# MVCA per-experiment tables
f5a = pd.read_csv(REL / "figures/fig5A_study1_rating.csv").set_index("experiment_id").d_z
f5b = pd.read_csv(REL / "figures/fig5B_study2_induced_mediator.csv").set_index("experiment_id").d_z
f6c = pd.read_csv(REL / "figures/fig6C_own_mediator_rank.csv").set_index("experiment_id").own_mediator_rank
idt = pd.read_csv(REL / "identification_test.csv"); idt = idt[idt.input.str.contains("scenario", case=False)].groupby("experiment").same_experiment.mean()
def digest_col(job, step, col):
    f = RV / job / step / "digest.md"
    if not f.exists(): return pd.Series(dtype=float)
    rows = [l for l in open(f, encoding="utf-8").read().splitlines() if l.startswith("| ") and not l.startswith("|---")]
    hdr = [h.strip() for h in rows[0].strip("|").split("|")]; D = pd.DataFrame([[c.strip() for c in r.strip("|").split("|")] for r in rows[1:]], columns=hdr)
    return pd.to_numeric(D.set_index(hdr[0])[col], errors="coerce")
V3J = json.load(open(HERE / "v3_all_models.json")) if (HERE / "v3_all_models.json").exists() else {}
RESP = pd.read_csv(HERE / "v3_all_models.csv") if (HERE / "v3_all_models.csv").exists() else None
def sp(a, b):
    a, b = a.align(b, join="inner"); m = a.notna() & b.notna(); a, b = a[m], b[m]
    if len(a) < 8: return None
    r, p = stats.spearmanr(a, b); return {"rho": round(float(r), 3), "p": round(float(p), 4), "n": int(len(a))}
ALL = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32", "Gemma 4 26B-A4B": "V720_gemma4_26b_a4b", "Qwen3.5 27B": "V830_v3sweep_qwen35_27b",
       "Qwen3.5 9B": "V831_v3sweep_qwen35_9b", "Qwen3.5 4B": "V832_v3sweep_qwen35_4b", "Qwen3.5 2B": "V833_v3sweep_qwen35_2b", "Qwen3.5 0.8B": "V834_v3sweep_qwen35_08b",
       "Gemma 4 E4B": "V835_v3sweep_gemma4_e4b", "OLMo 3 7B Instruct": "V836_v3sweep_olmo7", "Qwen3.8 27B": "V837_v3sweep_qwen38", "gpt-oss 20B": "V838_v3sweep_gptoss20"}
SH = {}
for name, job in ALL.items():
    sh = digest_col(job, "V3_exp", "own - others (situation, band 1) d_z")
    if sh.empty: continue
    SH[name] = sh; c = {"shift_vs_mvca_study1": sp(sh, f5a), "shift_vs_identification": sp(sh, idt)}
    if RESP is not None:
        rr = RESP[RESP.model == name].set_index("exp").dz; c["response_vs_mvca_study1"] = sp(rr, f5a); c["shift_vs_response"] = sp(sh, rr)
    if name in RUN1:
        c["v4b_own_push_vs_mvca_study2"] = sp(digest_col(job, "V4b_steer", "own push"), f5b)
    OUT["corr"][name] = c
# mean over the three run-1 models
M3 = pd.concat([SH[n] for n in RUN1], axis=1).mean(1); OUT["corr"]["mean of 3 main models"] = {"shift_vs_mvca_study1": sp(M3, f5a), "shift_vs_identification": sp(M3, idt)}
MA = pd.concat(list(SH.values()), axis=1).mean(1); OUT["corr"]["mean of all models"] = {"shift_vs_mvca_study1": sp(MA, f5a), "shift_vs_identification": sp(MA, idt)}
for nm, job in {"Qwen3.6 27B": "V810_steer2_qwen36", "Gemma 4 31B": "V811_steer2_gemma4_31b", "OLMo 3.1 32B": "V812_steer2_olmo32"}.items():
    f = RV / job / "V6b_steer2"
    if f.exists(): OUT["corr"].setdefault(nm, {})["v6b_note"] = "V6b per-experiment table available"
# rank agreement with MVCA
R = pd.DataFrame(RK); OUT["rank_vs_mvca"] = {n: sp(R[R.model == n].set_index("exp").rank_centred, f6c) for n in RUN1}
OUT["mvca_first"] = int((f6c == 1).sum())
json.dump(OUT, open(HERE / "cross_paper_v6.json", "w"), indent=1, default=str); print(json.dumps(OUT, indent=1, default=str))
