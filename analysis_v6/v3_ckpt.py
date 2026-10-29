"""Study 3 across the training checkpoints of OLMo 3 7B (V844-V84c, V836 = Instruct); otherwise identical to v3_all_models.py. Study 3 across all models with V3 results: (a) the silicon-sample response effect of each experiment (treatment - control on the 1-9
answer, in MVCA's predicted direction x_to_y_sign; items with digit mass >= .5; paired by variant x level x agent), (b) the own-minus-other
shift count from each digest, (c) the response effect by replication outcome (replicated vs not, the 28 experiments with large
preregistered replications), as in MVCA's replication benchmark.
Usage: python analysis_v6/v3_all_models.py -> analysis_v6/v3_ckpt.json / .csv"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent; BASE = HERE.parent; RV = BASE / "results_v6"
E = json.load(open(BASE / "materials_v6/upload_v6/experiments_v6.json", encoding="utf-8"))
sg = lambda s: int(float(str(s).replace("+", ""))) if str(s).strip() not in ("", "None") else 0
JOBS = {"step 1k": "V844_v3_olmo7_stage1_step1000", "step 3k": "V845_v3_olmo7_stage1_step3000", "step 10k": "V846_v3_olmo7_stage1_step10000",
        "step 30k": "V847_v3_olmo7_stage1_step30000", "step 300k": "V848_v3_olmo7_stage1_step300000", "step 1.41M": "V849_v3_olmo7_stage1_step1413814",
        "base (end of training)": "V84a_v3_olmo7_main", "SFT": "V84b_v3_olmo7_sft", "DPO": "V84c_v3_olmo7_dpo", "Instruct": "V836_v3sweep_olmo7"}
rows, summ = [], {}
for name, job in JOBS.items():
    f = RV / job / "V3_exp" / "items.csv"
    if not f.exists(): continue
    I = pd.read_csv(f); I = I[I.mass >= .5]
    dg = open(RV / job / "V3_exp" / "digest.md", encoding="utf-8").read(); m = re.search(r"own shift > 0 in (\d+)/(\d+)", dg) or re.search(r"> 0 in (\d+)/(\d+)", dg)
    for eid, g in I.groupby("exp"):
        s = sg(E[eid]["x_to_y_sign"]) or 1; k = ["vid", "level", "agent"]
        t = g[g.arm == "treatment"].set_index(k).ev; c = g[g.arm == "control"].set_index(k).ev; t, c = t.align(c, join="inner")
        d = (t - c) * s; n = len(d)
        if n < 3: continue
        tt = d.mean() / (d.std(ddof=1) / np.sqrt(n)) if d.std() > 0 else np.nan
        rows.append(dict(model=name, exp=eid, n=n, mean=d.mean(), dz=d.mean() / d.std(ddof=1) if d.std() > 0 else np.nan, t=tt, sig=bool(tt > 2) if tt == tt else False,
                         repl=E[eid].get("replication_status") or ""))
    D = pd.DataFrame([r for r in rows if r["model"] == name])
    rep = D[D.repl.str.lower().str.startswith("repl")]; fail = D[D.repl.str.lower().isin(["failed", "not replicated", "not_replicated", "fail"])]
    summ[name] = dict(own_shift_pos=f"{m.group(1)}/{m.group(2)}" if m else None, response_pos=f"{int((D.dz > 0).sum())}/{len(D)}", response_sig=int(D.sig.sum()),
                      repl_sig=f"{int(rep.sig.sum())}/{len(rep)}", fail_sig=f"{int(fail.sig.sum())}/{len(fail)}",
                      fisher_p_one_sided=round(float(stats.fisher_exact([[rep.sig.sum(), len(rep) - rep.sig.sum()], [fail.sig.sum(), len(fail) - fail.sig.sum()]], alternative="greater")[1]), 3) if len(rep) and len(fail) else None)
pd.DataFrame(rows).to_csv(HERE / "v3_ckpt.csv", index=False); json.dump(summ, open(HERE / "v3_ckpt.json", "w"), indent=1)
print(pd.DataFrame(summ).T.to_string()); print(sorted({r["repl"] for r in rows}))
# ---- own-minus-other shift (digest column) by replication outcome; combined diagnostic ----
SH = []
for name, job in JOBS.items():
    f = RV / job / "V3_exp" / "digest.md"
    if not f.exists(): continue
    for l in open(f, encoding="utf-8"):
        mm = re.match(r"\| (\w+) \| (-?[\d.]+|nan) \| (-?[\d.]+|nan) \|", l)
        if mm and mm.group(1) in E: SH.append(dict(model=name, exp=mm.group(1), shift_dz=float(mm.group(3)), repl=E[mm.group(1)].get("replication_status") or ""))
S2 = pd.DataFrame(SH).merge(pd.DataFrame(rows)[["model", "exp", "sig", "dz"]], on=["model", "exp"], how="left")
S2.to_csv(HERE / "v3_ckpt_shift.csv", index=False)
out = {}
for name, g in S2.groupby("model", sort=False):
    r, f_ = g[g.repl == "replicated"], g[g.repl == "failed"]
    u = stats.mannwhitneyu(r.shift_dz, f_.shift_dz, alternative="greater") if len(r) and len(f_) else None
    both_r, both_f = int(((r.shift_dz > 0) & r.sig).sum()), int(((f_.shift_dz > 0) & f_.sig).sum())
    out[name] = dict(shift_median_repl=round(float(r.shift_dz.median()), 2), shift_median_fail=round(float(f_.shift_dz.median()), 2), mw_p=round(float(u.pvalue), 3) if u else None,
                     shift_pos_repl=f"{int((r.shift_dz > 0).sum())}/{len(r)}", shift_pos_fail=f"{int((f_.shift_dz > 0).sum())}/{len(f_)}",
                     resp_sig_and_shift_pos_repl=f"{both_r}/{len(r)}", resp_sig_and_shift_pos_fail=f"{both_f}/{len(f_)}",
                     fisher_p_combined=round(float(stats.fisher_exact([[both_r, len(r) - both_r], [both_f, len(f_) - both_f]], alternative="greater")[1]), 3))
json.dump(out, open(HERE / "v3_ckpt_shift.json", "w"), indent=1); print(pd.DataFrame(out).T.to_string())

# shift vs response across experiments per checkpoint
cor = {}
for name, g in S2.groupby("model", sort=False):
    g = g.dropna(subset=["dz", "shift_dz"])
    if len(g) > 5: r_, p_ = stats.spearmanr(g.shift_dz, g.dz); cor[name] = {"rho_shift_response": round(float(r_), 3), "p": round(float(p_), 3), "n": len(g)}
json.dump(cor, open(HERE / "v3_ckpt_corr.json", "w"), indent=1); print(cor)
