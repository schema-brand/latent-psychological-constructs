"""Psych Science version (22 Oct 2026): activation of the mediator's representation in the 28 experiments that large replications
re-examined (14 replicated, 14 failed), 14 LLMs. (1) Activation (own-minus-others d_z, scenario, layers 30-50%) in the failed and in
the replicated experiments: per LLM, the number of experiments with activation > 0 and with activation above the median of the other
constructs (as in Fig. 4c), and pooled over LLMs (per-experiment mean over the 14 LLMs) with a bootstrap CI over experiments.
(2) Whether the LLMs' responses distinguish replicated from failed effects more strongly than the activation does: difference between
the two pooled rank-biserial correlations (z-scored within LLM, as in psychsci_stats_v6.py), bootstrap over experiments within group.
(3) Equivalence test (90% CI within +-.20) and BF01 for the activation's rank-biserial correlation. Writes repl_activation_v6.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
A = Path(__file__).resolve().parent; rng = np.random.default_rng(11); B = 5000
sb = pd.read_csv(A / "v3_shift_by_repl.csv"); s = sb[sb.repl.isin(["replicated", "failed"])].copy()
q = lambda v, lo=.025, hi=.975: [round(float(np.nanquantile(v, lo)), 3), round(float(np.nanquantile(v, hi)), 3)]
out = {"per_llm": {}}
for m, g in s.groupby("model"):
    r = {}
    for k in ("replicated", "failed"):
        v = g[g.repl == k].shift_dz.dropna().values
        r[k] = {"n": int(len(v)), "activation_gt0": int((v > 0).sum()), "mean_dz": round(float(v.mean()), 3),
                "binom_p_one_sided_gt0": round(float(stats.binomtest(int((v > 0).sum()), len(v), .5, alternative="greater").pvalue), 4)}
    out["per_llm"][m] = r
pe = s.groupby(["exp", "repl"]).shift_dz.mean().reset_index()
for k in ("replicated", "failed"):
    v = pe[pe.repl == k].shift_dz.values
    out[f"pooled_{k}"] = {"n": int(len(v)), "mean_dz": round(float(v.mean()), 3), "ci": q([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]),
                          "experiments_gt0": int((v > 0).sum()), "wilcoxon_p_one_sided": round(float(stats.wilcoxon(v, alternative="greater").pvalue), 4)}
for col in ("shift_dz", "dz"): s[col + "_z"] = s.groupby("model")[col].transform(lambda v: (v - v.mean()) / v.std())
pz = s.groupby(["exp", "repl"])[["shift_dz_z", "dz_z"]].mean().reset_index()
ra, fa = pz[pz.repl == "replicated"], pz[pz.repl == "failed"]
def rb(a, b): return 2 * stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1
est = {c: rb(ra[c].values, fa[c].values) for c in ("shift_dz_z", "dz_z")}
bs = []
for _ in range(B):
    i = rng.integers(0, len(ra), len(ra)); j = rng.integers(0, len(fa), len(fa))
    bs.append((rb(ra.shift_dz_z.values[i], fa.shift_dz_z.values[j]), rb(ra.dz_z.values[i], fa.dz_z.values[j])))
bs = np.array(bs); d = bs[:, 1] - bs[:, 0]
out["response_minus_activation_rank_biserial"] = {"activation": round(est["shift_dz_z"], 3), "response": round(est["dz_z"], 3),
    "difference": round(est["dz_z"] - est["shift_dz_z"], 3), "ci": q(d), "p_one_sided_le0": round(float(np.mean(d <= 0)), 4)}
ci90 = q(bs[:, 0], .05, .95)
out["activation_equivalence_20"] = {"ci90": ci90, "equivalent": bool(-.2 < ci90[0] and ci90[1] < .2)}
json.dump(out, open(A / "repl_activation_v6.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_llm"}, indent=1))
for m, r in out["per_llm"].items(): print(m, r["replicated"]["activation_gt0"], r["failed"]["activation_gt0"], r["failed"]["binom_p_one_sided_gt0"])
