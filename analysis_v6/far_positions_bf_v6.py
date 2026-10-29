"""Psych Science version (22 Oct 2026): Bayes factors (BF01, uniform prior; as in psychsci_stats_v6.py) for the correlations across the
60 experiments between the activation of the mediator's representation and the manipulation's effect on the response, at the
positions far from the response (passages that convey the manipulation, the scenario) at both depths, in the three main LLMs; and the
equivalence test (90% bootstrap CI of the mean rho over the three LLMs and four cells within +-.20). Also binomial tests for the induced
mediator's representation vs other constructs' (Study 4). Writes far_positions_bf_v6.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, special, integrate
A = Path(__file__).resolve().parent; rng = np.random.default_rng(5)
def bf01(r, n):
    def lik(rho): return (1 - rho ** 2) ** ((n - 1) / 2) * (1 - rho * r) ** (-(n - 1.5)) * special.hyp2f1(.5, .5, n - .5, (1 + rho * r) / 2)
    return float(lik(0.0) / integrate.quad(lambda p: lik(p) * .5, -1, 1, limit=200)[0])
ex = pd.read_csv(A / "example_s3_v6.csv"); models = list(ex.model.unique()); E = {m: ex[ex.model == m].set_index("exp") for m in models}
common = sorted(set.intersection(*[set(E[m].index) for m in models])); cells = ["x_B1", "x_B2", "body_B1", "body_B2"]
out = {"cells": {}}
for m in models:
    g = E[m].loc[common]
    for c in cells:
        r = stats.spearmanr(g[c], g.answer_dz)[0]; out["cells"][f"{m} {c}"] = {"rho": round(float(r), 3), "BF01": round(bf01(r, len(g)), 2)}
def mean_rho(idx): return np.mean([stats.spearmanr(E[m].loc[idx, c], E[m].loc[idx, "answer_dz"])[0] for m in models for c in cells])
bs = [mean_rho(list(rng.choice(common, len(common)))) for _ in range(2000)]
ci90 = [round(float(np.quantile(bs, .05)), 3), round(float(np.quantile(bs, .95)), 3)]
out["pooled_far"] = {"mean_rho": round(float(mean_rho(common)), 3), "ci90": ci90, "equivalent_20": bool(-.2 < ci90[0] and ci90[1] < .2),
                     "ci95": [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)]}
bfs = [v["BF01"] for v in out["cells"].values()]; out["BF01_range"] = [min(bfs), max(bfs)]
out["induced_vs_other_responses"] = {k: {"k": v, "n": 52, "p_one_sided": round(float(stats.binomtest(v, 52, .5, alternative="greater").pvalue), 3)}
                                     for k, v in (("Qwen3.6 27B", 31), ("Gemma 4 31B", 25), ("OLMo 3.1 32B", 20))}
json.dump(out, open(A / "far_positions_bf_v6.json", "w", encoding="utf-8"), indent=1); print(json.dumps(out, indent=1))
