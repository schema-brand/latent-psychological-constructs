"""Psych Science version (22 Oct 2026), Study 4: Bayes factors for the specificity of the induced mediator's representation's effect on
the responses (70% of depth, scenario to end of question). k = experiments (of 52) in which the mediator's representation had the larger
effect on the response than other constructs' representations (Table S6 Panel B: 31, 25, 20 in Qwen3.6 27B, Gemma 4 31B, OLMo 3.1 32B).
H0: rate = .5 (no specificity); H1: rate ~ Uniform(.5, 1) (the mediator's representation has the larger effect more often than
chance). BF01 = binom(k; 52, .5) / E_H1[binom(k; 52, p)]. Also the binomial test for the judgments in the experiments (own > others,
Table S6 Panel B: 34, 32, 39). Writes study4_specificity_bf_v6.json."""
import json
from pathlib import Path
from scipy import stats, integrate
out = {"responses_own_gt_others": {}, "judgments_own_gt_others": {}}
for m, k in (("Qwen3.6 27B", 31), ("Gemma 4 31B", 25), ("OLMo 3.1 32B", 20)):
    l0 = stats.binom.pmf(k, 52, .5); l1 = 2 * integrate.quad(lambda p: stats.binom.pmf(k, 52, p), .5, 1)[0]
    out["responses_own_gt_others"][m] = {"k": k, "n": 52, "BF01_one_sided": round(l0 / l1, 2), "p_one_sided": round(float(stats.binomtest(k, 52, .5, alternative="greater").pvalue), 3)}
for m, k in (("Qwen3.6 27B", 34), ("Gemma 4 31B", 32), ("OLMo 3.1 32B", 39)):
    out["judgments_own_gt_others"][m] = {"k": k, "n": 52, "p_one_sided": round(float(stats.binomtest(k, 52, .5, alternative="greater").pvalue), 4)}
json.dump(out, open(Path(__file__).with_name("study4_specificity_bf_v6.json"), "w", encoding="utf-8"), indent=1); print(json.dumps(out, indent=1))
