"""Exact tests for the count statistics of the draft (reviewer request): one-sided Fisher's exact tests by replication outcome, separately
for the effect on the mediator's representation (shift > 0) and for the answer (original effect significant), per LLM; exact binomial
tests for 'k of n' counts against their chance rate. Usage: python analysis_v6/exact_tests_v6.py -> analysis_v6/exact_tests_v6.json"""
import json
from pathlib import Path
from scipy import stats
HERE = Path(__file__).resolve().parent
A = json.load(open(HERE / "v3_all_models.json", encoding="utf-8")); S = json.load(open(HERE / "v3_shift_by_repl.json", encoding="utf-8"))
fr = lambda s: tuple(int(x) for x in s.split("/"))
OUT = {"by_replication": {}, "binomial": {}}
for m in A:
    if m not in S or "repl_sig" not in A[m]: continue
    (a, n1), (b, n2) = fr(S[m]["shift_pos_repl"]), fr(S[m]["shift_pos_fail"])
    p_shift = stats.fisher_exact([[a, n1 - a], [b, n2 - b]], alternative="greater")[1]
    (c, _), (d, _) = fr(A[m]["repl_sig"]), fr(A[m]["fail_sig"])
    p_ans = stats.fisher_exact([[c, n1 - c], [d, n2 - d]], alternative="greater")[1]
    OUT["by_replication"][m] = {"shift_pos": f"{a}/{n1} vs {b}/{n2}", "p_shift": round(p_shift, 4), "answer_sig": f"{c}/{n1} vs {d}/{n2}", "p_answer": round(p_ans, 4)}
    print(m, OUT["by_replication"][m])
def binom(k, n, p, key): OUT["binomial"][key] = {"k": k, "n": n, "chance": p, "p_one_sided": float(f"{stats.binomtest(k, n, p, alternative='greater').pvalue:.3g}")}; print(key, OUT["binomial"][key])
for k in (42, 49): binom(k, 60, .5, f"study3_own_gt_others_{k}of60")
for k in (5, 6): binom(k, 60, 1 / 66, f"study3_rank1_{k}of60")
for k in (13, 20): binom(k, 60, 5 / 66, f"study3_top5_{k}of60")
for k in (30, 31, 32, 39, 41, 45): binom(k, 52, .5, f"study4_answer_{k}of52")
for k in (45, 48, 55, 56, 59): binom(k, 66, .5, f"study4_causal_mtmm_{k}of66")
for k in (2, 3, 5): binom(k, 7, .5, f"study4_moderation_{k}of7")
for k in (25,): binom(k, 28, .5, f"study5_projection_{k}of28")
json.dump(OUT, open(HERE / "exact_tests_v6.json", "w"), indent=1)
