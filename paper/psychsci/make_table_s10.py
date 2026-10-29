"""Table S10 of the Psychological Science version (22 Oct 2026): estimates, 95% confidence intervals, equivalence tests and Bayes
factors for the main results of Studies 1-4, by study and LLM.

Reads the analysis files (never enters a number by hand):
  analysis_v6/psychsci_stats_v6.json  (psychsci_stats_v6.py)   CIs, BF01, equivalence, family-wise p, ceiling rule
  analysis_v6/repl_activation_v6.json (repl_activation_v6.py)  activation in replicated vs failed experiments
  analysis_v6/far_positions_bf_v6.json (far_positions_bf_v6.py) BF01 far from the response; induced vs other constructs
  analysis_v6/removal_ci_v6.json      (removal_ci_v6.py)       removal shares with CIs
  analysis_v6/qcoding_v6.json         (qcoding_v6.py)          question coding (Supplemental Note)
  analysis_v6/structure_v6.json       within/between-process similarities and their permutation p (Study 2)
  analysis_v6/criterion_v6.json       permutation p of the criterion correlations (Study 1)
Every number in the Estimate, 95% CI and Test columns is a formatted JSON value; an audit at the end asserts this, and the keys
used are listed in an HTML comment at the end of the table.

Rounding: half up on the JSON value. When the JSON value lies exactly on the rounding boundary (e.g. 1.125 to two decimals), the
unrounded value is recomputed from the analysis's own input file with the source script's formula, checked against the JSON value,
and used only to decide the direction; boundary values that cannot be recomputed this way are rounded half up and listed in the check.

Also checks every number of PS_DRAFT_v2.md's Studies 1-4 that the table reports (and the question-coding sentence of PS_SM_v2.md).
Usage: python draft_2026-10-21/psychsci/make_table_s10.py  ->  TABLE_S10.md, TABLE_S10_CHECK.md (same folder)
"""
import json, re
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path

PS = Path(__file__).resolve().parent; LAT = PS.parent.parent; A = LAT / "analysis_v6"; RV = LAT / "results_v6"
FILES = {"stats": "psychsci_stats_v6.json", "repl": "repl_activation_v6.json", "far": "far_positions_bf_v6.json",
         "rem": "removal_ci_v6.json", "qc": "qcoding_v6.json", "struct": "structure_v6.json", "crit": "criterion_v6.json"}
J = {k: json.load(open(A / f, encoding="utf-8")) for k, f in FILES.items()}
MAIN = ["Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B"]
JOB = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
MINUS = "−"; DASH = "—"; EN = "–"
USED = []          # JSON keys used, in order
TOKENS = set()     # number tokens produced by the formatters (audit)
TIES = []          # boundary values: (key, json value, unrounded or None, printed)


# ------------------------------------------------------------------ access to JSON values
class V:
    def __init__(self, x, key): self.x, self.key = x, key
    def __getitem__(self, i): return V(self.x[i], f"{self.key}[{i}]")


def g(f, *path):
    x = J[f]
    for p in path:
        assert p in x, f"missing key {FILES[f]}:{'/'.join(path)}"
        x = x[p]
    key = f"{FILES[f]}:{'/'.join(path)}"
    if key not in USED: USED.append(key)
    return V(x, key)


# ------------------------------------------------------------------ recomputation of boundary values (direction of rounding only)
@lru_cache(None)
def _ex():
    import pandas as pd
    ex = pd.read_csv(A / "example_s3_v6.csv"); E = {m: ex[ex.model == m].set_index("exp") for m in ex.model.unique()}
    common = sorted(set.intersection(*[set(e.index) for e in E.values()])); return E, common


@lru_cache(None)
def _sb():
    import pandas as pd
    return pd.read_csv(A / "v3_shift_by_repl.csv")


def _spearman(a, b):
    from scipy import stats
    return float(stats.spearmanr(a, b)[0])


@lru_cache(None)
def _stat(key):
    """Recompute the statistic behind a JSON key with the source script's formula, or None."""
    import pandas as pd
    m = re.fullmatch(r"psychsci_stats_v6\.json:study3/(?:positions|familywise)/(.+)/(\w+_B\d)/rho", key) or \
        re.fullmatch(r"far_positions_bf_v6\.json:cells/(.+) (\w+_B\d)/rho", key)
    if m:
        E, common = _ex(); e = E[m.group(1)].loc[common]; return _spearman(e[m.group(2)], e.answer_dz)
    m = re.fullmatch(r"psychsci_stats_v6\.json:study3_positions_with_ceiling_rule/(.+)/(\w+_B\d)", key)
    if m:
        E, common = _ex(); c = pd.read_csv(RV / JOB[m.group(1)] / "V3_exp" / "ceiling_check_v2.csv")
        drop = set(c.loc[c.blocked > .30, "exp"]); e = E[m.group(1)].loc[[x for x in common if x not in drop]]
        return _spearman(e[m.group(2)], e.answer_dz)
    m = re.fullmatch(r"psychsci_stats_v6\.json:study3/per_llm_scenario/(.+)/rho", key)
    if m:
        s = _sb(); s = s[s.model == m.group(1)].dropna(subset=["shift_dz", "dz"]); return _spearman(s.shift_dz, s.dz)
    m = re.fullmatch(r"psychsci_stats_v6\.json:study3/activation_by_llm/(.+)/mean_dz", key)
    if m:
        s = _sb(); return float(s[s.model == m.group(1)].shift_dz.dropna().mean())
    m = re.fullmatch(r"repl_activation_v6\.json:per_llm/(.+)/(replicated|failed)/mean_dz", key)
    if m:
        s = _sb(); return float(s[(s.model == m.group(1)) & (s.repl == m.group(2))].shift_dz.dropna().mean())
    m = re.fullmatch(r"repl_activation_v6\.json:pooled_(replicated|failed)/mean_dz", key)
    if m:
        s = _sb(); s = s[s.repl.isin(["replicated", "failed"])]; pe = s.groupby(["exp", "repl"]).shift_dz.mean().reset_index()
        return float(pe[pe.repl == m.group(1)].shift_dz.mean())
    m = re.fullmatch(r"qcoding_v6\.json:(\w+)/rho", key)
    if m:
        # REPOSITORY: the question codes are read from question_coding/ (was PS / "qcoding")
        Q = LAT / "question_coding"; c1 = json.load(open(Q / "coder1.json", encoding="utf-8")); c2 = json.load(open(Q / "coder2.json", encoding="utf-8"))
        ids = sorted(c1); score = pd.Series([(c1[i]["code"] + c2[i]["code"]) / 2 for i in ids], index=ids, name="score")
        ex = pd.read_csv(A / "example_s3_v6.csv"); d = ex.groupby("exp")[["x_B1", "body_B1", "q_B1", "q_B2", "dec_B1", "answer_dz"]].mean().join(score, how="inner")
        return _spearman(d.score, d[m.group(1)])
    m = re.fullmatch(r"psychsci_stats_v6\.json:study1_criterion/(.+)/r", key)
    if m:   # criterion_v6.py without its permutation null
        import numpy as np, torch
        job = RV / JOB[m.group(1)]
        D = pd.read_csv(job / "V4a_causal/causal_v4a.csv"); N = D[(D.part == "natural") & (D.mass >= .5)]
        w = N.pivot_table(index=["pid", "agent", "j"], columns="side", values="ev").dropna(); w["y"] = w.high - w.low
        Y = w.y.groupby(["pid", "j"]).mean().reset_index()
        L = torch.load(job / "V1_read/lowo_cos_v6.pt", weights_only=False); cons = L["constructs"]; li = L["layers"].index("B1")
        meta = pd.read_csv(job / "V1_read/pairs_meta.csv"); row = {p: i for i, p in enumerate(meta.pid)}
        Y = Y[Y.pid.isin(row) & Y.j.isin(cons)].copy(); Y["x"] = [float(L["proj"][row[p], li, cons.index(j)]) for p, j in zip(Y.pid, Y.j)]
        def centre(s, df): return s - df.groupby("pid")[s.name].transform("mean") - df.groupby("j")[s.name].transform("mean") + s.mean()
        return float(np.corrcoef(centre(Y.x, Y), centre(Y.y, Y))[0, 1])
    return None


def _fz(r, n, level=.95):   # psychsci_stats_v6.py fz_ci, unrounded
    import numpy as np
    from scipy import stats
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); s = stats.norm.ppf(.5 + level / 2); return [float(np.tanh(z - s * se)), float(np.tanh(z + s * se))]


def _json_at(key):
    f, path = key.split(":", 1); x = J[{v: k for k, v in FILES.items()}[f]]
    for part in re.findall(r"[^/\[\]]+|\[\d\]", path):
        x = x[int(part[1])] if part.startswith("[") else x[part]
    return x


def _verified(out, nd):
    """Keep a replication only if it reproduces every JSON value it covers (rounded as in the source script)."""
    return out if all(round(u, nd) == _json_at(k) for k, u in out.items()) else {}


@lru_cache(None)
def _removal():   # removal_ci_v6.py: same rules, seed and order of random draws; unrounded
    import numpy as np, pandas as pd
    def grp(c):
        w = ("__" + c.split("__")[1]) if "__" in c else ""
        for p_ in ("near_spec_k10", "rand_spec_k10", "unif"):
            if c.startswith(p_): return p_ + w
        return c
    rng_ = np.random.default_rng(3); out = {}
    for name, job in JOB.items():
        f = RV / job / "V4c_remove" / "remove_v4c.csv"
        if not f.exists(): continue
        D = pd.read_csv(f, usecols=["exp", "cond", "vid", "agent", "arm", "ev", "mass", "x_to_y_sign"]); k = ["exp", "cond", "vid", "agent"]
        W = D.pivot_table(index=k, columns="arm", values=["ev", "mass"]).reset_index(); W.columns = ["_".join(c).strip("_") for c in W.columns]
        sg = D.groupby("exp").x_to_y_sign.first(); W["eff"] = (W.ev_treatment - W.ev_control) * W.exp.map(sg)
        W["ok"] = (W.mass_treatment >= .5) & (W.mass_control >= .5)
        E = W[W.ok].groupby(["exp", "cond"]).eff.mean().reset_index(); E["grp"] = E.cond.map(grp); E = E.groupby(["exp", "grp"]).eff.mean().unstack()
        big = E[E["none"].abs() > .3]; S_ = big.div(big["none"], axis=0)
        for col in S_.columns:
            if col == "none": continue
            v = S_[col].dropna().values
            if len(v) < 5: continue
            bs = [np.median(v[rng_.integers(0, len(v), len(v))]) for _ in range(5000)]
            key = f"removal_ci_v6.json:{name}/{col}"
            out[key + "/median"] = float(np.median(v)); out[key + "/ci[0]"] = float(np.quantile(bs, .025)); out[key + "/ci[1]"] = float(np.quantile(bs, .975))
    return _verified(out, 3)


@lru_cache(None)
def _repl():      # repl_activation_v6.py: its first two bootstraps (pooled replicated, pooled failed), same seed and order; unrounded
    import numpy as np
    rng_ = np.random.default_rng(11); s = _sb(); s = s[s.repl.isin(["replicated", "failed"])]
    pe = s.groupby(["exp", "repl"]).shift_dz.mean().reset_index(); out = {}
    for k in ("replicated", "failed"):
        v = pe[pe.repl == k].shift_dz.values; bs = [v[rng_.integers(0, len(v), len(v))].mean() for _ in range(5000)]
        out[f"repl_activation_v6.json:pooled_{k}/ci[0]"] = float(np.nanquantile(bs, .025))
        out[f"repl_activation_v6.json:pooled_{k}/ci[1]"] = float(np.nanquantile(bs, .975))
    return _verified(out, 3)


def unrounded(key):
    u = _stat(key)
    if u is not None: return u
    m = re.fullmatch(r"(psychsci_stats_v6\.json:study3/positions/.+/\w+_B\d)/ci\[(\d)\]", key)
    if m:
        _, common = _ex(); return _fz(_stat(m.group(1) + "/rho"), len(common))[int(m.group(2))]
    m = re.fullmatch(r"(psychsci_stats_v6\.json:study3/per_llm_scenario/(.+))/(ci|ci90)\[(\d)\]", key)
    if m:
        return _fz(_stat(m.group(1) + "/rho"), J["stats"]["study3"]["per_llm_scenario"][m.group(2)]["n"], .95 if m.group(3) == "ci" else .90)[int(m.group(4))]
    m = re.fullmatch(r"(psychsci_stats_v6\.json:study3/per_llm_scenario/(.+))/equivalence_bound_reached", key)
    if m:   # the script takes the bound from the 90% interval rounded to four decimals
        lo, hi = _fz(_stat(m.group(1) + "/rho"), J["stats"]["study3"]["per_llm_scenario"][m.group(2)]["n"], .90)
        return max(abs(round(lo, 4)), abs(round(hi, 4)))
    m = re.fullmatch(r"(psychsci_stats_v6\.json:study1_criterion/(.+))/ci\[(\d)\]", key)
    if m:
        return _fz(_stat(m.group(1) + "/r"), J["stats"]["study1_criterion"][m.group(2)]["n_pairs"])[int(m.group(3))]
    if re.fullmatch(r"psychsci_stats_v6\.json:study3/familywise/.+/p_fwe_(?:within_llm|all24)", key):
        x = _json_at(key); k = round(x * 5001)   # permutation p = (count + 1) / 5001 exactly (NP = 5000); unique at four decimals
        assert round(k / 5001, 4) == x and round((k - 1) / 5001, 4) != x and round((k + 1) / 5001, 4) != x
        return k / 5001
    if key.startswith("removal_ci_v6.json:"): return _removal().get(key)
    if key.startswith("repl_activation_v6.json:pooled_"): return _repl().get(key)
    return None


# ------------------------------------------------------------------ APA formatting (each call registers its number tokens)
NUM = re.compile(r"(?<![A-Za-z0-9.−-])[−-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)%?")


def _reg(s):
    TOKENS.update(NUM.findall(s)); return s


def _q(v, nd, scale=1):
    d = Decimal(repr(float(v.x))) * scale; unit = Decimal(1).scaleb(-nd)
    if (abs(d) / unit) % 1 == Decimal("0.5"):                       # exactly on the boundary
        u = unrounded(v.key)
        if u is not None:
            assert abs(u - float(v.x)) < 5.0001e-4, (v.key, u, v.x)  # recomputation reproduces the JSON value
            r = (Decimal(repr(u)) * scale).quantize(unit, ROUND_HALF_UP)
        else:
            r = d.quantize(unit, ROUND_HALF_UP)
        TIES.append((v.key, v.x, u, f"{r}%" if scale == 100 else str(r)))
        return r
    return d.quantize(unit, ROUND_HALF_UP)


def _signed(r, body):
    return (MINUS if r < 0 else "") + body


def fr(v, nd=2):          # bounded by 1 (correlations, kappa, rank-biserial): no leading zero
    r = _q(v, nd); return _reg(_signed(r, f"{abs(r):.{nd}f}".lstrip("0") if abs(r) < 1 else f"{abs(r):.{nd}f}"))


def fd(v, nd=2):          # unbounded (d_z, BF01)
    r = _q(v, nd); return _reg(_signed(r, f"{abs(r):.{nd}f}"))


def fpct(v):              # share -> integer percentage
    r = _q(v, 0, 100); return _reg(_signed(r, f"{abs(r):.0f}%"))


def fp(v, nd=3):          # p: no leading zero, p < .001 below
    if float(v.x) < .001: return _reg("< .001")
    return "= " + fr(v, nd)


def fi(v): return _reg(str(int(v.x)))


def ci(lo, hi, f=fr): return f"[{f(lo)}, {f(hi)}]"


def rng(vals, f=fr):      # range over several JSON values (min to max)
    lo = min(vals, key=lambda v: float(v.x)); hi = max(vals, key=lambda v: float(v.x)); return f"{f(lo)} to {f(hi)}"


ROWS = []
def row(study, result, llm, est, ci_="—", test="—"): ROWS.append((str(study), result, llm, est, ci_ or DASH, test or DASH))


BAND = {"B1": f"layers 30{EN}50%", "B2": f"layers 60{EN}75%"}
POS = {"x": "manipulation passages", "body": "scenario", "q": "question", "dec": "last word fragment before the response"}

# ================================================================== Study 1
S = lambda m, k: g("stats", "study1_identification_mtmm", m, k)
for lab, k, nd in (("Same construct, different writers (convergent r)", "mtmm_convergent", 2),
                   ("Different constructs, same writer (r)", "mtmm_hetero_same_writer", 2),
                   ("Different constructs, different writers (r)", "mtmm_hetero_diff_writer", 2),
                   ("Different constructs: same minus different writer (difference in r)", "mtmm_hetero_same_minus_diff_writer", 3)):
    for m in MAIN:
        c = S(m, k + "_ci"); row(1, lab, m, fr(S(m, k), nd), ci(c[0], c[1], lambda v: fr(v, nd)))
for m in MAIN:
    c = S(m, "identification_B1_ci")
    row(1, f"Held-out writer's construct identified among 66 (layers 30{EN}50% of depth)", m, fpct(S(m, "identification_B1")), ci(c[0], c[1], fpct))
for m, key, lens in (("Qwen3.6 27B", "V610_qwen36|logitlens", "logit lens"), ("OLMo 3.1 32B", "V612_olmo32|logitlens", "logit lens"),
                     ("Gemma 4 31B", "V611_gemma4_31b|jlens", "Jacobian lens"), ("Qwen3.6 27B", "V610_qwen36|jlens", "Jacobian lens")):
    c = g("stats", "study1_content", key, "ci")
    row(1, f"Content validity: correct definition (of ten) chosen from the read-out words ({lens})", m,
        fpct(g("stats", "study1_content", key, "mean")), ci(c[0], c[1], fpct))
for m in MAIN:
    c = g("stats", "study1_criterion", m, "ci")
    row(1, "Criterion validity: similarity to the representation and difference in the LLM's judgment (r)", m,
        f"{fr(g('stats', 'study1_criterion', m, 'r'))} ({fi(g('stats', 'study1_criterion', m, 'n_pairs'))} pairs)", ci(c[0], c[1]),
        f"permutation p {fp(g('crit', m, 'perm_p'))}")

# ================================================================== Study 2
P = lambda *k: g("stats", "study2_profiles", *k)
c = P("identification_ci")
row(2, "Profile from one writer in one LLM matched to its construct among 66", "Three main LLMs", fpct(P("identification")), ci(c[0], c[1], fpct))
for m in MAIN:
    row(2, "Profile from one writer in one LLM matched to its construct among 66", m, fpct(P("by_llm", m)))
c = P("cos_same_construct_ci")
row(2, "Similarity of the profiles, same construct (cosine)", "Three main LLMs", fr(P("cos_same_construct")), ci(c[0], c[1]))
row(2, "Similarity of the profiles, different constructs (cosine)", "Three main LLMs", fr(P("cos_mean_profiles_different_constructs")))
for pair in ("Qwen3.6 27B ~ Gemma 4 31B", "Qwen3.6 27B ~ OLMo 3.1 32B", "Gemma 4 31B ~ OLMo 3.1 32B"):
    c = g("stats", "study2_structure_rho", pair, "ci")
    row(2, "Similarities among the constructs, correlated between LLMs (Spearman ρ)", pair.replace(" ~ ", " vs "),
        fr(g("stats", "study2_structure_rho", pair, "rho")), ci(c[0], c[1]))
for m, lab in [(m, m) for m in MAIN] + [("word counts", "Words of the situations")]:
    c = g("stats", "study2_process_difference", m, "ci")
    est = (f"{fr(g('stats', 'study2_process_difference', m, 'within_minus_between'), 3)} (within {fr(g('struct', 'theory_structure', m, 'within'), 3)}, "
           f"between {fr(g('struct', 'theory_structure', m, 'between'), 3)})")
    row(2, "Similarity of constructs within minus between MVCA's processes", lab, est, ci(c[0], c[1], lambda v: fr(v, 3)),
        f"permutation p {fp(g('struct', 'theory_structure', m, 'perm_p'))}")

# ================================================================== Study 3
T = lambda *k: g("stats", "study3", *k)
LAB_ACT = f"Activation of the mediator's representation, mediator minus other constructs (d_z; scenario, layers 30{EN}50%)"
c = T("activation_pooled_mean_dz", "ci"); row(3, LAB_ACT, "Pooled (14 LLMs)", fd(T("activation_pooled_mean_dz", "mean")), ci(c[0], c[1], fd))
for m in MAIN:
    c = T("activation_by_llm", m, "ci"); row(3, LAB_ACT, m, fd(T("activation_by_llm", m, "mean_dz")), ci(c[0], c[1], fd))
LAB_POS = "Experiments in which the mediator's representation was activated more than the other constructs' representations"
row(3, LAB_POS, "Range (14 LLMs)", f"{rng([T('activation_by_llm', m, 'positive') for m in J['stats']['study3']['activation_by_llm']], fi)} of "
    f"{fi(T('activation_by_llm', MAIN[0], 'n'))}")
for m in MAIN:
    c = T("activation_by_llm", m, "positive_ci")
    row(3, LAB_POS, m, f"{fi(T('activation_by_llm', m, 'positive'))} of {fi(T('activation_by_llm', m, 'n'))}", ci(c[0], c[1], fpct))
for k, lab in (("replicated", "replicated"), ("failed", "failed to replicate")):
    R = lambda *kk: g("repl", f"pooled_{k}", *kk); c = R("ci")
    row(3, f"Activation (d_z) in the experiments whose effects {lab} in human participants", "Pooled (14 LLMs)",
        f"{fd(R('mean_dz'))}; positive in {fi(R('experiments_gt0'))} of {fi(R('n'))}", ci(c[0], c[1], fd),
        f"Wilcoxon p {fp(R('wilcoxon_p_one_sided'))} (one-sided)")
for k, lab in (("replicated", "replicated"), ("failed", "failed to replicate")):
    for m in MAIN:
        R = lambda kk: g("repl", "per_llm", m, k, kk)
        row(3, f"Activation (d_z) in the experiments whose effects {lab} in human participants", m,
            f"{fd(R('mean_dz'))}; positive in {fi(R('activation_gt0'))} of {fi(R('n'))}", DASH,
            f"binomial p {fp(R('binom_p_one_sided_gt0'))} (one-sided)")
RB = lambda *k: T("replicated_vs_failed_pooled", *k)
c = RB("response", "ci")
row(3, "Replicated vs failed effects: LLMs' responses (rank-biserial r)", "Pooled (14 LLMs)", fr(RB("response", "rank_biserial")), ci(c[0], c[1]),
    f"Mann{EN}Whitney p {fp(RB('response', 'p_one_sided'))} (one-sided)")
c = RB("activation", "ci"); e90 = g("repl", "activation_equivalence_20", "ci90")
row(3, "Replicated vs failed effects: activation of the mediator's representation (rank-biserial r)", "Pooled (14 LLMs)",
    fr(RB("activation", "rank_biserial")), ci(c[0], c[1]),
    f"Mann{EN}Whitney p {fp(RB('activation', 'p_one_sided'))} (one-sided); 90% CI {ci(e90[0], e90[1])}, "
    f"{'equivalent' if g('repl', 'activation_equivalence_20', 'equivalent').x else 'not equivalent'} within ±.20")
D_ = lambda *k: g("repl", "response_minus_activation_rank_biserial", *k); c = D_("ci")
row(3, "Responses minus activation (difference in rank-biserial r)", "Pooled (14 LLMs)", fr(D_("difference")), ci(c[0], c[1]),
    f"bootstrap p {fp(D_('p_one_sided_le0'))} (one-sided)")
LAB_SC = f"Activation during the scenario (layers 30{EN}50%) and effect on the response, across the 60 experiments (Spearman ρ)"
sc = J["stats"]["study3"]["per_llm_scenario"]
row(3, LAB_SC, "Range (14 LLMs)", rng([T("per_llm_scenario", m, "rho") for m in sc]),
    DASH, f"BF01 = {rng([T('per_llm_scenario', m, 'BF01') for m in sc], fd)}")
PS14 = lambda *k: T("pooled_scenario_14", *k); c = PS14("ci"); c90 = PS14("ci90")
row(3, LAB_SC, "Pooled (14 LLMs)", fr(PS14("rho")), ci(c[0], c[1]),
    f"90% CI {ci(c90[0], c90[1])}, {'equivalent' if PS14('equivalence_20').x else 'not equivalent'} within ±.20")
for m in MAIN:
    for cell in ("x_B1", "x_B2", "body_B1", "body_B2", "q_B1", "q_B2", "dec_B1", "dec_B2"):
        pos, band = cell.split("_"); v = T("positions", m, cell, "rho"); c = T("positions", m, cell, "ci")
        fw = lambda k: T("familywise", m, cell, k)
        assert fw("rho").x == v.x
        test = f"family-wise p {fp(fw('p_fwe_within_llm'))} within LLM, p {fp(fw('p_fwe_all24'))} across LLMs"
        if pos in ("x", "body"):
            assert g("far", "cells", f"{m} {cell}", "rho").x == v.x
            test = f"BF01 = {fd(g('far', 'cells', f'{m} {cell}', 'BF01'))}; " + test
        if cell == "body_B1":
            s = lambda k: T("per_llm_scenario", m, k); assert s("rho").x == v.x; c90 = s("ci90")
            test += f"; 90% CI {ci(c90[0], c90[1])}, equivalent only within ±{fr(s('equivalence_bound_reached'))}"
        row(3, f"Activation and effect on the response (Spearman ρ): {POS[pos]}, {BAND[band]}", m, fr(v), ci(c[0], c[1]), test)
F = lambda *k: g("far", "pooled_far", *k); c = F("ci95"); c90 = F("ci90")
row(3, "Activation far from the response (manipulation passages and scenario, both depths) and effect on the response (mean ρ)",
    "Pooled (3 main LLMs)", fr(F("mean_rho")), ci(c[0], c[1]),
    f"90% CI {ci(c90[0], c90[1])}, {'equivalent' if F('equivalent_20').x else 'not equivalent'} within ±.20; "
    f"BF01 = {fd(g('far', 'BF01_range')[0])} to {fd(g('far', 'BF01_range')[1])} in each LLM, position and depth")
K = "pooled (question, last fragment) - (manipulation, scenario), both depths"; c = T("contrasts", K, "ci")
row(3, "Question and last word fragment minus manipulation passages and scenario (difference in ρ, both depths)", "Pooled (3 main LLMs)",
    fr(T("contrasts", K, "est")), ci(c[0], c[1]), f"bootstrap p {fp(T('contrasts', K, 'p_one_sided_le0'))} (one-sided)")
for m in MAIN:
    W = lambda k: g("stats", "study3_positions_with_ceiling_rule", m, k)
    row(3, f"Ceiling rule applied (experiments removed in which the comparison condition's response lay at the end of the 1{EN}9 scale toward "
           f"which the predicted effect moves in more than 30% of items): ρ at the scenario (layers 30{EN}50%), the question and the last word "
           f"fragment (layers 30{EN}50%, 60{EN}75%)", m,
        f"scenario {fr(W('body_B1'))}; question {fr(W('q_B1'))}, {fr(W('q_B2'))}; last fragment {fr(W('dec_B1'))}, {fr(W('dec_B2'))}", DASH,
        f"experiments retained: {fi(W('n'))} (removed: {fi(g('stats', 'ceiling_rule_flagged', m, 'n'))})")
Q = lambda *k: g("qc", *k)
row(3, "Question coding (Supplemental Note): agreement of the two coders", "Two coders",
    f"{fpct(Q('agreement'))} of {fi(Q('n'))} questions; weighted κ = {fr(Q('kappa_linear'))}")
for col, lab in (("q_B1", f"activation at the question (layers 30{EN}50%)"), ("answer_dz", "effect on the response")):
    c = Q(col, "ci")
    row(3, f"Question coding (Supplemental Note): mean code and {lab} (Spearman ρ)", "Mean of the three main LLMs", fr(Q(col, "rho")), ci(c[0], c[1]),
        f"p {fp(Q(col, 'p_two_sided'))} (two-sided)")

# ================================================================== Study 4
COND = (("own_spec_k10__mean", "the mediator's representation, without the shared directions"),
        ("own_k10__mean", "the mediator's representation, all 10 directions"),
        ("near_spec_k10__mean", "the representations of the most similar constructs, without the shared directions"),
        ("rand_spec_k10__mean", "the representations of other (random) constructs, without the shared directions"),
        ("unif__mean", "random patterns of the same size"),
        ("shared_k3__mean", "the three directions shared by all constructs"))
for m in MAIN:
    for k, lab in COND:
        Rm = lambda kk: g("rem", m, k, kk); c = Rm("ci")
        row(4, f"Share of the manipulation's effect on the response left after removal (mean replacement): {lab}", m,
            f"{fpct(Rm('median'))} (n = {fi(Rm('n'))})", ci(c[0], c[1], fpct))
for m in MAIN:
    I = lambda k: g("far", "induced_vs_other_responses", m, k)
    row(4, "Induced mediator's representation changed the response more than other constructs' representations (experiments)", m,
        f"{fi(I('k'))} of {fi(I('n'))}", DASH, f"binomial p {fp(I('p_one_sided'))} (one-sided)")

# ------------------------------------------------------------------ audit: every number in Estimate / CI / Test is a formatted JSON value
ALLOWED = [r"9[05]% CI", r"±\.20"]   # interval labels; equivalence bound of the keys *_equivalence_20 / equivalent_20 (fixed in the scripts)
for r in ROWS:
    for cell in r[3:]:
        t = cell
        for a in ALLOWED: t = re.sub(a, "", t)
        bad = [x for x in NUM.findall(t) if x not in TOKENS]
        assert not bad, (r, bad)

# ------------------------------------------------------------------ write TABLE_S10.md
NOTE = ("Note. CIs: bootstrap over constructs (Studies 1" + EN + "2) or experiments (Studies 3" + EN + "4), Fisher z for single correlations; "
        "BF01: Bayes factor for the absence of a correlation (uniform prior); equivalence bounds ±.20. Counts: exact (Clopper" + EN + "Pearson) "
        "intervals. Differences between mean similarities are given to three decimals. Family-wise p: max-statistic permutation test over the "
        "eight positions and depths of an LLM (within LLM) or of all three main LLMs (across LLMs). Removal: median over the experiments with "
        "an effect on the response larger than .3 scale points without removal; shared directions = the three directions common to the 66 constructs. "
        "Pooled: estimates averaged over LLMs, with experiments resampled jointly for all LLMs.")
L = ["**Table S10. Estimates, 95% confidence intervals, equivalence tests and Bayes factors for the main results, by study.**", "",
     "*Studies 1 to 4: the three main LLMs and pooled estimates.*", "",
     "| Study | Result | LLM | Estimate | 95% CI | Test (p, BF01, equivalence) |", "|---|---|---|---|---|---|"]
L += ["| " + " | ".join(r) + " |" for r in ROWS]
L += ["", NOTE, "", "<!-- sources (JSON file:key path; every number in the Estimate, 95% CI and Test columns is one of these values, "
      "formatted (asserted by make_table_s10.py); numbers in the Study, Result and LLM columns are design labels; values on a rounding "
      "boundary and how they were resolved are listed in TABLE_S10_CHECK.md):", *USED, "-->", ""]
(PS / "TABLE_S10.md").write_text("\n".join(L), encoding="utf-8")


# ================================================================== consistency check against PS_DRAFT_v2.md (Studies 1-4)
draft = (PS / "PS_DRAFT_v2.md").read_text(encoding="utf-8")
s1 = draft.index("### Study 1:"); s5 = draft.index("### Study 5:"); TXT = draft[s1:s5]
sm = (PS / "PS_SM_v2.md").read_text(encoding="utf-8")


def two(v): return fr(v, 2)


# claims: (source text, snippet copied from it, [(number as printed in the text, formatted JSON value(s) it should equal, key description)])
mn = lambda vals: min(vals, key=lambda v: float(v.x)); mx = lambda vals: max(vals, key=lambda v: float(v.x))
conv = [S(m, "mtmm_convergent") for m in MAIN]; idn = {m: S(m, "identification_B1") for m in MAIN}
pos14 = [T("activation_by_llm", m, "positive") for m in J["stats"]["study3"]["activation_by_llm"]]
farcells = [g("far", "cells", k, "rho") for k in J["far"]["cells"]]
rem = lambda m, k: g("rem", m, k, "median")
CLAIMS = [
 ("draft", "main LLMs: .58 to .60, 95% CIs from [.55, .60] to [.58, .63]",
  [(".58", fr(mn(conv)), "min convergent r (OLMo 3.1 32B)"), (".60", fr(mx(conv)), "max convergent r (Gemma 4 31B)"),
   ("[.55, .60]", ci(*S("OLMo 3.1 32B", "mtmm_convergent_ci")), "OLMo 3.1 32B CI"), ("[.58, .63]", ci(*S("Gemma 4 31B", "mtmm_convergent_ci")), "Gemma 4 31B CI")]),
 ("draft", "main LLMs: 91% [88%, 93%] to 92% [89%, 94%]",
  [("91%", fpct(idn["OLMo 3.1 32B"]), "OLMo 3.1 32B (lowest)"), ("[88%, 93%]", ci(*S("OLMo 3.1 32B", "identification_B1_ci"), f=fpct), "OLMo 3.1 32B CI"),
   ("92%", fpct(idn["Gemma 4 31B"]), "Gemma 4 31B (highest)"), ("[89%, 94%]", ci(*S("Gemma 4 31B", "identification_B1_ci"), f=fpct), "Gemma 4 31B CI")]),
 ("draft", "52% [44%, 61%] of the lists in Qwen3.6 27B and for 61% [52%, 70%] in OLMo 3.1 32B",
  [("52%", fpct(g("stats", "study1_content", "V610_qwen36|logitlens", "mean")), "Qwen logit lens"),
   ("[44%, 61%]", ci(*g("stats", "study1_content", "V610_qwen36|logitlens", "ci"), f=fpct), "Qwen logit lens CI"),
   ("61%", fpct(g("stats", "study1_content", "V612_olmo32|logitlens", "mean")), "OLMo logit lens"),
   ("[52%, 70%]", ci(*g("stats", "study1_content", "V612_olmo32|logitlens", "ci"), f=fpct), "OLMo logit lens CI")]),
 ("draft", "for 36% [28%, 43%] in Gemma 4 31B",
  [("36%", fpct(g("stats", "study1_content", "V611_gemma4_31b|jlens", "mean")), "Gemma Jacobian lens"),
   ("[28%, 43%]", ci(*g("stats", "study1_content", "V611_gemma4_31b|jlens", "ci"), f=fpct), "Gemma Jacobian lens CI")]),
 ("draft", "rated both situations of 395 pairs", [("395", fi(g("stats", "study1_criterion", m, "n_pairs")), f"n pairs, {m}") for m in MAIN]),
 ("draft", "(r = .34, 95% CI [.24, .42], to .44, 95% CI [.36, .52]; permutation p < .001",
  [(".34", fr(g("stats", "study1_criterion", "Gemma 4 31B", "r")), "Gemma 4 31B r"), ("[.24, .42]", ci(*g("stats", "study1_criterion", "Gemma 4 31B", "ci")), "Gemma CI"),
   (".44", fr(g("stats", "study1_criterion", "Qwen3.6 27B", "r")), "Qwen3.6 27B r"), ("[.36, .52]", ci(*g("stats", "study1_criterion", "Qwen3.6 27B", "ci")), "Qwen CI")]
  + [("< .001", fp(g("crit", m, "perm_p")), f"permutation p, {m}") for m in MAIN]),
 ("draft", "93% of cases (95% CI [90%, 96%]; 92% to 94% per LLM",
  [("93%", fpct(P("identification")), "pooled"), ("[90%, 96%]", ci(*P("identification_ci"), f=fpct), "pooled CI"),
   ("92%", fpct(mn([P("by_llm", m) for m in MAIN])), "lowest LLM"), ("94%", fpct(mx([P("by_llm", m) for m in MAIN])), "highest LLM")]),
 ("draft", "mean similarity of .79 (95% CI [.76, .82]), whereas those of different constructs had a mean similarity of .00",
  [(".79", fr(P("cos_same_construct")), "same construct"), ("[.76, .82]", ci(*P("cos_same_construct_ci")), "CI"),
   (".00", fr(P("cos_mean_profiles_different_constructs")), "different constructs")]),
 ("draft", "correlated at .94 (95% CI [.91, .95]) to .96 (95% CI [.94, .97])",
  [(".94", fr(g("stats", "study2_structure_rho", "Gemma 4 31B ~ OLMo 3.1 32B", "rho")), "lowest pair (Gemma vs OLMo)"),
   ("[.91, .95]", ci(*g("stats", "study2_structure_rho", "Gemma 4 31B ~ OLMo 3.1 32B", "ci")), "its CI"),
   (".96", fr(g("stats", "study2_structure_rho", "Qwen3.6 27B ~ Gemma 4 31B", "rho")), "highest pair (Qwen vs Gemma)"),
   ("[.94, .97]", ci(*g("stats", "study2_structure_rho", "Qwen3.6 27B ~ Gemma 4 31B", "ci")), "its CI")]),
 ("draft", "(mean similarity .076 to .088 within processes and .050 to .062 between them; permutation p < .001",
  [(".076", fr(mn([g("struct", "theory_structure", m, "within") for m in MAIN]), 3), "lowest within"),
   (".088", fr(mx([g("struct", "theory_structure", m, "within") for m in MAIN]), 3), "highest within"),
   (".050", fr(mn([g("struct", "theory_structure", m, "between") for m in MAIN]), 3), "lowest between"),
   (".062", fr(mx([g("struct", "theory_structure", m, "between") for m in MAIN]), 3), "highest between")]
  + [("< .001", fp(g("struct", "theory_structure", m, "perm_p")), f"permutation p, {m}") for m in MAIN]),
 ("draft", "d_z = 1.62 (95% CI [1.07, 2.12])",
  [("1.62", fd(T("activation_pooled_mean_dz", "mean")), "pooled 14 LLMs"), ("[1.07, 2.12]", ci(*T("activation_pooled_mean_dz", "ci"), f=fd), "CI")]),
 ("draft", "in 42 to 49 of the 60 experiments",
  [("42", fi(mn(pos14)), "fewest positive (14 LLMs)"), ("49", fi(mx(pos14)), "most positive (14 LLMs)"), ("60", fi(T("activation_by_llm", MAIN[0], "n")), "n")]),
 ("draft", "d_z = 1.12, 95% CI [0.28, 1.90]",
  [("1.12", fd(g("repl", "pooled_failed", "mean_dz")), "pooled failed"), ("[0.28, 1.90]", ci(*g("repl", "pooled_failed", "ci"), f=fd), "CI")]),
 ("draft", ".64, 95% CI [.29, .94]; for the activation, .16, 95% CI [−.27, .59]",
  [(".64", fr(RB("response", "rank_biserial")), "responses"), ("[.29, .94]", ci(*RB("response", "ci")), "CI"),
   (".16", fr(RB("activation", "rank_biserial")), "activation"), ("[−.27, .59]", ci(*RB("activation", "ci")), "CI")]),
 ("draft", "(Spearman ρ across the 60 experiments = −.08 to .15",
  [("−.08", fr(mn([T("per_llm_scenario", m, "rho") for m in sc])), "lowest (14 LLMs)"), (".15", fr(mx([T("per_llm_scenario", m, "rho") for m in sc])), "highest")]),
 ("draft", "ρ = .05 (95% CI [−.11, .21]), an equivalence test rejected correlations of .20 or larger (90% CI [−.09, .18])",
  [(".05", fr(PS14("rho")), "pooled 14"), ("[−.11, .21]", ci(*PS14("ci")), "CI"), ("[−.09, .18]", ci(*PS14("ci90")), "90% CI"),
   (".20", ".20" if PS14("equivalence_20").x else "not equivalent", "equivalence_20 = true")]),
 ("draft", "(BF01 = 3.33 to 6.21)", [("3.33", fd(T("BF01_range")[0]), "lowest BF01"), ("6.21", fd(T("BF01_range")[1]), "highest BF01")]),
 ("draft", "(ρ = −.02 to .12; BF01 = 4.25 to 6.11",
  [("−.02", fr(mn(farcells)), "lowest far cell"), (".12", fr(mx(farcells)), "highest far cell"),
   ("4.25", fd(g("far", "BF01_range")[0]), "lowest BF01"), ("6.11", fd(g("far", "BF01_range")[1]), "highest BF01")]),
 ("draft", "(ρ = .32, 95% CI [.07, .53], to .38, 95% CI [.14, .58]; family-wise p ≤ .037)",
  [(".32", fr(T("positions", "Qwen3.6 27B", "q_B1", "rho")), "Qwen question 30-50%"), ("[.07, .53]", ci(*T("positions", "Qwen3.6 27B", "q_B1", "ci")), "CI"),
   (".38", fr(T("positions", "Gemma 4 31B", "q_B1", "rho")), "Gemma question 30-50%"), ("[.14, .58]", ci(*T("positions", "Gemma 4 31B", "q_B1", "ci")), "CI"),
   (".037", fr(mx([T("familywise", "Qwen3.6 27B", "q_B1", "p_fwe_within_llm"), T("familywise", "Gemma 4 31B", "q_B1", "p_fwe_within_llm")]), 3),
    "largest family-wise p (within LLM) of the two cells")]),
 ("draft", "(ρ = .31 to .45; family-wise p ≤ .044)",
  [(".31", fr(T("positions", "OLMo 3.1 32B", "dec_B1", "rho")), "OLMo last fragment 30-50% (smallest of the three LLMs' largest)"),
   (".45", fr(T("positions", "Qwen3.6 27B", "dec_B1", "rho")), "Qwen last fragment 30-50% (largest)"),
   (".044", fr(mx([T("familywise", "Qwen3.6 27B", "dec_B1", "p_fwe_within_llm"), T("familywise", "Gemma 4 31B", "dec_B2", "p_fwe_within_llm"),
                   T("familywise", "OLMo 3.1 32B", "dec_B1", "p_fwe_within_llm")]), 3), "largest family-wise p (within LLM) of the three LLMs' best cells")]),
 ("draft", "(difference in ρ = .21, 95% CI [.07, .36])", [(".21", fr(T("contrasts", K, "est")), "pooled contrast"), ("[.07, .36]", ci(*T("contrasts", K, "ci")), "CI")]),
 ("draft", "in 31, 25, and 20 of the 52 experiments",
  [(n, fi(g("far", "induced_vs_other_responses", m, "k")), m) for n, m in zip(("31", "25", "20"), MAIN)] + [("52", fi(g("far", "induced_vs_other_responses", MAIN[0], "n")), "n")]),
 ("draft", "(binomial p ≥ .11)", [(".11", two(mn([g("far", "induced_vs_other_responses", m, "p_one_sided") for m in MAIN])), "smallest p, two decimals as in the text")]),
 ("draft", "to 68% (95% CI [55%, 85%]) and 39% (95% CI [4%, 58%])",
  [("68%", fpct(rem("Qwen3.6 27B", "own_spec_k10__mean")), "Qwen own, specific"), ("[55%, 85%]", ci(*g("rem", "Qwen3.6 27B", "own_spec_k10__mean", "ci"), f=fpct), "CI"),
   ("39%", fpct(rem("Gemma 4 31B", "own_spec_k10__mean")), "Gemma own, specific"), ("[4%, 58%]", ci(*g("rem", "Gemma 4 31B", "own_spec_k10__mean", "ci"), f=fpct), "CI")]),
 ("draft", "left 83% to 98% of it",
  [("83%", fpct(mn([rem(m, k) for m in MAIN[:2] for k in ("rand_spec_k10__mean", "unif__mean")])), "lowest of other constructs / random patterns, Qwen and Gemma"),
   ("98%", fpct(mx([rem(m, k) for m in MAIN[:2] for k in ("rand_spec_k10__mean", "unif__mean")])), "highest")]),
 ("draft", "reduced the manipulation's effect on the response to 17%", [("17%", fpct(rem("Gemma 4 31B", "near_spec_k10__mean")), "Gemma most similar, specific")]),
 ("draft", "left 99% (95% CI [79%, 107%])",
  [("99%", fpct(rem("OLMo 3.1 32B", "own_spec_k10__mean")), "OLMo own, specific"), ("[79%, 107%]", ci(*g("rem", "OLMo 3.1 32B", "own_spec_k10__mean", "ci"), f=fpct), "CI")]),
 ("sm", "they agreed on 87% of the experiments (weighted κ = .78)", [("87%", fpct(Q("agreement")), "agreement"), (".78", fr(Q("kappa_linear")), "weighted kappa")]),
 ("sm", "at ρ = .18 (95% CI [−.08, .42]) and with the manipulation's effect on the responses at ρ = .15 (95% CI [−.10, .40])",
  [(".18", fr(Q("q_B1", "rho")), "question 30-50%"), ("[−.08, .42]", ci(*Q("q_B1", "ci")), "CI"),
   (".15", fr(Q("answer_dz", "rho")), "effect on the response"), ("[−.10, .40]", ci(*Q("answer_dz", "ci")), "CI")]),
]
NOT_IN_TABLE = {   # numbers of Studies 1-4 that touch tabled results but come from other sources (reported, not checked here)
    ".04 to .07": "range over all 16 LLMs; the source JSONs hold the three main LLMs only (.04 to .06 in Table S10)",
}

out = ["# Table S10: consistency check against PS_DRAFT_v2.md (Studies 1-4)", "",
       "Built by make_table_s10.py. Each number of Studies 1-4 that Table S10 also reports was compared, after rounding to the precision "
       "printed in the text, with the formatted JSON value. The question-coding sentence of PS_SM_v2.md (Supplemental Note) is checked as well.", ""]
mism, n_ok, covered = [], 0, []
for src, snip, items in CLAIMS:
    text = TXT if src == "draft" else sm
    assert text.count(snip) >= 1, f"snippet not found in {src}: {snip}"
    at = text.index(snip)
    for num, tab, desc in items:
        assert num in snip, (num, snip)
        if src == "draft":   # mark the first occurrence of this number in the snippet that is not yet marked
            starts = [at + i.start() for i in re.finditer(re.escape(num), snip)]
            p0 = next((s0 for s0 in starts if s0 not in covered), starts[0]); covered += list(range(p0, p0 + len(num)))
        if num == tab: n_ok += 1
        else: mism.append((src, snip, num, tab, desc))
out += [f"## Mismatches: {len(mism)}", ""]
if mism:
    out += ["| Source | Text | Number in text | Table S10 | Value |", "|---|---|---|---|---|"]
    out += [f"| {s} | {sn} | {n} | {t} | {d} |" for s, sn, n, t, d in mism]
else:
    out += ["None."]
out += ["", f"Numbers compared: {n_ok + len(mism)} ({n_ok} equal after rounding).", ""]
out += ["## Values on a rounding boundary", "",
        "JSON values that lie exactly on the boundary of the printed precision (e.g. 1.125 to two decimals). Resolved = the unrounded value "
        "was recomputed from the analysis's own input file with the source script's formula (it reproduces the JSON value) and decided the direction.", "",
        "| JSON key | JSON value | Unrounded (recomputed) | Printed as |", "|---|---|---|---|"]
seen = set()
for key, x, u, r in TIES:
    if key in seen: continue
    seen.add(key)
    out.append(f"| {key} | {x} | {'%.6f' % u if u is not None else 'not recomputed (rounded half up)'} | {r} |")
out += ["", "## Notes on tabled results", "",
        "- Study 2, process structure: the permutation p (5,000 permutations of the process labels; structure_v6.json) is < .001 in each main LLM, "
        "whereas the bootstrap over constructs (psychsci_stats_v6.json) gives 95% CIs of the within-minus-between difference that include zero "
        "for Qwen3.6 27B and Gemma 4 31B and touch it for OLMo 3.1 32B. The text reports only the permutation p.",
        "- Study 3, far positions pooled (far_positions_bf_v6.json): 90% CI of the mean ρ over the three main LLMs and four far cells "
        f"{ci(*F('ci90'))}, not equivalent within ±.20 (the per-cell BF01 favor the absence of a correlation). The text reports the BF01 only; "
        "the equivalence test of the 14-LLM pooled scenario correlation (reported) is passed.",
        "- Study 3, per-LLM scenario correlations: no single LLM passes the ±.20 equivalence test (smallest bound reached "
        f"{fr(mn([T('per_llm_scenario', m, 'equivalence_bound_reached') for m in sc]))}); the text claims equivalence only for the pooled estimate.",
        "- Study 1, different constructs: same-writer correlations exceed different-writer correlations by "
        + "; ".join(f"{fr(S(m, 'mtmm_hetero_same_minus_diff_writer'), 3)} {ci(*S(m, 'mtmm_hetero_same_minus_diff_writer_ci'), f=lambda v: fr(v, 3))} ({m})" for m in MAIN)
        + "; the text says that the correlations were .04 to .07 'whether they came from the same writer or from different writers' (true at two decimals).",
        "- Study 2, process structure, bootstrap CIs of the within-minus-between difference: "
        + "; ".join(f"{fr(g('stats', 'study2_process_difference', m, 'within_minus_between'), 3)} {ci(*g('stats', 'study2_process_difference', m, 'ci'), f=lambda v: fr(v, 3))} ({m})" for m in MAIN) + ".", ""]
for k, v in NOT_IN_TABLE.items():
    assert k in TXT; out.append(f"- '{k}': {v}.")
# cross-check with Table S5 Panel C (same 24 correlations, SUPPLEMENTARY_TABLES_v1.md)
si = (PS.parent / "SUPPLEMENTARY_TABLES_v1.md").read_text(encoding="utf-8")
POSNAME = {"manipulation": "x", "scenario": "body", "question": "q", "last word fragment before the response": "dec"}
s5 = re.findall(r"^\| (Qwen3\.6 27B|Gemma 4 31B|OLMo 3\.1 32B) \| (manipulation|scenario|question|last word fragment before the response) \| "
                r"(30-50%|60-75%) \| ([−.\d]+) \|", si, re.M)
assert len(s5) == 24, len(s5)
diff5 = [(m, p_, b, v5, fr(T("positions", m, f"{POSNAME[p_]}_{'B1' if b == '30-50%' else 'B2'}", "rho"))) for m, p_, b, v5 in s5]
diff5 = [d for d in diff5 if d[3] != d[4]]
out += ["", f"## Cross-check with Table S5, Panel C (24 correlations): {len(diff5)} difference(s)", ""]
out += [f"- {m}, {p_}, {b}: Table S5 {v5}, Table S10 {v10} (JSON value on the boundary; the unrounded value is listed above)" for m, p_, b, v5, v10 in diff5] or ["None."]
# numbers of Studies 1-4 not compared (not in the source JSONs, or design numbers)
skip = re.compile(r"9[05]% CIs?|Qwen3\.6 27B|Gemma 4 31B|OLMo 3\.1 32B|Qwen3-Embedding-8B|Figs?\. [\d a-e,and]+|Tables? S\d+(?: to S\d+)?|Study \d|Studies \d(?: and \d)?|"
                  r"Supplemental Note \d|\[@[^\]]+\]|d_z")
mask = set(covered)
for mm in skip.finditer(TXT): mask.update(range(mm.start(), mm.end()))
rest = []
for mm in NUM.finditer(TXT):
    if any(i in mask for i in range(mm.start(), mm.end())): continue
    rest.append(f"{mm.group()} ({TXT[max(0, mm.start() - 45):mm.end() + 10].replace(chr(10), ' ').strip()})")
out += ["", f"## Numbers of Studies 1-4 not compared ({len(rest)})", "",
        "Design numbers and results whose values are not in the source JSONs (Table S10 does not report them; their sources are Tables S2 to S7).", ""]
out += ["- " + x for x in rest]
(PS / "TABLE_S10_CHECK.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"rows: {len(ROWS)}; keys: {len(USED)}; compared: {n_ok + len(mism)}; mismatches: {len(mism)}; boundary values: {len(seen)}")
for x in mism: print("MISMATCH", x)
for key, x, u, r in TIES: print("TIE", key, x, u, r)
