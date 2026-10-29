"""Statistics required by Psychological Science for the latent-representation paper (22 Oct 2026): confidence intervals,
equivalence tests and Bayes factors for claims of absence, tests of differences between text positions, family-wise
corrections, pooled comparisons, sensitivity (power) statements, and exclusion counts. Reads only local files.

Units of resampling: Studies 1-2 resample the 66 constructs (cluster bootstrap; pairs and similarities are nested in
constructs); Study 3 resamples the 60 experiments, jointly across LLMs when estimates are pooled (LLMs respond to the same
experiments and are not treated as independent). Counts get exact (Clopper-Pearson) intervals.
Writes analysis_v6/psychsci_stats_v6.json. Usage: python analysis_v6/psychsci_stats_v6.py [n_boot]"""
import json, re, sys, collections
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats, special, integrate

A = Path(__file__).resolve().parent; LAT = A.parent; RV = LAT / "results_v6"; MAT = LAT / "materials_v6"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rng = np.random.default_rng(20261022)
MAIN = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
OUT = {"_note": "95%% intervals unless stated; bootstrap B = %d" % B}


def q(a, lo=.025, hi=.975):
    a = np.asarray(a, float); a = a[np.isfinite(a)]; return [round(float(np.quantile(a, lo)), 4), round(float(np.quantile(a, hi)), 4)]


def cp(k, n, a=.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0; hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 4), round(float(hi), 4)]


def fz_ci(r, n, level=.95):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); s = stats.norm.ppf(.5 + level / 2); return [round(float(np.tanh(z - s * se)), 4), round(float(np.tanh(z + s * se)), 4)]


def bf01(r, n):  # BF01 for a correlation, uniform prior on rho (exact likelihood; Ly, Verhagen & Wagenmakers 2016), applied to rank r
    def lik(rho): return (1 - rho ** 2) ** ((n - 1) / 2) * (1 - rho * r) ** (-(n - 1.5)) * special.hyp2f1(.5, .5, n - .5, (1 + rho * r) / 2)
    return float(lik(0.0) / integrate.quad(lambda p: lik(p) * .5, -1, 1, limit=200)[0])


# ------------------------------------------------------------------ Studies 1-2: pair-level data of the three main LLMs
meta = pd.read_csv(RV / "V610_qwen36" / "V1_read" / "pairs_meta.csv")
cons = sorted(meta.construct.unique()); wrs = sorted(meta.writer.unique())
ci_ = meta.construct.map({c: i for i, c in enumerate(cons)}).values; wi_ = meta.writer.map({w: i for i, w in enumerate(wrs)}).values
S1 = {}
for name, job in MAIN.items():
    D = torch.load(RV / job / "V1_read" / "export_dz_v6.pt", map_location="cpu", weights_only=False)["dz_mean_B1"].float().numpy()
    # held-out-writer identification at the 30-50% band: representation = mean of the other five writers' pairs of each construct
    sums = np.zeros((len(cons), len(wrs), D.shape[1])); cnt = np.zeros((len(cons), len(wrs)))
    np.add.at(sums, (ci_, wi_), D); np.add.at(cnt, (ci_, wi_), 1)
    correct = np.zeros(len(D), bool)
    for w in range(len(wrs)):
        rep = (sums.sum(1) - sums[:, w]) / np.maximum(cnt.sum(1) - cnt[:, w], 1)[:, None]
        rep /= np.linalg.norm(rep, axis=1, keepdims=True)
        idx = np.where(wi_ == w)[0]; x = D[idx] / np.linalg.norm(D[idx], axis=1, keepdims=True)
        sim = x @ rep.T; own = sim[np.arange(len(idx)), ci_[idx]]
        correct[idx] = (sim > own[:, None]).sum(1) == 0          # ties count against the pair's construct (none >= own except itself)
    per_c = np.array([correct[ci_ == c].mean() for c in range(len(cons))]); n_c = np.array([(ci_ == c).sum() for c in range(len(cons))])
    acc = correct.mean()
    bs = []
    for _ in range(B):
        s = rng.integers(0, len(cons), len(cons)); bs.append((per_c[s] * n_c[s]).sum() / n_c[s].sum())
    # MTMM: means of each construct from each writer, correlated across the dimensions of the internal state
    M = sums / np.maximum(cnt, 1)[..., None]; valid = cnt > 0
    Z = M - M.mean(-1, keepdims=True); Z /= np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-12
    flat = Z.reshape(-1, Z.shape[-1]); R = flat @ flat.T; vflat = valid.reshape(-1)
    cc = np.repeat(np.arange(len(cons)), len(wrs)); ww = np.tile(np.arange(len(wrs)), len(cons))
    same_c = cc[:, None] == cc[None, :]; same_w = ww[:, None] == ww[None, :]; ok = vflat[:, None] & vflat[None, :] & ~np.eye(len(cc), dtype=bool)
    conv_c = np.array([R[(cc[:, None] == c) & (cc[None, :] == c) & ~same_w & ok].mean() for c in range(len(cons))])
    het_sw_c = np.array([R[(cc[:, None] == c) & ~same_c & same_w & ok].mean() for c in range(len(cons))])
    het_dw_c = np.array([R[(cc[:, None] == c) & ~same_c & ~same_w & ok].mean() for c in range(len(cons))])
    bsm = [(conv_c[s].mean(), het_sw_c[s].mean(), het_dw_c[s].mean()) for s in (rng.integers(0, len(cons), len(cons)) for _ in range(B))]
    bsm = np.array(bsm)
    S1[name] = {"identification_B1": round(float(acc), 4), "identification_B1_ci": q(bs), "per_construct_min": round(float(per_c.min()), 3),
                "constructs_ge_66pct": int((per_c >= .66).sum()),
                "mtmm_convergent": round(float(conv_c.mean()), 3), "mtmm_convergent_ci": q(bsm[:, 0]),
                "mtmm_hetero_same_writer": round(float(het_sw_c.mean()), 3), "mtmm_hetero_same_writer_ci": q(bsm[:, 1]),
                "mtmm_hetero_diff_writer": round(float(het_dw_c.mean()), 3), "mtmm_hetero_diff_writer_ci": q(bsm[:, 2]),
                "mtmm_hetero_same_minus_diff_writer": round(float(het_sw_c.mean() - het_dw_c.mean()), 4),
                "mtmm_hetero_same_minus_diff_writer_ci": q(bsm[:, 1] - bsm[:, 2])}
    print(name, S1[name], flush=True)
OUT["study1_identification_mtmm"] = S1

# digests: identification at the 30-50% band for all 16 LLMs (point estimates; pair-level exports exist only for the main LLMs)
JOBS = ["Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B", "Gemma 4 26B-A4B", "OLMo 3.1 32B base", "Gemma 4 31B base", "Qwen3.5 27B", "Qwen3.5 9B",
        "Qwen3.5 4B", "Qwen3.5 2B", "Qwen3.5 0.8B", "Gemma 4 E4B", "OLMo 3 7B Instruct", "Qwen3.8 27B", "gpt-oss 20B", "Granite 4.2 30B"]
id16 = {}
for name in JOBS:
    job = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32", "Gemma 4 26B-A4B": "V720_gemma4_26b_a4b",
           "OLMo 3.1 32B base": "V721_olmo32_base", "Gemma 4 31B base": "V722_gemma4_31b_base", "Qwen3.5 27B": "V730_qwen35_27b",
           "Qwen3.5 9B": "V731_qwen35_9b", "Qwen3.5 4B": "V732_qwen35_4b", "Qwen3.5 2B": "V733_qwen35_2b", "Qwen3.5 0.8B": "V734_qwen35_08b",
           "Gemma 4 E4B": "V735_gemma4_e4b", "OLMo 3 7B Instruct": "V736_olmo7", "Qwen3.8 27B": "V740_qwen38", "gpt-oss 20B": "V741_gptoss20",
           "Granite 4.2 30B": "V742_granite42"}[name]
    fdig = RV / job / "V1_read" / "digest.md"
    if not fdig.exists(): id16[name] = None; continue
    txt = fdig.read_text(encoding="utf-8")
    m = re.search(r"^\| B1 \| ([\d.]+) / ", txt, re.M); id16[name] = float(m.group(1)) if m else None
OUT["identification_B1_all_llms_digest"] = id16
OUT["identification_B1_range_local"] = [min(v for v in id16.values() if v), max(v for v in id16.values() if v)]

# criterion validity: Fisher-z CI over the 395 pairs (one pair per construct and writer)
crit = json.load(open(A / "criterion_v6.json", encoding="utf-8"))
OUT["study1_criterion"] = {m: {"r": v["criterion_r"], "ci": fz_ci(v["criterion_r"], v["n_pairs"]), "n_pairs": v["n_pairs"]} for m, v in crit.items()}

# content validity: rater identification of the meaning from read-out words, per construct -> bootstrap over constructs
rj = pd.read_csv(A / "readout_judge_v6_by_construct.csv")
OUT["study1_content"] = {}
for col in [c for c in rj.columns if "|" in c]:
    v = rj[col].dropna().values
    OUT["study1_content"][col] = {"mean": round(float(v.mean()), 4), "ci": q([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]), "n_constructs": int(len(v))}

# Study 2: construct profiles in the shared dimensions (shape_v6.npz) -> bootstrap over constructs
z = np.load(A / "shape_v6.npz"); U = z["unit"]; nC, nM, nW, K = U.shape
hits = np.full((nC, nM, nW), np.nan)
for m in range(nM):
    om = [x for x in range(nM) if x != m]
    for w in range(nW):
        ow = [x for x in range(nW) if x != w]
        T = U[:, om][:, :, ow].mean((1, 2)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        for c in range(nC):
            if np.any(U[c, m, w]): s = T @ U[c, m, w]; hits[c, m, w] = float(s.argmax() == c)
same = np.array([np.nanmean([U[c, i // nW, i % nW] @ U[c, j // nW, j % nW] for i in range(nM * nW) for j in range(i + 1, nM * nW)
                             if i // nW != j // nW and i % nW != j % nW and np.any(U[c, i // nW, i % nW]) and np.any(U[c, j // nW, j % nW])]) for c in range(nC)])
mean_prof = U.reshape(nC, -1, K).mean(1); mean_prof /= np.linalg.norm(mean_prof, axis=1, keepdims=True)
diff_c = np.array([np.mean([mean_prof[c] @ mean_prof[d] for d in range(nC) if d != c]) for c in range(nC)])
hc = np.nanmean(hits.reshape(nC, -1), 1); nh = np.sum(~np.isnan(hits.reshape(nC, -1)), 1)
bsh = []; bss = []
for _ in range(B):
    s = rng.integers(0, nC, nC); bsh.append((hc[s] * nh[s]).sum() / nh[s].sum()); bss.append(same[s].mean())
OUT["study2_profiles"] = {"identification": round(float(np.nansum(hits) / np.sum(~np.isnan(hits))), 4), "identification_ci": q(bsh),
                          "by_llm": {str(z["models"][m]): round(float(np.nanmean(hits[:, m])), 4) for m in range(nM)},
                          "cos_same_construct": round(float(same.mean()), 3), "cos_same_construct_ci": q(bss),
                          "cos_mean_profiles_different_constructs": round(float(diff_c.mean()), 3)}

# Study 2: similarity structure across LLMs (construct_geometry_v6.npy: words + 3 LLMs) and process structure
G = np.load(A / "construct_geometry_v6.npy"); IX = json.load(open(A / "construct_geometry_v6_index.json", encoding="utf-8"))
order = IX["order"]; groups = np.array(IX["groups"]); iu = np.triu_indices(G.shape[1], 1)
st = {}
for a in range(1, 4):
    for b in range(a + 1, 4):
        rho = stats.spearmanr(G[a][iu], G[b][iu])[0]; bs = []
        for _ in range(min(B, 1000)):
            s = rng.integers(0, G.shape[1], G.shape[1]); I, J = np.meshgrid(s, s, indexing="ij"); msk = I < J
            bs.append(stats.spearmanr(G[a][I[msk], J[msk]], G[b][I[msk], J[msk]])[0])
        st[f"{order[a]} ~ {order[b]}"] = {"rho": round(float(rho), 4), "ci": q(bs)}
OUT["study2_structure_rho"] = st
same_p = groups[:, None] == groups[None, :]
pr = {}
for a in range(4):
    def wb(idx):
        g = G[a][np.ix_(idx, idx)]; sp = same_p[np.ix_(idx, idx)]; off = ~np.eye(len(idx), dtype=bool) & (idx[:, None] != idx[None, :])
        return g[sp & off].mean() - g[~sp & off].mean()
    allidx = np.arange(G.shape[1]); est = wb(allidx)
    bs = [wb(rng.integers(0, G.shape[1], G.shape[1])) for _ in range(min(B, 1000))]
    sd = G[a][iu].std()
    pr[order[a]] = {"within_minus_between": round(float(est), 4), "ci": q(bs), "standardized_by_sd_of_similarities": round(float(est / sd), 3),
                    "standardized_ci": [round(x / sd, 3) for x in q(bs)]}
OUT["study2_process_difference"] = pr

# ------------------------------------------------------------------ Study 3
sb = pd.read_csv(A / "v3_shift_by_repl.csv"); ex = pd.read_csv(A / "example_s3_v6.csv")
s3 = {"per_llm_scenario": {}}
for m, g in sb.groupby("model"):
    g = g.dropna(subset=["shift_dz", "dz"]); n = len(g); r = stats.spearmanr(g.shift_dz, g.dz)[0]
    lo90 = fz_ci(r, n, .90); s3["per_llm_scenario"][m] = {"n": n, "rho": round(float(r), 3), "ci": fz_ci(r, n), "ci90": lo90,
                                                           "equivalence_bound_reached": round(float(max(abs(lo90[0]), abs(lo90[1]))), 3), "BF01": round(bf01(r, n), 2)}
W = sb.pivot(index="exp", columns="model", values="shift_dz"); Y = sb.pivot(index="exp", columns="model", values="dz"); exps = W.index.values
def mean_rho(idx): return np.mean([stats.spearmanr(W.loc[idx, c], Y.loc[idx, c], nan_policy="omit")[0] for c in W.columns])
bs = [mean_rho(rng.choice(exps, len(exps), replace=True)) for _ in range(min(B, 1000))]
s3["pooled_scenario_14"] = {"rho": round(float(mean_rho(exps)), 3), "ci": q(bs), "ci90": q(bs, .05, .95)}
s3["pooled_scenario_14"]["equivalence_20"] = bool(-.20 < s3["pooled_scenario_14"]["ci90"][0] and s3["pooled_scenario_14"]["ci90"][1] < .20)
bfs = [v["BF01"] for v in s3["per_llm_scenario"].values()]; s3["BF01_range"] = [min(bfs), max(bfs)]
# activation of the mediator's representation (planned pooled estimate): mean own-minus-others d_z over experiments, per LLM and pooled
act = {}
for m, g in sb.groupby("model"):
    v = g.shift_dz.dropna().values; act[m] = {"mean_dz": round(float(v.mean()), 3), "ci": q([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]),
                                               "positive": int((v > 0).sum()), "n": int(len(v)), "positive_ci": cp(int((v > 0).sum()), int(len(v)))}
s3["activation_by_llm"] = act
pooled = W.mean(1).values; s3["activation_pooled_mean_dz"] = {"mean": round(float(pooled.mean()), 3), "ci": q([pooled[rng.integers(0, len(pooled), len(pooled))].mean() for _ in range(B)])}
# positions x depths, three main LLMs
cells = ["x_B1", "x_B2", "body_B1", "body_B2", "q_B1", "q_B2", "dec_B1", "dec_B2"]
models = list(ex.model.unique()); E = {m: ex[ex.model == m].set_index("exp") for m in models}
common = sorted(set.intersection(*[set(E[m].index) for m in models]))
cell_res = {}
for m in models:
    g = E[m].loc[common]
    cell_res[m] = {c: {"rho": round(float(stats.spearmanr(g[c], g.answer_dz)[0]), 3), "ci": fz_ci(stats.spearmanr(g[c], g.answer_dz)[0], len(g))} for c in cells}
def rhos(idx):
    return {(m, c): stats.spearmanr(E[m].loc[idx, c], E[m].loc[idx, "answer_dz"])[0] for m in models for c in cells}
o = rhos(common); bsr = [rhos(list(rng.choice(common, len(common), replace=True))) for _ in range(min(B, 2000))]
def contrast(f):
    d = np.array([f(b) for b in bsr]); return {"est": round(float(f(o)), 3), "ci": q(d), "p_one_sided_le0": round(float(np.mean(d <= 0)), 4)}
con = {}
for m in models:
    for band in ("B1", "B2"):
        con[f"{m} question-scenario {band}"] = contrast(lambda r, m=m, b=band: r[(m, f"q_{b}")] - r[(m, f"body_{b}")])
        con[f"{m} lastfragment-scenario {band}"] = contrast(lambda r, m=m, b=band: r[(m, f"dec_{b}")] - r[(m, f"body_{b}")])
for band in ("B1", "B2"):
    con[f"pooled question-scenario {band}"] = contrast(lambda r, b=band: np.mean([r[(m, f'q_{b}')] - r[(m, f'body_{b}')] for m in models]))
    con[f"pooled lastfragment-scenario {band}"] = contrast(lambda r, b=band: np.mean([r[(m, f'dec_{b}')] - r[(m, f'body_{b}')] for m in models]))
con["pooled (question, last fragment) - (manipulation, scenario), both depths"] = contrast(
    lambda r: np.mean([np.mean([r[(m, c)] for c in cells if c[0] in "qd"]) - np.mean([r[(m, c)] for c in cells if c[0] in "xb"]) for m in models]))
con["pooled scenario B1 (3 main)"] = contrast(lambda r: np.mean([r[(m, "body_B1")] for m in models]))
d3 = np.array([np.mean([b[(m, "body_B1")] for m in models]) for b in bsr]); con["pooled scenario B1 (3 main)"]["ci90"] = q(d3, .05, .95)
s3["positions"] = cell_res; s3["contrasts"] = con
# family-wise max-statistic permutation (association), within each LLM (8 cells) and over all 24 cells
NP = 5000; Rk = {m: {c: stats.rankdata(E[m].loc[common, c]) for c in cells} for m in models}; Ry = {m: stats.rankdata(E[m].loc[common, "answer_dz"]) for m in models}
def sp(a, b): a = a - a.mean(); b = b - b.mean(); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
obs_c = {(m, c): sp(Rk[m][c], Ry[m]) for m in models for c in cells}; mxw = {m: [] for m in models}; mxa = []
for _ in range(NP):
    perm = rng.permutation(len(common)); vals = {(m, c): sp(Rk[m][c], Ry[m][perm]) for m in models for c in cells}
    for m in models: mxw[m].append(max(vals[(m, c)] for c in cells))
    mxa.append(max(vals.values()))
mxa = np.array(mxa)
s3["familywise"] = {m: {c: {"rho": round(obs_c[(m, c)], 3), "p_fwe_within_llm": round(float((np.sum(np.array(mxw[m]) >= obs_c[(m, c)]) + 1) / (NP + 1)), 4),
                            "p_fwe_all24": round(float((np.sum(mxa >= obs_c[(m, c)]) + 1) / (NP + 1)), 4)} for c in cells} for m in models}
# replicated vs failed (14 vs 14), pooled over LLMs via per-experiment means of z-scored values
s = sb[sb.repl.isin(["replicated", "failed"])].copy()
for col in ("shift_dz", "dz"): s[col + "_z"] = s.groupby("model")[col].transform(lambda v: (v - v.mean()) / v.std())
pe = s.groupby(["exp", "repl"])[["shift_dz_z", "dz_z"]].mean().reset_index(); rep = {}
for col, lab in (("shift_dz_z", "activation"), ("dz_z", "response")):
    a = pe[pe.repl == "replicated"][col].values; b = pe[pe.repl == "failed"][col].values
    U_, p = stats.mannwhitneyu(a, b, alternative="greater"); rb = 2 * U_ / (len(a) * len(b)) - 1
    bsd = [2 * stats.mannwhitneyu(rng.choice(a, len(a)), rng.choice(b, len(b))).statistic / (len(a) * len(b)) - 1 for _ in range(min(B, 2000))]
    rep[lab] = {"rank_biserial": round(float(rb), 3), "ci": q(bsd), "p_one_sided": round(float(p), 4), "n_replicated": int(len(a)), "n_failed": int(len(b))}
s3["replicated_vs_failed_pooled"] = rep
OUT["study3"] = s3

# ceiling rule (fixed 20 Oct before the results; not applied before): drop LLM x experiment if the comparison arm sits at the blocking end in > 30% of items
ceil = {}
for name, job in MAIN.items():
    f = RV / job / "V3_exp" / "ceiling_check_v2.csv"
    if f.exists():
        c = pd.read_csv(f); ceil[name] = sorted(c.loc[c.blocked > .30, "exp"].tolist())
OUT["ceiling_rule_flagged"] = {k: {"n": len(v), "experiments": v} for k, v in ceil.items()}
alt = {}
for m in models:
    keep = [e for e in common if e not in ceil.get(m, [])]; g = E[m].loc[keep]
    alt[m] = {c: round(float(stats.spearmanr(g[c], g.answer_dz)[0]), 3) for c in ("body_B1", "q_B1", "q_B2", "dec_B1", "dec_B2")}; alt[m]["n"] = len(keep)
OUT["study3_positions_with_ceiling_rule"] = alt

# ------------------------------------------------------------------ sensitivity (power) statements
def rho_detectable(n, power=.8, alpha=.05):  # one-sided, Fisher z approximation
    return float(np.tanh((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) / np.sqrt(n - 3)))
crit60 = int(stats.binom.ppf(.95, 60, .5)) + 1
pw = [p for p in np.arange(.5, 1, .01) if 1 - stats.binom.cdf(crit60 - 1, 60, p) >= .8]
OUT["sensitivity"] = {"rho_detectable_n60_80pct_one_sided": round(rho_detectable(60), 3), "count_test_critical_of_60": crit60,
                      "count_test_rate_detectable_80pct": round(float(pw[0]), 2) if pw else None,
                      "dz_detectable_n18_80pct_two_sided": round(float((stats.t.ppf(.975, 17) + stats.t.ppf(.8, 17)) / np.sqrt(18)), 2),
                      "dz_detectable_n7_80pct_two_sided": round(float((stats.t.ppf(.975, 6) + stats.t.ppf(.8, 6)) / np.sqrt(7)), 2)}

# ------------------------------------------------------------------ exclusions and pair counts
raw = sum(1 for _ in open(MAT / "pairs_v6_raw.jsonl", encoding="utf-8")); fin = sum(1 for _ in open(MAT / "pairs_v6_final.jsonl", encoding="utf-8"))
used = len(meta); excl = json.load(open(MAT / "exclusions_v6.json", encoding="utf-8"))
OUT["pairs"] = {"composed_raw_lines": raw, "final_lines": fin, "analyzed_pairs": used,
                "exclusions_v6_json": dict(collections.Counter(v.split(" ")[0] for v in excl.values()))}

json.dump(OUT, open(A / "psychsci_stats_v6.json", "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ["identification_B1_range_local", "study2_profiles", "sensitivity", "pairs", "ceiling_rule_flagged"]}, indent=1, default=float))
print(json.dumps(OUT["study3"]["pooled_scenario_14"]), OUT["study3"]["BF01_range"])
