"""Study 2 (structure across models) for the three main models, from the V1 exports (dz_mean_B1: standardized high - low pair differences,
mean over the situation's tokens, band 1) and the word-count differences of the same pairs (baseline).
(1) Construct geometry: 66 x 66 similarity of construct centroids computed across writers (centroid of construct i from writers in half 1,
    of construct j from half 2, averaged over both splits, so no similarity rests on one writer's style); the off-diagonal entries are
    correlated between models (Spearman), raw and after removing the word-count geometry (residuals of each model's entries on the word-count
    entries).
(2) Stimulus geometry: cosine similarity among 2,000 randomly drawn pairs, correlated between models, raw and with the word-count similarity
    partialled out.
(3) Theory structure: mean cross-writer similarity of constructs in the same MVCA appraisal process (spec group) vs different processes,
    with a label-permutation test (5,000), for each model and for the word counts.
Usage: python analysis_v6/structure_v6.py -> analysis_v6/structure_v6.json"""
import json, re, collections, itertools
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy import stats
HERE = Path(__file__).resolve().parent; BASE = HERE.parent; RV = BASE / "results_v6"
P = json.load(open(BASE / "materials_v6/upload_v6/pairs_v6.json", encoding="utf-8")); byid = {p["pid"]: p for p in P}
SPEC = json.load(open(BASE / "materials_v6/upload_v6/spec_v6.json", encoding="utf-8"))["constructs"]
JOBS = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
meta = pd.read_csv(RV / "V610_qwen36/V1_read/pairs_meta.csv"); order = [byid[x] for x in meta.pid]
for j in JOBS.values(): assert list(pd.read_csv(RV / j / "V1_read/pairs_meta.csv").pid) == list(meta.pid), "pair order differs"
CONS = sorted({p["construct"] for p in order}); WR = sorted({p["writer"] for p in order})
ci = np.array([CONS.index(p["construct"]) for p in order]); wi = np.array([WR.index(p["writer"]) for p in order])
tok = lambda s: re.findall(r"[a-z]+", s.lower())
df = collections.Counter(w for p in order for side in ("high", "low") for w in set(tok(p[side]))); V = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 2))}
Xw = torch.zeros(2 * len(order), len(V))
for i, p in enumerate(order):
    for s, side in enumerate(("high", "low")):
        for w, n in collections.Counter(w for w in tok(p[side]) if w in V).items(): Xw[2 * i + s, V[w]] = float(n)
Xw = Xw / (Xw.std(0) + 1e-6); DW = (Xw[0::2] - Xw[1::2]).numpy()
REPS = {"word counts": DW}
for n, j in JOBS.items():
    Z = torch.load(RV / j / "V1_read/export_dz_v6.pt", weights_only=False)["dz_mean_B1"].float(); REPS[n] = torch.nan_to_num(Z).numpy()
splits = [s for s in itertools.combinations(range(len(WR)), len(WR) // 2) if 0 in s]
def geometry(Z):
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9); G = np.zeros((len(CONS), len(CONS))); k = 0
    for s in splits:
        a = np.isin(wi, s); A = np.stack([Zn[a & (ci == c)].mean(0) for c in range(len(CONS))]); B = np.stack([Zn[~a & (ci == c)].mean(0) for c in range(len(CONS))])
        A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True); M = A @ B.T; G += (M + M.T) / 2; k += 1
    return G / k
GEO = {n: geometry(Z) for n, Z in REPS.items()}; iu = np.triu_indices(len(CONS), 1)
def resid(y, x): b = np.polyfit(x, y, 1); return y - np.polyval(b, x)
OUT = {"n_constructs": len(CONS), "n_pairs": len(order), "construct_geometry": {}, "stimulus_geometry": {}, "theory_structure": {}}
names = list(JOBS)
for a, b in itertools.combinations(["word counts"] + names, 2):
    ya, yb = GEO[a][iu], GEO[b][iu]; r = stats.spearmanr(ya, yb)[0]
    rp = stats.spearmanr(resid(ya, GEO["word counts"][iu]), resid(yb, GEO["word counts"][iu]))[0] if "word counts" not in (a, b) else None
    OUT["construct_geometry"][f"{a} ~ {b}"] = {"rho": round(float(r), 3), "rho_beyond_words": None if rp is None else round(float(rp), 3)}
rng = np.random.default_rng(5); idx = rng.choice(len(order), 2000, replace=False)
def stim(Z): Zs = Z[idx] / (np.linalg.norm(Z[idx], axis=1, keepdims=True) + 1e-9); return (Zs @ Zs.T)[np.triu_indices(2000, 1)]
ST = {n: stim(Z) for n, Z in REPS.items()}
for a, b in itertools.combinations(names, 2):
    r = stats.spearmanr(ST[a], ST[b])[0]; rp = stats.spearmanr(resid(ST[a], ST["word counts"]), resid(ST[b], ST["word counts"]))[0]
    OUT["stimulus_geometry"][f"{a} ~ {b}"] = {"rho": round(float(r), 3), "rho_beyond_words": round(float(rp), 3)}
for a in names: OUT["stimulus_geometry"][f"word counts ~ {a}"] = {"rho": round(float(stats.spearmanr(ST["word counts"], ST[a])[0]), 3)}
grp = np.array([SPEC[c].get("group") or "" for c in CONS]); same = (grp[:, None] == grp[None, :])[iu]
for n in ["word counts"] + names:
    g = GEO[n][iu]; obs = g[same].mean() - g[~same].mean(); null = []
    for _ in range(5000):
        pg = rng.permutation(grp); s2 = (pg[:, None] == pg[None, :])[iu]; null.append(g[s2].mean() - g[~s2].mean())
    OUT["theory_structure"][n] = {"within_minus_between": round(float(obs), 4), "perm_p": round(float((np.array(null) >= obs).mean()), 4), "within": round(float(g[same].mean()), 4), "between": round(float(g[~same].mean()), 4)}
# beyond words: residual geometry within vs between processes
for n in names:
    g = resid(GEO[n][iu], GEO["word counts"][iu]); obs = g[same].mean() - g[~same].mean(); null = []
    for _ in range(5000):
        pg = rng.permutation(grp); s2 = (pg[:, None] == pg[None, :])[iu]; null.append(g[s2].mean() - g[~s2].mean())
    OUT["theory_structure"][n + " (beyond words)"] = {"within_minus_between": round(float(obs), 4), "perm_p": round(float((np.array(null) >= obs).mean()), 4)}
np.save(HERE / "construct_geometry_v6.npy", np.stack([GEO[n] for n in ["word counts"] + names])); json.dump({"order": ["word counts"] + names, "constructs": CONS, "groups": list(grp)}, open(HERE / "construct_geometry_v6_index.json", "w"))
json.dump(OUT, open(HERE / "structure_v6.json", "w"), indent=1); print(json.dumps(OUT, indent=1))
