"""Sentence-encoder comparison for Studies 1-2 (reviewer request: a strong model of sentence meaning as the text baseline).
Encoder: Qwen3-Embedding-8B (V770_embed/V0_embed: normalized embeddings of the high and low text of every v6 pair).
Study 1: leave-one-writer-out nearest-centroid identification (same lowo as bow_floor_v6 / V1 lowo6) of the encoder's pair differences
(raw and with dimensions standardized over texts), and the cross-table with the three main LLMs (V1 export dz_mean_B1, band 1):
identification by the LLM in the pairs the encoder misses, and the discordant pairs (LLM right / encoder wrong vs the reverse).
Study 2: the encoder's cross-writer construct geometry (structure_v6 geometry()), correlated with the LLMs'; LLM-LLM correlations after
removing the encoder's geometry, and after removing encoder and word-count geometry together (multiple regression residuals); stimulus
geometry (the same 2,000 pairs as structure_v6) beyond the encoder; the theory structure (within- vs between-process similarity) in the
encoder and in the LLMs beyond the encoder (5,000 label permutations).
Usage: python analysis_v6/encoder_floor_v6.py -> analysis_v6/encoder_floor_v6.json"""
import json, re, collections, itertools
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
HERE = Path(__file__).resolve().parent; BASE = HERE.parent; RV = BASE / "results_v6"
P = json.load(open(BASE / "materials_v6/upload_v6/pairs_v6.json", encoding="utf-8")); byid = {p["pid"]: p for p in P}
SPEC = json.load(open(BASE / "materials_v6/upload_v6/spec_v6.json", encoding="utf-8"))["constructs"]
JOBS = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
meta = pd.read_csv(RV / "V610_qwen36/V1_read/pairs_meta.csv"); order = [byid[x] for x in meta.pid]
CONS = sorted({p["construct"] for p in order}); WR = sorted({p["writer"] for p in order})
ci = np.array([CONS.index(p["construct"]) for p in order]); wi = np.array([WR.index(p["writer"]) for p in order])
nW, nC = len(WR), len(CONS)
# encoder, aligned to the V1 pair order
em = pd.read_csv(RV / "V770_embed/V0_embed/pairs_meta.csv"); E = torch.load(RV / "V770_embed/V0_embed/emb_v6.pt", weights_only=False)["emb"].float().numpy()
pos = {pid: i for i, pid in enumerate(em.pid)}; ix = np.array([pos[x] for x in meta.pid])
EH, EL = E[0::2][ix], E[1::2][ix]
sd = np.concatenate([EH, EL]).std(0) + 1e-6
ENC = {"encoder_raw": EH - EL, "encoder_std": EH / sd - EL / sd}

def lowo(Z):
    S = np.zeros((nW, nC, Z.shape[1])); N = np.zeros((nW, nC))
    np.add.at(S, (wi, ci), Z); np.add.at(N, (wi, ci), 1); M = S / np.maximum(N, 1)[..., None]; has = N > 0
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12); rank = np.zeros(len(Z), dtype=int)
    for w in range(nW):
        oth = has.copy(); oth[w] = False; cnt = oth.sum(0)
        C = (M * oth[..., None]).sum(0) / np.maximum(cnt, 1)[:, None]; C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        ii = np.where(wi == w)[0]; Sm = Zn[ii] @ C.T; Sm[:, cnt == 0] = -9
        rank[ii] = (Sm >= Sm[np.arange(len(ii)), ci[ii]][:, None]).sum(1)
    return rank
OUT = {"encoder": "Qwen/Qwen3-Embedding-8B", "n_pairs": len(order), "identification": {}, "cross_table": {}}
RK = {}
for k, Z in ENC.items():
    r = lowo(Z); RK[k] = r; OUT["identification"][k] = {"first": round(float((r == 1).mean()), 4), "top5": round(float((r <= 5).mean()), 4)}; print(k, OUT["identification"][k], flush=True)
ek = max(RK, key=lambda k: (RK[k] == 1).mean()); he = RK[ek] == 1; OUT["encoder_used"] = ek
REPS = {}
for n, j in JOBS.items():
    Z = torch.nan_to_num(torch.load(RV / j / "V1_read/export_dz_v6.pt", weights_only=False)["dz_mean_B1"].float()).numpy(); REPS[n] = Z
    hm = lowo(Z) == 1
    OUT["cross_table"][n] = {"llm_first": round(float(hm.mean()), 4), "llm_first_when_encoder_fails": round(float(hm[~he].mean()), 4),
                             "encoder_first_when_llm_fails": round(float(he[~hm].mean()), 4), "n_encoder_fails": int((~he).sum()),
                             "llm_right_encoder_wrong": int((hm & ~he).sum()), "encoder_right_llm_wrong": int((he & ~hm).sum()),
                             "mcnemar_p": float(f"{stats.binomtest(int((hm & ~he).sum()), int((hm != he).sum()), .5).pvalue:.3g}")}
    print(n, OUT["cross_table"][n], flush=True)
# word counts (as structure_v6)
tok = lambda s: re.findall(r"[a-z]+", s.lower())
dfc = collections.Counter(w for p in order for side in ("high", "low") for w in set(tok(p[side]))); V = {w: i for i, w in enumerate(sorted(w for w, c in dfc.items() if c >= 2))}
Xw = np.zeros((2 * len(order), len(V)), dtype=np.float32)
for i, p in enumerate(order):
    for s, side in enumerate(("high", "low")):
        for w, c in collections.Counter(w for w in tok(p[side]) if w in V).items(): Xw[2 * i + s, V[w]] = c
Xw = Xw / (Xw.std(0) + 1e-6); REPS["word counts"] = Xw[0::2] - Xw[1::2]; REPS["encoder"] = ENC[ek]; del Xw
splits = [s for s in itertools.combinations(range(nW), nW // 2) if 0 in s]
def geometry(Z):
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9); G = np.zeros((nC, nC)); k = 0
    for s in splits:
        a = np.isin(wi, s); A = np.stack([Zn[a & (ci == c)].mean(0) for c in range(nC)]); B = np.stack([Zn[~a & (ci == c)].mean(0) for c in range(nC)])
        A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True); M = A @ B.T; G += (M + M.T) / 2; k += 1
    return G / k
GEO = {n: geometry(Z) for n, Z in REPS.items()}; iu = np.triu_indices(nC, 1)
def resid(y, *xs):
    X = np.column_stack([np.ones_like(y)] + list(xs)); b = np.linalg.lstsq(X, y, rcond=None)[0]; return y - X @ b
names = list(JOBS); g = {n: GEO[n][iu] for n in GEO}
OUT["construct_geometry"] = {f"encoder ~ {n}": round(float(stats.spearmanr(g["encoder"], g[n])[0]), 3) for n in names + ["word counts"]}
for a, b in itertools.combinations(names, 2):
    OUT["construct_geometry"][f"{a} ~ {b}"] = {"rho": round(float(stats.spearmanr(g[a], g[b])[0]), 3),
        "beyond_encoder": round(float(stats.spearmanr(resid(g[a], g["encoder"]), resid(g[b], g["encoder"]))[0]), 3),
        "beyond_encoder_and_words": round(float(stats.spearmanr(resid(g[a], g["encoder"], g["word counts"]), resid(g[b], g["encoder"], g["word counts"]))[0]), 3)}
rng = np.random.default_rng(5); idx = rng.choice(len(order), 2000, replace=False)
def stim(Z): Zs = Z[idx] / (np.linalg.norm(Z[idx], axis=1, keepdims=True) + 1e-9); return (Zs @ Zs.T)[np.triu_indices(2000, 1)]
ST = {n: stim(REPS[n]) for n in names + ["encoder"]}
OUT["stimulus_geometry"] = {f"encoder ~ {n}": round(float(stats.spearmanr(ST["encoder"], ST[n])[0]), 3) for n in names}
for a, b in itertools.combinations(names, 2):
    OUT["stimulus_geometry"][f"{a} ~ {b}"] = {"rho": round(float(stats.spearmanr(ST[a], ST[b])[0]), 3),
        "beyond_encoder": round(float(stats.spearmanr(resid(ST[a], ST["encoder"]), resid(ST[b], ST["encoder"]))[0]), 3)}
grp = np.array([SPEC[c].get("group") or "" for c in CONS]); same = (grp[:, None] == grp[None, :])[iu]
def perm(v):
    obs = v[same].mean() - v[~same].mean(); null = []
    for _ in range(5000):
        pg = rng.permutation(grp); s2 = (pg[:, None] == pg[None, :])[iu]; null.append(v[s2].mean() - v[~s2].mean())
    return {"within_minus_between": round(float(obs), 4), "perm_p": round(float((np.array(null) >= obs).mean()), 4)}
OUT["theory_structure"] = {"encoder": dict(perm(g["encoder"]), within=round(float(g["encoder"][same].mean()), 4), between=round(float(g["encoder"][~same].mean()), 4))}
for n in names:
    OUT["theory_structure"][n + " (beyond encoder)"] = perm(resid(g[n], g["encoder"]))
    OUT["theory_structure"][n + " (beyond encoder and words)"] = perm(resid(g[n], g["encoder"], g["word counts"]))
json.dump(OUT, open(HERE / "encoder_floor_v6.json", "w"), indent=1); print(json.dumps(OUT, indent=1))
