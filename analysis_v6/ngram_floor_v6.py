"""Stronger word floors for Study 1 identification (reviewer request: a unigram count is a weak lexical control).
Same leave-one-writer-out nearest-centroid procedure as bow_floor_v6.py / V1 lowo6 (ties count against the own construct).
Text features: word n-grams (1, 1-2, 1-3; lower case, letters only; n-grams in >= 2 texts), as counts, binary or tf-idf, raw or
standardized over texts; plus latent semantic analysis (LSA: truncated SVD of the 1-2-gram tf-idf matrix, 100/300/1000 dimensions).
Also the cross-table with the three main LLMs (V1 export dz_mean_B1): identification by the LLM in the pairs the strongest word floor misses.
Usage: python analysis_v6/ngram_floor_v6.py  -> analysis_v6/ngram_floor_v6.json"""
import json, re, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
HERE = Path(__file__).resolve().parent; BASE = HERE.parent; RV = BASE / "results_v6"
P = json.load(open(BASE / "materials_v6" / "upload_v6" / "pairs_v6.json", encoding="utf-8"))
CONS = sorted({p["construct"] for p in P}); WR = sorted({p["writer"] for p in P})
ci = np.array([CONS.index(p["construct"]) for p in P]); wi = np.array([WR.index(p["writer"]) for p in P])
texts = [p[side] for p in P for side in ("high", "low")]
nW, nC = len(WR), len(CONS); g = wi * nC + ci
G = sp.csr_matrix((np.ones(len(P)), (g, np.arange(len(P)))), shape=(nW * nC, len(P)))
N = np.asarray(G.sum(1)).ravel().reshape(nW, nC)

def lowo(Z):
    """Z: (pairs x dims), sparse or dense. Returns the rank of the own construct (1 = identified)."""
    dense = not sp.issparse(Z)
    M = (G @ Z); M = np.asarray(M.todense()) if sp.issparse(M) else np.asarray(M)
    M = M.reshape(nW, nC, -1) / np.maximum(N, 1)[..., None]
    if dense: Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    else:
        nr = np.sqrt(np.asarray(Z.multiply(Z).sum(1)).ravel()) + 1e-12; Zn = sp.diags(1 / nr) @ Z
    rank = np.zeros(len(P), dtype=int); has = N > 0
    for w in range(nW):
        oth = has.copy(); oth[w] = False; cnt = oth.sum(0)
        C = (M * oth[..., None]).sum(0) / np.maximum(cnt, 1)[:, None]; C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        ii = np.where(wi == w)[0]
        if not len(ii): continue
        S = Zn[ii] @ C.T; S = np.asarray(S); S[:, cnt == 0] = -9
        own = S[np.arange(len(ii)), ci[ii]][:, None]; rank[ii] = (S >= own).sum(1)
    return rank

def feats(ngram, kind, std):
    cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-z]+\b", ngram_range=(1, ngram), min_df=2,
                         binary=(kind == "binary"), dtype=np.float64)
    X = cv.fit_transform(texts).tocsr()
    if kind == "tfidf":
        dfreq = np.asarray((X > 0).sum(0)).ravel(); X = X @ sp.diags(np.log(X.shape[0] / dfreq))
    if std:
        m = np.asarray(X.mean(0)).ravel(); m2 = np.asarray(X.multiply(X).mean(0)).ravel(); sd = np.sqrt(np.maximum(m2 - m * m, 0)) + 1e-6
        X = X @ sp.diags(1 / sd)
    X = X.tocsr(); return (X[0::2] - X[1::2]).tocsr(), X.shape[1]

OUT = {"n_pairs": len(P), "chance_first": round(1 / nC, 4), "floors": {}}
best = (0, None, None)
for ngram in (1, 2, 3):
    for kind in ("counts", "binary", "tfidf"):
        for std in (False, True):
            D, nv = feats(ngram, kind, std); r = lowo(D); key = f"ng{ngram}_{kind}_{'std' if std else 'raw'}"
            OUT["floors"][key] = {"vocab": int(nv), "first": round(float((r == 1).mean()), 4), "top5": round(float((r <= 5).mean()), 4)}
            print(key, OUT["floors"][key], flush=True)
            if (r == 1).mean() > best[0]: best = ((r == 1).mean(), key, r)
# latent semantic analysis on the 1-2-gram tf-idf texts
cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-z]+\b", ngram_range=(1, 2), min_df=2, dtype=np.float64)
X = cv.fit_transform(texts).tocsr(); dfreq = np.asarray((X > 0).sum(0)).ravel(); X = (X @ sp.diags(np.log(X.shape[0] / dfreq))).tocsr()
for k in (100, 300, 1000):
    L = TruncatedSVD(n_components=k, random_state=2026).fit_transform(X); L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
    D = L[0::2] - L[1::2]; r = lowo(D); key = f"lsa{k}"
    OUT["floors"][key] = {"first": round(float((r == 1).mean()), 4), "top5": round(float((r <= 5).mean()), 4)}
    print(key, OUT["floors"][key], flush=True)
    if (r == 1).mean() > best[0]: best = ((r == 1).mean(), key, r)
OUT["strongest_floor"] = {"key": best[1], "first": round(float(best[0]), 4)}
hw = best[2] == 1
import torch
for job in ["V610_qwen36", "V611_gemma4_31b", "V612_olmo32"]:
    meta = pd.read_csv(RV / job / "V1_read" / "pairs_meta.csv"); pos = {p["pid"]: i for i, p in enumerate(P)}; idx = np.array([pos[x] for x in meta.pid])
    Zm = torch.load(RV / job / "V1_read" / "export_dz_v6.pt")["dz_mean_B1"].float().numpy(); ok = ~np.isnan(Zm[:, 0]); Zm = np.nan_to_num(Zm)
    Zf = np.zeros((len(P), Zm.shape[1]), dtype=np.float32); Zf[idx] = Zm; okf = np.zeros(len(P), bool); okf[idx] = ok
    hm = lowo(Zf) == 1
    OUT[job] = {"model_first": round(float(hm[okf].mean()), 4), "model_first_when_strongest_floor_fails": round(float(hm[okf & ~hw].mean()), 4),
                "n_floor_fails": int((okf & ~hw).sum())}
    print(job, OUT[job], flush=True)
json.dump(OUT, open(HERE / "ngram_floor_v6.json", "w"), indent=1)
