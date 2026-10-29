"""Study 1: does the model's representation identify the construct beyond the words of the situations?
Model: V1 export dz_mean_B1 (standardized high - low pair difference, mean over the situation's tokens, band 1) of the three main models.
Words: the word-count difference of the same pair (bow_floor_v6.py, standardized counts).
(1) leave-one-writer-out identification (V1 lowo6) of model, words, and the cross-table (model right when words fail);
(2) cross-fitted removal of the words: for each held-out writer, a ridge map words -> model state is fitted on the other five writers and
    its prediction subtracted from the held-out writer's pairs; identification on these out-of-sample residuals (all writers), for several
    ridge penalties (the most removing one is the conservative estimate); words reduced to their top 2,000 principal components.
Usage: python analysis_v6/beyond_words_v6.py  -> analysis_v6/beyond_words_v6.json"""
import json, re, collections, sys
from pathlib import Path
import pandas as pd, torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent; BASE = HERE.parent; RV = BASE / "results_v6"
P = json.load(open(BASE / "materials_v6" / "upload_v6" / "pairs_v6.json", encoding="utf-8"))
tok = lambda s: re.findall(r"[a-z]+", s.lower())
df = collections.Counter(w for p in P for side in ("high", "low") for w in set(tok(p[side])))
V = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 2))}
def words_diff(order):
    X = torch.zeros(2 * len(order), len(V))
    for i, p in enumerate(order):
        for s, side in enumerate(("high", "low")):
            for w, n in collections.Counter(w for w in tok(p[side]) if w in V).items(): X[2 * i + s, V[w]] = float(n)
    X = X / (X.std(0) + 1e-6); return X[0::2] - X[1::2]
def lowo(Z, ci, wi, nW, nC):
    S = torch.zeros(nW, nC, Z.shape[1]); N = torch.zeros(nW, nC)
    S.index_put_((wi, ci), Z, accumulate=True); N.index_put_((wi, ci), torch.ones(len(Z)), accumulate=True)
    Mc = S / N.clamp(min=1)[..., None]; has = N > 0; rank = torch.zeros(len(Z), dtype=torch.long); Zn = F.normalize(Z, dim=-1)
    for w in range(nW):
        oth = has.clone(); oth[w] = False; cnt = oth.sum(0)
        C = (Mc * oth[..., None]).sum(0) / cnt.clamp(min=1)[:, None]; Cn = F.normalize(C, dim=-1); ii = (wi == w).nonzero().flatten()
        if not len(ii): continue
        sims = Zn[ii] @ Cn.T; sims[:, cnt == 0] = -9; rank[ii] = (sims >= sims.gather(1, ci[ii][:, None])).sum(1)
    return rank
OUT = {}
for job in sys.argv[1:] or ["V610_qwen36", "V611_gemma4_31b", "V612_olmo32"]:
    meta = pd.read_csv(RV / job / "V1_read" / "pairs_meta.csv"); byid = {p["pid"]: p for p in P}; order = [byid[x] for x in meta.pid]
    Zm = torch.load(RV / job / "V1_read" / "export_dz_v6.pt")["dz_mean_B1"].float(); ok = ~torch.isnan(Zm[:, 0]); Zm = torch.nan_to_num(Zm)
    CONS = sorted({p["construct"] for p in order}); WR = sorted({p["writer"] for p in order})
    ci = torch.tensor([CONS.index(p["construct"]) for p in order]); wi = torch.tensor([WR.index(p["writer"]) for p in order])
    Xw = words_diff(order); U, S, Vt = torch.linalg.svd(Xw - Xw.mean(0), full_matrices=False); Xr = U[:, :2000] * S[:2000]
    rm, rw = lowo(Zm, ci, wi, len(WR), len(CONS)), lowo(Xw, ci, wi, len(WR), len(CONS)); hm, hw = rm == 1, rw == 1
    R = {"n": int(ok.sum()), "model_first": round(float(hm[ok].float().mean()), 4), "words_first": round(float(hw[ok].float().mean()), 4),
         "model_first_when_words_fail": round(float(hm[ok & ~hw].float().mean()), 4), "model_first_when_words_right": round(float(hm[ok & hw].float().mean()), 4),
         "words_first_when_model_fails": round(float(hw[ok & ~hm].float().mean()), 4), "residual": {}}
    for lam in (1e1, 1e2, 1e3, 1e4):
        Res = torch.zeros_like(Zm)
        for w in range(len(WR)):
            tr, te = (wi != w) & ok, (wi == w)
            A = Xr[tr]; Wt = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]), A.T @ Zm[tr]); Res[te] = Zm[te] - Xr[te] @ Wt
        rr = lowo(Res, ci, wi, len(WR), len(CONS)); var_left = float((Res[ok] ** 2).sum() / (Zm[ok] ** 2).sum())
        R["residual"][f"{lam:g}"] = {"first": round(float((rr[ok] == 1).float().mean()), 4), "top5": round(float((rr[ok] <= 5).float().mean()), 4), "share_of_variance_left": round(var_left, 3)}
    OUT[job] = R; print(job, json.dumps(R))
json.dump(OUT, open(HERE / "beyond_words_v6.json", "w"), indent=1)
