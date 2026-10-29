"""Study 1 discriminant validity among the closest alternatives (reviewer request): leave-one-writer-out identification (as V1 lowo6;
ties count against the pair's construct) restricted (a) to the constructs of the pair's own MVCA process (chance 1/size of the process),
and (b) to each mediator and the rival construct that the original article ruled out (8 pairs; chance 50%). Three main LLMs (V1 export
dz_mean_B1), the sentence encoder (Qwen3-Embedding-8B, standardized) and single-word counts (standardized) for comparison.
Usage: python analysis_v6/discriminant_v6.py -> analysis_v6/discriminant_v6.json"""
import json, re, collections
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; BASE = HERE.parent; RV = BASE / "results_v6"
P = json.load(open(BASE / "materials_v6/upload_v6/pairs_v6.json", encoding="utf-8")); byid = {p["pid"]: p for p in P}
SPEC = json.load(open(BASE / "materials_v6/upload_v6/spec_v6.json", encoding="utf-8"))["constructs"]
EXPS = json.load(open(BASE / "materials_v6/upload_v6/experiments_v6.json", encoding="utf-8"))
meta = pd.read_csv(RV / "V610_qwen36/V1_read/pairs_meta.csv"); order = [byid[x] for x in meta.pid]
CONS = sorted({p["construct"] for p in order}); WR = sorted({p["writer"] for p in order})
ci = np.array([CONS.index(p["construct"]) for p in order]); wi = np.array([WR.index(p["writer"]) for p in order]); nW, nC = len(WR), len(CONS)
grp = np.array([SPEC[c].get("group") or "" for c in CONS])
RIV = [(CONS.index(EXPS[s["sources"][0]["experiment"]]["construct"]), CONS.index(c)) for c, s in SPEC.items() if s["role"] == "rival"]

def centroids(Z):
    S = np.zeros((nW, nC, Z.shape[1])); N = np.zeros((nW, nC)); np.add.at(S, (wi, ci), Z); np.add.at(N, (wi, ci), 1)
    return S / np.maximum(N, 1)[..., None], N > 0

def sims(Z):
    """sims[i, c] = cosine of pair i with construct c's centroid from the other writers."""
    M, has = centroids(Z); Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12); out = np.full((len(Z), nC), -9.0)
    for w in range(nW):
        oth = has.copy(); oth[w] = False; cnt = oth.sum(0)
        C = (M * oth[..., None]).sum(0) / np.maximum(cnt, 1)[:, None]; C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        ii = np.where(wi == w)[0]; s = Zn[ii] @ C.T; s[:, cnt == 0] = -9; out[ii] = s
    return out

def evaluate(Z):
    S = sims(Z); own = S[np.arange(len(S)), ci]
    same = grp[None, :] == grp[ci][:, None]
    within = ((S >= own[:, None]) & same).sum(1) == 1                  # own construct is the unique best within its process
    chance_within = float(np.mean(1 / same.sum(1)))
    riv = []
    for m, r in RIV:
        ii = np.where((ci == m) | (ci == r))[0]; other = np.where(ci[ii] == m, r, m)
        riv.append(float((own[ii] > S[ii, other]).mean()))
    return {"within_process": round(float(within.mean()), 4), "chance_within": round(chance_within, 4),
            "mediator_vs_rival_mean": round(float(np.mean(riv)), 4), "mediator_vs_rival_min": round(float(np.min(riv)), 4), "n_rival_pairs": len(RIV)}
OUT = {}
for n, j in {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}.items():
    Z = torch.nan_to_num(torch.load(RV / j / "V1_read/export_dz_v6.pt", weights_only=False)["dz_mean_B1"].float()).numpy()
    OUT[n] = evaluate(Z); print(n, OUT[n], flush=True)
em = pd.read_csv(RV / "V770_embed/V0_embed/pairs_meta.csv"); Em = torch.load(RV / "V770_embed/V0_embed/emb_v6.pt", weights_only=False)["emb"].float().numpy()
pos = {pid: i for i, pid in enumerate(em.pid)}; ix = np.array([pos[x] for x in meta.pid]); H, L = Em[0::2][ix], Em[1::2][ix]; sd = np.concatenate([H, L]).std(0) + 1e-6
OUT["sentence encoder"] = evaluate(H / sd - L / sd); print("encoder", OUT["sentence encoder"], flush=True)
tok = lambda s: re.findall(r"[a-z]+", s.lower())
dfc = collections.Counter(w for p in order for side in ("high", "low") for w in set(tok(p[side]))); V = {w: i for i, w in enumerate(sorted(w for w, c in dfc.items() if c >= 2))}
X = np.zeros((2 * len(order), len(V)), dtype=np.float32)
for i, p in enumerate(order):
    for s, side in enumerate(("high", "low")):
        for w, c in collections.Counter(w for w in tok(p[side]) if w in V).items(): X[2 * i + s, V[w]] = c
X = X / (X.std(0) + 1e-6); OUT["single words"] = evaluate(X[0::2] - X[1::2]); print("words", OUT["single words"], flush=True)
json.dump(OUT, open(HERE / "discriminant_v6.json", "w"), indent=1)
