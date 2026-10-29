"""Word-count floor for Study 1 identification (same procedure as code_v6/V1_read.py lowo6): each text = counts of its words (lower case,
letters only; words in >= 2 texts), dimensions standardized over all texts (as 'std mean'), pair difference high - low, leave-one-writer-out
nearest-centroid identification among the 66 constructs (ties count against the own construct). Also binary presence and tf-idf variants.
Reports first-place and top-5 rates overall and on the third of the pairs used for the OLMo 7B checkpoints (V1_FRAC=3, if reproducible).
Usage: python analysis_v6/bow_floor_v6.py  -> analysis_v6/bow_floor_v6.json"""
import json, re, collections
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent; BASE = HERE.parent
P = json.load(open(BASE / "materials_v6" / "upload_v6" / "pairs_v6.json", encoding="utf-8"))
CONS = sorted({p["construct"] for p in P}); WR = sorted({p["writer"] for p in P})
tok = lambda s: re.findall(r"[a-z]+", s.lower())
df = collections.Counter(w for p in P for side in ("high", "low") for w in set(tok(p[side])))
V = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 2))}
def mat(kind):
    X = torch.zeros(2 * len(P), len(V))
    for i, p in enumerate(P):
        for s, side in enumerate(("high", "low")):
            c = collections.Counter(w for w in tok(p[side]) if w in V)
            for w, n in c.items(): X[2 * i + s, V[w]] = (1.0 if kind == "binary" else float(n))
    if kind == "tfidf": X = X * torch.log(2 * len(P) / torch.tensor([df[w] for w in V], dtype=torch.float))
    return X
ci = torch.tensor([CONS.index(p["construct"]) for p in P]); wi = torch.tensor([WR.index(p["writer"]) for p in P])
def lowo(Z, ci, wi):
    nW, nC = len(WR), len(CONS); S = torch.zeros(nW, nC, Z.shape[1]); N = torch.zeros(nW, nC)
    S.index_put_((wi, ci), Z, accumulate=True); N.index_put_((wi, ci), torch.ones(len(Z)), accumulate=True)
    Mc = S / N.clamp(min=1)[..., None]; has = N > 0; rank = torch.zeros(len(Z), dtype=torch.long); Zn = F.normalize(Z, dim=-1)
    for w in range(nW):
        oth = has.clone(); oth[w] = False; cnt = oth.sum(0)
        C = (Mc * oth[..., None]).sum(0) / cnt.clamp(min=1)[:, None]; Cn = F.normalize(C, dim=-1); ii = (wi == w).nonzero().flatten()
        if not len(ii): continue
        sims = Zn[ii] @ Cn.T; sims[:, cnt == 0] = -9; rank[ii] = (sims >= sims.gather(1, ci[ii][:, None])).sum(1)
    return rank
OUT = {"n_pairs": len(P), "vocab": len(V), "chance_first": round(1 / len(CONS), 4)}
for kind in ("counts", "binary", "tfidf"):
    X = mat(kind)
    for std in (True, False):
        Xs = X / (X.std(0) + 1e-6) if std else X
        D = Xs[0::2] - Xs[1::2]; r = lowo(D, ci, wi)
        OUT[f"{kind}_{'std' if std else 'raw'}"] = {"first": round(float((r == 1).float().mean()), 4), "top5": round(float((r <= 5).float().mean()), 4)}
    print(kind, OUT[f"{kind}_std"], OUT[f"{kind}_raw"])
json.dump(OUT, open(HERE / "bow_floor_v6.json", "w"), indent=1); print(OUT)
