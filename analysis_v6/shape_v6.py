"""Shape of each construct's representation in a coordinate system shared by the three main LLMs.

For every pair, each LLM's difference vector (high minus low situation, mean over word fragments, layers 30-50% of the way
through the LLM; export_dz_v6.pt) is centered on the mean over all pairs and reduced to its first P principal components.
The shared coordinate system is the first K principal components of the three LLMs' reduced vectors taken together (each
LLM scaled to equal total variance); these components are defined from all pairs without construct labels.
Each LLM's vectors are mapped onto the shared coordinates by ridge regression. For construct c, the mapping is estimated
from the pairs of the other 65 constructs only, so that agreement of c's shape across LLMs cannot arise from the mapping.
The shape of construct c for writer w in LLM m is the mean of the mapped vectors of the pairs that w composed for c.

Outputs: shape_v6.npz (shapes: 66 constructs x 3 LLMs x 6 writers x K, unit-normalized), shape_v6.json (identification
of each construct among 66 from one writer in one LLM, using templates from the other two LLMs and the other five writers;
similarity of shapes of the same and of different constructs)."""
import json, numpy as np, pandas as pd, torch
from pathlib import Path
R = Path(__file__).resolve().parent.parent / 'results_v6'
OUT = Path(__file__).resolve().parent
MODELS = [('Qwen3.6 27B', 'V610_qwen36'), ('Gemma 4 31B', 'V611_gemma4_31b'), ('OLMo 3.1 32B', 'V612_olmo32')]
P, K, LAM = 256, 16, 1e-2

meta = pd.read_csv(R / MODELS[0][1] / 'V1_read' / 'pairs_meta.csv')
cons = sorted(meta.construct.unique()); wrs = sorted(meta.writer.unique())
ci = meta.construct.map({c: i for i, c in enumerate(cons)}).values; wi = meta.writer.map({w: i for i, w in enumerate(wrs)}).values

Z = []
for name, d in MODELS:
    D = torch.load(R / d / 'V1_read' / 'export_dz_v6.pt', map_location='cpu', weights_only=False)['dz_mean_B1'].float()
    D = D - D.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(D, q=P, center=False, niter=4)
    z = (D @ V).numpy(); z /= np.sqrt((z ** 2).sum(1).mean())
    print(name, tuple(D.shape), 'variance in', P, 'PCs:', round(float((S ** 2).sum() / (D ** 2).sum()), 3))
    Z.append(z)
cat = np.concatenate(Z, 1); u, s, vt = np.linalg.svd(cat - cat.mean(0), full_matrices=False)
Sh = u[:, :K] * s[:K]                                   # shared coordinates of every pair (label-free)
print('share of the joint variance in the', K, 'shared dimensions:', round(float((s[:K] ** 2).sum() / (s ** 2).sum()), 3))

def ridge(X, Y):
    return np.linalg.solve(X.T @ X + LAM * np.trace(X.T @ X) / X.shape[1] * np.eye(X.shape[1]), X.T @ Y)

shapes = np.zeros((len(cons), len(MODELS), len(wrs), K)); npairs = np.zeros((len(cons), len(wrs)), int)
for c in range(len(cons)):
    tr, te = ci != c, ci == c
    for m in range(len(MODELS)):
        B = ridge(Z[m][tr], Sh[tr]); Y = Z[m][te] @ B
        for w in range(len(wrs)):
            sel = wi[te] == w; npairs[c, w] = sel.sum()
            if sel.any(): shapes[c, m, w] = Y[sel].mean(0)
nrm = np.linalg.norm(shapes, axis=-1, keepdims=True); unit = np.where(nrm > 0, shapes / np.maximum(nrm, 1e-12), 0)

# identification: shape of (c, m, w) vs templates of all constructs from the other two LLMs and the other five writers
hits, hits_m = 0, np.zeros(len(MODELS)); n = 0; ranks = []
for m in range(len(MODELS)):
    om = [x for x in range(len(MODELS)) if x != m]
    for w in range(len(wrs)):
        ow = [x for x in range(len(wrs)) if x != w]
        T = unit[:, om][:, :, ow].mean((1, 2)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        for c in range(len(cons)):
            if npairs[c, w] == 0: continue
            sim = T @ unit[c, m, w]; r = int((sim > sim[c]).sum()); ranks.append(r)
            hits += r == 0; hits_m[m] += r == 0; n += 1
same, diff = [], []
flat = unit.reshape(len(cons), -1, K)
for a in range(len(cons)):
    for i in range(flat.shape[1]):
        for j in range(i + 1, flat.shape[1]):
            if (i // len(wrs)) != (j // len(wrs)) and (i % len(wrs)) != (j % len(wrs)): same.append(flat[a, i] @ flat[a, j])
    for b in range(a + 1, len(cons)):
        diff.append(float((flat[a] @ flat[b].T).mean()))
res = {'K': K, 'P': P, 'n_shapes': n, 'identification_cross_llm_cross_writer': hits / n,
       'by_llm': {MODELS[m][0]: hits_m[m] / (n / len(MODELS)) for m in range(len(MODELS))},
       'median_rank_of_true_construct': float(np.median(ranks)), 'chance': 1 / len(cons),
       'mean_cos_same_construct_other_llm_other_writer': float(np.mean(same)), 'mean_cos_different_constructs': float(np.mean(diff)),
       'constructs': cons, 'writers': wrs, 'models': [m[0] for m in MODELS]}
json.dump(res, open(OUT / 'shape_v6.json', 'w'), indent=1)
np.savez_compressed(OUT / 'shape_v6.npz', shapes=shapes, unit=unit, npairs=npairs, constructs=np.array(cons), writers=np.array(wrs),
                    models=np.array([m[0] for m in MODELS]))
print({k: v for k, v in res.items() if k not in ('constructs', 'writers', 'models')})
