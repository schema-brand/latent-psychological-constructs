# V1 (Studies 1-2, reading; also each training checkpoint of the origins analysis): every v6 contrast pair read once, both texts as a
# user turn with no agent, at the layer set (10% grid + band 1 + band 2). Mean reading (mean over the text's tokens) and end reading (last
# token of the text). Per layer: leave-one-writer-out identification among the 66 constructs (dimensions standardized over texts; raw for
# comparison), first-place and top-5 rates; Campbell-Fiske cells at the bands and at L*. L* = the first grid layer (>= 10% of depth) whose
# standardized accuracy reaches 95% of the best grid layer (preregistered intervention layer). Saves the direction source for V3-V4 (cell
# means per writer x construct x low pole, per-layer std, natural sizes, top-10 subspaces at L* and the bands), compact exports for the
# local analysis, the logit-lens readout of every construct (end reading, 50-100% depth) and, with V1_FACET = k, the agent facet.
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
diag = load6(MID, REV); nL = CUR["n_layers"]; d = CUR["d"]; B1, B2, _ = bands(); LS = layer_set(); GRID = grid6(nL)
P = pairs6(SMALL, frac=C6("V1_FRAC", 1)); CONS = cons6(); WR = sorted({p["writer"] for p in P}); nP = len(P)
ci = torch.tensor([CONS.index(p["construct"]) for p in P]); wi = torch.tensor([WR.index(p["writer"]) for p in P])
iB1 = [LS.index(l) for l in B1]; iB2 = [LS.index(l) for l in B2]; NAMES = [str(l) for l in LS] + ["B1", "B2"]; RO_L = [l for l in GRID if l >= 0.5 * nL]
groups = {str(l): [l] for l in LS}
DM = torch.zeros(nP, len(LS), d, dtype=STORE_DTYPE); DE = torch.zeros_like(DM); TB1 = torch.zeros(2 * nP, d, dtype=torch.float16)
SUM = {rd: torch.zeros(len(NAMES), d, dtype=torch.float64) for rd in ("mean", "end")}; SQ = {rd: torch.zeros(len(NAMES), d, dtype=torch.float64) for rd in ("mean", "end")}
RS = torch.zeros(len(CONS), 2, len(RO_L), d); RN = torch.zeros(len(CONS)); iRO = [LS.index(l) for l in RO_L]; RUNS = []
def bandify(X): return torch.cat([X.float(), X[:, iB1].float().mean(1, keepdim=True), X[:, iB2].float().mean(1, keepdim=True)], 1)
CH = C6("V1_CH", 1024); NMISS = 0           # pairs per batched call (run 1: 64; texts of equal token length run side by side)
for k0 in range(0, nP, CH):
    pp = P[k0:k0 + CH]; prompts, marks = [], []
    for p in pp:
        for side in ("high", "low"): prompts.append(chat6(p[side], None, gen=False)); marks.append({"mean": ("mean", p[side]), "end": ("last", p[side])})
    r = run_marks6(prompts, marks, groups, prefix=""); RUNS.append(r)
    for rd, X in (("mean", r["mean"]), ("end", r["end"])):
        Xb = torch.nan_to_num(bandify(X).double()); SUM[rd] += Xb.sum(0); SQ[rd] += (Xb ** 2).sum(0)
        if rd == "mean": NMISS += int(torch.isnan(X[:, 0, 0].float()).sum())
        D_ = (X[0::2].float() - X[1::2].float()).to(STORE_DTYPE)
        (DM if rd == "mean" else DE)[k0:k0 + len(pp)] = D_
    TB1[2 * k0:2 * k0 + 2 * len(pp)] = r["mean"][:, iB1].float().mean(1).half()
    for n_, p in enumerate(pp):
        c = CONS.index(p["construct"]); RS[c, 0] += r["end"][2 * n_, iRO].float(); RS[c, 1] += r["end"][2 * n_ + 1, iRO].float(); RN[c] += 1
    del r; beat(f"V1 read {k0 + len(pp)}/{nP}")
NT = 2 * nP; SD = {rd: ((SQ[rd] / NT - (SUM[rd] / NT) ** 2).clamp(min=0).sqrt() + 1e-6).float() for rd in SUM}
def layer_view(X, j): return X[:, j].float() if j < len(LS) else X[:, iB1 if NAMES[j] == "B1" else iB2].float().mean(1)
def lowo6(Z):
    """Leave-one-writer-out nearest-centroid identification: centroid = mean over the other writers of their cell means (unit length).
    Returns own-rank per pair (1 = first) and the cosines with all 66 centroids [n, 66]."""
    nW, nC = len(WR), len(CONS); S = torch.zeros(nW, nC, Z.shape[1]); N = torch.zeros(nW, nC)
    S.index_put_((wi, ci), Z, accumulate=True); N.index_put_((wi, ci), torch.ones(len(Z)), accumulate=True)
    Mc = S / N.clamp(min=1)[..., None]; has = N > 0; rank = torch.zeros(len(Z), dtype=torch.long); COS = torch.full((len(Z), nC), float("nan"))
    Zn = F.normalize(Z, dim=-1)
    for w in range(nW):
        oth = has.clone(); oth[w] = False; cnt = oth.sum(0)
        C = (Mc * oth[..., None]).sum(0) / cnt.clamp(min=1)[:, None]; Cn = F.normalize(C, dim=-1)
        ii = (wi == w).nonzero().flatten()
        if not len(ii): continue
        sims = Zn[ii] @ Cn.T; sims[:, cnt == 0] = -9; COS[ii] = sims
        rank[ii] = (sims >= sims.gather(1, ci[ii][:, None])).sum(1)          # ties count against the own construct (zero differences rank last)
    return rank, COS
def mtmm6(Z):
    """Cell directions (writer x construct, unit) -> mean cosine: same construct / different writers (convergent), different constructs /
    same writer (method), different constructs and writers (baseline); convergent per construct."""
    cells = {}
    for i_, p in enumerate(P): cells.setdefault((wi[i_].item(), ci[i_].item()), []).append(i_)
    keys = sorted(cells); V = F.normalize(torch.stack([Z[cells[k]].mean(0) for k in keys]), dim=-1); Cm = V @ V.T
    kw = torch.tensor([k[0] for k in keys]); kc = torch.tensor([k[1] for k in keys]); sw = kw[:, None] == kw[None]; sc = kc[:, None] == kc[None]
    off = ~torch.eye(len(keys), dtype=torch.bool)
    conv_c = {CONS[c]: round(float(Cm[(kc[:, None] == c) & (kc[None] == c) & ~sw].mean()), 3) for c in range(len(CONS)) if ((kc[:, None] == c) & (kc[None] == c) & ~sw).any()}
    return dict(convergent=round(float(Cm[sc & ~sw].mean()), 3), method=round(float(Cm[~sc & sw & off].mean()), 3), baseline=round(float(Cm[~sc & ~sw].mean()), 3), convergent_by_construct=conv_c)
prof = {"std_mean": {}, "raw_mean": {}, "std_end": {}}; PROJ = torch.zeros(nP, len(NAMES), len(CONS), dtype=torch.float16)
for j, nm in enumerate(NAMES):
    for key, X, rd, std in (("std_mean", DM, "mean", True), ("raw_mean", DM, "mean", False), ("std_end", DE, "end", True)):
        Z = layer_view(X, j); Z = Z / SD[rd][j] if std else Z
        rank, COS = lowo6(Z); acc_c = {CONS[c]: round(float((rank[ci == c] == 1).float().mean()), 3) for c in range(len(CONS)) if (ci == c).any()}
        prof[key][nm] = dict(acc=round(float((rank == 1).float().mean()), 4), top5=round(float((rank <= 5).float().mean()), 4), acc_mediators=round(float(np.mean([acc_c[c] for c in acc_c if spec6()[c]["role"] == "mediator"])), 4), by_construct=acc_c)
        if key == "std_mean": PROJ[:, j] = COS.half()
    beat(f"V1 profile {nm}")
gacc = {l: prof["std_mean"][str(l)]["acc"] for l in GRID}; best = max(gacc[l] for l in GRID if l >= 0.1 * nL)
LSTAR = int(C6("V_LSTAR") or next(l for l in GRID if l >= 0.1 * nL and gacc[l] >= 0.95 * best)); jL = LS.index(LSTAR) if LSTAR in LS else None
SUBL = sorted(set(B1) | set(B2) | {LSTAR})
R = {"version": "v6", "model": MID, "rev": REV, "small": bool(SMALL), "n_layers": nL, "d": d, "layers": LS, "grid": GRID, "B1": B1, "B2": B2, "Lstar": LSTAR, "SUBL": SUBL,
     "Lstar_rule": "first grid layer >= 10% depth with std mean-reading LOWO accuracy >= 95% of the best grid layer", "grid_acc": {str(l): gacc[l] for l in GRID}, "best_grid_acc": best,
     "n_pairs": nP, "n_texts_unmatched": NMISS, "writers": WR, "load": diag, "profile": prof}
for nm, j in (("B1", NAMES.index("B1")), ("B2", NAMES.index("B2")), ("Lstar", jL)):
    if j is not None: R[f"mtmm_{nm}"] = mtmm6(layer_view(DM, j) / SD["mean"][j])
beat("V1 mtmm done")
# direction source
DS = {"version": "v6", "model": MID, "rev": REV, "layers": LS, "B1": B1, "B2": B2, "Lstar": LSTAR, "SUBL": SUBL, "cells": {}, "std": SD["mean"][:len(LS)].half(), "norm": {}, "sub": {},
      "mu": {l: (SUM["mean"][LS.index(l)] / NT).float() for l in SUBL}}            # mean text state per layer (mean replacement in V4c)
cidx = {}
for i_, p in enumerate(P): cidx.setdefault((p["writer"], p["construct"], p["low_type"]), []).append(i_)
for k_, ii in cidx.items(): DS["cells"][k_] = (DM[ii].float().mean(0).half(), len(ii))
dev = _dev()
for c in CONS:
    ii = (ci == CONS.index(c)).nonzero().flatten()
    for l in LS: DS["norm"][(c, l)] = float(DM[ii, LS.index(l)].float().norm(dim=-1).mean())
    for l in SUBL:
        j = LS.index(l); Z = DM[ii, j].float().to(dev)          # raw differences: standardizing and scaling back inflates a few huge-valued
        U = torch.linalg.svd(Z, full_matrices=False)[2][:10].cpu(); DS["sub"][(c, l)] = F.normalize(U, dim=-1).half()   # dimensions (Gemma), so removal hit them
torch.save(DS, f"{OUT}/dirsrc.pt"); beat("V1 dirsrc saved")
# exports for the local analysis (standardized difference vectors and text PCs; pair metadata)
pd.DataFrame([{k: p[k] for k in ("pid", "writer", "construct", "role", "low_type", "domain")} for p in P]).to_csv(f"{OUT}/pairs_meta.csv", index=False)
EX = {"model": MID, "rev": REV, "layers": NAMES, "Lstar": LSTAR}
jb1 = NAMES.index("B1"); EX["sd_mean_B1"] = SD["mean"][jb1].half(); EX["raw_norm_mean_B1"] = layer_view(DM, jb1).norm(dim=-1).half()
torch.save({"model": MID, "rev": REV, "dz_mean_B1": (layer_view(DM, jb1) / SD["mean"][jb1]).half()}, f"{OUT}/export_dz_v6.pt")      # < 90 MB for d <= 5376
for nm, X, rd in (("mean_B2", DM, "mean"), ("end_B1", DE, "end"), ("mean_Lstar", DM, "mean")):
    j = NAMES.index("B2") if nm == "mean_B2" else (NAMES.index("B1") if nm == "end_B1" else jL)
    if j is None: continue
    Z = (layer_view(X, j) / SD[rd][j]).to(dev); U, Sv, _ = torch.linalg.svd(Z, full_matrices=False); k = min(1024, len(Sv))
    EX[f"pcs_{nm}"] = (U[:, :k] * Sv[:k]).half().cpu(); EX[f"sv_{nm}"] = Sv.cpu(); del Z, U
Zt = (TB1.float() - TB1.float().mean(0)) / (TB1.float().std(0) + 1e-6); U, Sv, _ = torch.linalg.svd(Zt.to(dev), full_matrices=False)
EX["text_pcs_mean_B1"] = (U[:, :256] * Sv[:256]).half().cpu(); EX["text_sv_mean_B1"] = Sv.cpu(); del U, Zt
torch.save(EX, f"{OUT}/export_v6.pt"); torch.save({"proj": PROJ, "layers": NAMES, "constructs": CONS}, f"{OUT}/lowo_cos_v6.pt"); beat("V1 exports saved")
# logit-lens readout (end reading, mean high state vs mean low state of each construct)
RO = {}
try:
    for c in range(len(CONS)):
        if RN[c] == 0: continue
        RO[CONS[c]] = {str(l): logit_lens_contrast(RS[c, 0, k] / RN[c], RS[c, 1, k] / RN[c]) for k, l in enumerate(RO_L)}
    R["readout_ok"] = True
except Exception:
    R["readout_ok"] = False; R["readout_error"] = traceback.format_exc()[-600:]
save_json("readout_logitlens.json", RO)
# agent facet: the first L4 pair of every writer x construct cell read under k MVCA agents (band 1, band 2, L*)
NF = C6("V1_FACET", 0)
if NF:
    first = {}
    for i_, p in enumerate(P):
        k_ = (p["writer"], p["construct"])
        if k_ not in first or (p["low_type"] == "L4" and P[first[k_]]["low_type"] != "L4"): first[k_] = i_
    PF = sorted(first.values()); AG = agents6()[:NF]; GF = {"B1": B1, "B2": B2, "L": [LSTAR]}; FAC = torch.zeros(len(PF), NF, 3, d, dtype=torch.float16)
    for a_, ag in enumerate(AG):
        prompts, marks = [], []
        for i_ in PF:
            for side in ("high", "low"): t = P[i_][side]; prompts.append(chat6(t, SYS + ag["persona"], gen=False)); marks.append({"mean": ("mean", t)})
        r = run_marks6(prompts, marks, GF, prefix="", tag=f"V1 facet {ag['id']}"); RUNS.append(r)
        FAC[:, a_] = (r["mean"][0::2].float() - r["mean"][1::2].float()).half(); del r
    torch.save({"diff": FAC, "pair_index": PF, "agents": [a["id"] for a in AG], "groups": list(GF)}, f"{OUT}/facet_v6.pt")
R["timing"] = timing(RUNS); R["wall_s"] = round(time.time() - T0); save_json("summary.json", R)
L = [f"# V1 reading ({MID}{'@' + REV if REV else ''}; {nP} pairs, {len(WR)} writers; L* = {LSTAR} of {nL})", "",
     f"Timing: {R['timing']}", "", "| layer | std mean: first / top5 / mediators | raw mean: first | std end: first |", "|---|---|---|---|"]
for nm in NAMES: L.append(f"| {nm} | {prof['std_mean'][nm]['acc']} / {prof['std_mean'][nm]['top5']} / {prof['std_mean'][nm]['acc_mediators']} | {prof['raw_mean'][nm]['acc']} | {prof['std_end'][nm]['acc']} |")
for nm in ("B1", "B2", "Lstar"):
    if f"mtmm_{nm}" in R: m = R[f"mtmm_{nm}"]; L.append(f"\nMTMM {nm}: convergent {m['convergent']}, same writer {m['method']}, baseline {m['baseline']}")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(L)); del DM, DE, PROJ; gc.collect(); log(f"V1 {MID} done L*={LSTAR}"); beat("V1 ok")
