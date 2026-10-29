# V5c (follow-up 3, 20 Oct: how much, and which group, must be removed). Run 1 removed 10 dimensions at the V4c layers: Qwen3.6 lost the
# effect specifically, Gemma 4 31B lost more when the nearest constructs were removed, OLMo 3.1 32B lost nothing.
# Reading pass: every v6 pair read again (as V1: the text alone as a user turn, mean over its tokens) at the V4c layers and every layer
# from L* to the last -> raw pair differences per construct and layer, the mean text state per layer (mean replacement), the 3 components
# shared by the 66 construct means, and uniform random subspaces in the span of the centred cell differences.
# Removal (mean replacement, every position, both arms of each experiment, condition level A) over two layer ranges: "v4c" = the V4c layers
# (L* and bands 1-2) and "toend" = every layer from L* to the last. Conditions per range:
#   own_spec_k10 / k30 / k100: the construct-specific part (shared components taken out) of the top 10 / 30 / 100 components;
#   unif_k10 / k30 / k100: uniform random subspaces of the same size;
#   fam5_k50 = own + 4 nearest constructs (specific k10 each); near5_k50 = the 5 nearest without the own construct; rand5_k50 = the 5
#   fixed random comparison constructs; unif_k50.
# own_spec_k10 on "v4c" repeats V4c's own_spec_k10__mean (built-in check); summary also checks the re-read subspaces against V1's (L*).
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; nL = CUR["n_layers"]; d = CUR["d"]; SUBL = S1["SUBL"]; CONS = cons6(); CMP = comps6(); E = mat6("experiments_v6.json"); dev = _dev()
EIDS = list(E)[:C6("SMALL_EXPS", 3)] if SMALL else (C6("V5C_EXPS") or list(E))
NV = C6("SMALL_NV", 2) if SMALL else C6("V5C_NV", 4); NA = C6("SMALL_NA", 1) if SMALL else C6("V5C_NA", 2); KS = C6("V5C_KS", [10, 30, 100]); KMAX = max(KS)
RANGES = {"v4c": sorted(SUBL), "toend": list(range(L, nL + 1))}; LR = sorted(set(RANGES["v4c"]) | set(RANGES["toend"]))
# ---- reading pass ----
P = pairs6(SMALL); nP = len(P); CH = C6("V1_CH", 1024); groups = {str(l): [l] for l in LR}
DM = torch.zeros(nP, len(LR), d, dtype=STORE_DTYPE); SUMT = torch.zeros(len(LR), d, dtype=torch.float64); NT = 0; RUNS = []
for k0 in range(0, nP, CH):
    pp = P[k0:k0 + CH]; prompts, marks = [], []
    for p in pp:
        for side in ("high", "low"): prompts.append(chat6(p[side], None, gen=False)); marks.append({"mean": ("mean", p[side])})
    r = run_rows6(prompts, marks, groups, prefix=""); RUNS.append(r); X = r["mean"]
    SUMT += torch.nan_to_num(X.double()).sum(0); NT += X.shape[0]; DM[k0:k0 + len(pp)] = (X[0::2].float() - X[1::2].float()).to(STORE_DTYPE); del r, X
    beat(f"V5c read {k0 + len(pp)}/{nP}")
MU = {l: (SUMT[j] / NT).float().to(dev) for j, l in enumerate(LR)}
cidx = {c: [i for i, p in enumerate(P) if p["construct"] == c] for c in CONS}
cells = {}
for i, p in enumerate(P): cells.setdefault((p["writer"], p["construct"], p["low_type"]), []).append(i)
CK = sorted(cells); _SUB, _SH, _Q = {}, {}, {}
def topk(c, l, k):
    """Top-k right singular vectors of construct c's raw pair differences at layer l (unit rows; as V1, which kept 10)."""
    if (c, l) not in _SUB:
        Z = DM[cidx[c], LR.index(l)].float().to(dev); _SUB[(c, l)] = F.normalize(torch.linalg.svd(Z, full_matrices=False)[2][:KMAX], dim=-1).half().cpu()
    return _SUB[(c, l)][:k].float().to(dev)
def shared(l):
    if l not in _SH:
        Dc = torch.stack([DM[cidx[c], LR.index(l)].float().mean(0) for c in CONS if cidx[c]]).to(dev); _SH[l] = torch.linalg.svd(Dc, full_matrices=False)[2][:3]
    return _SH[l]
def spec(c, k, layers):
    out = {}
    for l in layers:
        U = topk(c, l, k); S = shared(l); U = U - (U @ S.T) @ S; out[l] = ortho6(U)
    return out
def union(subs, layers): return {l: ortho6(torch.cat([s[l] for s in subs])) for l in layers}
def unif(layers, k, g):
    out = {}
    for l in layers:
        if l not in _Q:
            j = LR.index(l); M = torch.stack([DM[cells[k_], j].float().mean(0) for k_ in CK]); M = M - M.mean(0); _Q[l] = torch.linalg.qr(M.T.to(dev))[0]
        Q = _Q[l]; out[l] = ortho6((Q @ torch.randn(Q.shape[1], k, generator=g).to(dev)).T)
    return out
g = torch.Generator().manual_seed(70); UNIF = {(rn, k): unif(ls, k, g) for rn, ls in RANGES.items() for k in KS + [50]}
beat("V5c subspaces ready")
# check: re-read top-10 subspaces at L* vs V1's (mean squared cosine between the two 10-dim subspaces; 1 = identical)
chk = []
for c in sorted({E[e]["construct"] for e in EIDS}):
    if (c, L) in DS["sub"]:
        A = topk(c, L, 10); B = F.normalize(DS["sub"][(c, L)].float().to(dev), dim=-1); chk.append(float(((ortho6(A) @ ortho6(B).T) ** 2).sum() / min(len(A), len(B))))
# ---- removal ----
rows = []
for eid in EIDS:
    e = E[eid]; c = e["construct"]; sx = sgn(e["x_to_y_sign"]) or 1
    I = exp_items6([eid], nv=NV, na=NA, arms=("treatment", "control"), levels=("A",)); prompts = [it["prompt"] for it in I]
    near = CMP[c]["nearest"]; rnd = CMP[c]["random"]; conds = [("none", None, None, 0)]
    for rn, ls in RANGES.items():
        mus = {l: MU[l] for l in ls}
        for k in KS: conds += [(f"own_spec_k{k}__{rn}", spec(c, k, ls), mus, k), (f"unif_k{k}__{rn}", UNIF[(rn, k)], mus, k)]
        s10 = {x: spec(x, 10, ls) for x in [c] + near[:5] + rnd[:5]}
        conds += [(f"fam5_k50__{rn}", union([s10[x] for x in [c] + near[:4]], ls), mus, 50), (f"near5_k50__{rn}", union([s10[x] for x in near[:5]], ls), mus, 50),
                  (f"rand5_k50__{rn}", union([s10[x] for x in rnd[:5]], ls), mus, 50), (f"unif_k50__{rn}", UNIF[(rn, 50)], mus, 50)]
    R_ = run_rows6([(pr, [] if Us is None else [("proj", Us, None, mus)]) for _, Us, mus, _ in conds for pr in prompts]); RUNS.append(R_); nt = len(prompts)
    for ci_, (name, _, _, k) in enumerate(conds):
        for kk, it in enumerate(I):
            o = nt * ci_ + kk
            rows.append(dict(exp=eid, construct=c, cond=name, k=k, vid=it["vid"], agent=it["agent"], arm=it["arm"], ev=float(R_["ev"][o]), mass=float(R_["mass"][o]), x_to_y_sign=sx,
                             p9=";".join(f"{x:.4f}" for x in R_["p9"][o].float().tolist())))
    del conds, R_; beat(f"V5c {eid}")
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/remove2_v5c.csv", index=False)
# summary: share of the unedited effect left (experiments with |effect| > .3; items with digit mass >= .5 in both arms)
S = {"version": "v6.1", "model": MID, "ranges": RANGES, "ks": KS, "subspace_check_Lstar_median": round(float(np.median(chk)), 4) if chk else None,
     "subspace_check_Lstar_min": round(float(np.min(chk)), 4) if chk else None, "mass_lt_.5_by_cond": D.groupby("cond").mass.apply(lambda m: round(float((m < .5).mean()), 4)).to_dict()}
W = D.pivot_table(index=["exp", "cond", "vid", "agent"], columns="arm", values=["ev", "mass"]).reset_index(); W.columns = ["_".join(x).strip("_") for x in W.columns]
W = W[(W.mass_treatment >= .5) & (W.mass_control >= .5)]; W["eff"] = (W.ev_treatment - W.ev_control) * W.exp.map(D.groupby("exp").x_to_y_sign.first())
Ef = W.groupby(["exp", "cond"]).eff.mean().unstack() if len(W) else pd.DataFrame()
big = Ef[Ef["none"].abs() > .3] if "none" in Ef else Ef.iloc[0:0]; Sh = big.div(big["none"], axis=0) if len(big) else big
S["n_experiments_effect_gt_.3"] = int(len(big)); S["median_share_left"] = {k: round(float(v), 3) for k, v in Sh.median().items()} if len(big) else {}
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
Lg = [f"# V5c removal: size and group ({MID}; ranges v4c {RANGES['v4c'][0]}-{RANGES['v4c'][-1]}, toend {L}-{nL})", "",
      f"re-read subspaces vs V1 at L*: median {S['subspace_check_Lstar_median']}, min {S['subspace_check_Lstar_min']}; {S['n_experiments_effect_gt_.3']} experiments with |effect| > .3; timing {S['timing']}", "",
      "| condition | median share of the effect left |", "|---|---|"] + [f"| {k} | {v} |" for k, v in S["median_share_left"].items()]
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); del DM, _SUB; gc.collect(); log(f"V5c {MID} done"); beat("V5c ok")
