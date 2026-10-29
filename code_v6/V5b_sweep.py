# V5b (follow-up 2, 20 Oct: where a push moves the answer). The MVCA Study 2 experiments whose answers followed MVCA in all three run-1
# models (25; config V5B_EXPS), the arm into which MVCA inserted the mediator statement. The experiment's own construct direction and one
# span-random direction (both estimated at each depth) are pushed at three depths (L*, the reading-layer nearest 50% and nearest 70% of
# depth) on two token ranges: "sit" = the situation's tokens (as V4b) and "sit2end" = from the situation's first token to the end of the
# prompt (also the question and the answer prefix), at s = +-2 and +-4 natural shifts of the construct at that depth. Push effect =
# (answer - unedited answer) x MVCA's mediator -> response sign. Depth L* on "sit" at +-2 repeats V4b's own push (built-in check).
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; nL = CUR["n_layers"]; E = mat6("experiments_v6.json"); LSd = DS["layers"]
EIDS = [k for k in (C6("V5B_EXPS") or []) if k in E] or [k for k, e in E.items() if "2" in str(e["studies"]).split(";")]
EIDS = EIDS[:3] if SMALL else EIDS
NV = 2 if SMALL else C6("V5B_NV", 4); NA = 1 if SMALL else C6("V5B_NA", 3)
def near_layer(f): return min((l for l in LSd if l > L), key=lambda l: abs(l - f * nL), default=L)
DEPTHS = [("Lstar", L), ("d50", near_layer(.5)), ("d70", near_layer(.7))]
g = torch.Generator().manual_seed(69); SPANd = {l: span_random6(DS, l, 1, g)[0] for _, l in DEPTHS}
rows, RUNS = [], []
for eid in EIDS:
    e = E[eid]; c = e["construct"]; arm = (e.get("study2_insert_into") or ["treatment"])[0]; ms = sgn(e["m_to_y_sign"])
    I = exp_items6([eid], nv=NV, na=NA, arms=(arm,), levels=("A",)); prompts = [it["prompt"] for it in I]; bodies = [it["body"] for it in I]
    conds = []
    for dn, l in DEPTHS:
        size = natural6(DS, c, l); u_own = unit6(direction6(DS, c, l))
        for rng in ("sit", "sit2end"):
            for kind, u in (("own", u_own), ("span", SPANd[l])):
                for s in (2, -2, 4, -4): conds.append((dn, l, rng, kind, s, s * size * u))
    R_ = run_rows6(list(prompts) + [(pr, [("add", l, vec, bd if rng == "sit" else ("from", bd))]) for _, l, rng, _, _, vec in conds for pr, bd in zip(prompts, bodies)])
    RUNS.append(R_); nt = len(prompts)
    for ci_, (dn, l, rng, kind, s, _) in enumerate(conds):
        o = nt * (ci_ + 1)
        for k, it in enumerate(I):
            rows.append(dict(exp=eid, construct=c, depth=dn, layer=l, range=rng, kind=kind, dose=s, vid=it["vid"], agent=it["agent"], arm=arm, m_to_y_sign=ms,
                             ev=float(R_["ev"][o + k]), ev0=float(R_["ev"][k]), d=float((R_["ev"][o + k] - R_["ev"][k]) * (ms or 1)), mass=float(R_["mass"][o + k]),
                             mass0=float(R_["mass"][k]), p9=";".join(f"{x:.4f}" for x in R_["p9"][o + k].float().tolist())))
    beat(f"V5b {eid}")
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/sweep_v5b.csv", index=False)
S = {"version": "v6.1", "model": MID, "depths": {dn: l for dn, l in DEPTHS}, "n_layers": nL, "mass_min": round(float(D.mass.min()), 3), "cells": {}}
Dk = D[(D.mass >= .5) & (D.mass0 >= .5)]
for (dn, rng), g_ in Dk.groupby(["depth", "range"]):
    for s_abs in (2, 4):
        h = g_[g_.dose.abs() == s_abs].copy(); h["u"] = np.sign(h.dose) * h.d / 2
        pe = h.groupby(["exp", "kind"]).u.mean().unstack()
        if "own" not in pe: continue
        S["cells"][f"{dn}|{rng}|{s_abs}"] = dict(own_pos=int((pe["own"] > 0).sum()), own_gt_span=int((pe["own"] > pe.get("span", 0)).sum()), n=int(pe["own"].notna().sum()),
                                                 own_mean=round(float(pe["own"].mean()), 4), span_mean=round(float(pe["span"].mean()), 4) if "span" in pe else None)
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
Lg = [f"# V5b push sweep ({MID}; depths {S['depths']} of {nL})", "", f"timing {S['timing']}", "", "| depth | tokens | dose | own push > 0 | own > span | own mean | span mean |", "|---|---|---|---|---|---|---|"]
for k_, v in S["cells"].items():
    dn, rng, sa = k_.split("|"); Lg.append(f"| {dn} | {rng} | +-{sa} | {v['own_pos']}/{v['n']} | {v['own_gt_span']}/{v['n']} | {v['own_mean']} | {v['span_mean']} |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); log(f"V5b {MID} done"); beat("V5b ok")
