# V6d (run 3, 21 Oct): V4d (context moderation; the 7 moderated experiments, MVCA's base arm at both levels A/B, same items) repeated at
# V6b's two push locations: "d50|sit" (reading layer nearest 50% of depth, situation tokens) and "d70|sit2end" (nearest 70%, situation to the
# end of the prompt). Directions estimated at the push layer: own, nearest-1 and 2 span-random, s = +-2 natural shifts. MVCA's prediction:
# (push at A - push at B) x x_to_m_sign x x_to_y_sign > 0 (analysis_v6/moderation_v6.py).
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; nL = CUR["n_layers"]; CMP = comps6(); E = mat6("experiments_v6.json"); LSd = DS["layers"]
EIDS = [k for k, e in E.items() if e.get("moderation")]; EIDS = EIDS[:2] if SMALL else EIDS
NV = 2 if SMALL else C6("V4D_NV", 6); NA = 1 if SMALL else C6("V4D_NA", 3)
def near_layer(f): return min((l for l in LSd if l > L), key=lambda l: abs(l - f * nL), default=L)
LOCS = [(f"{d}|{r}", near_layer({"d50": .5, "d70": .7}[d]), r) for d, r in C6("V6B_LOCS", [("d50", "sit"), ("d70", "sit2end")])]
SPAN = {}
for _, l, _ in LOCS:
    if l not in SPAN: g = torch.Generator().manual_seed(171 + l); SPAN[l] = span_random6(DS, l, 2, g)
rows, RUNS = [], []
for eid in EIDS:
    e = E[eid]; c = e["construct"]; arm = e.get("cp3_base_role") or "treatment"
    I = exp_items6([eid], nv=NV, na=NA, arms=(arm,), levels=("A", "B")); prompts = [it["prompt"] for it in I]; bodies = [it["body"] for it in I]
    conds = []
    for loc, l, rng in LOCS:
        size = natural6(DS, c, l); nb = CMP[c]["nearest"][0]
        dirs = [("own", c, unit6(direction6(DS, c, l))), ("near", nb, unit6(direction6(DS, nb, l)))] + [("span", f"span{k}", u) for k, u in enumerate(SPAN[l])]
        conds += [(loc, l, rng, kind, name, s, s * size * u) for kind, name, u in dirs for s in (2, -2)]
    R_ = run_rows6(list(prompts) + [(pr, [("add", l, vec, bd if rng == "sit" else ("from", bd))]) for _, l, rng, _, _, _, vec in conds for pr, bd in zip(prompts, bodies)])
    RUNS.append(R_); nt = len(prompts)
    p9r = lambda k: ";".join(f"{x:.4f}" for x in R_["p9"][k].float().tolist())
    for ci_, (loc, l, rng, kind, name, s, _) in enumerate(conds):
        o = nt * (ci_ + 1)
        for k, it in enumerate(I):
            rows.append(dict(exp=eid, construct=c, loc=loc, layer=l, range=rng, kind=kind, dir=name, dose=s, vid=it["vid"], agent=it["agent"], arm=arm, level=it["level"],
                             cond_level=it["cond_level"], ev=float(R_["ev"][o + k]), ev0=float(R_["ev"][k]), d=float(R_["ev"][o + k] - R_["ev"][k]), mass=float(R_["mass"][o + k]),
                             sign3=sgn(e["study3_interaction_sign"]), p9=p9r(o + k), p9_0=p9r(k)))
    beat(f"V6d {eid}")
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/moderation2_v6d.csv", index=False)
S = {"version": "v6.2", "model": MID, "locations": {loc: l for loc, l, _ in LOCS}, "mass_min": round(float(D.mass.min()), 3), "experiments": {}}
for (loc, eid), g_ in D[D.mass >= .5].groupby(["loc", "exp"]):
    e = E[eid]; s = sgn(e["x_to_m_sign"]) * sgn(e["x_to_y_sign"])
    q = push_of(g_[g_.kind == "own"], ["level", "vid", "agent"]).rename("push").reset_index().pivot_table(index=["vid", "agent"], columns="level", values="push").dropna()
    if "A" in q and "B" in q: S["experiments"][f"{loc}|{eid}"] = dz(((q["A"] - q["B"]) * s).values)
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
Lg = [f"# V6d moderation at mid depth ({MID}; locations {S['locations']})", "", f"timing {S['timing']}", "", "| location | experiment | aligned (A - B) mean | d_z | t |", "|---|---|---|---|---|"]
for k_, v in S["experiments"].items():
    loc, eid = k_.rsplit("|", 1); Lg.append(f"| {loc} | {eid} | {v['mean']} | {v['dz']} | {v['t']} |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); log(f"V6d {MID} done"); beat("V6d ok")
