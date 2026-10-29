# V6b (run 3, 21 Oct): V4b (MVCA Study 2 experiments, the arm into which MVCA inserted the mediator statement, same items and the same
# direction set) repeated at two push locations fixed before the Gemma and OLMo sweeps were seen, from the Qwen3.6 sweep (V5b: pushes at
# L* moved answers in 14-15/25 experiments, at ~50% depth in 20/25 with 4-5x larger effects):
#   primary  "d50|sit"     = the reading layer nearest 50% of depth, the situation's tokens (V4b's token range);
#   secondary "d70|sit2end" = the reading layer nearest 70% of depth, from the situation's first token to the end of the prompt.
# Directions as V4b, each estimated at the push layer: own (s = +-1, +-2, +-4 natural shifts of the construct at that layer), 3 nearest,
# 3 fixed random constructs and 2 span-random directions (s = +-2). Push effect = (answer - unedited answer) x MVCA's mediator -> response sign.
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; nL = CUR["n_layers"]; CMP = comps6(); E = mat6("experiments_v6.json"); LSd = DS["layers"]
EIDS = [k for k, e in E.items() if "2" in str(e["studies"]).split(";")]; EIDS = EIDS[:3] if SMALL else EIDS
NV = 2 if SMALL else C6("V4B_NV", 6); NA = 1 if SMALL else C6("V4B_NA", 3); NN = C6("V4B_NEAR", 3); NR = C6("V4B_RAND", 3)
def near_layer(f): return min((l for l in LSd if l > L), key=lambda l: abs(l - f * nL), default=L)
LOCS = [(f"{d}|{r}", near_layer({"d50": .5, "d70": .7}[d]), r) for d, r in C6("V6B_LOCS", [("d50", "sit"), ("d70", "sit2end")])]
SPAN = {}
for _, l, _ in LOCS:
    if l not in SPAN: g = torch.Generator().manual_seed(71 + l); SPAN[l] = span_random6(DS, l, 2, g)
rows, RUNS = [], []
for eid in EIDS:
    e = E[eid]; c = e["construct"]; arm = (e.get("study2_insert_into") or ["treatment"])[0]; ms = sgn(e["m_to_y_sign"])
    I = exp_items6([eid], nv=NV, na=NA, arms=(arm,), levels=("A",)); prompts = [it["prompt"] for it in I]; bodies = [it["body"] for it in I]
    conds = []
    for loc, l, rng in LOCS:
        size = natural6(DS, c, l)
        dirs = [("own", c, [1, -1, 2, -2, 4, -4])] + [("near", x, [2, -2]) for x in CMP[c]["nearest"][:NN]] + [("rand", x, [2, -2]) for x in CMP[c]["random"][:NR]]
        dirs = [(k, n, unit6(direction6(DS, n, l)), s) for k, n, s in dirs] + [("span", f"span{k}", u, [2, -2]) for k, u in enumerate(SPAN[l])]
        conds += [(loc, l, rng, kind, name, s, s * size * u) for kind, name, u, doses in dirs for s in doses]
    R_ = run_rows6(list(prompts) + [(pr, [("add", l, vec, bd if rng == "sit" else ("from", bd))]) for _, l, rng, _, _, _, vec in conds for pr, bd in zip(prompts, bodies)])
    RUNS.append(R_); nt = len(prompts)
    p9r = lambda k: ";".join(f"{x:.4f}" for x in R_["p9"][k].float().tolist())
    for ci_, (loc, l, rng, kind, name, s, _) in enumerate(conds):
        o = nt * (ci_ + 1)
        for k, it in enumerate(I):
            rows.append(dict(exp=eid, construct=c, loc=loc, layer=l, range=rng, kind=kind, dir=name, dose=s, vid=it["vid"], agent=it["agent"], arm=arm, m_to_y_sign=ms,
                             ev=float(R_["ev"][o + k]), ev0=float(R_["ev"][k]), d=float((R_["ev"][o + k] - R_["ev"][k]) * (ms or 1)), mass=float(R_["mass"][o + k]),
                             mass0=float(R_["mass"][k]), p9=p9r(o + k), p9_0=p9r(k)))
    beat(f"V6b {eid}")
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/steer2_v6b.csv", index=False)
S = {"version": "v6.2", "model": MID, "Lstar": L, "n_layers": nL, "locations": {loc: l for loc, l, _ in LOCS}, "mass_min": round(float(D.mass.min()), 3), "by_location": {}}
Dk = D[(D.mass >= .5) & (D.mass0 >= .5)]
for loc, g_ in Dk.groupby("loc"):
    pv = push_of(g_, ["exp", "kind", "dir", "vid", "agent"]).rename("push").reset_index()
    own = pv[pv.kind == "own"].groupby("exp").push.mean(); oth = pv[pv.kind.isin(["near", "rand"])].groupby("exp").push.mean(); spn = pv[pv.kind == "span"].groupby("exp").push.mean()
    S["by_location"][loc] = dict(n=int(len(own)), own_pos=int((own > 0).sum()), own_gt_others=int((own > oth.reindex(own.index)).sum()), own_gt_span=int((own > spn.reindex(own.index)).sum()),
                                 own_mean=round(float(own.mean()), 4), others_mean=round(float(oth.mean()), 4), span_mean=round(float(spn.mean()), 4),
                                 per_experiment={k: round(float(v), 4) for k, v in own.items()})
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
Lg = [f"# V6b steering at mid depth ({MID}; L* {L} of {nL}; locations {S['locations']})", "", f"timing {S['timing']}", "",
      "| location | own push > 0 | own > others | own > span | own mean | others mean | span mean |", "|---|---|---|---|---|---|---|"]
for loc, v in S["by_location"].items(): Lg.append(f"| {loc} | {v['own_pos']}/{v['n']} | {v['own_gt_others']}/{v['n']} | {v['own_gt_span']}/{v['n']} | {v['own_mean']} | {v['others_mean']} | {v['span_mean']} |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); log(f"V6b {MID} done"); beat("V6b ok")
