# V4b (Study 4b; MVCA Study 2): steering the answer in the published experiments. For the 52 MVCA Study 2 experiments, the arm into which
# MVCA inserted the mediator statement (treatment in all 52) is read with the situation's tokens moved at L* along the experiment's own
# construct direction (s = +-1, +-2, +-4 natural shifts of that construct), along its 3 nearest constructs, 3 fixed random constructs and
# 2 span-random directions (s = +-2). Push effect = (answer - unedited answer) x MVCA's mediator -> response sign; positive minus negative
# push corresponds to MVCA's "mediator statement minus denial".
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; CONS = cons6(); CMP = comps6(); E = mat6("experiments_v6.json")
EIDS = [k for k, e in E.items() if "2" in str(e["studies"]).split(";")]; EIDS = EIDS[:3] if SMALL else ([k for k in EIDS if k in C6("V4B_EXPS")] if C6("V4B_EXPS") else EIDS)
NV = 2 if SMALL else C6("V4B_NV", 6); NA = 1 if SMALL else C6("V4B_NA", 3); NN = C6("V4B_NEAR", 3); NR = C6("V4B_RAND", 3)
g = torch.Generator().manual_seed(65); SPAN = span_random6(DS, L, 2, g); rows, RUNS = [], []
for eid in EIDS:
    e = E[eid]; c = e["construct"]; arm = (e.get("study2_insert_into") or ["treatment"])[0]; ms = sgn(e["m_to_y_sign"]); size = natural6(DS, c, L)
    I = exp_items6([eid], nv=NV, na=NA, arms=(arm,), levels=("A",)); prompts = [it["prompt"] for it in I]; bodies = [it["body"] for it in I]
    dirs = [("own", c, [1, -1, 2, -2, 4, -4])] + [("near", x, [2, -2]) for x in CMP[c]["nearest"][:NN]] + [("rand", x, [2, -2]) for x in CMP[c]["random"][:NR]]
    dirs = [(k, n, unit6(direction6(DS, n, L)), s) for k, n, s in dirs] + [("span", f"span{k}", u, [2, -2]) for k, u in enumerate(SPAN)]
    conds = [(kind, name, s, s * size * u) for kind, name, u, doses in dirs for s in doses]
    R_ = run_rows6(list(prompts) + [(pr, [("add", L, vec, bd)]) for _, _, _, vec in conds for pr, bd in zip(prompts, bodies)]); RUNS.append(R_); nt = len(prompts)
    p9r = lambda k: ";".join(f"{x:.4f}" for x in R_["p9"][k].float().tolist())
    for ci_, (kind, name, s, _) in enumerate(conds):
        o = nt * (ci_ + 1)
        for k, it in enumerate(I):
            rows.append(dict(exp=eid, construct=c, kind=kind, dir=name, dose=s, vid=it["vid"], agent=it["agent"], arm=arm, m_to_y_sign=ms, ev=float(R_["ev"][o + k]), ev0=float(R_["ev"][k]),
                             d=float((R_["ev"][o + k] - R_["ev"][k]) * (ms or 1)), mass=float(R_["mass"][o + k]), mass0=float(R_["mass"][k]), p9=p9r(o + k), p9_0=p9r(k)))
    beat(f"V4b {eid}")
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/steer_v4b.csv", index=False)
S = {"version": "v6", "model": MID, "layer": L, "mass_min": round(float(D.mass.min()), 3), "share_mass_lt_.5": round(float((D.mass < .5).mean()), 4), "experiments": {}}
for eid, g_ in D.groupby("exp"):
    pv = push_of(g_, ["kind", "dir", "vid", "agent"]).rename("push").reset_index(); own = pv[pv.kind == "own"].set_index(["vid", "agent"]).push
    oth = pv[pv.kind.isin(["near", "rand"])].groupby(["vid", "agent"]).push.mean(); spn = pv[pv.kind == "span"].groupby(["vid", "agent"]).push.mean()
    dose = g_[g_.kind == "own"].groupby("dose").d.mean()
    S["experiments"][eid] = dict(own=dz(own.values), own_minus_others=dz((own - oth.reindex(own.index)).values), own_minus_span=dz((own - spn.reindex(own.index)).values),
                                 dose={f"{float(k):g}": round(float(v), 3) for k, v in dose.items()}, m_to_y_sign=int(g_.m_to_y_sign.iloc[0]))
ex = S["experiments"]; S["n_own_pos"] = int(sum((v["own"]["mean"] or 0) > 0 for v in ex.values())); S["n_own_minus_others_pos"] = int(sum((v["own_minus_others"]["mean"] or 0) > 0 for v in ex.values()))
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
Lg = [f"# V4b steering in the MVCA Study 2 experiments ({MID}; situation tokens at hidden layer {L})", "", f"own push > 0: {S['n_own_pos']}/{len(ex)}; own - others > 0: {S['n_own_minus_others_pos']}/{len(ex)}; timing {S['timing']}", "",
      "| experiment | own push | own - others d_z | dose -4/-2/-1/+1/+2/+4 |", "|---|---|---|---|"]
for k, v in ex.items(): Lg.append(f"| {k} | {v['own']['mean']} | {v['own_minus_others']['dz']} | {' / '.join(str(v['dose'].get(s)) for s in ('-4', '-2', '-1', '1', '2', '4'))} |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); log(f"V4b {MID} done"); beat("V4b ok")
