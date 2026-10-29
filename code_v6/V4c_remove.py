# V4c (Study 4c, necessity): removing the representation. Both arms of each experiment (condition level A) are read with a subspace removed
# at L* and at every layer of bands 1 and 2, in two matched forms:
#   zeroing ("__zero"): the state's component in the subspace is set to zero (the test-plan-3 method);
#   mean replacement ("__mean"): only the state's deviation from the mean text state is removed within the subspace.
# Subspaces come from the raw pair differences (top 10 components per construct and layer):
#   own_k10: the experiment's own construct; own_spec_k10: its components after the 3 components shared by all 66 constructs are taken out
#   (construct-specific part); near_spec_k10 / rand_spec_k10: the specific parts of 2 nearest and 2 fixed random constructs; unif_k10: 2
#   random 10-dimensional subspaces drawn uniformly within the span of real cell differences; shared_k3: the shared components (an
#   evaluative axis common to many constructs).
# Removal is at every position (the situation's information also reaches the question tokens through the layers below L*); a short set
# zeroes on the situation's tokens only ("__sitzero"). Smoke (19 Oct, small samples): in Qwen3.6, zeroing the own construct left 44% of the
# effect vs 92% for a uniform random subspace; mean replacement left every effect intact; in Gemma 4 zeroing had been non-specific.
# Outcome: treatment - control answer x MVCA's X -> response sign, compared with the unedited effect. Primary necessity test, per model:
# own_spec_k10 vs near_spec_k10, rand_spec_k10 and unif_k10 under zeroing; mean replacement reported alongside.
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; SUBL = S1["SUBL"]; CONS = cons6(); CMP = comps6(); E = mat6("experiments_v6.json"); EIDS = list(E)[:C6("SMALL_EXPS", 3)] if SMALL else (C6("V4C_EXPS") or list(E))
NV = C6("SMALL_NV", 2) if SMALL else C6("V4C_NV", 4); NA = C6("SMALL_NA", 1) if SMALL else C6("V4C_NA", 2)
g = torch.Generator().manual_seed(66); SH = shared6(DS, SUBL, 3); UNI = [unif_sub6(DS, SUBL, 10, g) for _ in range(2)]; rows, RUNS = [], []
for eid in EIDS:
    e = E[eid]; c = e["construct"]; sx = sgn(e["x_to_y_sign"]) or 1
    I = exp_items6([eid], nv=NV, na=NA, arms=("treatment", "control"), levels=("A",)); prompts = [it["prompt"] for it in I]; bodies = [it["body"] for it in I]
    own10 = sub6(DS, c, 10, SUBL); spec10 = specific6(DS, c, 10, SUBL, SH)
    near = [(f"near_spec_k10_{x}", specific6(DS, x, 10, SUBL, SH)) for x in CMP[c]["nearest"][:2]]; rand = [(f"rand_spec_k10_{x}", specific6(DS, x, 10, SUBL, SH)) for x in CMP[c]["random"][:2]]
    unif = [(f"unif{k}_k10", U) for k, U in enumerate(UNI)]
    full = [("own_k10", own10), ("own_spec_k10", spec10)] + near + rand + unif + [("shared_k3", SH)]
    conds = [("none", None, "all")] + [(f"{n}__zero", U, "zero") for n, U in full] + [(f"{n}__mean", U, "mean") for n, U in full]
    conds += [(f"{n}__sitzero", U, "sitzero") for n, U in [("own_spec_k10", spec10), unif[0]]]
    # one batched call: every condition x item (removal edits act on their own row only)
    R_ = run_rows6([(pr, [] if Us is None else [("proj", Us, bd if where == "sitzero" else None, DS.get("mu") if where == "mean" else None)])
                    for name, Us, where in conds for pr, bd in zip(prompts, bodies)]); RUNS.append(R_); nt = len(prompts)
    for ci_, (name, Us, where) in enumerate(conds):
        for k, it in enumerate(I):
            o = nt * ci_ + k
            rows.append(dict(exp=eid, construct=c, cond=name, where=where, vid=it["vid"], agent=it["agent"], arm=it["arm"], ev=float(R_["ev"][o]), mass=float(R_["mass"][o]), x_to_y_sign=sx, p9=";".join(f"{x:.4f}" for x in R_["p9"][o].float().tolist())))
    beat(f"V4c {eid}")
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/remove_v4c.csv", index=False)
S = {"version": "v6", "model": MID, "layers": SUBL, "experiments": {}, "mass_by_cond": D.groupby("cond").mass.min().round(3).to_dict()}
for eid, g_ in D.groupby("exp"):
    t = g_[g_.arm == "treatment"].set_index(["cond", "vid", "agent"]).ev; c_ = g_[g_.arm == "control"].set_index(["cond", "vid", "agent"]).ev
    eff = ((t - c_.reindex(t.index)) * g_.x_to_y_sign.iloc[0]).groupby(level=0).mean(); e0 = float(eff.get("none", np.nan))
    ex = dict(effect=round(e0, 3), mass_min=round(float(g_.mass.min()), 3))
    for k, v in eff.items():
        if k == "none": continue
        n, w = k.split("__"); grp = next((pre for pre in ("near_spec_k10", "rand_spec_k10", "unif") if n.startswith(pre)), n); ex.setdefault(f"{grp}__{w}", []).append(float(v))
    ex = {k: (round(float(np.mean(v)), 3) if isinstance(v, list) else v) for k, v in ex.items()}; S["experiments"][eid] = ex
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
COLS = [f"{a}__{w}" for w in ("zero", "mean") for a in ("own_k10", "own_spec_k10", "near_spec_k10", "rand_spec_k10", "unif", "shared_k3")] + ["own_spec_k10__sitzero", "unif__sitzero"]
Lg = [f"# V4c removal: zeroing and mean replacement ({MID}; layers {SUBL[0]}-{SUBL[-1]}; every position unless 'sit')", "", f"timing {S['timing']}; min digit mass by condition {S['mass_by_cond']}", "",
      "| experiment | effect | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
for k, v in S["experiments"].items(): Lg.append(f"| {k} | {v['effect']:+.2f} | " + " | ".join(str(v.get(c)) for c in COLS) + " |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); log(f"V4c {MID} done"); beat("V4c ok")
