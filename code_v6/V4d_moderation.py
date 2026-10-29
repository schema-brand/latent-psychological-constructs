# V4d (Study 4d; MVCA Study 3): context moderation. For the 7 moderated experiments, MVCA's base arm is read at both levels of the
# moderator (A = the level of the Study 1 comparison, B = the other level) with the situation's tokens moved at L* along the own construct
# direction, the nearest construct and 2 span-random directions (s = +-2 natural shifts). Interaction = push at B - push at A; MVCA's
# predicted sign (study3_interaction_sign) and the level ids are kept with every row so the local analysis can align the direction.
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; CMP = comps6(); E = mat6("experiments_v6.json"); EIDS = [k for k, e in E.items() if e.get("moderation")]; EIDS = EIDS[:2] if SMALL else EIDS
NV = 2 if SMALL else C6("V4D_NV", 6); NA = 1 if SMALL else C6("V4D_NA", 3)
g = torch.Generator().manual_seed(67); SPAN = span_random6(DS, L, 2, g); rows, RUNS = [], []
for eid in EIDS:
    e = E[eid]; c = e["construct"]; arm = e.get("cp3_base_role") or "treatment"; size = natural6(DS, c, L)
    I = exp_items6([eid], nv=NV, na=NA, arms=(arm,), levels=("A", "B")); prompts = [it["prompt"] for it in I]; bodies = [it["body"] for it in I]
    dirs = [("own", c, unit6(direction6(DS, c, L))), ("near", CMP[c]["nearest"][0], unit6(direction6(DS, CMP[c]["nearest"][0], L)))] + [("span", f"span{k}", u) for k, u in enumerate(SPAN)]
    conds = [(kind, name, s, s * size * u) for kind, name, u in dirs for s in (2, -2)]
    R_ = run_rows6(list(prompts) + [(pr, [("add", L, vec, bd)]) for _, _, _, vec in conds for pr, bd in zip(prompts, bodies)]); RUNS.append(R_); nt = len(prompts)
    p9r = lambda k: ";".join(f"{x:.4f}" for x in R_["p9"][k].float().tolist())
    for ci_, (kind, name, s, _) in enumerate(conds):
        o = nt * (ci_ + 1)
        for k, it in enumerate(I):
            rows.append(dict(exp=eid, construct=c, kind=kind, dir=name, dose=s, vid=it["vid"], agent=it["agent"], arm=arm, level=it["level"], cond_level=it["cond_level"],
                             ev=float(R_["ev"][o + k]), ev0=float(R_["ev"][k]), d=float(R_["ev"][o + k] - R_["ev"][k]), mass=float(R_["mass"][o + k]), sign3=sgn(e["study3_interaction_sign"]),
                             p9=p9r(o + k), p9_0=p9r(k)))
    beat(f"V4d {eid}")
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/moderation_v4d.csv", index=False)
S = {"version": "v6", "model": MID, "layer": L, "experiments": {}, "mass_min": round(float(D.mass.min()), 3)}
for eid, g_ in D.groupby("exp"):
    pv = push_of(g_, ["kind", "dir", "level", "vid", "agent"]).rename("push").reset_index(); out = {}
    for kind in ("own", "near", "span"):
        q = pv[pv.kind == kind].groupby(["level", "vid", "agent"]).push.mean()
        a = q.xs("A", level=0) if "A" in q.index.get_level_values(0) else None; b = q.xs("B", level=0) if "B" in q.index.get_level_values(0) else None
        if a is not None and b is not None: out[kind] = dict(push_A=round(float(a.mean()), 3), push_B=round(float(b.mean()), 3), interaction_B_minus_A=dz((b - a.reindex(b.index)).values))
    S["experiments"][eid] = dict(out, predicted_sign=int(g_.sign3.iloc[0]), levels={lv: str(g_[g_.level == lv].cond_level.iloc[0]) for lv in ("A", "B") if (g_.level == lv).any()})
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
Lg = [f"# V4d moderation ({MID}; hidden layer {L})", "", f"timing {S['timing']}", "", "| experiment | levels A / B | own push A / B | own interaction d_z | span interaction d_z | predicted sign |", "|---|---|---|---|---|---|"]
for k, v in S["experiments"].items():
    o = v.get("own", {}); sp = v.get("span", {}); Lg.append(f"| {k} | {v['levels'].get('A')} / {v['levels'].get('B')} | {o.get('push_A')} / {o.get('push_B')} | {(o.get('interaction_B_minus_A') or {}).get('dz')} | {(sp.get('interaction_B_minus_A') or {}).get('dz')} | {v['predicted_sign']} |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); log(f"V4d {MID} done"); beat("V4d ok")
