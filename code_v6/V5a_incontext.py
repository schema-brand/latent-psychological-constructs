# V5a (follow-up 1, 20 Oct: where the causal chain breaks). The experiment's situation is followed by the first-person judgment item of the
# experiment's own construct (the 66 items of V4a) instead of the experiment's response question.
#   part "check": both arms of all 60 experiments (condition level A), unedited -> does the manipulation move the construct as the agent
#   judges it (X -> M, the model's own manipulation check)? Signed by MVCA's x_to_m_sign.
#   part "pushed": the 52 MVCA Study 2 experiments, the arm into which MVCA inserted the mediator statement, with exactly V4b's edits (own
#   +-1/2/4, 3 nearest, 3 fixed random constructs and the same 2 span-random directions at +-2; situation tokens at L*; same items) ->
#   does the push move the construct judgment in context? Compared item by item with V4b's push on the response (local analysis).
# The judgment item asks about the construct's high pole, so a push along the high pole is expected to raise the judgment (d = ev - ev0).
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; CMP = comps6(); E = mat6("experiments_v6.json"); JI = judge_items6(); PER = {a["id"]: a["persona"] for a in agents6()}
EALL = list(E)[:3] if SMALL else list(E); E2 = [k for k in EALL if "2" in str(E[k]["studies"]).split(";")]
NV = 2 if SMALL else C6("V5A_NV", 6); NA = 1 if SMALL else C6("V5A_NA", 3); NN = C6("V4B_NEAR", 3); NR = C6("V4B_RAND", 3)
def jprompt(it): return chat6(it["body"] + "\n\n" + JI[E[it["exp"]]["construct"]] + JUDGE_FMT6, SYS + PER[it["agent"]])
def p9r(r, k): return ";".join(f"{x:.4f}" for x in r["p9"][k].float().tolist())
rows, RUNS = [], []
# part check: X -> M on the construct judgment, both arms, every experiment
IC = exp_items6(EALL, nv=NV, na=NA, arms=("treatment", "control"), levels=("A",))
r = run_rows6([jprompt(it) for it in IC], tag="V5a check"); RUNS.append(r)
for k, it in enumerate(IC):
    e = E[it["exp"]]
    rows.append(dict(part="check", exp=it["exp"], construct=e["construct"], kind="none", dir="none", dose=0, vid=it["vid"], agent=it["agent"], arm=it["arm"],
                     x_to_m_sign=sgn(e["x_to_m_sign"]), ev=float(r["ev"][k]), ev0=float(r["ev"][k]), d=0.0, mass=float(r["mass"][k]), mass0=float(r["mass"][k]), p9=p9r(r, k)))
beat("V5a check done")
# part pushed: V4b's edits on the construct judgment in context
g = torch.Generator().manual_seed(65); SPAN = span_random6(DS, L, 2, g)          # same seed and call as V4b -> the same span-random directions
for eid in (E2[:3] if SMALL else E2):
    e = E[eid]; c = e["construct"]; arm = (e.get("study2_insert_into") or ["treatment"])[0]; size = natural6(DS, c, L)
    I = exp_items6([eid], nv=NV, na=NA, arms=(arm,), levels=("A",)); prompts = [jprompt(it) for it in I]; bodies = [it["body"] for it in I]
    dirs = [("own", c, [1, -1, 2, -2, 4, -4])] + [("near", x, [2, -2]) for x in CMP[c]["nearest"][:NN]] + [("rand", x, [2, -2]) for x in CMP[c]["random"][:NR]]
    dirs = [(k, n, unit6(direction6(DS, n, L)), s) for k, n, s in dirs] + [("span", f"span{k}", u, [2, -2]) for k, u in enumerate(SPAN)]
    conds = [(kind, name, s, s * size * u) for kind, name, u, doses in dirs for s in doses]
    R_ = run_rows6(list(prompts) + [(pr, [("add", L, vec, bd)]) for _, _, _, vec in conds for pr, bd in zip(prompts, bodies)]); RUNS.append(R_); nt = len(prompts)
    for ci_, (kind, name, s, _) in enumerate(conds):
        o = nt * (ci_ + 1)
        for k, it in enumerate(I):
            rows.append(dict(part="pushed", exp=eid, construct=c, kind=kind, dir=name, dose=s, vid=it["vid"], agent=it["agent"], arm=arm, x_to_m_sign=sgn(e["x_to_m_sign"]),
                             ev=float(R_["ev"][o + k]), ev0=float(R_["ev"][k]), d=float(R_["ev"][o + k] - R_["ev"][k]), mass=float(R_["mass"][o + k]), mass0=float(R_["mass"][k]), p9=p9r(R_, o + k)))
    beat(f"V5a {eid}")
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/incontext_v5a.csv", index=False)
# summary: manipulation check (treatment - control on the construct judgment, x MVCA's X -> M sign) and the push on the judgment in context
S = {"version": "v6.1", "model": MID, "layer": L, "mass_min": round(float(D.mass.min()), 3), "share_mass_lt_.5": round(float((D.mass < .5).mean()), 4), "check": {}, "pushed": {}}
C_ = D[(D.part == "check") & (D.mass >= .5)]
for eid, g_ in C_.groupby("exp"):
    t = g_[g_.arm == "treatment"].set_index(["vid", "agent"]).ev; c_ = g_[g_.arm == "control"].set_index(["vid", "agent"]).ev
    S["check"][eid] = dz(((t - c_.reindex(t.index)) * (g_.x_to_m_sign.iloc[0] or 1)).values)
Pd = D[(D.part == "pushed") & (D.mass >= .5) & (D.mass0 >= .5)]
for eid, g_ in Pd.groupby("exp"):
    pv = push_of(g_, ["kind", "dir", "vid", "agent"]).rename("push").reset_index(); own = pv[pv.kind == "own"].set_index(["vid", "agent"]).push
    oth = pv[pv.kind.isin(["near", "rand"])].groupby(["vid", "agent"]).push.mean()
    S["pushed"][eid] = dict(own_push=round(float(own.mean()), 4), own_minus_others=dz((own - oth.reindex(own.index)).values))
S["n_check_positive"] = int(sum((v["mean"] or 0) > 0 for v in S["check"].values())); S["n_check"] = len(S["check"])
S["n_own_push_positive"] = int(sum(v["own_push"] > 0 for v in S["pushed"].values())); S["n_pushed"] = len(S["pushed"])
S["n_own_minus_others_positive"] = int(sum((v["own_minus_others"]["mean"] or 0) > 0 for v in S["pushed"].values()))
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
Lg = [f"# V5a construct judgment in the experiment's context ({MID}; hidden layer {L})", "",
      f"manipulation check (treatment - control on the construct judgment, MVCA's X -> M sign) > 0: {S['n_check_positive']}/{S['n_check']}; "
      f"own push on the judgment > 0: {S['n_own_push_positive']}/{S['n_pushed']}; own - others > 0: {S['n_own_minus_others_positive']}/{S['n_pushed']}; timing {S['timing']}", "",
      "| experiment | check d_z | own push | own - others d_z |", "|---|---|---|---|"]
for eid in S["check"]:
    pu = S["pushed"].get(eid, {}); Lg.append(f"| {eid} | {S['check'][eid]['dz']} | {pu.get('own_push', '')} | {(pu.get('own_minus_others') or {}).get('dz', '')} |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); log(f"V5a {MID} done"); beat("V5a ok")
