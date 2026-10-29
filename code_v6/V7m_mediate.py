# V7m (individuality, mediation by removal; 21 Oct, after V7i): when MVCA's characteristic statement is in the agent profile ("About you: ..."),
# does the statement's moderation of the experiment's effect run through the model's internal direction for that characteristic?
# 1. Reading as V7i (every characteristic pair as a user turn, mean over its tokens) at the V1 layers from ~10% to ~90% of depth ->
#    characteristic direction per layer (mean high - low, unit) and its confound-adjusted version (least squares on the 7 indicators + rated
#    wealth, pleasantness, sociability differences).
# 2. Version A rows of V7i (both arms of the test's experiment, statement at the high / low level, and none), unedited (= V7i A), and with ONE
#    direction removed (projected out) at every V1 layer from ~10% to ~85% of depth: own direction, own adjusted, 2 span-random directions, the
#    directions of 2 other characteristics. Two places: "stmt" = the statement's tokens only; "after" = the statement and every later token.
# 3. Check: projection of the decision token's state on the characteristic direction at the layer nearest 90% of depth (after the removal range).
# Reading: interaction = ((T - C) at the larger level - (T - C) at the other level) x x_to_y_sign (as V7i); mediation = the interaction shrinks
# with the own direction removed more than with random or other-characteristic directions removed, while the experiment's own effect (T - C
# without statement) is a separate column.
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
nL = CUR["n_layers"]; LSd = DS["layers"]; IND = mat6("indiv_v6.json"); AGT = {a["id"]: a for a in agents6()}
RL = [l for l in LSd if 0.1 * nL <= l <= 0.85 * nL] or [S1["Lstar"]]
L90 = min((l for l in LSd if l > RL[-1]), key=lambda l: abs(l - 0.9 * nL), default=RL[-1])
LY = sorted(set(RL) | {L90}); GR = {str(l): [l] for l in LY}
NV = 2 if SMALL else C6("V7I_NV", 6); NA = 1 if SMALL else C6("V7I_NA", 3); RUNS = []
# ---- 1. reading ----
P = IND["pairs"]; P = [p for i, p in enumerate(P) if not SMALL or i % 8 == 0]; TR = sorted(IND["traits"])
prompts, marks = [], []
for p in P:
    for side in ("high", "low"): prompts.append(chat6(p[side].strip(), None, gen=False)); marks.append({"m": ("mean", p[side].strip())})
r = run_rows6(prompts, marks, GR, prefix="", tag="V7m read"); RUNS.append(r); X = r["m"].float(); del r
DIF = X[0::2] - X[1::2]; ok = ~torch.isnan(DIF[:, 0, 0]); del X
DIR = {}
CF = ["wealth", "pleasantness", "sociability"]; okl = [i for i in range(len(P)) if ok[i]]
Z = torch.tensor([[1.0 if P[i]["construct"] == t else 0.0 for t in TR] + [float((P[i].get("confound_diff") or {}).get(c) or 0.0) for c in CF] for i in okl])
for j, l in enumerate(LY):
    for t in TR:
        ii = [i for i, p in enumerate(P) if p["construct"] == t and ok[i]]; DIR[(t, l)] = F.normalize(DIF[ii, j].mean(0), dim=0)
    B_ = torch.linalg.lstsq(Z, DIF[okl, j]).solution
    for k, t in enumerate(TR): DIR[(t + "|adj", l)] = F.normalize(B_[k], dim=0)
del DIF; beat("V7m reading done")
# ---- 2. rows ----
def items(eid): return exp_items6([eid], nv=NV, na=NA, arms=("treatment", "control"), levels=("A",))
def prompt_with(it, extra):
    a = AGT[it["agent"]]; per = a["persona"] + (("\n\nAbout you: " + extra) if extra else "")
    return chat6(it["body"] + "\n\n" + it["q"] + FMT9, SYS + per), per
TESTS = [te for te in IND["tests"] if te["test"] != "thaler_selfmonitoring_null2"]       # null2 = the same materials as null (V7i)
TESTS = TESTS[:2] if SMALL else TESTS; ST = IND["statements"]; rows = []; g = torch.Generator().manual_seed(91)
for te in TESTS:
    eid, t = te["experiment"], te["trait"]; I = items(eid); st = {s["level"]: s["text"] for s in ST if s["test"] == te["test"]}
    others = [o for o in TR if o != t]; others = [others[k] for k in torch.randperm(len(others), generator=g)[:2].tolist()]
    span = {l: span_random6(DS, l, 2, g) for l in RL}
    kinds = [("own", lambda l: DIR[(t, l)]), ("ownadj", lambda l: DIR[(t + "|adj", l)])] + \
            [(f"rand{k}", (lambda l, k=k: F.normalize(span[l][k].float(), dim=0))) for k in range(2)] + \
            [(f"other{k}", (lambda l, o=o: DIR[(o, l)])) for k, o in enumerate(others)]
    U = {nm: {l: f(l)[None, :] for l in RL} for nm, f in kinds}
    pr, mk, meta = [], [], []
    for it in I:
        fills = fills6(eid, AGT[it["agent"]])
        for lev in ("high", "low", "none"):
            stmt = st[lev].format_map(_Keep(fills or {})) if lev != "none" else None
            p_, per = prompt_with(it, stmt)
            conds = [("asis", "none", None)] + ([] if stmt is None else [(nm, loc, ("proj", U[nm], where, None)) for nm, _ in kinds
                                                                          for loc, where in (("stmt", "About you: " + stmt), ("after", ("from", "About you: " + stmt)))])
            for nm, loc, ed in conds:
                pr.append(p_ if ed is None else (p_, [ed])); mk.append({"dec": ("dec", None)}); meta.append((it, lev, nm, loc))
    R_ = run_rows6(pr, mk, {str(L90): [L90]}, tag=f"V7m {te['test']}"); RUNS.append(R_)
    for k, (it, lev, nm, loc) in enumerate(meta):
        row = dict(test=te["test"], experiment=eid, trait=t, null=int(te["null_test"]), larger=te["larger_level"], x_to_y=int(te["x_to_y_sign"]), level=lev,
                   kind=nm, loc=loc, vid=it["vid"], agent=it["agent"], arm=it["arm"], ev=float(R_["ev"][k]), mass=float(R_["mass"][k]))
        if "dec" in R_:
            v = R_["dec"][k, 0].float(); row["proj_dec_L90"] = np.nan if bool(torch.isnan(v[0])) else float(v @ DIR[(t, L90)])
        rows.append(row)
    beat(f"V7m {te['test']}")
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/mediate.csv", index=False)
# ---- summary ----
def effect(g_):
    w = g_[g_.mass >= .5].pivot_table(index=["vid", "agent"], columns="arm", values="ev")
    return (w["treatment"] - w["control"]) if {"treatment", "control"} <= set(w.columns) else pd.Series(dtype=float)
S = {"version": "v6.3-mediate", "model": MID, "removal_layers": RL, "check_layer": L90, "tests": {}}
for te in TESTS:
    a = D[D.test == te["test"]]; big, oth = te["larger_level"], ("low" if te["larger_level"] == "high" else "high"); sx = int(te["x_to_y_sign"])
    out = {"experiment_effect_no_statement": dz((effect(a[a.level == "none"]) * sx).values)}
    for (nm, loc), g_ in a[a.level != "none"].groupby(["kind", "loc"]):
        eb, eo = effect(g_[g_.level == big]), effect(g_[g_.level == oth]); d = ((eb - eo.reindex(eb.index)) * sx).dropna()
        pj = g_[g_.level == "high"].proj_dec_L90.mean() - g_[g_.level == "low"].proj_dec_L90.mean() if "proj_dec_L90" in g_ else np.nan
        out[f"{nm}|{loc}"] = {"interaction": dz(d.values), "mean": round(float(d.mean()), 4) if len(d) else None, "proj_high_minus_low": round(float(pj), 4),
                              "mass_lt_.5": round(float((g_.mass < .5).mean()), 3)}
    S["tests"][te["test"]] = dict(null=int(te["null_test"]), **out)
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
Lg = [f"# V7m mediation by removal ({MID}; removal at V1 layers {RL[0]}-{RL[-1]}, check layer {L90})", "", f"timing {S['timing']}", "",
      "| test | null | no statement: effect d_z | as is: interaction mean (d_z) | own stmt | own after | adj after | rand after (2) | other after (2) | proj high-low as is / own after |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for k, v in S["tests"].items():
    f = lambda c: "" if c not in v else f"{v[c]['mean']} ({(v[c]['interaction'] or {}).get('dz')})"
    Lg.append(f"| {k} | {v['null']} | {(v['experiment_effect_no_statement'] or {}).get('dz')} | {f('asis|none')} | {f('own|stmt')} | {f('own|after')} | {f('ownadj|after')} | "
              f"{f('rand0|after')}; {f('rand1|after')} | {f('other0|after')}; {f('other1|after')} | {(v.get('asis|none') or {}).get('proj_high_minus_low')} / {(v.get('own|after') or {}).get('proj_high_minus_low')} |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); log(f"V7m {MID} done"); beat("V7m ok")
