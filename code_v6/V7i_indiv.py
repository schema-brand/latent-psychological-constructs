# V7i (individuality; MVCA Study 4 counterpart, PI-approved 20 Oct): does a person characteristic of the agent change how the experiment's
# manipulation moves the answer, as MVCA's six theory tests predict (and not in the two self-monitoring null tests)?
# Materials: materials/v6/indiv_v6.json (7 characteristics, contrast pairs by six writers; MVCA's high/low characteristic statements; tests).
# 1. Reading: every characteristic pair read as V1 (the text alone as a user turn, mean over its tokens) at L* and at the reading layer nearest
#    50% of depth -> characteristic direction = mean high - low difference (raw), natural size = its norm; leave-one-writer-out identification
#    among the 7 characteristics. Confound-adjusted direction ("adj", robustness): least squares over all pairs of the state difference on the
#    7 characteristic indicators plus the pair's rated high - low differences in wealth, pleasantness and sociability; the characteristic's
#    coefficient is its direction with the parts that go with those confounds removed.
# 2. Version A (MVCA's statements): both arms of the test's experiment (condition level A) with "About you: <statement>" appended to the agent
#    profile at the high and at the low level, and without a statement. Answer; and the states of the statement's tokens and of the decision
#    token projected on the characteristic direction (does the statement move the internal characteristic?).
# 3. Version B (internal push): both arms with the agent profile's tokens moved along the characteristic direction (s = +-2 natural shifts)
#    at L* and at ~50% depth; control = 2 span-random directions of the same size.
# Prediction (MVCA): ((T - C) at the larger level - (T - C) at the other level) x x_to_y_sign > 0; null tests: no difference.
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; nL = CUR["n_layers"]; LSd = DS["layers"]; IND = mat6("indiv_v6.json"); E = mat6("experiments_v6.json"); AGT = {a["id"]: a for a in agents6()}
def near_layer(f): return min((l for l in LSd if l > L), key=lambda l: abs(l - f * nL), default=L)
LOCS = [("Lstar", L), ("d50", near_layer(.5))]; LY = sorted({l for _, l in LOCS}); GR = {str(l): [l] for l in LY}
NV = 2 if SMALL else C6("V7I_NV", 6); NA = 1 if SMALL else C6("V7I_NA", 3); RUNS = []
# ---- 1. reading ----
P = IND["pairs"]; P = [p for i, p in enumerate(P) if not SMALL or i % 8 == 0]; TR = sorted(IND["traits"])
prompts, marks = [], []
for p in P:
    for side in ("high", "low"): prompts.append(chat6(p[side].strip(), None, gen=False)); marks.append({"m": ("mean", p[side].strip())})
r = run_rows6(prompts, marks, GR, prefix="", tag="V7i read"); RUNS.append(r); X = r["m"].float()          # [2n, layers, d]
DIF = X[0::2] - X[1::2]; ok = ~torch.isnan(DIF[:, 0, 0])
DIR, SIZE = {}, {}
for j, l in enumerate(LY):
    for t in TR:
        ii = [i for i, p in enumerate(P) if p["construct"] == t and ok[i]]
        v = DIF[ii, j].mean(0); DIR[(t, l)] = F.normalize(v, dim=0); SIZE[(t, l)] = float(v.norm())
CF = ["wealth", "pleasantness", "sociability"]; okl = [i for i in range(len(P)) if ok[i]]
Z = torch.tensor([[1.0 if P[i]["construct"] == t else 0.0 for t in TR] + [float((P[i].get("confound_diff") or {}).get(c) or 0.0) for c in CF] for i in okl])
for j, l in enumerate(LY):
    B_ = torch.linalg.lstsq(Z, DIF[okl, j]).solution
    for k, t in enumerate(TR): DIR[(t + "|adj", l)] = F.normalize(B_[k], dim=0); SIZE[(t + "|adj", l)] = float(B_[k].norm())
WR = sorted({p["writer"] for p in P}); hit = []
for i, p in enumerate(P):
    if not ok[i]: continue
    cen = {t: F.normalize(DIF[[k for k, q in enumerate(P) if q["construct"] == t and q["writer"] != p["writer"] and ok[k]], 0].mean(0), dim=0) for t in TR}
    sims = {t: float(F.normalize(DIF[i, 0], dim=0) @ c) for t, c in cen.items()}; hit.append(max(sims, key=sims.get) == p["construct"])
READ = {"n_pairs": len(P), "lowo_acc_Lstar": round(float(np.mean(hit)), 3), "chance": round(1 / len(TR), 3),
        "cos_raw_adj": {f"{t}|{l}": round(float(DIR[(t, l)] @ DIR[(t + '|adj', l)]), 3) for t in TR for l in LY},
        "size": {f"{t}|{l}": round(SIZE[(t, l)], 3) for t in TR for l in LY}}
beat("V7i reading done")
# ---- items ----
def items(eid): return exp_items6([eid], nv=NV, na=NA, arms=("treatment", "control"), levels=("A",))
def prompt_with(it, extra):
    a = AGT[it["agent"]]; per = a["persona"] + (("\n\nAbout you: " + extra) if extra else "")
    return chat6(it["body"] + "\n\n" + it["q"] + FMT9, SYS + per), per
TESTS = IND["tests"]; ST = IND["statements"]; rowsA, rowsB = [], []
for te in (TESTS[:2] if SMALL else TESTS):
    eid, t = te["experiment"], te["trait"]; I = items(eid)
    st = {s["level"]: s["text"] for s in ST if s["test"] == te["test"]}
    pr, mk, meta = [], [], []
    for it in I:
        fills = fills6(eid, AGT[it["agent"]])
        for lev in ("high", "low", "none"):
            stmt = st[lev].format_map(_Keep(fills or {})) if lev != "none" else None
            p_, per = prompt_with(it, stmt); pr.append(p_)
            mk.append({"stmt": ("mean", "About you: " + stmt) if stmt else ("mean", per[-60:]), "dec": ("dec", None)}); meta.append((it, lev))
    R_ = run_rows6(pr, mk, GR, tag=f"V7i A {te['test']}"); RUNS.append(R_)
    for k, (it, lev) in enumerate(meta):
        row = dict(test=te["test"], experiment=eid, trait=t, null=int(te["null_test"]), larger=te["larger_level"], x_to_y=int(te["x_to_y_sign"]), level=lev,
                   vid=it["vid"], agent=it["agent"], arm=it["arm"], ev=float(R_["ev"][k]), mass=float(R_["mass"][k]))
        for nm in ("stmt", "dec"):
            if nm in R_:
                for j, l in enumerate(LY):
                    v = R_[nm][k, j].float(); bad = bool(torch.isnan(v[0]))
                    row[f"proj_{nm}_{l}"] = np.nan if bad else float(v @ DIR[(t, l)]); row[f"projadj_{nm}_{l}"] = np.nan if bad else float(v @ DIR[(t + "|adj", l)])
        rowsA.append(row)
    beat(f"V7i A {te['test']}")
done = set()
for te in (TESTS[:2] if SMALL else TESTS):
    eid, t = te["experiment"], te["trait"]
    if (eid, t) in done: continue
    done.add((eid, t)); I = items(eid); g = torch.Generator().manual_seed(77)
    conds = []
    for loc, l in LOCS:
        span = span_random6(DS, l, 2, g)
        for kind, u in [("own", DIR[(t, l)]), ("ownadj", DIR[(t + "|adj", l)])] + [(f"span{k}", s_) for k, s_ in enumerate(span)]:
            for s in (2, -2): conds.append((loc, l, kind, s, s * SIZE[(t, l)] * u))
    base = [it["prompt"] for it in I]; pers = [AGT[it["agent"]]["persona"] for it in I]
    R_ = run_rows6(base + [(pr_, [("add", l, vec, per)]) for _, l, _, _, vec in conds for pr_, per in zip(base, pers)]); RUNS.append(R_); nt = len(I)
    for ci_, (loc, l, kind, s, _) in enumerate([(None, None, "none", 0, None)] + conds):
        for k, it in enumerate(I):
            o = nt * ci_ + k
            rowsB.append(dict(experiment=eid, trait=t, loc=loc or "none", layer=l, kind=kind, dose=s, vid=it["vid"], agent=it["agent"], arm=it["arm"],
                              ev=float(R_["ev"][o]), mass=float(R_["mass"][o])))
    beat(f"V7i B {eid} {t}")
DA, DB = pd.DataFrame(rowsA), pd.DataFrame(rowsB); DA.to_csv(f"{OUT}/indiv_A.csv", index=False); DB.to_csv(f"{OUT}/indiv_B.csv", index=False)
# ---- summary ----
def effect(g_):
    w = g_[g_.mass >= .5].pivot_table(index=["vid", "agent"], columns="arm", values="ev")
    return (w["treatment"] - w["control"]) if {"treatment", "control"} <= set(w.columns) else pd.Series(dtype=float)
S = {"version": "v6.2-indiv", "model": MID, "locations": dict(LOCS), "reading": READ, "A": {}, "B": {}}
for te in (TESTS[:2] if SMALL else TESTS):
    a = DA[DA.test == te["test"]]; big, oth = te["larger_level"], ("low" if te["larger_level"] == "high" else "high"); sx = int(te["x_to_y_sign"])
    eb, eo = effect(a[a.level == big]), effect(a[a.level == oth]); d = ((eb - eo.reindex(eb.index)) * sx).dropna()
    pj = {f"{pre}{nm}_{l}": round(float(a[a.level == "high"][f"{pre}_{nm}_{l}"].mean() - a[a.level == "low"][f"{pre}_{nm}_{l}"].mean()), 4)
          for pre in ("proj", "projadj") for nm in ("stmt", "dec") for l in LY if f"{pre}_{nm}_{l}" in a}
    S["A"][te["test"]] = dict(null=int(te["null_test"]), interaction=dz(d.values), effect_larger=round(float(eb.mean() * sx), 3), effect_other=round(float(eo.mean() * sx), 3),
                              trait_projection_high_minus_low=pj)
    b = DB[(DB.experiment == te["experiment"]) & (DB.trait == te["trait"])]; out = {}
    for (loc, kind), g_ in b[b.kind != "none"].groupby(["loc", b.kind.str.replace(r"\d", "", regex=True)]):
        sb = 2 if big == "high" else -2
        eb_, eo_ = effect(g_[g_.dose == sb]), effect(g_[g_.dose == -sb]); dd = ((eb_ - eo_.reindex(eb_.index)) * sx).dropna()
        out[f"{loc}|{kind}"] = dz(dd.values)
    S["B"][te["test"]] = dict(null=int(te["null_test"]), **out)
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
Lg = [f"# V7i individuality ({MID}; locations {dict(LOCS)})", "", f"characteristic identification (leave one writer out, 7 characteristics): {READ['lowo_acc_Lstar']} (chance {READ['chance']}); timing {S['timing']}", "",
      "| test | null | A: interaction d_z (t) | A: statement moves trait (dec, L*) | B own L* d_z (t) | B own d50 d_z (t) | B span d50 d_z | B adjusted d50 d_z (t) |", "|---|---|---|---|---|---|---|---|"]
for k, v in S["A"].items():
    b = S["B"].get(k, {}); f = lambda x: f"{(x or {}).get('dz')} ({(x or {}).get('t')})"
    Lg.append(f"| {k} | {v['null']} | {f(v['interaction'])} | {v['trait_projection_high_minus_low'].get(f'projdec_{L}')} | {f(b.get('Lstar|own'))} | {f(b.get('d50|own'))} | {(b.get('d50|span') or {}).get('dz')} | {f(b.get('d50|ownadj'))} |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); log(f"V7i {MID} done"); beat("V7i ok")
