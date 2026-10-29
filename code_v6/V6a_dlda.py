# V6a (run 3, 21 Oct): robustness of V4a to how a construct direction is estimated. Identical to V4a (same texts, agents, comparison sets,
# doses, natural matrix; run with the job's 4a config) except that every construct direction (own, nearest, random, single-writer) is the
# diagonal discriminant direction: the mean high - low difference divided by the per-dimension variance of the text states at that layer
# (DS["std"]^2), instead of the plain mean difference. Push size stays s x the construct's natural shift (norm of the mean difference), so
# only the direction changes. Span-random directions are unchanged.
# Original V4a header: causal multitrait-multimethod matrix on first-person construct judgments. An MVCA agent reads a situation and answers
# one judgment item built from an MVCA scale definition (1-9).
# Pushed matrix: neutral-pole (L4) texts of construct j, judgment j; the states of the situation's tokens at L* are moved along a direction
# by s x the natural shift of construct j along its own direction (norm of the mean high - low difference at L*). Directions: own (j), the
# 5 constructs nearest to j by rated definition similarity, 5 fixed random constructs, and span-random directions (random combinations of
# real cell differences); construct directions are estimated without the text's writer. s = +-2 for all; +-1 and +-4 also for the own
# direction. Writer replication (V4A_WRITER): the own direction estimated from each other single writer, s = +-2.
# Natural matrix: high vs low texts of construct i judged on every j whose comparison set contains i (and on i itself).
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; CONS = cons6(); CMP = comps6(); P = pairs6(); AG = agents6(); WR = sorted({p["writer"] for p in P})
NT = C6("V4A_NT", 2); NNEAR = C6("V4A_NEAR", 5); NRAND = C6("V4A_RAND", 5); NSPAN = C6("V4A_SPAN", 4); OWNX = C6("V4A_OWN", [1, -1, 4, -4])
WREP = C6("V4A_WRITER", True); NNAT = C6("V4A_NAT", 1); NATW = C6("V4A_NAT_WRITERS") or WR
JS = CONS[:3] if SMALL else (C6("V4A_JUDGE") or CONS); WU = WR[:2] if SMALL else WR
VAR = {l: (DS["std"][DS["layers"].index(l)].float() ** 2 + 1e-6) for l in [L]}
def ddir6(DS_, c, l, **kw):
    v = direction6(DS_, c, l, **kw); return None if v is None else v / VAR[l]
g = torch.Generator().manual_seed(64); SPAN = span_random6(DS, L, NSPAN, g); SIZE = {c: natural6(DS, c, L) for c in CONS}
def compset(j): return [("own", j)] + [("near", c) for c in CMP[j]["nearest"][:NNEAR]] + [("rand", c) for c in CMP[j]["random"][:NRAND]]
def pick6(c, w, k, prefer="L4"):
    pp = [p for p in P if p["construct"] == c and p["writer"] == w]; return sorted(pp, key=lambda p: p["low_type"] != prefer)[:k]
rows, RUNS = [], []
def p9s(r, k): return ";".join(f"{x:.4f}" for x in r["p9"][k].float().tolist())      # full 1-9 answer distribution (non-saturating measures)
for j in JS:
    for wk, w in enumerate(WU):
        tw = pick6(j, w, 1 if SMALL else NT)
        if not tw: continue
        texts = [p["low"] for p in tw]; ags = [AG[(wk * NT + t) % len(AG)] for t in range(len(tw))]
        prompts = [judge6(t, j, a["persona"]) for t, a in zip(texts, ags)]
        dirs = [(k, c, unit6(ddir6(DS, c, L, exclude_writer=w))) for k, c in compset(j) if ddir6(DS, c, L, exclude_writer=w) is not None]
        dirs += [("span", f"span{k}", u) for k, u in enumerate(SPAN)]
        if WREP: dirs += [("writer", f"own_from_{w2}", unit6(ddir6(DS, j, L, writers=[w2]))) for w2 in WR if w2 != w and ddir6(DS, j, L, writers=[w2]) is not None]
        conds = [(kind, name, s, s * SIZE[j] * u) for kind, name, u in dirs for s in [2, -2] + (OWNX if kind == "own" else [])]
        # one batched call: the unedited prompts, then every condition (the same prompts with the situation's tokens moved at L*)
        r = run_rows6(list(prompts) + [(pr, [("add", L, vec, tx)]) for _, _, _, vec in conds for pr, tx in zip(prompts, texts)]); RUNS.append(r); nt = len(prompts)
        ev0, m0 = r["ev"][:nt], r["mass"][:nt]
        base = dict(j=j, writer=w, layer=L)
        for k_, (p, a, e_, m_) in enumerate(zip(tw, ags, ev0, m0)): rows.append(dict(base, part="pushed", kind="none", i="none", pid=p["pid"], agent=a["id"], low_type=p["low_type"], dose=0, ev=float(e_), ev0=float(e_), mass=float(m_), p9=p9s(r, k_)))
        for ci_, (kind, name, s, _) in enumerate(conds):
            o = nt * (ci_ + 1)
            for k_, (p, a, e0) in enumerate(zip(tw, ags, ev0)):
                rows.append(dict(base, part="pushed", kind=kind, i=name, pid=p["pid"], agent=a["id"], low_type=p["low_type"], dose=s, ev=float(r["ev"][o + k_]), ev0=float(e0), mass=float(r["mass"][o + k_]), p9=p9s(r, o + k_)))
    beat(f"V6a pushed {j}")
# natural matrix (all prompts in one batched call)
NI = JS if SMALL else CONS; NQ, NM = [], []
for ik, i in enumerate(NI):
    jset = sorted({i} | {j for j in JS if i in [c for _, c in compset(j)]})
    for wk, w in enumerate(NATW if not SMALL else WU):
        for p in pick6(i, w, NNAT, prefer="L4" if wk % 2 else "L1"):
            a = AG[(ik + wk) % len(AG)]
            for side in ("high", "low"):
                for j in jset: NQ.append(judge6(p[side], j, a["persona"])); NM.append(dict(part="natural", kind="natural", i=i, j=j, pid=p["pid"], writer=w, agent=a["id"], low_type=p["low_type"], side=side))
r = run_rows6(NQ, tag="V6a natural"); RUNS.append(r)
for k_, m_ in enumerate(NM): rows.append(dict(m_, ev=float(r["ev"][k_]), mass=float(r["mass"][k_]), p9=p9s(r, k_)))
beat("V6a natural done")
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/causal_dlda_v6a.csv", index=False)
# ---- summary ----
Pu = D[(D.part == "pushed") & (D.kind != "none")].copy(); Pu["d"] = Pu.ev - Pu.ev0
pt = push_of(Pu, ["j", "kind", "i", "pid"]).rename("push").reset_index()            # push per text and direction
S = {"version": "v6.2-dlda", "model": MID, "layer": L, "size": {c: round(SIZE[c], 3) for c in JS}, "mass_min": round(float(D.mass.min()), 3), "share_mass_lt_.5": round(float((D.mass < .5).mean()), 4), "column": {}}
for j, g_ in pt.groupby("j"):
    own = g_[g_.kind == "own"].set_index("pid").push; oth = g_[g_.kind.isin(["near", "rand"])].groupby("pid").push.mean()
    near = g_[g_.kind == "near"].groupby("pid").push.mean(); rnd = g_[g_.kind == "rand"].groupby("pid").push.mean(); spn = g_[g_.kind == "span"].groupby("pid").push.mean()
    by_dir = g_[g_.kind.isin(["own", "near", "rand", "span"])].groupby("i").push.mean()
    dose = Pu[(Pu.j == j) & (Pu.kind == "own")].groupby("dose").d.mean()
    S["column"][j] = dict(own=dz(own.values), own_minus_others=dz((own - oth.reindex(own.index)).values), own_minus_near=dz((own - near.reindex(own.index)).values),
                          own_minus_random=dz((own - rnd.reindex(own.index)).values), own_minus_span=dz((own - spn.reindex(own.index)).values),
                          rank=int(1 + (by_dir.drop(j, errors="ignore") > by_dir.get(j, np.nan)).sum()), n_dirs=int(len(by_dir)), dose={f"{float(k):g}": round(float(v), 3) for k, v in dose.items()},
                          writer_rep_positive=round(float((g_[g_.kind == "writer"].groupby("i").push.mean() > 0).mean()), 3) if (g_.kind == "writer").any() else None,
                          kind=CMP[j].get("kind"))
col = S["column"]; S["n_own_minus_others_pos"] = int(sum((v["own_minus_others"]["mean"] or 0) > 0 for v in col.values()))
S["n_own_minus_others_t2"] = int(sum((v["own_minus_others"]["t"] or 0) > 2 for v in col.values())); S["n_rank1"] = int(sum(v["rank"] == 1 for v in col.values()))
N = D[D.part == "natural"]
if len(N):
    h = N[N.side == "high"].set_index(["i", "j", "pid"]).ev; l_ = N[N.side == "low"].set_index(["i", "j", "pid"]).ev
    nat = (h - l_.reindex(h.index)).groupby(level=[0, 1]).mean(); pushm = pt[pt.kind.isin(["own", "near", "rand"])].groupby(["i", "j"]).push.mean()
    common = [k for k in nat.index if k in pushm.index]
    if len(common) > 5:
        a = pd.Series([nat[k] for k in common], index=pd.MultiIndex.from_tuples(common)); b = pd.Series([pushm[k] for k in common], index=a.index)
        def centre(x): x = x.copy(); x = x - x.groupby(level=0).transform("mean"); return x - x.groupby(level=1).transform("mean")
        S["r_natural_pushed"] = round(float(np.corrcoef(a, b)[0, 1]), 3); S["r_natural_pushed_centred"] = round(float(np.corrcoef(centre(a), centre(b))[0, 1]), 3); S["n_cells"] = len(common)
    S["natural_own"] = {i: round(float(nat.get((i, i), np.nan)), 3) for i in NI if (i, i) in nat.index}
S["timing"] = timing(RUNS); S["wall_s"] = round(time.time() - T0); save_json("summary.json", S)
Lg = [f"# V6a causal MTMM, diagonal discriminant directions ({MID}; push on the situation tokens at hidden layer {L}; 1 = natural shift of the judged construct)", "",
      f"own - others > 0: {S['n_own_minus_others_pos']}/{len(col)} (t > 2: {S['n_own_minus_others_t2']}); own direction first: {S['n_rank1']}/{len(col)}; r(natural, pushed) = {S.get('r_natural_pushed')} (centred {S.get('r_natural_pushed_centred')}); timing {S['timing']}", "",
      "| judgment | kind | own push | own - others d_z | rank | dose -4/-2/-1/+1/+2/+4 |", "|---|---|---|---|---|---|"]
for j, v in col.items(): Lg.append(f"| {j} | {v['kind']} | {v['own']['mean']} | {v['own_minus_others']['dz']} | {v['rank']}/{v['n_dirs']} | {' / '.join(str(v['dose'].get(k)) for k in ('-4', '-2', '-1', '1', '2', '4'))} |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(Lg)); log(f"V6a {MID} done"); beat("V6a ok")
