# V3 (Study 3): the 60 published experiments as the agent reads them. Every variant (MVCA frozen stimulus v00 + everyday rewordings) x
# condition level (A; B for moderated experiments) x eligible MVCA agent x arm, read once with no edit. Records the agent's answer (expected
# value over the digits 1-9, the silicon-sample response) and the states at the manipulation span (union of its passages), the situation
# (mean over the body), the question and the decision token, averaged over band 1, band 2 and L*. Saves: items.csv (answers), proj_v3.pt
# (every item's states projected on the 66 construct directions, raw and standardized), shift_v3.pt (treatment - control shift per variant
# and level, mean over agents, full vectors) and p9 (digit distributions for the bin check D1).
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
LST = S1["Lstar"]; B1, B2 = S1["B1"], S1["B2"]; G = {"B1": B1, "B2": B2, "L": [LST]}; GN = list(G); MK = ["x", "body", "q", "dec"]
CONS = cons6(); E = mat6("experiments_v6.json"); EIDS = list(E)[:3] if SMALL else (C6("V3_EXPS") or list(E))
def gdir(c, ls): return torch.stack([direction6(DS, c, l) for l in ls]).mean(0)
SDg = {g: torch.stack([DS["std"][DS["layers"].index(l)].float() for l in ls]).mean(0) for g, ls in G.items()}
DIRg = {g: torch.stack([gdir(c, ls) for c in CONS]) for g, ls in G.items()}
UR = {g: F.normalize(DIRg[g], dim=-1) for g in G}; US = {g: F.normalize(DIRg[g] / SDg[g], dim=-1) for g in G}
rows, PR, PS, P9, SH, SHK, RUNS = [], [], [], [], [], [], []
IALL = exp_items6(EIDS, nv=2 if SMALL else None, na=2 if SMALL else None)          # all items in one batched call (equal token lengths run together)
MALL = [{"x": ("union", it["xs"]), "body": ("mean", it["body"]), "q": ("mean", it["q"]), "dec": ("dec", None)} for it in IALL]
RALL = run_marks6([it["prompt"] for it in IALL], MALL, G, tag="V3"); RUNS.append(RALL)
for eid in EIDS:
    e = E[eid]; ix = [k for k, it in enumerate(IALL) if it["exp"] == eid]; I = [IALL[k] for k in ix]
    r = {k_: (v_[ix] if k_ in ("ev", "mass", "p9") or k_ in MK else v_) for k_, v_ in RALL.items()}
    S = torch.stack([r[m].float() if m in r else torch.full((len(I), len(G), CUR["d"]), float("nan")) for m in MK], 1)      # [n, 4, G, d]
    PR.append(torch.stack([S[:, :, gi] @ UR[g].T for gi, g in enumerate(GN)], 2).half())                                  # [n, 4, G, 66]
    PS.append(torch.stack([(S[:, :, gi] / SDg[g]) @ US[g].T for gi, g in enumerate(GN)], 2).half()); P9.append(r["p9"])
    idx = {(it["vid"], it["level"], it["agent"], it["arm"]): k for k, it in enumerate(I)}
    for (vid, lev) in dict.fromkeys((it["vid"], it["level"]) for it in I):
        ags = [a for (v_, l_, a, arm) in idx if v_ == vid and l_ == lev and arm == "treatment" and (v_, l_, a, "control") in idx]
        if not ags: continue
        D_ = torch.stack([S[idx[(vid, lev, a, "treatment")]] - S[idx[(vid, lev, a, "control")]] for a in ags]); SH.append(torch.nanmean(D_, 0).half()); SHK.append((eid, vid, lev, len(ags)))
    for k, it in enumerate(I):
        rows.append(dict(exp=eid, vid=it["vid"], level=it["level"], cond_level=it["cond_level"], arm=it["arm"], agent=it["agent"], writer=it["writer"], ev=float(r["ev"][k]),
                         mass=float(r["mass"][k]), n_xs=len(it["xs"]), x_found=bool(not torch.isnan(S[k, 0, 0, 0])), construct=e["construct"]))
    del r, S; gc.collect()
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/items.csv", index=False)
torch.save({"raw": torch.cat(PR), "std": torch.cat(PS), "marks": MK, "groups": GN, "constructs": CONS}, f"{OUT}/proj_v3.pt")
torch.save({"shift": torch.stack(SH), "keys": SHK, "marks": MK, "groups": GN}, f"{OUT}/shift_v3.pt"); torch.save(torch.cat(P9), f"{OUT}/p9_v3.pt")
# quick look: response effect and own-construct shift (situation mean, band 1, standardized: own minus the mean of the other 65)
PSa = torch.cat(PS).float(); S_ = {"model": MID, "Lstar": LST, "n_items": len(D), "mass_min": round(float(D.mass.min()), 3), "share_mass_lt_.5": round(float((D.mass < .5).mean()), 4),
                                    "x_found_share": round(float(D.x_found.mean()), 4), "experiments": {}}
P9a = torch.cat(P9).float(); top = P9a.argmax(1); D["top_digit"] = (top + 1).numpy()
for eid, g_ in D.groupby("exp"):
    e = E[eid]; sx = sgn(e["x_to_y_sign"]); A_ = g_[g_.level == "A"]
    t = A_[A_.arm == "treatment"].set_index(["vid", "agent"]).ev; c = A_[A_.arm == "control"].set_index(["vid", "agent"]).ev; resp = (t - c.reindex(t.index)) * (sx or 1)
    oc = CONS.index(e["construct"]); ii = torch.tensor(A_.index.values); ti = A_.arm.values == "treatment"
    pj = PSa[ii][:, MK.index("body"), GN.index("B1")]; own_minus = (pj[:, oc] - (pj.sum(1) - pj[:, oc]) / (len(CONS) - 1)).numpy()
    A2 = A_.assign(om=own_minus); ts = A2[A2.arm == "treatment"].set_index(["vid", "agent"]).om; cs = A2[A2.arm == "control"].set_index(["vid", "agent"]).om
    shift = (ts - cs.reindex(ts.index)).groupby(level=0).mean() * (sgn(e["x_to_m_sign"]) or 1)
    S_["experiments"][eid] = dict(response=dz(resp.groupby(level=0).mean().values), own_minus_others_body_B1=dz(shift.values), extreme_bin_share=round(float(g_.top_digit.isin([1, 9]).mean()), 3),
                                  n_variants=int(A_.vid.nunique()))
ex = S_["experiments"]; S_["D1_flag"] = [k for k, v in ex.items() if v["extreme_bin_share"] > .15]
S_["n_own_shift_positive"] = int(sum((v["own_minus_others_body_B1"]["mean"] or 0) > 0 for v in ex.values())); S_["timing"] = timing(RUNS); S_["wall_s"] = round(time.time() - T0)
save_json("summary.json", S_)
L = [f"# V3 experiments ({MID}; {len(D)} items; L* = {LST})", "", f"Timing {S_['timing']}; digit mass min {S_['mass_min']}; manipulation span found {S_['x_found_share']}; own shift > 0 in {S_['n_own_shift_positive']}/{len(ex)}; D1 flags {S_['D1_flag']}", "",
     "| experiment | response d_z | own - others (situation, band 1) d_z | extreme-bin share |", "|---|---|---|---|"]
for k, v in ex.items(): L.append(f"| {k} | {v['response']['dz']} | {v['own_minus_others_body_B1']['dz']} | {v['extreme_bin_share']} |")
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(L)); log(f"V3 {MID} done"); beat("V3 ok")
