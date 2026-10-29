# V8 (20 Oct): check of the batched routine (run_rows6, run 2 on) against run 1's one-at-a-time routine, on the run-1 model of this job.
# Part A, in this job: the same prompts and edits through run_rows6 (batched) and run_marks6_b1 (run 1's routine): 64 reading texts
# (mean state at L*), 24 experiment items unedited, with the own construct pushed (+2 natural shifts, situation tokens at L*), and with the
# own construct-specific subspace removed (mean replacement, V4c layers, every position) -> answer differences, state cosines, time per row.
# Part B, against run 1's files: this job's V3 / V4a / V4b / V4c outputs on subsets (job config) vs run 1's rows with the same keys
# (V8_REF = run 1's job of this model) -> n matched, r, median / 99th percentile / max absolute difference of the answer, time per pass.
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV); S1, DS = load_v1(MID, REV); assert DS is not None, "V1_read results for this model are missing"
L = S1["Lstar"]; SUBL = S1["SUBL"]; E = mat6("experiments_v6.json"); P = pairs6(SMALL); RES = {"model": MID, "version": "v6.1"}
def cmp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d_ = np.abs(a - b)
    return dict(n=int(len(a)), r=round(float(np.corrcoef(a, b)[0, 1]), 6) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None,
                median_abs=round(float(np.median(d_)), 5), p99_abs=round(float(np.percentile(d_, 99)), 5), max_abs=round(float(d_.max()), 5))
# ---- Part A ----
pp = P[::max(1, len(P) // 64)][:64]; tx = [chat6(p["high"], None, gen=False) for p in pp]; mk = [{"mean": ("mean", p["high"])} for p in pp]
ra = run_rows6(tx, mk, {"L": [L]}, prefix=""); rb = run_marks6_b1(tx, mk, {"L": [L]}, prefix="")
cs = F.cosine_similarity(ra["mean"][:, 0].float(), rb["mean"][:, 0].float(), dim=-1)
RES["A_reading_state_cos"] = dict(min=round(float(cs.min()), 6), median=round(float(cs.median()), 6))
I = exp_items6(list(E)[:8], nv=1, na=3, arms=("treatment",), levels=("A",))[:24]
SH = shared6(DS, SUBL, 3); rows, plain, push, rem = [], [], [], []
for it in I:
    c = E[it["exp"]]["construct"]; vec = 2 * natural6(DS, c, L) * unit6(direction6(DS, c, L)); Us = specific6(DS, c, 10, SUBL, SH)
    rows += [it["prompt"], (it["prompt"], [("add", L, vec, it["body"])]), (it["prompt"], [("proj", Us, None, DS.get("mu"))])]
    plain.append(run_marks6_b1([it["prompt"]])); push.append(run_marks6_b1([it["prompt"]], hook_fn=add_at(L, vec, it["body"])))
    rem.append(run_marks6_b1([it["prompt"]], hook_fn=proj_at(Us, None, DS.get("mu"))))
rr = run_rows6(rows)
for nm, k_, ref in (("unedited", 0, plain), ("pushed", 1, push), ("removed", 2, rem)):
    RES[f"A_items_{nm}"] = cmp(rr["ev"][k_::3], [x["ev"][0] for x in ref])
t_b1 = sum(x["sec"] for x in plain + push + rem) / (3 * len(I)); RES["A_sec_per_row"] = dict(one_at_a_time=round(t_b1, 4), batched=round(rr["sec"] / len(rows), 4), speedup=round(t_b1 / max(rr["sec"] / len(rows), 1e-9), 2))
beat("V8 part A done")
# ---- Part B ----
REF = C6("V8_REF"); keys = {"V3_exp": ("items.csv", ["exp", "vid", "level", "arm", "agent"]), "V4a_causal": ("causal_v4a.csv", ["j", "writer", "part", "kind", "i", "pid", "agent", "dose", "side"]),
                           "V4b_steer": ("steer_v4b.csv", ["exp", "kind", "dir", "dose", "vid", "agent"]), "V4c_remove": ("remove_v4c.csv", ["exp", "cond", "vid", "agent", "arm"])}
for st, (fn, ks) in keys.items():
    mine, ref = f"{_BASE}/{st}/{fn}", f"{ROOT}/results/{REF}/{st}/{fn}"
    if not (REF and os.path.exists(mine) and os.path.exists(ref)): RES[f"B_{st}"] = "missing"; continue
    a = pd.read_csv(mine); b = pd.read_csv(ref)
    for D_ in (a, b):
        for k_ in ks:
            if k_ in D_: D_[k_] = D_[k_].map(lambda x: "" if pd.isna(x) else (f"{x:g}" if isinstance(x, float) else str(x)))
    ks_ = [k_ for k_ in ks if k_ in a and k_ in b]; m = a.merge(b, on=ks_, suffixes=("", "_ref"))
    RES[f"B_{st}"] = dict(cmp(m.ev, m.ev_ref), rows_this_job=int(len(a)))
    try:
        sa = json.load(open(f"{_BASE}/{st}/summary.json")).get("timing", {}); sb_ = json.load(open(f"{ROOT}/results/{REF}/{st}/summary.json")).get("timing", {})
        RES[f"B_{st}"]["sec_per_pass"] = dict(this_job=sa.get("sec_per_forward"), run1=sb_.get("sec_per_forward"))
    except Exception: pass
RES["wall_s"] = round(time.time() - T0); save_json("summary.json", RES)
open(f"{OUT}/digest.md", "w", encoding="utf-8").write("# V8 batched vs one-at-a-time\n\n" + "\n".join(f"- {k}: {v}" for k, v in RES.items())); log(f"V8 {MID} done"); beat("V8 ok")
