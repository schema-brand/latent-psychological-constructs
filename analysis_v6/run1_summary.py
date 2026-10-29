"""Run 1 cross-model summary (Qwen3.6-27B, Gemma 4 31B, OLMo 3.1 32B) under the fixed analysis rules. Writes analysis_v6/run1_summary.json."""
import json, re, torch, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent; RV = HERE / "results_v6"
E = json.load(open(HERE / "materials_v6/upload_v6/experiments_v6.json", encoding="utf-8"))
sg = lambda s: int(float(str(s).replace("+", ""))) if str(s).strip() not in ("", "None") else 0
M = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
def table(md):
    rows = [l for l in open(md, encoding="utf-8").read().splitlines() if l.startswith("| ") and not l.startswith("| ---") ]
    hdr = [h.strip() for h in rows[0].strip("|").split("|")]
    return pd.DataFrame([[c.strip() for c in r.strip("|").split("|")] for r in rows[1:]], columns=hdr)
OUT, V3, J, B = {}, {}, {}, {}
for name, job in M.items():
    o = {}
    # V3 response effects (EV) and robustness (log ratio upper vs lower half of 1-9), items with mass >= .5
    I = pd.read_csv(RV / job / "V3_exp/items.csv"); P = torch.load(RV / job / "V3_exp/p9_v3.pt", map_location="cpu", weights_only=False).float().numpy()
    I["lr"] = np.log((P[:, 5:].sum(1) + 1e-4) / (P[:, :4].sum(1) + 1e-4)); I = I[I.mass >= .5]
    ceil = pd.read_csv(RV / job / "V3_exp/ceiling_check_v2.csv").set_index("exp").blocked
    eff = {}
    for eid, g in I.groupby("exp"):
        s = sg(E[eid]["x_to_y_sign"]) or 1; k = ["vid", "level", "agent"]
        t = g[g.arm == "treatment"].set_index(k); c = g[g.arm == "control"].set_index(k); t, c = t.align(c, join="inner", axis=0)
        d = (t.ev - c.ev) * s; dl = (t.lr - c.lr) * s
        eff[eid] = dict(ev=d.mean(), dz=d.mean() / d.std() if d.std() > 0 else np.nan, lr=dl.mean())
    F = pd.DataFrame(eff).T; F["blocked"] = ceil.reindex(F.index); V3[name] = F
    keep = F[F.blocked <= .30]
    o["response_pos"] = f"{int((F.ev > 0).sum())}/{len(F)}"; o["response_pos_rule"] = f"{int((keep.ev > 0).sum())}/{len(keep)}"
    o["response_pos_logratio_rule"] = f"{int((keep.lr > 0).sum())}/{len(keep)}"; o["response_dz_gt_.5_rule"] = f"{int((keep.dz > .5).sum())}/{len(keep)}"
    o["blocked_gt_.30"] = int((F.blocked > .30).sum())
    T = table(RV / job / "V3_exp/digest.md"); col = [c for c in T.columns if c.startswith("own - others")][0]
    sh = pd.to_numeric(T.set_index("experiment")[col], errors="coerce"); o["scenario_shift_own_gt_others"] = f"{int((sh > 0).sum())}/{len(sh)}"
    V3[name]["shift"] = sh.reindex(F.index)
    # V4a judgments
    T = table(RV / job / "V4a_causal/digest.md").set_index("judgment"); J[name] = T
    a = pd.to_numeric(T["own - others d_z"], errors="coerce"); o["judg_own_gt_others"] = f"{int((a > 0).sum())}/{len(a)}"
    S = json.load(open(RV / job / "V4a_causal/summary.json")); o["judg_t2"] = S["n_own_minus_others_t2"]; o["judg_rank1"] = S["n_rank1"]; o["judg_r_nat_push"] = round(S["r_natural_pushed"], 2)
    st = T.assign(v=a).groupby("kind").v.mean(); o["judg_state_vs_fact_dz"] = f"{st.get('state', np.nan):.2f} / {st.get('fact', np.nan):.2f}"
    # V4b steering
    T = table(RV / job / "V4b_steer/digest.md").set_index("experiment"); B[name] = T
    op = pd.to_numeric(T["own push"], errors="coerce"); od = pd.to_numeric(T["own - others d_z"], errors="coerce")
    o["steer_own_push_pos"] = f"{int((op > 0).sum())}/{len(op)}"; o["steer_own_gt_others"] = f"{int((od > 0).sum())}/{len(od)}"
    # V4d
    R = pd.read_csv(RV / job / "V4d_moderation/moderation_aligned.csv"); r = R[R.kind == "own"]
    o["moderation_agree"] = f"{int((r.aligned > 0).sum())}/7 (t>2: {int((r.t > 2).sum())}; t<-2: {int((r.t < -2).sum())})"
    OUT[name] = o
names = list(M)
cross = {}
for i in range(3):
    for j in range(i + 1, 3):
        a, b = names[i], names[j]; key = f"{a} ~ {b}"
        x, y = V3[a].ev.astype(float), V3[b].ev.astype(float).reindex(V3[a].index)
        ja = pd.to_numeric(J[a]["own - others d_z"], errors="coerce"); jb = pd.to_numeric(J[b]["own - others d_z"], errors="coerce").reindex(ja.index)
        ba = pd.to_numeric(B[a]["own - others d_z"], errors="coerce"); bb = pd.to_numeric(B[b]["own - others d_z"], errors="coerce").reindex(ba.index)
        sa, sb = V3[a]["shift"].astype(float), V3[b]["shift"].astype(float).reindex(V3[a].index)
        cross[key] = dict(response_r=round(x.corr(y), 2), scenario_shift_r=round(sa.corr(sb), 2), judgment_r=round(ja.corr(jb), 2), steering_r=round(ba.corr(bb), 2))
allpos = lambda f: int(sum(all(V3[n].ev.get(e, -1) > 0 for n in names) for e in V3[names[0]].index))
OUT["_cross"] = cross; OUT["_response_pos_all_three"] = allpos(None)
json.dump(OUT, open(HERE / "analysis_v6/run1_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(OUT, indent=1, ensure_ascii=False))
