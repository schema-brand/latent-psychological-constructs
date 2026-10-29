"""Per-experiment effects on the activation of the mediator's representation at each position (three main LLMs), to choose an example
experiment for the text. Same computation as depth_s3_v6.py (own minus mean of the other 65, d_z over wording labels, signed by x_to_m),
for positions x/body/q/dec and bands B1/B2; answer d_z from v3_all_models.csv. Writes example_s3_v6.csv."""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent; LAT = HERE.parent; RV = LAT / "results_v6"
E = json.load(open(LAT / "materials_v6" / "upload_v6" / "experiments_v6.json", encoding="utf-8"))
def sgn(s):
    try: return int(float(str(s).replace("+", ""))) if str(s).strip() not in ("", "None") else 0
    except ValueError: return 0
def dz(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean() / x.std(ddof=1)) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan
JOBS = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32"}
ANS = pd.read_csv(HERE / "v3_all_models.csv")
rows = []
for name, job in JOBS.items():
    P = torch.load(RV / job / "V3_exp" / "proj_v3.pt", map_location="cpu", weights_only=False)
    PS = P["std"].float().numpy(); MK, GN, CONS = P["marks"], P["groups"], P["constructs"]
    D = pd.read_csv(RV / job / "V3_exp" / "items.csv")
    a = ANS[ANS.model == name].set_index("exp").dz
    for eid, g_ in D.groupby("exp"):
        e = E[eid]; A_ = g_[g_.level == "A"]; oc = CONS.index(e["construct"])
        r = dict(model=name, exp=eid, label=e.get("label"), construct=e["construct"], repl=e.get("replication_status") or "", answer_dz=a.get(eid, np.nan))
        for m in ("x", "body", "q", "dec"):
            for g in ("B1", "B2"):
                pj = PS[A_.index.values][:, MK.index(m), GN.index(g)]
                om = pj[:, oc] - (pj.sum(1) - pj[:, oc]) / (len(CONS) - 1)
                A2 = A_.assign(om=om); ts = A2[A2.arm == "treatment"].set_index(["vid", "agent"]).om; cs = A2[A2.arm == "control"].set_index(["vid", "agent"]).om
                s = (ts - cs.reindex(ts.index)).groupby(level=0).mean() * (sgn(e["x_to_m_sign"]) or 1)
                r[f"{m}_{g}"] = dz(s.values)
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(HERE / "example_s3_v6.csv", index=False)
m = df.groupby(["exp", "label", "construct", "repl"])[["answer_dz", "x_B1", "body_B1", "q_B1", "q_B2", "dec_B1", "dec_B2"]].agg(["mean", "min"]).round(2)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 80); pd.set_option("display.max_columns", 30)
mm = df.groupby(["exp", "label", "construct", "repl"])[["answer_dz", "body_B1", "q_B1", "dec_B1", "dec_B2"]].mean().round(2)
print(mm.sort_values("body_B1", ascending=False).to_string())
