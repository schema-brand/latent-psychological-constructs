"""Diagnostic for 4c: how much of the experiment's treatment-minus-control state shift (V3, situation tokens, band 1, standardized)
lies in the experiment construct's top-10 pair-difference subspace (from the exported band-1 standardized pair differences), compared
with the subspaces of the other constructs. Usage: python analysis_v6/subspace_share_v6.py JOB ..."""
import sys, json, torch, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent; RV = HERE / "results_v6"
E = json.load(open(HERE / "materials_v6/upload_v6/experiments_v6.json", encoding="utf-8"))
for job in sys.argv[1:]:
    X = torch.load(RV / job / "V1_read/export_v6.pt", map_location="cpu", weights_only=False); sd = X["sd_mean_B1"].float()
    DZ = torch.load(RV / job / "V1_read/export_dz_v6.pt", map_location="cpu", weights_only=False)["dz_mean_B1"].float()
    meta = pd.read_csv(RV / job / "V1_read/pairs_meta.csv"); cons = sorted(meta.construct.unique())
    U = {}
    for c in cons:
        Z = DZ[torch.tensor((meta.construct == c).values)]; U[c] = torch.linalg.svd(Z, full_matrices=False)[2][:10]
    S = torch.load(RV / job / "V3_exp/shift_v3.pt", map_location="cpu", weights_only=False)
    mi, gi = S["marks"].index("body"), S["groups"].index("B1"); sh = S["shift"][:, mi, gi].float() / sd
    rows = []
    for k, key in enumerate(S["keys"]):
        eid = key[0] if isinstance(key, (list, tuple)) else str(key).split("|")[0]
        c = E[eid]["construct"]; s = sh[k]; n2 = float(s @ s)
        if n2 == 0 or c not in U: continue
        sh_c = {cc: float(((U[cc] @ s) ** 2).sum()) / n2 for cc in cons}
        others = [v for cc, v in sh_c.items() if cc != c]
        rows.append(dict(exp=eid, own=sh_c[c], others_median=float(np.median(others)), rank=1 + sum(v > sh_c[c] for v in others)))
    R = pd.DataFrame(rows).groupby("exp").mean()
    print(f"== {job}: share of the shift in the own 10-dim subspace: median {R.own.median():.3f} vs other constructs {R.others_median.median():.3f}; "
          f"own ranked first of 66 in {int((R['rank'] <= 1.5).sum())}/{len(R)}, median rank {R['rank'].median():.1f}")
