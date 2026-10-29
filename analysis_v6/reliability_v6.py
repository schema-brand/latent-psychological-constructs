"""Within-model split-half reliability of experiment-level (V4b steering, V3 response) and judgment-level (V4a) effects, to compare with
cross-model agreement. Halves: even vs odd variant (V3/V4b) or writer halves (V4a). Spearman-Brown corrected r."""
import sys, numpy as np, pandas as pd
from pathlib import Path
RV = Path(__file__).resolve().parent.parent / "results_v6"
sb = lambda r: 2 * r / (1 + r)
for job in sys.argv[1:]:
    B = pd.read_csv(RV / job / "V4b_steer/steer_v4b.csv", usecols=["exp", "kind", "dose", "vid", "agent", "m_to_y_sign", "d", "mass", "mass0"])
    B = B[(B.mass >= .5) & (B.mass0 >= .5) & (B.dose.abs() == 2)]
    B["u"] = np.sign(B.dose) * B.d * B.m_to_y_sign / 2; B["half"] = B.vid.str[1:].astype(int) % 2
    own = B[B.kind == "own"].groupby(["exp", "half"]).u.mean(); oth = B[B.kind != "own"].groupby(["exp", "half"]).u.mean()
    o = own.unstack(); od = (own - oth).unstack()
    I = pd.read_csv(RV / job / "V3_exp/items.csv"); I = I[I.mass >= .5]; I["half"] = I.vid.str[1:].astype(int) % 2
    t = I[I.arm == "treatment"].groupby(["exp", "half"]).ev.mean(); c = I[I.arm == "control"].groupby(["exp", "half"]).ev.mean(); rr = (t - c).unstack()
    A = pd.read_csv(RV / job / "V4a_causal/causal_v4a.csv", usecols=["j", "writer", "part", "kind", "dose", "ev", "ev0", "mass"])
    A = A[(A.part == "pushed") & (A.mass >= .5) & (A.dose.abs() == 2)]; A["u"] = np.sign(A.dose) * (A.ev - A.ev0) / 2
    ws = sorted(A.writer.unique()); A["half"] = A.writer.map({w: k % 2 for k, w in enumerate(ws)})
    ao = A[A.kind == "own"].groupby(["j", "half"]).u.mean(); aa = A[A.kind.isin(["near", "rand"])].groupby(["j", "half"]).u.mean(); ad = (ao - aa).unstack()
    f = lambda X: round(sb(X[0].corr(X[1])), 2)
    print(f"{job}: split-half reliability (corrected): response {f(rr)}, steering own push {f(o)}, steering own-minus-others {f(od)}, judgment own-minus-others {f(ad)}")
