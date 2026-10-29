"""Per-construct identification of the held-out writer's pairs at each main LLM's best layer (from V1_read/lowo_cos_v6.pt).
Supports 'each construct had an identifiable representation': minimum per-construct accuracy across the 66 constructs."""
import torch, pandas as pd, numpy as np, json
from pathlib import Path
RV = Path(__file__).resolve().parent.parent / "results_v6"; out = {}
for name, job in (("Qwen3.6 27B", "V610_qwen36"), ("Gemma 4 31B", "V611_gemma4_31b"), ("OLMo 3.1 32B", "V612_olmo32")):
    d = torch.load(RV / job / "V1_read" / "lowo_cos_v6.pt", map_location="cpu", weights_only=False)
    P = d["proj"].float().numpy(); C = d["constructs"]; L = d["layers"]
    meta = pd.read_csv(RV / job / "V1_read" / "pairs_meta.csv"); y = np.array([C.index(c) for c in meta["construct"]])
    acc = [(P[:, li, :].argmax(1) == y).mean() for li in range(len(L))]; bl = int(np.argmax(acc))
    per = pd.Series(P[:, bl, :].argmax(1) == y).groupby(y).mean()
    out[name] = dict(best_layer=L[bl], overall=round(float(acc[bl]), 4), per_construct_min=round(float(per.min()), 3), per_construct_median=round(float(per.median()), 3),
                     lowest={C[i]: round(float(per[i]), 3) for i in per.nsmallest(3).index})
    print(name, out[name])
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
