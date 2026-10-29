"""Exploratory test (Psych Science version, 22 Oct 2026): does the experiment's question concern the mediator?
Two independent coders (LLM agents, blind to the results; question_coding/coder1.json, coder2.json) coded
each of the 60 questions: 2 = asks about the mediator itself, 1 = makes it salient, 0 = does not evoke it. The score is the mean
of the two codes. Relates the score, across experiments, to the activation of the mediator's representation while the main
LLMs processed the question (mean of the three main LLMs, layers 30-50% and 60-75%), to the activation during the scenario,
and to the manipulation's effect on the response. Bootstrap over experiments. Writes analysis_v6/qcoding_v6.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

A = Path(__file__).resolve().parent; Q = A.parent / "question_coding"   # REPOSITORY: was A.parent / "draft_2026-10-21" / "psychsci" / "qcoding"
c1 = json.load(open(Q / "coder1.json", encoding="utf-8")); c2 = json.load(open(Q / "coder2.json", encoding="utf-8"))
ids = sorted(c1); a = np.array([c1[i]["code"] for i in ids]); b = np.array([c2[i]["code"] for i in ids])
score = pd.Series((a + b) / 2, index=ids)
ex = pd.read_csv(A / "example_s3_v6.csv"); m = ex.groupby("exp")[["x_B1", "body_B1", "q_B1", "q_B2", "dec_B1", "answer_dz"]].mean()
d = m.join(score.rename("score"), how="inner")
rng = np.random.default_rng(7); out = {"n": int(len(d)), "agreement": round(float((a == b).mean()), 3),
                                       "kappa_linear": round(float(cohen_kappa_score(a, b, weights="linear")), 3),
                                       "codes_coder1": {k: int((a == k).sum()) for k in (0, 1, 2)}, "codes_coder2": {k: int((b == k).sum()) for k in (0, 1, 2)}}
for col in ("q_B1", "q_B2", "dec_B1", "body_B1", "x_B1", "answer_dz"):
    r = stats.spearmanr(d.score, d[col])[0]
    bs = []
    for _ in range(5000):
        s = rng.integers(0, len(d), len(d)); bs.append(stats.spearmanr(d.score.values[s], d[col].values[s])[0])
    out[col] = {"rho": round(float(r), 3), "ci": [round(float(np.nanquantile(bs, .025)), 3), round(float(np.nanquantile(bs, .975)), 3)],
                "p_two_sided": round(float(stats.spearmanr(d.score, d[col])[1]), 4)}
out["examples"] = {e: float(score[e]) for e in ("gray_wegner_moral_typecasting", "reed_identity_salience") if e in score}
json.dump(out, open(A / "qcoding_v6.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
