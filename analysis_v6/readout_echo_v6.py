"""Skeptic check for the readout blind judging (readout_judge_v6.py): are the read-out words only an echo of the words in the situations?
For each list: echo share = share of its word tokens (letters only, >= 3 letters, lower case) that occur as words in the construct's own HIGH
situations; text-match pick = the candidate whose high situations contain most of the list's words (a judge that only matches surface words).
Then the raters' hit rate split by echo share, and among lists whose words mostly do NOT occur in the situations.
Usage: python analysis_v6/readout_echo_v6.py  (after readout_judge_v6.py score)  -> analysis_v6/readout_echo_v6.json"""
import json, re, collections
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; BASE = HERE.parent
P = json.load(open(BASE / "materials_v6" / "upload_v6" / "pairs_v6.json", encoding="utf-8"))
X = {x["item"]: x for x in json.load(open(HERE / "jobs" / "R1_items.json", encoding="utf-8"))}
W = collections.defaultdict(set)
for p in P: W[p["construct"]] |= set(re.findall(r"[a-z]+", p["high"].lower()))
def words(toks): return [w for w in (re.sub(r"[^a-z]", "", t.lower()) for t in toks) if len(w) >= 3]
rows = []
for x in X.values():
    ws = words(x["tokens"])
    if not ws: rows.append(dict(item=x["item"], n_words=0, echo=None, text_hit=None)); continue
    echo = sum(w in W[x["construct"]] for w in ws) / len(ws)
    sc = {c: sum(w in W[c] for w in ws) for c in x["cands"]}; best = max(sc.values())
    text_hit = (sc[x["construct"]] == best and best > 0) / max(1, sum(v == best for v in sc.values()))   # ties split
    rows.append(dict(item=x["item"], n_words=len(ws), echo=echo, text_hit=text_hit))
E = pd.DataFrame(rows); D = pd.read_csv(HERE / "readout_judge_v6.csv").merge(E, on="item")
D["echo_band"] = pd.cut(D.echo, [-.01, .1, .2, .4, 1.0], labels=["<=10%", "10-20%", "20-40%", ">40%"])
S = {"text_match_hit_by_model_source": E.merge(pd.DataFrame(X.values())[["item", "model", "source"]], on="item").groupby(["model", "source"]).text_hit.mean().round(3).reset_index().to_dict("records"),
     "echo_mean_by_model_source": D.groupby(["model", "source"]).echo.mean().round(3).reset_index().to_dict("records"),
     "rater_hit_by_echo_band": D.groupby(["model", "source", "echo_band"], observed=True).agg(n=("hit", "size"), hit=("hit", "mean"), hit3=("hit3", "mean")).round(3).reset_index().to_dict("records"),
     "rater_hit_when_text_match_fails": D[D.text_hit == 0].groupby(["model", "source"]).agg(n=("hit", "size"), hit=("hit", "mean"), hit3=("hit3", "mean")).round(3).reset_index().to_dict("records")}
json.dump(S, open(HERE / "readout_echo_v6.json", "w"), indent=1, default=str)
for k, v in S.items(): print(k); [print("  ", r) for r in v]
