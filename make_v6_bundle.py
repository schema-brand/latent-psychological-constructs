"""Bundle the frozen v6 materials for the GPU runner (Drive MVCA_latent/runner/materials/v6/). Usage: python make_v6_bundle.py
Writes materials_v6/upload_v6/: spec_v6.json, pairs_v6.json (final pairs), experiments_v6.json (60 experiments with v00 + rewordings),
agents_v6.json, judgment_items_v6.json, comparisons_v6.json (per construct: 5 nearest constructs by rated definition similarity, 5 random
other constructs (fixed seed), state/fact kind), MANIFEST copy."""
import json, csv, random, shutil, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"; UP = M6 / "upload_v6"; UP.mkdir(exist_ok=True)
S = json.loads((M6 / "spec_v6.json").read_text(encoding="utf-8")); C = S["constructs"]; keys = sorted(C)
(UP / "spec_v6.json").write_text(json.dumps({"constructs": C, "general_rules": S["general_rules"]}, ensure_ascii=False), encoding="utf-8")
P = [json.loads(l) for l in open(M6 / "pairs_v6_final.jsonl", encoding="utf-8")]
(UP / "pairs_v6.json").write_text(json.dumps([{k: p[k] for k in ("pid", "writer", "construct", "role", "low_type", "domain", "high", "low")} for p in P], ensure_ascii=False), encoding="utf-8")
E = {}; MV = HERE / "external" / "agent-psychonomicus" / "materials" / "experiments"   # REPOSITORY: was the MVCA Toolkit copy
for f in sorted((M6 / "experiments").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8")); m = json.loads((MV / f.name).read_text(encoding="utf-8"))
    role = {a["id"]: a["role"] for a in m["arms"]}; st = m.get("statements") or {}; ins = (st.get("insert_into") or {}).get("default") or []
    d["arm_roles"] = role; d["study2_insert_into"] = [role.get(a, a) for a in ins]; d["mediator_statement_pole"] = st.get("mediator_statement_pole") or m.get("mediator_statement_pole")
    d["cp3_base_role"] = role.get(d.get("cp3_base_arm")) if d.get("cp3_base_arm") else None
    E[d["id"]] = d
(UP / "experiments_v6.json").write_text(json.dumps(E, ensure_ascii=False), encoding="utf-8")
for f in ("agents_v6.json", "judgment_items_v6.json", "MANIFEST_v6.json"): shutil.copy(M6 / f, UP / f)
sim = collections.defaultdict(dict)
for r in csv.DictReader(open(M6 / "coding_v6" / "definition_similarity.csv", encoding="utf-8")):
    if r["pair"].count("|") != 1: continue   # 24 DeepSeek rows answered with item numbers instead of names
    a, b = r["pair"].split("|"); v = [float(x) for x in (r["sim_glm"], r["sim_deepseek"]) if x not in ("", "None")]
    if v: sim[a][b] = sim[b][a] = sum(v) / len(v)
SF = json.loads((M6 / "coding_v6" / "state_fact_final.json").read_text())
rng = random.Random(606); comp = {}; tie = {c: random.Random(f"tie{c}").random() for c in keys}
miss = sum(1 for i, a in enumerate(keys) for b in keys[i + 1:] if b not in sim[a]); print("unrated construct pairs:", miss)
for k in keys:
    near = sorted((c for c in keys if c != k), key=lambda c: (-sim[k].get(c, 0), tie[c]))[:5]
    rest = [c for c in keys if c != k and c not in near]; rnd = rng.sample(rest, 5)
    comp[k] = {"nearest": near, "random": rnd, "kind": SF.get(k, {}).get("kind"), "experiments": [s["experiment"] for s in C[k]["sources"]], "role": C[k]["role"], "near_sim": [round(sim[k].get(c, 0), 2) for c in near]}
(UP / "comparisons_v6.json").write_text(json.dumps(comp, indent=1), encoding="utf-8")
print(len(C), "constructs;", len(P), "pairs;", len(E), "experiments;", sum(len(e["variants"]) for e in E.values()), "variants");
print("e.g. ownership nearest:", comp["ownership"]["nearest"])
print("study2 insert roles:", collections.Counter(tuple(e["study2_insert_into"]) for e in E.values() if "2" in e["studies"]), "| cp3 base roles:", {k: e["cp3_base_role"] for k, e in E.items() if e.get("moderation")},
      "| statement pole:", collections.Counter(e["mediator_statement_pole"] for e in E.values() if "2" in e["studies"]))
