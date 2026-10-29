"""Materials v6 (OPENROUTER_PLAN_2026-10-19.md; PAPER_OUTLINE_v1_2026-10-19.md): assemble spec_v6.json from the spec parts
(materials_v6/spec_parts/*.json, written from the MVCA scales) plus the comparison and reused rival constructs of v4, and write the
writing jobs for the runner (one JSONL per writer and wave) and the identical prompt files for the local Claude writer.
Usage: python make_or_jobs_v6.py spec          # assemble and check spec_v6.json, write MATERIALS_SPEC_v6.md
       python make_or_jobs_v6.py jobs <wave>    # wave 1 = first call of every construct; wave 2 = all other calls"""
import json, sys, glob, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"; JOBS = M6 / "jobs"; CP = M6 / "claude_prompts"
for d in (JOBS, CP): d.mkdir(parents=True, exist_ok=True)
V4 = json.loads((HERE / "materials_v4" / "spec_v4.json").read_text(encoding="utf-8"))
GENERAL_RULES = V4["general_rules"]
WRITERS = {"deepseek": "deepseek/deepseek-v4.1-flash-1017", "kimi": "moonshotai/kimi-k3", "muse": "meta/muse-spark-1.3",
           "grok": "x-ai/grok-4.7", "minimax": "minimax/minimax-m3", "claude": "local:claude-sonnet (subagent)"}
EFFORT = {"kimi": "medium"}
REASONING_OFF = {"minimax"}
DOMAINS = dict(V4["domains"])
DOMAINS["minimax"] = ["chess club", "pet shop", "self-service car wash", "boat rental dock", "quiz night at a community hall", "rooftop terrace",
                      "stationery shop", "climbing gym", "village pharmacy", "photography walk", "toy library", "seaside promenade"]
COMPARISON_POINTS = "how pleasant the situation is, how good the object, cause or request is, how attractive the surroundings are, and how much 'you' is involved"
V4_COMPARISONS = {"favorable_evaluation": "comparison", "pleasantness": "comparison", "setting_appeal": "comparison",
                  "affective_reaction": "rival", "request_contrast": "rival"}

GENERIC_BANNED = {"kind", "related", "matter", "matters", "hard", "facts", "contact", "possib*", "personal*", "heart", "sick", "fan", "fans", "a while", "long time"}

def load_spec():
    C = {}
    for f in sorted(glob.glob(str(M6 / "spec_parts" / "*.json"))):
        group = Path(f).stem
        for c in json.loads(Path(f).read_text(encoding="utf-8"))["constructs"]:
            c["group"] = group.replace("_", " ")
            c["banned"] = [w for w in c.get("banned", []) if w.lower() not in GENERIC_BANNED]
            if c["construct"] in C: raise SystemExit(f"duplicate construct {c['construct']} ({group} and {C[c['construct']]['group']})")
            C[c["construct"]] = c
    for k, role in V4_COMPARISONS.items():
        v = V4["comparisons"][k]
        C[k] = {"construct": k, "role": role, "group": "comparison", "sources": [{"experiment": s["experiment"], "scale": s["scale"]} for s in v["mvca_scales"]],
                "definition": v["definition"], "high": v["high"], "low": {"L4": v["low"]}, "banned": v["banned"], "number_exempt": False,
                "differ_only_in": None, "avoid_manipulation": None, "anchors": None, "writability": "easy", "v4_format": True}
    return C

def check_spec(C):
    problems = []
    for k, c in C.items():
        for f in ("definition", "high", "low", "banned"):
            if not c.get(f): problems.append(f"{k}: missing {f}")
        if not c.get("v4_format"):
            for f in ("anchors", "differ_only_in", "avoid_manipulation"):
                if not c.get(f): problems.append(f"{k}: missing {f}")
            if set(c.get("low", {})) != {"L1", "L4"}: problems.append(f"{k}: low poles {list(c.get('low', {}))}")
    return problems

def calls_for(C):
    calls = []
    for w in WRITERS:
        for k, c in C.items():
            if c.get("v4_format"): plan = [("L4", 8, 0), ("L4", 8, 1)]
            elif c["role"] == "rival": plan = [("L1", 8, 0), ("L4", 8, 1)]
            else: plan = [("L1", 6, 0), ("L1", 6, 1), ("L4", 6, 0), ("L4", 6, 1)]
            for i, (low, n, j) in enumerate(plan):
                calls.append(dict(writer=w, construct=k, low=low, n=n, j=j, idx=i, call=f"{w}__{k}__{low}_{i}"))
    return calls

def domains_of(call):
    doms = DOMAINS[call["writer"]]; n = call["n"]
    start = 3 if call["j"] == 0.5 else ((6 * call["j"]) % 12 if n == 6 else (4 * call["j"]) % 12)
    return (doms * 2)[start:start + n]

def prompt_v6(call, c):
    use = domains_of(call); n = call["n"]
    if c.get("v4_format"):
        point = (f"Construct: {c['definition']}.\nHIGH version: {c['high']}\nLOW version: {c['low']['L4']}\n"
                 f"The two versions must not differ in anything other than the construct above.")
    else:
        a = c["anchors"]; anchor = "1" if call["low"] == "L1" else "4"
        point = (f"Construct: {c['definition']}.\nThe point in which the two versions differ: {c['differ_only_in']}.\n"
                 f"HIGH version: {c['high']}\nLOW version: {c['low'][call['low']]}\n"
                 f"Reference scale for the person in the situation: 1 = {a['1']}; 4 = {a['4']}; 7 = {a['7']}. "
                 f"The HIGH version should sit at 7 and the LOW version at {anchor}.\n"
                 f"Everything else stays the same in both versions, including {COMPARISON_POINTS}.\n"
                 f"Do not produce this state through the following route, which belongs to a published study: {c['avoid_manipulation']}. Use other everyday routes.")
    rules = "\n".join(f"{i+1}. {r}" for i, r in enumerate(GENERAL_RULES))
    return (f"You are writing short everyday situations for a set of research materials. Each item is a PAIR: the same situation written twice, "
            f"a HIGH version and a LOW version, that differ in one point only.\n\nTHE ONE POINT THAT DIFFERS\n{point}\n\nRULES\n{rules}\n\n"
            f"BANNED WORDS (never use them, in any form; '*' means any ending): {', '.join(c['banned'])}\n\n"
            f"SETTINGS: write exactly {n} pairs, one in each of these settings, in this order: {'; '.join(use)}.\n\n"
            'OUTPUT: only a JSON object, no other text: {"pairs": [{"domain": "<setting>", "high": "<HIGH version>", "low": "<LOW version>"}]} '
            f"with exactly {n} pairs.")

def request(call, c):
    w = call["writer"]
    meta = {"writer": w, "construct": call["construct"], "role": c["role"], "low": call["low"], "n": call["n"], "banned": c["banned"],
            "domains": DOMAINS[w], "number_exempt": bool(c.get("number_exempt")), "prompt_version": "v6.0", "call": call["call"]}
    return {"id": call["call"], "model": WRITERS[w], "prompt": prompt_v6(call, c),
            "params": {"max_tokens": 32000, "temperature": 0.8, "reasoning": {"enabled": False} if w in REASONING_OFF else {"effort": EFFORT.get(w, "low")}}, "check": "pairs", "meta": meta}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "spec"
    C = load_spec(); probs = check_spec(C)
    if mode == "spec":
        (M6 / "spec_v6.json").write_text(json.dumps({"version": "v6", "writers": WRITERS, "domains": DOMAINS, "general_rules": GENERAL_RULES, "constructs": C},
                                                    indent=1, ensure_ascii=False), encoding="utf-8")
        roles = {}
        for c in C.values(): roles[c["role"]] = roles.get(c["role"], 0) + 1
        print(len(C), "constructs", roles, "| problems:", probs or "none")
        sys.exit()
    if probs: raise SystemExit("spec problems: " + "; ".join(probs))
    wave = int(sys.argv[2]); calls = [x for x in calls_for(C) if (x["idx"] == 0) == (wave == 1)]
    by = {}
    for x in calls: by.setdefault(x["writer"], []).append(x)
    for w, xs in by.items():
        if w == "claude":
            for x in xs: (CP / f"{x['call']}.txt").write_text(prompt_v6(x, C[x["construct"]]), encoding="utf-8")
            (M6 / f"claude_calls_w{wave}.json").write_text(json.dumps([x["call"] for x in xs], indent=0), encoding="utf-8")
        else:
            (JOBS / f"B{wave}_{w}.jsonl").write_text("\n".join(json.dumps(request(x, C[x["construct"]]), ensure_ascii=False) for x in xs) + "\n", encoding="utf-8")
        print(w, len(xs), "calls", sum(x["n"] for x in xs), "pairs")
