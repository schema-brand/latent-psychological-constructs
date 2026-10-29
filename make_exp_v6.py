"""Materials v6, experiments (PAPER_OUTLINE_v1 study 3 / 4b-4d): for each of the 60 MVCA experiments, the frozen MVCA stimulus as variant
v00 (the Study 1 cells for moderated experiments, with both moderator levels kept), and rewriting jobs for 12 everyday variants written by
two writers (6 each), rotating over the six writers. Moderated experiments (MVCA Study 3 cells) get each variant at both condition levels.
Usage: python make_exp_v6.py jobs        # writes materials_v6/jobs/C1_<writer>.jsonl and Claude prompt files
       python make_exp_v6.py v00         # writes materials_v6/experiments/<id>.json with v00 (needs exp_questions_v00.json)"""
import json, sys, csv, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"; JOBS = M6 / "jobs"; CP = M6 / "claude_prompts"; EX = M6 / "experiments"
for d in (JOBS, CP, EX): d.mkdir(parents=True, exist_ok=True)
MV = HERE / "external" / "agent-psychonomicus" / "materials"   # REPOSITORY: was the MVCA Toolkit copy of the same files
_s = importlib.util.spec_from_file_location("cm3", HERE / "check_materials_v3.py"); cm3 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cm3)
_j = importlib.util.spec_from_file_location("j6", HERE / "make_or_jobs_v6.py"); j6 = importlib.util.module_from_spec(_j); _j.loader.exec_module(j6)
WRITERS = list(j6.WRITERS); E = {r["id"]: r for r in csv.DictReader(open(MV / "EXPERIMENTS.csv", encoding="utf-8"))}
RULES = [
    "Real, not hypothetical: second person, present tense ('You are at ...'). Never 'imagine', 'suppose', 'picture' or 'pretend'.",
    "Everyday frame only: no study, survey, questionnaire, researcher, participant, experiment, lab or market research. Move the original procedure into a natural everyday setting where the same manipulation occurs naturally, keeping its causal logic intact.",
    "Minimal pair: the treatment and control texts differ only in the manipulated feature; all other text is identical, character for character.",
    "Restate the manipulated feature near the end, where the person is about to answer, in parallel form in both texts. The restatement describes the situation, never a feeling or an effect.",
    "No stated mental states: never 'you feel', 'you like it', 'it seems special to you', 'you owe' and the like.",
    "The question is identical in both texts and asks for one judgement on a 1-9 scale in the original's direction (higher = more of the original response). For money or quantities, give 9 ordered bins written out in full, chosen so that typical answers fall near the middle; otherwise label the endpoints (1 = ..., 9 = ...). The question must not contain the manipulated feature or a banned word.",
    "The texts must fit any adult: no gender, age, family members, job or city for 'you'; other people may have first names. No brands, real people or famous places.",
    "Natural English with varied sentence structure across variants; every variant uses a different object, setting or relationship.",
]

def mv(eid): return json.loads((MV / "experiments" / f"{eid}.json").read_text(encoding="utf-8"))

def frozen(d):
    """Treatment and control texts of the Study 1 comparison; for moderated experiments also the other condition level."""
    c1, c3 = d.get("cp1") or {}, d.get("cp3") or {}
    cells = {c["id"]: c for c in (c3.get("cells") or [])}
    if c1.get("treatment_unit") in cells:
        T, C = cells[c1["treatment_unit"]], cells[c1["control_unit"]]; lev = c1.get("condition_level")
        other = [l for l in (c3.get("condition") or {}).get("levels", []) if l["id"] != lev]
        out = {"body_treatment": T["stimulus"], "body_control": C["stimulus"], "condition_level": lev}
        if other:
            ot = next((c for c in cells.values() if c["arm"] == T["arm"] and c["condition_level"] == other[0]["id"]), None)
            oc = next((c for c in cells.values() if c["arm"] == C["arm"] and c["condition_level"] == other[0]["id"]), None)
            if ot and oc: out.update(body_treatment_B=ot["stimulus"], body_control_B=oc["stimulus"], condition_level_B=other[0]["id"])
        return out
    T = next(a for a in d["arms"] if a["role"] == "treatment"); C = next(a for a in d["arms"] if a["role"] == "control")
    return {"body_treatment": T["stimulus"], "body_control": C["stimulus"]}

def banned(d):
    lt = d.get("leakage_terms") or {}
    return sorted(set(lt.get("theory", []) + lt.get("mediator", []) + cm3.GLOBAL + cm3.STUDY + cm3.HINT + cm3.HYPO), key=str.lower)

def moderation(d):
    c3 = d.get("cp3") or {}; cond = c3.get("condition") or {}; lev = (d.get("cp1") or {}).get("condition_level")
    if not c3.get("cells") or not cond.get("levels") or not lev: return None
    A = next(l for l in cond["levels"] if l["id"] == lev); B = next(l for l in cond["levels"] if l["id"] != lev)
    return {"name": cond.get("name"), "A": A, "B": B}

def prompt(eid, w, k):
    d = mv(eid); fz = frozen(d); mod = moderation(d); doms = j6.DOMAINS[w]; use = (doms * 2)[(3 * k) % 12:(3 * k) % 12 + 6]
    rules = "\n".join(f"{i+1}. {r}" for i, r in enumerate(RULES))
    modtxt, modfields = "", ""
    if mod:
        modtxt = (f"\n\nSITUATIONAL CONDITION (write every variant under both levels)\nThe experiment also varies {mod['name']}.\n"
                  f"Level A (use in body_treatment and body_control): {mod['A']['label']}. Original wording: {mod['A']['text']}\n"
                  f"Level B (use in body_treatment_B and body_control_B): {mod['B']['label']}. Original wording: {mod['B']['text']}\n"
                  "The B texts are the A texts with only the condition passage changed to level B; everything else stays identical.")
        modfields = '"body_treatment_B": "...", "body_control_B": "...", '
    return (f"You are rewriting a published psychology experiment as short everyday situations. Write 6 variants of the experiment, each a "
            f"PAIR of texts: a treatment text and a control text that differ only in the manipulation.\n\n"
            f"THE EXPERIMENT\nManipulation (treatment vs control): {d['design']['x']}\nResponse measured: {d['design']['y']}\n"
            f"Published result: {d['design'].get('published_direction')}\n\nThe original procedure (for reference only; do not copy its setting):\n"
            f"TREATMENT: {fz['body_treatment']}\nCONTROL: {fz['body_control']}{modtxt}\n\nRULES\n{rules}\n\n"
            f"BANNED WORDS (never use them, in any form; '*' means any ending): {', '.join(banned(d))}. Words that describe the manipulated act itself are allowed inside the manipulated passages.\n\n"
            f"LENGTHS AND SETTINGS: variants 1-2 short (50-80 words), 3-4 medium (90-130 words), 5-6 long (140-200 words), counted on the treatment text; "
            f"use these settings, one per variant, in order: {'; '.join(use)} (adapt the setting so the manipulation occurs naturally in it).\n\n"
            'OUTPUT: only a JSON object, no other text: {"variants": [{"vid": "v01", "setting": "<short label>", "length": "short|medium|long", '
            f'"body_treatment": "...", "body_control": "...", {modfields}"x_span_treatment": ["<exact substring(s) of body_treatment that carry the manipulation, one per differing passage, in order>"], '
            '"x_span_control": ["<the corresponding exact substrings of body_control, same number and order>"], "question": "<identical in both texts>", '
            '"note": "<one sentence: how this variant implements the manipulation faithfully>"}]} with exactly 6 variants (vid v01-v06).'), banned(d), bool(mod)

def assignment():
    """Two writers per experiment, rotating so that each writer gets 20 experiments and every pair of writers appears."""
    ids = sorted(E); A = {}
    for i, eid in enumerate(ids):
        a = WRITERS[i % 6]; b = WRITERS[(i // 6 + i + 1) % 6]
        if b == a: b = WRITERS[(i + 3) % 6]
        A[eid] = [a, b]
    return A

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "jobs"
    if mode == "jobs":
        A = assignment(); by = {w: [] for w in WRITERS}; cmeta = {}
        for eid, ws in A.items():
            for k, w in enumerate(ws):
                p, ban, moded = prompt(eid, w, k); cid = f"{w}__exp__{eid}"
                if w == "claude":
                    (CP / f"{cid}.txt").write_text(p, encoding="utf-8"); by[w].append(cid)
                    cmeta[cid] = {"writer": w, "experiment": eid, "n": 6, "banned": ban, "moderated": moded, "prompt_version": "exp_v6.0", "call": cid}; continue
                by[w].append({"id": cid, "model": j6.WRITERS[w], "prompt": p, "check": "variants",
                              "params": {"max_tokens": 32000, "temperature": 0.8, "reasoning": {"effort": j6.EFFORT.get(w, "low")}},
                              "meta": {"writer": w, "experiment": eid, "n": 6, "banned": ban, "moderated": moded, "prompt_version": "exp_v6.0", "call": cid}})
        for w, xs in by.items():
            if w == "claude": (M6 / "claude_calls_exp.json").write_text(json.dumps(xs, indent=0), encoding="utf-8")
            else: (JOBS / f"C1_{w}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in xs) + "\n", encoding="utf-8")
            print(w, len(xs))
        json.dump(A, open(M6 / "exp_writer_assignment.json", "w"), indent=1)
        json.dump(cmeta, open(M6 / "claude_exp_meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    elif mode == "v00":
        Q = {q["experiment"]: q for q in json.loads((M6 / "exp_questions_v00.json").read_text(encoding="utf-8"))}
        for eid, r in E.items():
            d = mv(eid); fz = frozen(d); mod = moderation(d)
            v00 = dict(vid="v00", setting="MVCA frozen stimulus", length="long", question=Q[eid]["question_1_9"], **fz)
            out = dict(id=eid, label=r["label"], set=r["set"], replication_status=r["replication_status"], home_process=r["home_process"],
                       x_to_m_sign=r["x_to_m_sign"], m_to_y_sign=r["m_to_y_sign"], x_to_y_sign=r["x_to_y_sign"], study3_interaction_sign=r["study3_interaction_sign"],
                       studies=r["studies"], mediator_key=next(s["key"] for s in d["rater_scales"] if s["role"] == "mediator"), moderation=mod,
                       cp3_base_arm=(d.get("cp3") or {}).get("base_arm"), higher_means=Q[eid].get("higher_means"), variants=[v00])
            (EX / f"{eid}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(len(E), "experiment files")
