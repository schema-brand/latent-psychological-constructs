"""Materials v6, coding (OPENROUTER_PLAN_2026-10-19.md, phase E), by two raters blind to the study's hypotheses:
E1 state vs fact: is each construct a state the person holds (feeling, self-view, evaluation, motive) or a judgement that follows from facts
   in the situation (an inference about what happened, what others did, how things are)?  (PAPER_OUTLINE study 4a, preregistered moderator)
E2 conceptual similarity of every pair of construct definitions (1-7) and whether the two name the same psychological state (construct
   families; PAPER_OUTLINE study 1 MTMM and study 2 theory structure).
Usage: python make_or_code_v6.py   -> materials_v6/jobs/E_<rater>.jsonl"""
import json, itertools, random
from pathlib import Path
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"; JOBS = M6 / "jobs"
RATERS = {"deepseek": "deepseek/deepseek-v4.1-flash-1017", "glm": "z-ai/glm-5.3"}
C = json.loads((M6 / "spec_v6.json").read_text(encoding="utf-8"))["constructs"]
keys = sorted(C); rng = random.Random(5)
E1 = """Below are definitions of psychological constructs, each describing something about "the person". For each one, decide which kind it is:
- "state": a state the person holds: a feeling, a view of themself, an evaluation, a desire or motive, or a way of thinking that the person brings to the situation and that could be stronger or weaker in the same situation;
- "fact": a judgement that follows directly from facts in the situation: an inference about what happened, what others did or intend, or how things are, which the facts of the situation largely settle.
Give your best single answer and your confidence (1 = a guess, 5 = certain). Judge only the definition.

CONSTRUCTS
{items}

OUTPUT: only a JSON object: {{"codes": [{{"id": "<id>", "kind": "state" | "fact", "confidence": <1-5>}}]}} with one entry per construct."""
E2 = """Below are pairs of definitions of psychological constructs. For each pair, rate how similar in meaning the two constructs are
(1 = unrelated, 4 = related but clearly different, 7 = the same psychological state described in other words), and say whether they name
the same psychological state ("same": true or false). Judge only the definitions.

PAIRS
{items}

OUTPUT: only a JSON object: {{"pairs": [{{"id": "<id>", "similarity": <1-7>, "same": true | false}}]}} with one entry per pair."""

def defn(k): return C[k]["definition"]
jobs = {r: [] for r in RATERS}
order = keys[:]; rng.shuffle(order)
for i in range(0, len(order), 11):
    chunk = order[i:i + 11]; items = "\n".join(f"[{k}] {defn(k)}" for k in chunk)
    for r, m in RATERS.items():
        jobs[r].append({"id": f"E1__{r}__{i:03d}", "model": m, "prompt": E1.format(items=items), "check": "json",
                        "params": {"max_tokens": 16000, "temperature": 0.2, "reasoning": {"effort": "low"}}, "meta": {"task": "E1", "ids": chunk}})
pairs = list(itertools.combinations(keys, 2)); rng.shuffle(pairs)
for i in range(0, len(pairs), 25):
    chunk = pairs[i:i + 25]
    items = "\n".join(f"[{a}|{b}] A: {defn(a)}\n    B: {defn(b)}" if rng.random() < .5 else f"[{a}|{b}] A: {defn(b)}\n    B: {defn(a)}" for a, b in chunk)
    for r, m in RATERS.items():
        jobs[r].append({"id": f"E2__{r}__{i:04d}", "model": m, "prompt": E2.format(items=items), "check": "json",
                        "params": {"max_tokens": 16000, "temperature": 0.2, "reasoning": {"effort": "low"}}, "meta": {"task": "E2", "ids": [f"{a}|{b}" for a, b in chunk]}})
for r, xs in jobs.items():
    (JOBS / f"E_{r}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in xs) + "\n", encoding="utf-8"); print(r, len(xs))
