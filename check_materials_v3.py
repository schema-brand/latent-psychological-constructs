"""Check a materials_v3 file (exp_*.json, family_*.json or awareness.json) against MATERIALS_SPEC_v3.md.
Usage: python check_materials_v3.py materials_v3/<file>.json [more files]. Prints errors (E) and warnings (W); exit 1 on errors."""
import json, re, sys, difflib
from pathlib import Path

GLOBAL = ["theory", "theoretical*", "mechanism*", "lens", "lenses", "hypothes*", "mediat*", "construct*", "bias*", "heuristic*",
          "psycholog*", "cognitive", "framing", "frame", "framed", "effect", "effects", "research shows", "studies show", "people tend",
          "tend to", "most people", "experiment*", "manipulat*", "participant*", "condition", "conditions", "self-report*", "nudge*"]
STUDY = ["study", "studies", "survey*", "questionnaire*", "researcher*", "research", "lab", "laboratory", "psychologist*", "scientist*",
         "subject*", "respondent*"]
HINT = ["you feel", "you may feel", "you might feel", "you realize", "you realise", "you notice*", "makes you", "you sense", "you wonder",
        "naturally", "of course", "obviously", "you should", "you remember*", "you recall*", "you think back", "comes back to you",
        "on your mind", "in your mind", "stays with you", "stayed with you", "you can't help", "in mind"]
HYPO = ["imagine*", "suppose*", "pretend*", "picture yourself", "what if"]
FAMILY = {
    "ownership": ["ownership", "owner*", "own", "mine", "my own", "your own", "yours", "belong*", "possess*", "attach*", "personal",
                  "endow*", "perceived", "take it home", "take home"],
    "loss": ["loss", "losses", "lose", "losing", "lost", "give up", "giving up", "part with", "parting", "endow*", "owner*", "ownership",
             "gain*", "forgo*"],
    "reciprocity": ["reciproc*", "owe*", "owing", "obligat*", "indebted", "debt", "return the favor", "return the favour", "fair*",
                    "concession*", "compromise*", "meet halfway", "halfway", "in return", "repay*", "even", "payback"],
    "self_perception": ["kind of person", "sort of person", "type of person", "self-image", "self-view", "self-concept", "identity",
                        "consistent", "consistency", "commit*", "generous", "helpful", "altruis*", "good person", "civic-minded", "charitable"],
    "anchoring": ["anchor*", "starting point", "starting value", "adjust*", "insufficient*", "estimate from", "start from", "reference point",
                  "ballpark"],
}
BANDS = {"short": (50, 80), "medium": (90, 130), "long": (140, 200)}

def rx(term):
    t = re.escape(term.lower()).replace(r"\*", r"[\w'-]*").replace(r"\ ", r"\s+")
    return re.compile(r"(?<![\w'-])" + t + r"(?![\w'-])" if not term.endswith("*") else r"(?<![\w'-])" + t, re.I)

def hits(text, terms, allowed_terms=(), spans=()):
    """Leakage hits: list of (term, context, inside_span). spans: substrings of text whose matches are flagged inside_span."""
    out, low = [], [a.lower() for a in allowed_terms]
    rng = [(text.find(s), text.find(s) + len(s)) for s in spans if s and s in text]
    for term in terms:
        if term.lower() in low: continue
        for m in rx(term).finditer(text):
            inside = any(a <= m.start() < b for a, b in rng)
            out.append((term, " ".join(text[max(0, m.start() - 30): m.end() + 30].split()), inside))
    return out

def words(t): return len(re.findall(r"[A-Za-z0-9$'’-]+", t))

def as_list(x): return [x] if isinstance(x, str) else list(x or [])

def outside_span_diff(a, b, sa, sb):
    """True if a and b become identical when each declared span pair (in order) is replaced by the same placeholder."""
    if len(sa) != len(sb) or not sa: return False
    for i, (x, y) in enumerate(zip(sa, sb)):
        if x not in a or y not in b: return False
        a = a.replace(x, f"<{i}>", 1); b = b.replace(y, f"<{i}>", 1)
    return a == b

def check_exp(d, E, W):
    fam = d.get("family")
    if fam not in FAMILY: E.append(f"family {fam!r} unknown")
    for k in ("id", "mediator", "x", "predicted_sign", "original_dv", "leakage_terms", "statements", "variants"):
        if k not in d: E.append(f"missing {k}")
    terms = list(dict.fromkeys((d.get("leakage_terms") or []) + FAMILY.get(fam, [])))
    x_allowed = d.get("x_allowed", [])
    st = d.get("statements", {})
    for k, n in (("kf_mediator", 4), ("kf_denial", 4), ("kf_matched_control", 4)):
        xs = st.get(k, [])
        if len(xs) != n: E.append(f"statements.{k}: {len(xs)} items, need {n}")
        for s in xs:
            if not 9 <= words(s) <= 18: W.append(f"statements.{k} length {words(s)}: {s}")
            for t, c, _ in hits(s, terms): E.append(f"statements.{k} leakage '{t}': ...{c}...")
    vs = d.get("variants", [])
    if len(vs) != 12: E.append(f"{len(vs)} variants, need 12")
    lens = [v.get("length") for v in vs]
    for b in BANDS:
        if lens.count(b) != 4: E.append(f"{lens.count(b)} {b} variants, need 4")
    seen = set()
    for v in vs:
        vid = v.get("vid", "?")
        for k in ("body_treatment", "body_control", "x_span_treatment", "x_span_control", "setting", "length"):
            if not v.get(k): E.append(f"{vid}: missing {k}")
        bt, bc = v.get("body_treatment", ""), v.get("body_control", "")
        st_, sc_ = as_list(v.get("x_span_treatment")), as_list(v.get("x_span_control"))
        qs = [v.get("question")] if v.get("question") else [v.get("question_treatment"), v.get("question_control")]
        if not all(qs): E.append(f"{vid}: question missing")
        for x in st_:
            if x not in bt: E.append(f"{vid}: x_span_treatment piece not found in body_treatment: {x[:60]}")
        for x in sc_:
            if x not in bc: E.append(f"{vid}: x_span_control piece not found in body_control: {x[:60]}")
        if len(st_) != len(sc_): E.append(f"{vid}: x_span_treatment and x_span_control must have the same number of pieces")
        if not outside_span_diff(bt, bc, st_, sc_):
            sm = difflib.SequenceMatcher(None, bt.split(), bc.split())
            dif = [(op, " ".join(bt.split()[i1:i2]), " ".join(bc.split()[j1:j2])) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
            E.append(f"{vid}: arms differ outside the declared spans (list every differing passage as a piece of x_span_treatment / x_span_control, in order): {dif[:4]}")
        b = BANDS.get(v.get("length"))
        if b and not b[0] <= words(bt) <= b[1]: E.append(f"{vid}: body_treatment {words(bt)} words, band {v.get('length')} {b}")
        for name, text, span in (("body_treatment", bt, st_), ("body_control", bc, sc_)):
            for t, c, inside in hits(text, terms + GLOBAL + STUDY + HINT + HYPO, allowed_terms=x_allowed, spans=span):
                (W if inside else E).append(f"{vid} {name} leakage '{t}'{' (inside x_span: allowed only if it IS the manipulation)' if inside else ''}: ...{c}...")
        for q in qs:
            if not q: continue
            for t, c, _ in hits(q, terms + GLOBAL + STUDY + HINT + HYPO): E.append(f"{vid} question leakage '{t}': ...{c}...")
            if not ("1" in q and "9" in q): E.append(f"{vid}: question lacks 1-9 labels: {q[:80]}")
        if "?" in bt or "?" in bc: W.append(f"{vid}: question mark inside body")
        key = bt[:60]
        if key in seen: E.append(f"{vid}: duplicate body opening")
        seen.add(key)

def check_family(d, E, W):
    fam = d.get("family"); terms = FAMILY.get(fam, [])
    if fam not in FAMILY: E.append(f"family {fam!r} unknown")
    ps = d.get("pairs", [])
    if len(ps) != 20: E.append(f"{len(ps)} pairs, need 20")
    for p in ps:
        h, l = p.get("high", ""), p.get("low", "")
        if not (40 <= words(h) <= 85 and 40 <= words(l) <= 85): E.append(f"{p.get('pid')}: lengths {words(h)}/{words(l)} outside 40-85")
        if h and l and abs(words(h) - words(l)) > 0.12 * max(words(h), words(l)) + 1: W.append(f"{p.get('pid')}: high/low length mismatch {words(h)}/{words(l)}")
        for nm, t in (("high", h), ("low", l)):
            for term, c, _ in hits(t, terms + GLOBAL + STUDY + HINT + HYPO): E.append(f"{p.get('pid')} {nm} leakage '{term}': ...{c}...")

def check_awareness(d, E, W):
    for k, n in (("study_prefixes", 6), ("neutral_prefixes", 6), ("study_vs_everyday", 30), ("hypothetical_vs_real", 30),
                 ("fiction_vs_self", 20), ("awareness_questions", 5)):
        if len(d.get(k, [])) != n: E.append(f"{k}: {len(d.get(k, []))} items, need {n}")
    for s in d.get("neutral_prefixes", []):
        for t, c, _ in hits(s, STUDY + GLOBAL + HYPO): E.append(f"neutral_prefix leakage '{t}': {c}")
    for p in d.get("study_vs_everyday", []):
        for t, c, _ in hits(p.get("everyday", ""), STUDY + GLOBAL + HINT + HYPO): E.append(f"{p.get('pid')} everyday leakage '{t}': ...{c}...")
        if not hits(p.get("study", ""), STUDY + ["participant*", "experiment*"]): E.append(f"{p.get('pid')}: study version has no research words")
    for p in d.get("hypothetical_vs_real", []):
        for t, c, _ in hits(p.get("real", ""), STUDY + GLOBAL + HINT + HYPO): E.append(f"{p.get('pid')} real leakage '{t}': ...{c}...")
        if not hits(p.get("hypothetical", ""), HYPO): E.append(f"{p.get('pid')}: hypothetical version has no imagine/suppose")
    for p in d.get("fiction_vs_self", []):
        for nm in ("self", "fiction"):
            for t, c, _ in hits(p.get(nm, ""), STUDY + GLOBAL + HYPO): E.append(f"{p.get('pid')} {nm} leakage '{t}': ...{c}...")

def main(paths):
    bad = 0
    for path in paths:
        d = json.load(open(path, encoding="utf-8")); E, W = [], []
        name = Path(path).name
        (check_exp if name.startswith("exp_") else check_family if name.startswith("family_") else check_awareness)(d, E, W)
        print(f"== {name}: {len(E)} errors, {len(W)} warnings")
        for e in E: print("  E", e)
        for w in W: print("  W", w)
        bad += len(E)
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main(sys.argv[1:])
