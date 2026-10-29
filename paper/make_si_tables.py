"""Build the Supplementary Tables S1-S9 of the latent-representation manuscript (04_DRAFT_v7.md) from the analysis files.

Reads only local files under latent_2026-10-17/ (materials_v6, analysis_v6, results_v6) and the companion paper's release data
(release_2026-10-15/agent-psychonomicus/data). Values that exist only in the run-2/3 digests copied from Drive
(analysis_v6/run23_digests_2026-10-21.md, "RD") are parsed from that file; values that exist only in DEVLOG.md (the V7i individuality
digests, which are not local) are entered by hand below, each with its DEVLOG line. Nothing is written into the manuscript.

Usage: python draft_2026-10-21/make_si_tables.py  ->  draft_2026-10-21/SUPPLEMENTARY_TABLES_v1.md
"""
import json, re, collections
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                                   # latent_2026-10-17
AN, RV, MAT = BASE / "analysis_v6", BASE / "results_v6", BASE / "materials_v6"
UP = MAT / "upload_v6"
REL = BASE / "external/agent-psychonomicus/data"   # REPOSITORY: was BASE.parent / "release_2026-10-15/..."
OUT = HERE / "SUPPLEMENTARY_TABLES_v1.md"

# ------------------------------------------------------------------ formatting
MINUS = "−"
def _r(x, nd):
    return Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
def pct(x):
    """share (0-1) -> 'NN%' without decimals"""
    if x is None or (isinstance(x, float) and np.isnan(x)): return "n.a."
    v = _r(100 * float(x), 0)
    return (MINUS if v < 0 else "") + f"{abs(v)}%"
def cor(x, nd=2):
    """correlation-style number, two decimals, no leading zero"""
    if x is None or (isinstance(x, float) and np.isnan(x)): return "n.a."
    v = _r(float(x), nd); s = f"{abs(v):.{nd}f}"
    if abs(v) < 1: s = s.lstrip("0")
    return (MINUS if v < 0 else "") + s
def num(x, nd=2):
    """effect size (d_z, mean) with leading zero"""
    if x is None or (isinstance(x, float) and np.isnan(x)): return "n.a."
    v = _r(float(x), nd)
    return (MINUS if v < 0 else "") + f"{abs(v):.{nd}f}"
def pval(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return "n.a."
    p = float(p)
    if p < 0.0001: return "< 0.0001"
    if p < 0.001: return num(p, 4)
    if p < 0.01: return num(p, 3)
    return num(p, 2)
def P(p):
    v = pval(p); return "P " + v if v.startswith("<") else "P = " + v
def binom_p(k, n, p0):
    return float(stats.binomtest(int(k), int(n), p0, alternative="greater").pvalue)
def kn(s):
    """'42/60' -> (42, 60)"""
    a, b = str(s).split("/"); return int(a), int(b)
def table(head, rows):
    L = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows: L.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(L)
def jload(p): return json.load(open(p, encoding="utf-8"))

# ------------------------------------------------------------------ materials
SPEC = jload(UP / "spec_v6.json")["constructs"]
PAIRS = jload(UP / "pairs_v6.json")
EXPS = jload(UP / "experiments_v6.json")
WRITERS = ["claude", "deepseek", "grok", "kimi", "minimax", "muse"]
WNAME = {"claude": "Claude Sonnet", "deepseek": "DeepSeek V4.1 Flash", "grok": "Grok 4.7", "kimi": "Kimi K3", "minimax": "MiniMax M3", "muse": "Muse Spark 1.3"}
WSHORT = {"claude": "Cl", "deepseek": "DS", "grok": "Gr", "kimi": "Ki", "minimax": "MM", "muse": "Mu"}
PROC_ORDER = ["categorization", "social inference", "self-relevance", "affective response", "goal relevance", "causal inference", "comparison"]

PLAIN = {  # plain-word names (authors' construct keys -> words)
    "analytic_thinking": "Analytic thinking", "aversive_reaction": "Aversive reaction to another's action",
    "bad_outcome_in_mind": "Anticipation of a bad outcome", "claim_reliance": "Reliance on stated claims",
    "feeling_reliance": "Reliance on feelings", "felt_activation": "Felt activation", "prize_excitement": "Excitement about a prize",
    "processing_ease": "Ease of processing", "warmth_toward_others": "Warmth toward others",
    "action_imagery_vividness": "Vivid imagery of an action", "conservative_party_association": "Association with a conservative party",
    "expected_usability": "Expected ease of use", "focal_traits_shared": "Focal object's features found in the other",
    "heat_in_mind": "Heat on one's mind", "high_starting_value": "Starting value (anchor)", "object_seen_as_person": "Object seen as a person",
    "perceived_capability": "Perceived capability of a product", "perceived_fit": "Perceived fit of a new product with its maker",
    "price_near_top": "Price near the top of its range", "restriction_severity": "Severity of a restriction",
    "small_routine_expense": "Payment seen as a small routine expense", "action_outcome_confidence": "Confidence that one's actions produce results",
    "blame_attribution": "Blame for a bad outcome", "choice_uncertainty": "Uncertainty about the better option",
    "decision_importance": "Importance of getting a choice right", "earned_indulgence": "Earned indulgence",
    "economic_system_fairness": "Fairness of the economic system", "expected_price_level": "Expected (fair) price level",
    "outcome_rarity_inference": "Outcome seen as arising only after many attempts", "affective_reaction": "Emotional reaction to the object",
    "favorable_evaluation": "Favorable evaluation", "pleasantness": "Pleasantness", "request_contrast": "Request seen as small next to a larger one",
    "setting_appeal": "Appeal of the setting", "experience_focus": "Focus on the experience rather than the money",
    "gain_coding": "Outcome coded as a gain", "gift_reaches_cause": "Gift reaching the cause in full", "goal_progress": "Goal progress",
    "loss_framing": "Giving up coded as a loss", "perceived_stakes": "Perceived stakes", "power_desire": "Desire for power",
    "spending_slack": "Spending slack", "strength_weighting": "Weighting of the options' strengths", "accomplishment": "Accomplishment",
    "category_identity_importance": "Importance of a category identity", "competence_confidence": "Confidence in one's competence",
    "future_self_continuity": "Continuity with the future self", "helper_self_view": "View of oneself as a helper",
    "ingroup_closeness": "Closeness to one's group", "interdependent_self": "Interdependent view of the self", "ownership": "Psychological ownership",
    "personal_connection": "Personal connection to a product", "worldview_defense": "Defense of one's worldview",
    "affiliation_goal": "Wish to be liked by a particular person", "environmental_concern": "Environmental concern",
    "height_as_power": "Physical height as a sign of power", "misidentification_concern": "Concern about being taken for an outgroup member",
    "moral_agency": "Moral agency of another person", "others_as_partners": "Others seen as partners", "prestige_transfer": "Prestige of the source",
    "reciprocity_obligation": "Obligation to reciprocate", "room_to_use_more": "Room to use more", "status_desire": "Desire for status",
    "target_autonomy": "Autonomy of another person", "uniqueness": "Desire for uniqueness", "usual_amount_high": "Large amounts seen as usual",
}
assert set(PLAIN) == set(SPEC), set(SPEC) ^ set(PLAIN)
EXP_LABEL = {k: v["label"] for k, v in EXPS.items()}

def short_def(c):
    d = SPEC[c]["definition"].strip()
    d = re.sub(r"^[A-Za-z][A-Za-z\- ]{2,40}: (?=the person|The person)", "", d)        # drop 'label: ' prefixes
    d = d[0].upper() + d[1:]
    return d if d.endswith(".") else d + "."

# ------------------------------------------------------------------ LLMs
LLM = collections.OrderedDict([
    # job: (display, developer, nominal parameters, instruction-tuned, HF id)
    ("V610_qwen36", ("Qwen3.6 27B", "Alibaba (Qwen)", "27B", "yes", "Qwen/Qwen3.6-27B")),
    ("V611_gemma4_31b", ("Gemma 4 31B", "Google", "31B", "yes", "google/gemma-4-31B-it")),
    ("V612_olmo32", ("OLMo 3.1 32B", "Ai2", "32B", "yes", "allenai/Olmo-3.1-32B-Instruct")),
    ("V720_gemma4_26b_a4b", ("Gemma 4 26B", "Google", "26B (A4B)", "yes", "google/gemma-4-26B-A4B-it")),
    ("V730_qwen35_27b", ("Qwen3.5 27B", "Alibaba (Qwen)", "27B", "yes", "Qwen/Qwen3.5-27B")),
    ("V731_qwen35_9b", ("Qwen3.5 9B", "Alibaba (Qwen)", "9B", "yes", "Qwen/Qwen3.5-9B")),
    ("V732_qwen35_4b", ("Qwen3.5 4B", "Alibaba (Qwen)", "4B", "yes", "Qwen/Qwen3.5-4B")),
    ("V733_qwen35_2b", ("Qwen3.5 2B", "Alibaba (Qwen)", "2B", "yes", "Qwen/Qwen3.5-2B")),
    ("V734_qwen35_08b", ("Qwen3.5 0.8B", "Alibaba (Qwen)", "0.8B", "yes", "Qwen/Qwen3.5-0.8B")),
    ("V735_gemma4_e4b", ("Gemma 4 E4B", "Google", "E4B", "yes", "google/gemma-4-E4B-it")),
    ("V736_olmo7", ("OLMo 3 7B", "Ai2", "7B", "yes", "allenai/Olmo-3-7B-Instruct")),
    ("V740_qwen38", ("Qwen3.8 27B", "Alibaba (Qwen)", "27B", "yes", "Qwen/Qwen3.8-27B")),
    ("V741_gptoss20", ("gpt-oss 20B", "OpenAI", "20B", "yes", "openai/gpt-oss-20b")),
    ("V742_granite42", ("Granite 4.2 30B", "IBM", "30B", "yes", "ibm-granite/granite-4.2-30b")),
    ("V721_olmo32_base", ("OLMo 3 32B base", "Ai2", "32B", "no", "allenai/Olmo-3-1125-32B")),
    ("V722_gemma4_31b_base", ("Gemma 4 31B base", "Google", "31B", "no", "google/gemma-4-31B")),
])
MAIN = ["V610_qwen36", "V611_gemma4_31b", "V612_olmo32"]
NAME = {k: v[0] for k, v in LLM.items()}
# Study 3 / sweep jobs per model (run 3 numbering) and v3_all_models.json names
S3JOB = {"V610_qwen36": "V610_qwen36", "V611_gemma4_31b": "V611_gemma4_31b", "V612_olmo32": "V612_olmo32",
         "V730_qwen35_27b": "V830_v3sweep_qwen35_27b", "V731_qwen35_9b": "V831_v3sweep_qwen35_9b", "V732_qwen35_4b": "V832_v3sweep_qwen35_4b",
         "V733_qwen35_2b": "V833_v3sweep_qwen35_2b", "V734_qwen35_08b": "V834_v3sweep_qwen35_08b", "V735_gemma4_e4b": "V835_v3sweep_gemma4_e4b",
         "V736_olmo7": "V836_v3sweep_olmo7", "V740_qwen38": "V837_v3sweep_qwen38", "V741_gptoss20": "V838_v3sweep_gptoss20"}
V3NAME = {"V610_qwen36": "Qwen3.6 27B", "V611_gemma4_31b": "Gemma 4 31B", "V612_olmo32": "OLMo 3.1 32B", "V730_qwen35_27b": "Qwen3.5 27B",
          "V731_qwen35_9b": "Qwen3.5 9B", "V732_qwen35_4b": "Qwen3.5 4B", "V733_qwen35_2b": "Qwen3.5 2B", "V734_qwen35_08b": "Qwen3.5 0.8B",
          "V735_gemma4_e4b": "Gemma 4 E4B", "V736_olmo7": "OLMo 3 7B Instruct", "V740_qwen38": "Qwen3.8 27B", "V741_gptoss20": "gpt-oss 20B"}

# ------------------------------------------------------------------ run-2/3 digests (RD)
RD = open(AN / "run23_digests_2026-10-21.md", encoding="utf-8").read().splitlines()
def rd_overview_v1():
    out = {}
    for ln in RD:
        m = re.match(r"\| (V7\d\d) \| (\S+) \| (\d+) \| (\d+) / (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) @(\d+) \| ([\d.]+) \| ([\d.]+) / ([\d.]+) / ([\d.]+) \|", ln)
        if m:
            g = m.groups()
            out[g[0]] = dict(model=g[1], pairs=int(g[2]), Lstar=int(g[3]), n_layers=int(g[4]), acc0=float(g[5]), accL=float(g[6]), best=float(g[7]),
                             best_layer=int(g[8]), B1=float(g[9]), conv=float(g[10]), same=float(g[11]), base=float(g[12]))
    return out
def rd_v4a():
    out = {}
    for ln in RD:
        m = re.match(r"\| (V\d+) \| (V4a|V6a) \| .*? \| (\d+) \| (\d+)/66 \((\d+)\) \| (\d+)/66 \| (-?[\d.]+) \((-?[\d.]+)\) \| (\d+) \|", ln)
        if m:
            g = m.groups(); out[(g[0], g[1])] = dict(layer=int(g[2]), pos=int(g[3]), t2=int(g[4]), first=int(g[5]), r=float(g[6]), rc=float(g[7]), nfwd=int(g[8]))
    return out
def rd_section(job):
    s = [i for i, ln in enumerate(RD) if ln.startswith(f"## {job}")]
    if not s: return []
    i0 = s[0]; i1 = next((i for i in range(i0 + 1, len(RD)) if RD[i].startswith("## ")), len(RD)); return RD[i0:i1]
def rd_rows(job, first_col_regex):
    rows = []
    for ln in rd_section(job):
        if ln.startswith("|") and re.match(first_col_regex, ln[1:].strip()):
            rows.append([c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", ln.strip())[1:-1]])
    return rows
RDV1 = rd_overview_v1(); RDV4A = rd_v4a()

# ------------------------------------------------------------------ Study 1 identification per LLM
def v1_row(job):
    if job in MAIN:
        s = jload(RV / job / "V1_read/summary.json"); sm = s["profile"]["std_mean"]
        best = max(((int(k), v["acc"]) for k, v in sm.items() if k not in ("B1", "B2")), key=lambda x: x[1])
        return dict(pairs=s["n_pairs"], n_layers=s["n_layers"], Lstar=s["Lstar"], accL=sm[str(s["Lstar"])]["acc"], best=best[1], best_layer=best[0],
                    B1=sm["B1"]["acc"], top5=sm["B1"]["top5"], med=sm["B1"]["acc_mediators"], acc0=sm["0"]["acc"],
                    conv=s["mtmm_B1"]["convergent"], same=s["mtmm_B1"]["method"], base=s["mtmm_B1"]["baseline"], B1layers=s["B1"])
    key = job.split("_")[0]; r = dict(RDV1[key]); r["top5"] = None; return r
V1 = {j: v1_row(j) for j in LLM}

# ------------------------------------------------------------------ read-out
RO = pd.read_csv(AN / "readout_judge_v6_all_models.csv").set_index("model")
RJ = pd.read_csv(AN / "readout_judge_v6.csv"); RJ2 = pd.read_csv(AN / "readout_judge_v6_R2.csv")
def jlens_mid(job):
    g = RJ[(RJ.model == job) & (RJ.source == "jlens")]
    if not len(g): return None
    L = sorted(g.layer.unique()); x = g[g.layer.isin(L[1:-1])]
    return dict(hit=x.hit.mean(), hit3=x.hit3.mean(), layers=L[1:-1], n=len(x))
def agreement(job):
    D = RJ if job in MAIN else RJ2
    W = D[D.model == job].pivot_table(index="item", columns="rater", values="hit", aggfunc="first").dropna().astype(float)
    a, b = W.iloc[:, 0], W.iloc[:, 1]; po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return po, (po - pe) / (1 - pe), len(W)
RJS = jload(AN / "readout_judge_v6.json"); RJS2 = jload(AN / "readout_judge_v6_R2.json")
BOTH = {**RJS["constructs_hit_by_both_raters_any_layer"], **RJS2["constructs_hit_by_both_raters_any_layer"]}
ECHO = jload(AN / "readout_echo_v6.json")

# ------------------------------------------------------------------ Study 3
V3 = jload(AN / "v3_all_models.json"); SHR = jload(AN / "v3_shift_by_repl.json"); EX = jload(AN / "exact_tests_v6.json"); CP = jload(AN / "cross_paper_v6.json")

# ------------------------------------------------------------------ S1
def s1():
    cnt = collections.Counter((p["construct"], p["writer"]) for p in PAIRS); low = collections.Counter((p["construct"], p["low_type"]) for p in PAIRS)
    med_of = collections.defaultdict(list)
    for k, e in EXPS.items(): med_of[e["construct"]].append(e["label"])
    rows = []
    for c in sorted(SPEC, key=lambda c: (PROC_ORDER.index(SPEC[c]["group"]), SPEC[c]["role"] != "mediator", PLAIN[c])):
        v = SPEC[c]; src = [EXP_LABEL.get(s["experiment"], s["experiment"]) for s in v["sources"]]
        if v["role"] == "mediator": role = "Mediator: " + "; ".join(sorted(med_of[c]))
        elif v["role"] == "rival": role = "Rival: " + "; ".join(src)
        else: role = "Comparison: " + "; ".join(src)
        tot = sum(cnt[(c, w)] for w in WRITERS)
        rows.append([PLAIN[c], role, v["group"], short_def(c), tot, " / ".join(str(cnt[(c, w)]) for w in WRITERS), f"{low[(c, 'L1')]} / {low[(c, 'L4')]}"])
    tot = len(PAIRS); byw = " / ".join(str(sum(cnt[(c, w)] for c in SPEC)) for w in WRITERS)
    rows.append(["**All 66 constructs**", "55 mediators, 8 rivals, 3 comparisons", "", "", f"{tot:,}", byw, f"{sum(v for (c, t), v in low.items() if t == 'L1')} / {sum(v for (c, t), v in low.items() if t == 'L4')}"])
    pr = collections.Counter(v["group"] for v in SPEC.values())
    head = ["Construct", "Role (experiment)", "MVCA process", "Definition (from MVCA's rating scale)", "Pairs retained", "Per writer (" + " / ".join(WSHORT[w] for w in WRITERS) + ")", "Low pole: opposite / absent"]
    notes = ("Notes. Construct: plain-word name of the construct key in spec_v6.json. Role: mediator of the named experiment(s) (55 distinct mediators for 60 experiments; "
             "five mediators serve two or three experiments); rival = construct that the original article had ruled out (named experiment); comparison = MVCA's "
             "comparison scale (named experiments). MVCA process: the process in which the construct arises according to its theory (spec_v6.json `group`; "
             + ", ".join(f"{g} {pr[g]}" for g in PROC_ORDER) + "; 'comparison' = comparison of alternatives, which holds the three comparison scales and two rivals). "
             "Definition: the scale definition from which the pairs were written (label prefixes removed). Pairs retained: pairs left after the two exclusion rules "
             f"(own-scale rating rule, near-duplicates), total and per writer ({', '.join(WSHORT[w] + ' = ' + WNAME[w] for w in WRITERS)}). Low pole: number of pairs "
             "whose low situation expresses the opposite state (scale point 1) / leaves the state absent (scale point 4). Source: materials_v6/upload_v6/spec_v6.json, "
             "pairs_v6.json (8,311 pairs), experiments_v6.json.")
    return "**Table S1. The 66 constructs, their roles and processes, and the retained pairs of situations.**\n\n" + table(head, rows) + "\n\n" + notes

# ------------------------------------------------------------------ S2
def entered(job):
    e = ["S3"]
    if job in MAIN: e = ["S3 (identification, MTMM, baselines, criterion, Study 2 structure)"]
    e.append("S4 (logit lens" + (", Jacobian lens)" if job in ("V610_qwen36", "V611_gemma4_31b") else ")"))
    if job in MAIN: e += ["S5", "S6 (judgments, responses at three depths, removal)", "S7", "S8"]
    elif job == "V720_gemma4_26b_a4b": e += ["S5", "S6 (judgments, responses at three depths, removal by size)", "S7", "S8"]
    elif job in ("V730_qwen35_27b", "V731_qwen35_9b", "V732_qwen35_4b", "V733_qwen35_2b", "V734_qwen35_08b", "V735_gemma4_e4b", "V736_olmo7"):
        e += ["S5", "S6 (judgments, short design; responses in 25 experiments)", "S8A-B"]
    elif job == "V740_qwen38": e += ["S5", "S6 (responses in 25 experiments)"]
    elif job == "V741_gptoss20": e += ["S5", "S6 (responses in 25 experiments)", "S8A-B"]
    elif job == "V742_granite42": e += ["S5", "S8A-B"]
    return "; ".join(e)
def s2():
    rows = []
    for j, (nm, dev, par, it, hf) in LLM.items():
        v = V1[j]; jl = jlens_mid(j)
        ro = pct(RO.loc[j, "hit"]) + (f"; J-lens {pct(jl['hit'])}" if jl else "")
        rows.append([nm, f"`{hf}`", dev, par, v["n_layers"], it, f"{pct(v['best'])} ({v['best_layer']})", f"{pct(v['accL'])} ({v['Lstar']})", ro, entered(j)])
    head = ["LLM", "Hugging Face id", "Developer", "Parameters", "Layers", "Instruction-tuned", "Identification, best layer (layer)", "Identification, identification layer (layer)", "Meaning identified (read-out)", "Entered"]
    A = table(head, rows)
    # panel B: OLMo 3 7B training
    tr = pd.read_csv(AN / "olmo7_training_v1_read.csv"); t4 = pd.read_csv(AN / "olmo7_training_v4a.csv").set_index("ckpt")
    rmap = {"stage1-step0": "V7500_olmo7_stage1_step0", "stage1-step1000": "V7501_olmo7_stage1_step1000", "stage1-step3000": "V7502_olmo7_stage1_step3000",
            "stage1-step10000": "V7503_olmo7_stage1_step10000", "stage1-step30000": "V7504_olmo7_stage1_step30000", "stage1-step300000": "V7506_olmo7_stage1_step300000",
            "stage1-step1413814": "V7509_olmo7_stage1_step1413814", "main": "V7512_olmo7_main"}
    cmap = {"stage1-step10000": ("10k", "V820_src4a_olmo7_stage1_step10000"), "stage1-step300000": ("300k", "V821_src4a_olmo7_stage1_step300000"),
            "stage1-step1413814": ("1.41M", "V822_src4a_olmo7_stage1_step1413814"), "main": ("main", "V823_src4a_olmo7_main")}
    def natown(job):
        s = jload(RV / job / "V4a_causal/summary.json"); return float(np.mean(list(s["natural_own"].values())))
    rows = []
    lab = {"main": "end of training (main = stage 3 final)"}
    for _, r in tr.iterrows():
        ck = r.checkpoint; stage = {1: "1 (text prediction)", 2: "2 (mid-training)", 3: "3"}[int(r.stage)]
        ro = pct(RO.loc[rmap[ck], "hit"]) if ck in rmap else "not judged"
        if ck in cmap:
            k, job = cmap[ck]; c = t4.loc[k]
            ind = f"{int(c.own_minus_others_pos)} ({int(c.t2)})"; rr = cor(c.r_nat_push); ns = f"{num(natown(job))} / {num(c.natural_shift_mean)}"
        else: ind = rr = ns = "not run"
        rows.append([ck, stage, f"{int(r.step):,}", int(r.Lstar), pct(r.acc_layer0), pct(r.acc_layer16), pct(r.acc_best_mid), ro, ind, rr, ns])
    for k, job, nm in (("SFT", "V824_src4a_olmo7_sft", "after supervised tuning (Instruct-SFT)"), ("DPO", "V825_src4a_olmo7_dpo", "after preference tuning (Instruct-DPO)")):
        c = t4.loc[k]
        idn = {"SFT": "87% (16; 2,853 pairs, RD)", "DPO": "88% (DEVLOG 01:58 UTC: .875; digest not local)"}[k]
        a0 = pct(RDV1["V760"]["acc0"]) if k == "SFT" else "n.a."
        rows.append([nm, "post-training", "", 10, a0, "n.a.", idn, "not judged", f"{int(c.own_minus_others_pos)} ({int(c.t2)})", cor(c.r_nat_push), f"{num(natown(job))} / {num(c.natural_shift_mean)}"])
    rows.append(["Instruct (= OLMo 3 7B above)", "post-training", "", V1["V736_olmo7"]["Lstar"], pct(V1["V736_olmo7"]["acc0"]), "n.a.", f"{pct(V1['V736_olmo7']['best'])} (all 8,311 pairs)",
                 pct(RO.loc["V736_olmo7", "hit"]) + " (all pairs)", f"{RDV4A[('V736', 'V4a')]['pos']} ({RDV4A[('V736', 'V4a')]['t2']})", cor(RDV4A[("V736", "V4a")]["r"]), "n.a."])
    head = ["Checkpoint", "Stage", "Updates", "Identification layer", "Identification, layer 0", "Identification, layer 16", "Identification, best middle layer",
            "Meaning identified (read-out)", "Induced judgments: own > others, of 66 (t > 2)", "r (induced, natural)", "Judgment difference high - low: own item / all 11 items"]
    B = table(head, rows)
    c3 = jload(AN / "v3_ckpt.json"); cc = jload(AN / "v3_ckpt_corr.json"); cs = jload(AN / "v3_ckpt_shift.json"); rows = []
    for k in c3:
        v = c3[k]; e = cs[k]
        rows.append([k, v["own_shift_pos"], f"{v['response_pos']} / {v['response_sig']}", f"{v['repl_sig']} vs {v['fail_sig']} ({P(v['fisher_p_one_sided'])})",
                     f"{e['shift_pos_repl']} vs {e['shift_pos_fail']}", f"{cor(cc[k]['rho_shift_response'])} ({P(cc[k]['p'])})"])
    C = table(["Checkpoint", "Larger effect on mediator than on the other 65 (of 60)", "Responses: original effect positive / significant (of 60)",
               "Responses: significant, replicated vs failed (of 14; Fisher P)", "Representation: positive, replicated vs failed (of 14)", "Spearman rho, representation vs response"], rows)
    AG = jload(UP / "agents_v6.json")["agents"]
    D = table(["Profile", "Age", "Gender", "Occupation", "City"], [[a["id"], a.get("age", ""), a.get("gender", ""), a.get("occupation", ""), a.get("city", "")] for a in AG])
    notes = ("Notes. Panel A. Identification: share of the held-out writer's pairs assigned to the correct construct among 66 (chance 1.5%; standardized internal "
             "state averaged over the situation's tokens), at the layer where it is highest and at the identification layer (first layer at 10% of the depth or "
             "deeper with at least 95% of the best grid layer's rate). Meaning identified: share of read-out lists for which the raters chose the correct definition "
             "among ten (chance 10%), logit lens, four middle read-out layers; Jacobian lens (Qwen3.6 27B and Gemma 4 31B only), three middle of its five layers. "
             "Parameters: nominal size in the model name (A4B, E4B as named by the developer). Instruction-tuned: as classified in the manuscript (only the two base "
             "versions were examined before instruction tuning). Entered: the Supplementary Tables in which the LLM's results appear. Panel B. OLMo 3 7B (Olmo-3-1025-7B) at "
             "its public checkpoints, read on one third of the pairs (2,853 pairs, six writers). Induced judgments and r: short design of Study 4 (rank among 10 "
             "directions, doses +-2 and +-4, natural matrix from three writers, 198 pairs). Judgment difference: mean high - low difference of the LLM's 1-9 judgment, on "
             "the construct's own item (diagonal of the natural matrix; V82x summary.json `natural_own`) / over all 11 judged items (olmo7_training_v4a.csv "
             "`natural_shift_mean`, recomputed here from causal_v4a.csv as the mean over all 528 cells). 'main' is the final stage-3 checkpoint and gives the same "
             "values as stage3-step11921. Panel C. Study 3 at the OLMo 3 7B checkpoints (directions from the checkpoint's own V1 reading on 2,853 pairs; "
             "measures as in table S5; 'base' = end of training before instruction tuning; Instruct = the OLMo 3 7B of table S5). Panel D. The ten agent profiles of "
             "MVCA's test sample (p_151 to p_160; persona texts verbatim in materials_v6/upload_v6/agents_v6.json) used as system prompts in Studies 1 (criterion "
             "validity), 3, 4 and 5. 54 experiments used all ten profiles; five experiments restricted to US residents used nine (carter_flag_priming, "
             "caruso_currency_priming, greenberg_mortality_salience_worldview, hyman_sheatsley_norm_of_reciprocity, rugg_forbid_allow), and White & Argo's identity "
             "threat used eight (MVCA excludes the non-binary profile). Sources: results_v6/V61x/V1_read/summary.json; analysis_v6/run23_digests_2026-10-21.md (RD) for all other LLMs; "
             "readout_judge_v6_all_models.csv, readout_judge_v6.csv; olmo7_training_v1_read.csv, olmo7_training_v4a.csv; results_v6/V82x/V4a_causal; DEVLOG.md 01:58 UTC (DPO); "
             "v3_ckpt.json, v3_ckpt_shift.json, v3_ckpt_corr.json; agents_v6.json; PRERUN_ADJUSTMENTS_v6.md.")
    return ("**Table S2. The LLMs examined, their identification of the constructs, and the analyses in which they entered.**\n\n*Panel A. LLMs.*\n\n" + A +
            "\n\n*Panel B. OLMo 3 7B over training.*\n\n" + B + "\n\n*Panel C. Study 3 over the training of OLMo 3 7B.*\n\n" + C +
            "\n\n*Panel D. Agent profiles.*\n\n" + D + "\n\n" + notes)

# ------------------------------------------------------------------ S3
def s3():
    rows = []
    for j in LLM:
        v = V1[j]
        rows.append([NAME[j], f"{v['pairs']:,}", f"{pct(v['best'])} ({v['best_layer']})", f"{pct(v['accL'])} ({v['Lstar']})", pct(v["B1"]),
                     pct(v["top5"]) if v.get("top5") else "n.a.", pct(v["acc0"]), f"{cor(v['conv'])} / {cor(v['same'])} / {cor(v['base'])}"])
    sft = RDV1["V760"]
    rows.append(["OLMo 3 7B after supervised tuning", f"{sft['pairs']:,}", f"{pct(sft['best'])} ({sft['best_layer']})", f"{pct(sft['accL'])} ({sft['Lstar']})", pct(sft["B1"]), "n.a.",
                 pct(sft["acc0"]), f"{cor(sft['conv'])} / {cor(sft['same'])} / {cor(sft['base'])}"])
    A = table(["LLM", "Pairs", "Best layer (layer)", "Identification layer (layer)", "Layers at 30-50% of depth", "Top 5 (30-50%)", "Input layer (0)",
               "MTMM (30-50%): same construct, different writer / different construct, same writer / different both"], rows)
    # panel B baselines
    bw = jload(AN / "bow_floor_v6.json"); ng = jload(AN / "ngram_floor_v6.json")["floors"]; en = jload(AN / "encoder_floor_v6.json")["identification"]
    tr = pd.read_csv(AN / "olmo7_training_v1_read.csv").set_index("checkpoint").loc["stage1-step0"]
    rows = [["Single words: counts, standardized (8,542 words)", pct(bw["counts_std"]["first"]), pct(bw["counts_std"]["top5"])],
            ["Single words: presence, standardized", pct(bw["binary_std"]["first"]), pct(bw["binary_std"]["top5"])],
            ["Single words: weighted by rarity, raw", pct(bw["tfidf_raw"]["first"]), pct(bw["tfidf_raw"]["top5"])],
            ["Sequences of up to 2 words, weighted by rarity (101,985)", pct(ng["ng2_tfidf_raw"]["first"]), pct(ng["ng2_tfidf_raw"]["top5"])],
            ["Sequences of up to 3 words, weighted by rarity (282,350; strongest)", pct(ng["ng3_tfidf_raw"]["first"]), pct(ng["ng3_tfidf_raw"]["top5"])],
            ["Latent semantic analysis, 100 / 300 / 1,000 dimensions", " / ".join(pct(ng[k]["first"]) for k in ("lsa100", "lsa300", "lsa1000")), " / ".join(pct(ng[k]["top5"]) for k in ("lsa100", "lsa300", "lsa1000"))],
            ["Sentence encoder (Qwen3-Embedding-8B), standardized / raw", f"{pct(en['encoder_std']['first'])} / {pct(en['encoder_raw']['first'])}", f"{pct(en['encoder_std']['top5'])} / {pct(en['encoder_raw']['top5'])}"],
            ["OLMo 3 7B before training (step 0; 2,853 pairs): layer 0 / layer 16 / best middle layer", f"{pct(tr.acc_layer0)} / {pct(tr.acc_layer16)} / {pct(tr.acc_best_mid)}", f"n.a. / {pct(tr.top5_layer16)} / n.a."],
            ["Trained LLMs, best layer (16 LLMs)", f"{pct(min(V1[j]['best'] for j in LLM))} to {pct(max(V1[j]['best'] for j in LLM))}", ""]]
    B = table(["Baseline", "Correct construct (first of 66)", "Top 5"], rows)
    # panel C cross-tables
    bwd = jload(AN / "beyond_words_v6.json"); ngj = jload(AN / "ngram_floor_v6.json"); ec = jload(AN / "encoder_floor_v6.json")["cross_table"]
    n = 8311; rows = []
    for j in MAIN:
        b = bwd[j]; e = ec[NAME[j]]; g = ngj[j]
        rows.append([NAME[j], pct(b["model_first"]), f"{pct(b['model_first_when_words_fail'])}", f"{pct(g['model_first_when_strongest_floor_fails'])}",
                     f"{pct(e['llm_first_when_encoder_fails'])}", f"{pct(b['words_first_when_model_fails'])} / {pct(e['encoder_first_when_llm_fails'])}",
                     f"{e['llm_right_encoder_wrong']:,} / {e['encoder_right_llm_wrong']:,} (P = {e['mcnemar_p']:.0e})".replace("e-", "e" + MINUS)])
    nw = round(n * (1 - bwd["V610_qwen36"]["words_first"])); nf = ngj["V610_qwen36"]["n_floor_fails"]; ne = ec["Qwen3.6 27B"]["n_encoder_fails"]
    C = table(["LLM", "LLM, all pairs", f"Where single words fail (n = {nw:,}, {pct(nw / n)})", f"Where word sequences up to 3 fail (n = {nf:,}, {pct(nf / n)})",
               f"Where the sentence encoder fails (n = {ne:,}, {pct(ne / n)})", "Single words / encoder right where the LLM fails", "Pairs LLM right, encoder wrong / encoder right, LLM wrong (McNemar)"], rows)
    # panel D criterion
    cr = jload(AN / "criterion_v6.json"); rows = []
    for j in MAIN:
        c = cr[NAME[j]]
        rows.append([NAME[j], c["n_pairs"], int(c["items_per_pair"]), cor(c["criterion_r"]), cor(c["null_95"]), "< 0.001 (2,000 permutations)", pct(c["closest_centroid_is_largest_shift"]),
                     pct(c["chance"]), f"{num(c['own_item_shift_mean'])} / {num(c['other_items_shift_mean'])}"])
    D = table(["LLM", "Pairs", "Items judged per pair", "r (similarity to representation, judgment difference)", "Permutation null, 95th percentile", "P",
               "Most similar construct = largest judgment difference", "Chance", "Judgment difference (1-9): own item / other items"], rows)
    # panel E Study 2
    st = jload(AN / "structure_v6.json"); en2 = jload(AN / "encoder_floor_v6.json"); rows = []
    for a, b in (("Qwen3.6 27B", "Gemma 4 31B"), ("Qwen3.6 27B", "OLMo 3.1 32B"), ("Gemma 4 31B", "OLMo 3.1 32B")):
        k = f"{a} ~ {b}"; cg = st["construct_geometry"][k]; ce = en2["construct_geometry"][k]; sg = st["stimulus_geometry"][k]; se = en2["stimulus_geometry"][k]
        rows.append([f"{a} vs {b}", cor(cg["rho"]), cor(cg["rho_beyond_words"]), cor(ce["beyond_encoder"]), cor(ce["beyond_encoder_and_words"]), cor(sg["rho"]), cor(sg["rho_beyond_words"]), cor(se["beyond_encoder"])])
    for m in ("Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B"):
        rows.append([f"Words vs {m}", cor(st["construct_geometry"][f"word counts ~ {m}"]["rho"]), "", "", "", cor(st["stimulus_geometry"][f"word counts ~ {m}"]["rho"]), "", ""])
        rows.append([f"Sentence encoder vs {m}", cor(en2["construct_geometry"][f"encoder ~ {m}"]), "", "", "", cor(en2["stimulus_geometry"][f"encoder ~ {m}"]), "", ""])
    rows.append(["Sentence encoder vs words", cor(en2["construct_geometry"]["encoder ~ word counts"]), "", "", "", "", "", ""])
    E1 = table(["Comparison", "Constructs (2,145 similarities)", "beyond words", "beyond encoder", "beyond encoder and words", "Pairs of situations (2,000)", "beyond words", "beyond encoder"], rows)
    ts = st["theory_structure"]; te = en2["theory_structure"]; rows = []
    for m in ("Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B"):
        rows.append([m, num(ts[m]["within"], 3), num(ts[m]["between"], 3), num(ts[m]["within_minus_between"], 3), pval(max(ts[m]["perm_p"], 1 / 5000)) if ts[m]["perm_p"] > 0 else "< 0.0002",
                     f"{num(ts[m + ' (beyond words)']['within_minus_between'], 3)} (P {'< 0.0002' if ts[m + ' (beyond words)']['perm_p'] == 0 else '= ' + pval(ts[m + ' (beyond words)']['perm_p'])})",
                     f"{num(te[m + ' (beyond encoder)']['within_minus_between'], 3)} ({P(te[m + ' (beyond encoder)']['perm_p'])})",
                     f"{num(te[m + ' (beyond encoder and words)']['within_minus_between'], 3)} ({P(te[m + ' (beyond encoder and words)']['perm_p'])})"])
    rows.append(["Words", num(ts["word counts"]["within"], 3), num(ts["word counts"]["between"], 3), num(ts["word counts"]["within_minus_between"], 3), pval(ts["word counts"]["perm_p"]), "", "", ""])
    rows.append(["Sentence encoder", num(te["encoder"]["within"], 3), num(te["encoder"]["between"], 3), num(te["encoder"]["within_minus_between"], 3), pval(te["encoder"]["perm_p"]), "", "", ""])
    E2 = table(["Representation", "Mean similarity within processes", "between processes", "Difference", "Permutation P", "Difference beyond words", "beyond encoder", "beyond encoder and words"], rows)
    DI = jload(AN / "discriminant_v6.json"); rows = []
    for nm in ("Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B", "sentence encoder", "single words"):
        x = DI[nm]; rows.append([nm[0].upper() + nm[1:], pct(x["within_process"]), pct(x["chance_within"]), f"{pct(x['mediator_vs_rival_mean'])} (lowest {pct(x['mediator_vs_rival_min'])})", x["n_rival_pairs"]])
    F = table(["Representation", "Identified among the constructs of the pair's own process", "Chance", "Mediator vs its rival: correct (mean over pairs; lowest pair)", "Mediator-rival pairs"], rows)
    notes = ("Notes. Panel A. Identification as in table S2 (chance 1.5%); 'layers at 30-50% of depth' = the internal state averaged over these layers, the "
             "layer band used for the MTMM correlations, the baselines' comparisons, criterion validity, Study 2 and Study 3. MTMM: mean correlation between "
             "the writer-by-construct mean pair differences of the same construct from different writers / of different constructs from the same writer / of "
             "different constructs from different writers. Panel B. The baselines use the same writer-held-out procedure on the situations' words (8,542 "
             "single words in bow_floor_v6.json; the n-gram run counts 8,534), on the sentence encoder's embeddings, and on the untrained OLMo 3 7B. Panel C. "
             "Identification by the LLM (30-50% of depth) on the pairs that each baseline misidentified; 'single words' = standardized counts (67%); McNemar "
             "test on the pairs on which the LLM and the encoder disagree. Panel D. Criterion validity: for each pair and each of the 11 judged items, the "
             "pair's similarity to the construct's representation (writer held out, 30-50% of depth) and the difference between the LLM's 1-9 judgments in the high "
             "and the low situation, after removing pair and item means; one pair per construct and writer (395 pairs), each judged under one agent profile. "
             "Panel E (Study 2). Spearman correlations of the 66 x 66 construct similarities (representations estimated from complementary halves of the "
             "writers, ten splits) and of the similarities among 2,000 random pairs of situations; 'beyond' = residuals after regressing on the words' or the "
             "encoder's similarities. Process structure: mean construct similarity within vs between the seven MVCA processes, 5,000 label permutations. "
             "Sources: V61x/V1_read/summary.json; RD (other LLMs; OLMo 3 7B after supervised tuning read on 2,853 pairs); bow_floor_v6.json, ngram_floor_v6.json, "
             "encoder_floor_v6.json, beyond_words_v6.json, olmo7_training_v1_read.csv, criterion_v6.json, structure_v6.json, discriminant_v6.json. Panel F. Writer-held-out "
             "identification restricted to the constructs of the pair's own MVCA process (chance = 1 / number of constructs in the process, averaged over pairs) and to the "
             "8 pairs of a mediator and the rival construct that the original article ruled out (chance 50%); 30-50% of depth; sentence encoder standardized, single "
             "words standardized counts.")
    return ("**Table S3. Study 1 (and Study 2): identification, multitrait-multimethod correlations, baselines, criterion validity and structure across LLMs.**\n\n"
            "*Panel A. Identification of the held-out writer's construct and MTMM correlations.*\n\n" + A + "\n\n*Panel B. Baselines (all 8,311 pairs unless stated).*\n\n" + B +
            "\n\n*Panel C. Cross-tables for the three main LLMs.*\n\n" + C + "\n\n*Panel D. Criterion validity (the LLM's own judgment of the construct).*\n\n" + D +
            "\n\n*Panel E. Study 2: structure across the three main LLMs.*\n\n" + E1 + "\n\n" + E2 +
            "\n\n*Panel F. Discrimination among the closest alternatives (three main LLMs and baselines).*\n\n" + F + "\n\n" + notes)

# ------------------------------------------------------------------ S4
def s4():
    allsrc = {(d["model"], d["source"]): d for d in RJS["by_model_source"]}
    tm = {(d["model"], d["source"]): d["text_hit"] for d in ECHO["text_match_hit_by_model_source"]}
    tf = {(d["model"], d["source"]): d for d in ECHO["rater_hit_when_text_match_fails"]}
    rows = []
    for j in LLM:
        po, k, n = agreement(j)
        jl = jlens_mid(j)
        ll_all = allsrc.get((j, "logitlens")); jl_all = allsrc.get((j, "jlens"))
        wm = f"{pct(tm[(j, 'logitlens')])}" if (j, "logitlens") in tm else "not run"
        wf = f"{pct(tf[(j, 'logitlens')]['hit'])} (n = {tf[(j, 'logitlens')]['n']})" if (j, "logitlens") in tf else "not run"
        rows.append([NAME[j], f"{pct(RO.loc[j, 'hit'])} / {pct(RO.loc[j, 'hit3'])}", f"{pct(jl['hit'])} / {pct(jl['hit3'])}" if jl else "not run",
                     f"{pct(ll_all['hit'])}" + (f"; J-lens {pct(jl_all['hit'])}" if jl_all else "") if ll_all else "not judged",
                     wm + (f"; J-lens {pct(tm[(j, 'jlens')])}" if (j, "jlens") in tm else ""),
                     wf + (f"; J-lens {pct(tf[(j, 'jlens')]['hit'])} (n = {tf[(j, 'jlens')]['n']})" if (j, "jlens") in tf else ""),
                     f"{pct(po)} / {cor(k)}", BOTH.get(j, "")])
    A = table(["LLM", "Logit lens: correct / in top 3", "Jacobian lens: correct / in top 3", "All read-out layers (main LLMs)", "Word-matching procedure: correct",
               "Raters correct where word matching failed", "Rater agreement on correct (% / Cohen's kappa)", "Constructs identified by both raters at some layer (of 66)"], rows)
    BC = pd.read_csv(AN / "readout_judge_v6_by_construct.csv").set_index("construct")
    cols = ["V610_qwen36|logitlens", "V610_qwen36|jlens", "V611_gemma4_31b|jlens", "V612_olmo32|logitlens"]
    def prow(c): return [PLAIN[c], SPEC[c]["group"]] + [pct(BC.loc[c, x]) for x in cols] + [pct(BC.loc[c, "mean"])]
    top = BC.sort_values("mean", ascending=False).head(10).index; bot = BC.sort_values("mean").head(10).index
    hb = ["Construct", "Process", "Qwen3.6 27B logit lens", "Qwen3.6 27B J-lens", "Gemma 4 31B J-lens", "OLMo 3.1 32B logit lens", "Mean"]
    B1 = table(hb, [prow(c) for c in top]); B2 = table(hb, [prow(c) for c in bot])
    # summary counts and relation to induction (recomputed)
    cnt = {x: (int((BC[x] >= .3).sum()), int((BC[x] >= .5).sum())) for x in cols}
    rel = []
    for x, job in zip(cols, ["V610_qwen36", "V610_qwen36", "V611_gemma4_31b", "V612_olmo32"]):
        s = jload(RV / job / "V4a_causal/summary.json")["column"]
        own = pd.Series({k: v["own"]["mean"] for k, v in s.items()}); om = pd.Series({k: v["own_minus_others"]["mean"] for k, v in s.items()}); omd = pd.Series({k: v["own_minus_others"]["dz"] for k, v in s.items()})
        rel.append([x.replace("|", ", ").replace("V610_qwen36", "Qwen3.6 27B").replace("V611_gemma4_31b", "Gemma 4 31B").replace("V612_olmo32", "OLMo 3.1 32B").replace("logitlens", "logit lens").replace("jlens", "J-lens"),
                    f"{cnt[x][0]} / {cnt[x][1]}", cor(np.corrcoef(BC[x].reindex(own.index), own)[0, 1]), cor(np.corrcoef(BC[x].reindex(om.index), om)[0, 1]), cor(np.corrcoef(BC[x].reindex(omd.index), omd)[0, 1])])
    C = table(["Read-out", "Constructs identified in >= 30% / >= 50% of their lists", "r with induced effect on own judgment", "r with own minus others (mean)", "r with own minus others (d_z)"], rel)
    notes = ("Notes. Panel A. Correct: share of read-out lists (30 word fragments most promoted by the construct's high-minus-low difference at the last token) for "
             "which the raters chose the construct's own definition among ten (chance 10%; top 3 chance 30%), mean over the two raters (DeepSeek V4.1 Flash without "
             "reasoning, GLM 5.3). Logit lens: four middle of the six read-out layers at 50-100% of depth; Jacobian lens: three middle of its five layers (Qwen3.6 27B, "
             "Gemma 4 31B; no lens was available for the other LLMs). 'All read-out layers': the first judging round over all six (logit lens) or five (J-lens) "
             "layers. Word-matching procedure: chooses the definition whose high situations share the most words with the list (all read-out layers; run for the main "
             "LLMs only); 'raters correct where word matching failed' = raters' hit on the lists on which that procedure chose a wrong definition. Agreement: share of "
             "lists on which both raters agree whether the correct definition was chosen, and Cohen's kappa (overall: first round 85% / .68 on 1,847 lists; second "
             "round 84% / .68 on 5,533 lists); computed here from readout_judge_v6.csv and readout_judge_v6_R2.csv. Panels B and C. Constructs ranked by their mean "
             "hit over the four first-round read-outs of the main LLMs (middle layers; readout_judge_v6_by_construct.csv). Panel C relates this hit to the induced "
             "effect on the construct's own judgment in Study 4 (V61x/V4a_causal/summary.json, 66 constructs), recomputed here. Sources: readout_judge_v6.json, "
             "readout_judge_v6_R2.json, readout_judge_v6_all_models.csv, readout_echo_v6.json, readout_judge_v6_by_construct.csv.")
    return ("**Table S4. Content validity: meaning of the representations identified from their read-out words.**\n\n*Panel A. Read-out per LLM.*\n\n" + A +
            "\n\n*Panel B. Constructs identified most often (ten highest means).*\n\n" + B1 + "\n\n*Constructs identified least often (ten lowest means).*\n\n" + B2 +
            "\n\n*Panel C. Number of constructs identified, and relation to the induced effect on judgments.*\n\n" + C + "\n\n" + notes)

# ------------------------------------------------------------------ S5
def s5():
    SHC = pd.read_csv(AN / "v3_shift_by_repl.csv"); ANS = pd.read_csv(AN / "v3_all_models.csv")
    f5a = pd.read_csv(REL / "figures/fig5A_study1_rating.csv").set_index("experiment_id").d_z
    def sp(x, y):
        x, y = x.align(y, join="inner"); r, pv = stats.spearmanr(x, y); return float(r), float(pv)
    order = list(V3NAME) + ["V742_granite42", "V720_gemma4_26b_a4b"]
    v3n = dict(V3NAME); v3n["V742_granite42"] = "Granite 4.2 30B"; v3n["V720_gemma4_26b_a4b"] = "Gemma 4 26B-A4B"
    rows, rhos, shifts = [], [], {}
    SBR = jload(AN / "v3_shift_by_repl.json")                 # median d_z of the representation effect by replication outcome; one-sided Mann-Whitney
    for j in order:
        nm = v3n[j]; v = V3[nm]; k, n = kn(v["own_shift_pos"]); e = EX["by_replication"][nm]
        sh = SHC[SHC.model == nm].set_index("exp").shift_dz; shifts[nm] = sh
        r1, p1 = sp(sh, f5a); r2, p2 = sp(sh, ANS[ANS.model == nm].set_index("exp").dz); rhos.append(r1)
        rk = CP["rank"].get(nm); rkt = f"{rk['first_centred']} / {rk['top5_centred']} ({num(rk['median_centred'], 1).rstrip('0').rstrip('.')})" if rk else "not run"
        rows.append([NAME[j], f"{k}/{n} ({P(binom_p(k, n, .5))})", rkt, f"{cor(r1)} ({P(p1)})", f"{cor(r2)} ({P(p2)})",
                     f"{v['response_pos']} / {v['response_sig']}", f"{e['shift_pos']} ({P(e['p_shift'])})", f"{SBR[nm]['shift_median_repl']:.2f} vs {SBR[nm]['shift_median_fail']:.2f} ({P(SBR[nm]['mw_p'])})", f"{e['answer_sig']} ({P(e['p_answer'])})"])
    m3 = pd.concat([shifts[v3n[j]] for j in MAIN], axis=1).mean(axis=1); ma = pd.concat(list(shifts.values()), axis=1).mean(axis=1)
    r3, p3 = sp(m3, f5a); ra, pa = sp(ma, f5a)
    rows.append(["Shift averaged over the 3 main LLMs", "", "", f"{cor(r3)} ({P(p3)})", "", "", "", "", ""])
    rows.append(["Shift averaged over the 14 LLMs", "", "", f"{cor(ra)} ({P(pa)})", "", "", "", "", ""])
    rows.append(["Mean of the 14 correlations", "", "", cor(np.mean(rhos)), "", "", "", "", ""])
    head = ["LLM", "Larger effect on mediator than on the other 65 (of 60)", "Mediator first / in top five of 66 (median rank)", "Spearman rho with MVCA Study 1",
            "Spearman rho, effect on representation vs on response", "Responses: original effect positive (of 60) / significant", "Representation: positive effect, replicated vs failed (of 14; Fisher P)",
            "Representation: median d_z, replicated vs failed (Mann-Whitney P)", "Responses: original effect significant, replicated vs failed (of 14; Fisher P)"]
    A = table(head, rows)
    RL = jload(AN / "reliability_s3_v6.json"); rows = []
    for nm in ("Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B"):
        x = RL[nm]; rows.append([nm, x["n_exp"], x["n_wordings"], cor(x["reliability_shift"]), cor(x["reliability_answer"]), cor(x["r_shift_answer"]), cor(x["rho_shift_answer"]), cor(x["r_corrected"])])
    B = table(["LLM", "Experiments", "Wording labels", "Split-half reliability: effect on representation", "Split-half reliability: effect on response",
               "Pearson r between the effects", "Spearman rho between the effects", "r corrected for attenuation"], rows)
    notes = ("Notes. Panel A. Effect on the mediator's representation: per experiment, the treatment-minus-control difference in the scenario's internal state (mean over its "
             "word fragments and the layers at 30-50% of depth) projected on the mediator's representation minus its mean projection on the other 65, signed by the "
             "theory's predicted direction (X -> M); positive count with exact one-sided binomial P against .5. Its d_z is computed over the seven wording labels "
             "(v00 = MVCA's scenario, v01-v06 = rewordings; each rewording label pools the rewordings of the two writers assigned to the experiment, 6 to 12 rewordings "
             "in all), each averaged over the agent profiles. Rank: rank of the mediator among the 66 constructs after removing the component common to all experiments "
             "(three main LLMs only; chance: first 0.9, top five 4.5 of 60; MVCA's process statements: own mediator first in 20 to 27 of 60 depending on its LLM, 24 "
             "averaged; release data construct_activation/per_experiment_<llm>.csv). Spearman rho with MVCA Study 1: across the 60 experiments, with the effect of the "
             "manipulation on the rated mediator in MVCA's cognitive agents (release data figures/fig5A_study1_rating.csv); recomputed here from v3_shift_by_repl.csv "
             "(identical to cross_paper_v6.json for its 12 LLMs; that file has not been regenerated for Granite 4.2 30B and Gemma 4 26B). The 'averaged' rows "
             "correlate the effect on the representation averaged over LLMs with MVCA; the last row averages the 14 correlations. Effect on the response: treatment minus control on the 1-9 "
             "response in the predicted direction, items with digit mass >= .5, d_z over wording-by-profile combinations; positive / t > 2. Replication: the 28 "
             "experiments re-examined by large preregistered replications (14 replicated, 14 failed); one-sided Fisher's exact tests on the counts; median d_z of the effect on the mediator's representation in the two sets, one-sided Mann-Whitney test (replicated > failed). Panel B. Split-half reliability "
             "over the 35 splits of the seven wording labels into three and four, Spearman-Brown corrected; corrected r = r / sqrt(product of the reliabilities). "
             "Sources: v3_all_models.json/.csv, v3_shift_by_repl.json/.csv and exact_tests_v6.json (14-LLM versions of 21 Oct), cross_paper_v6.json (ranks), "
             "study3_rank_v6.csv, reliability_s3_v6.json. Panel C. The effect on the mediator's representation computed as in Panel A from the internal states "
             "at the manipulation span (union of its passages), the scenario (mean over its word fragments), the question (mean over its word fragments; identical in "
             "both conditions) and the last word fragment before the response, averaged over the layers at 30-50% or 60-75% of depth, with the representations estimated "
             "at the same layers; rho = Spearman correlation with the effect on the response across the 60 experiments. Null: 2,000 random assignments of another "
             "experiment's mediator to each experiment (no experiment keeps its own construct); mean and 95th percentile of rho, one-sided permutation P. "
             "Positive: experiments with a positive effect on the mediator's representation (of 60); first: experiments in which the mediator is the most affected of "
             "the 66. Sources: depth_s3_v6.py / depth_s3_spec_v6.py and their .json files (V61x V3_exp/proj_v3.pt); the scenario at 30-50% reproduces Panel A exactly.")
    SP = jload(AN / "depth_s3_spec_v6.json"); rows = []
    PLAB = {"x": "manipulation", "body": "scenario", "q": "question", "dec": "last word fragment before the response"}
    for nm in ("Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B"):
        for pk, pl in PLAB.items():
            for g, gl in (("B1", "30-50%"), ("B2", "60-75%")):
                r = SP[nm][f"{pk}_{g}"]
                rows.append([nm, pl, gl, cor(r["rho_own"]), f"{cor(r['null_mean'])} / {cor(r['null_95'])}", P(r["p_perm"]), f"{r['own_pos']} / {r['own_first']}"])
    C = table(["LLM", "Position", "Depth", "rho, effect on representation vs on response", "Other experiments' mediators: mean / 95th percentile of rho",
               "Permutation P", "Positive / first (of 60)"], rows)
    EX3 = pd.read_csv(AN / "example_s3_v6.csv"); rows = []
    for eid, lab in (("reed_identity_salience", "Identity salience (interdependent self)"), ("gray_wegner_moral_typecasting", "Moral typecasting (moral agency)")):
        for nm in ("Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B"):
            r = EX3[(EX3.exp == eid) & (EX3.model == nm)].iloc[0]
            rows.append([lab, nm, f"{r.x_B1:.2f}", f"{r.body_B1:.2f}", f"{r.q_B1:.2f}", f"{r.dec_B1:.2f}", f"{r.answer_dz:.2f}"])
    Dt = table(["Experiment (mediator)", "LLM", "Activation, manipulation passages", "Activation, scenario", "Activation, question", "Activation, last fragment before the response", "Effect on the response"], rows)
    notes += (" Panel D. The two example experiments of the text: d_z of the manipulation's activation of the mediator's representation (30-50% of depth; computed as in Panel A) at "
              "each position, and d_z of the manipulation's effect on the response (Panel A). Source: example_s3_v6.py / example_s3_v6.csv.")
    return ("**Table S5. Study 3: activation of the mediator's representation by the published manipulations and effects on the responses, by LLM.**\n\n*Panel A. Effects by LLM.*\n\n" + A +
            "\n\n*Panel B. Reliability of the two effects across wordings (three main LLMs).*\n\n" + B +
            "\n\n*Panel C. Correlation between the two effects by position and depth (three main LLMs).*\n\n" + C +
            "\n\n*Panel D. The two example experiments (three main LLMs).*\n\n" + Dt + "\n\n" + notes)

# ------------------------------------------------------------------ S6
def removal_shares(job):
    D = pd.read_csv(RV / job / "V4c_remove" / "remove_v4c.csv", usecols=["exp", "cond", "vid", "agent", "arm", "ev", "mass", "x_to_y_sign"])
    def grp(c):
        w = ("__" + c.split("__")[1]) if "__" in c else ""
        for p in ("near_spec_k10", "rand_spec_k10", "unif"):
            if c.startswith(p): return p + w
        return c
    k = ["exp", "cond", "vid", "agent"]; W = D.pivot_table(index=k, columns="arm", values=["ev", "mass"]).reset_index(); W.columns = ["_".join(c).strip("_") for c in W.columns]
    sg = D.groupby("exp").x_to_y_sign.first(); W["eff"] = (W.ev_treatment - W.ev_control) * W.exp.map(sg)
    W["ok"] = (W.mass_treatment >= .5) & (W.mass_control >= .5); W["grp"] = W.cond.map(grp)
    excl = W.groupby("grp").ok.apply(lambda x: 1 - x.mean())
    E = W[W.ok].groupby(["exp", "cond"]).eff.mean().reset_index(); E["grp"] = E.cond.map(grp); E = E.groupby(["exp", "grp"]).eff.mean().unstack()
    big = E[E["none"].abs() > .3]; S = big.div(big["none"], axis=0)
    return len(big), S.median(), excl
def s6():
    # panel A judgments
    rows = []
    for j in ["V610_qwen36", "V611_gemma4_31b", "V612_olmo32"]:
        s = jload(RV / j / "V4a_causal/summary.json"); col = s["column"]
        own_pos = sum((v["own"]["mean"] or 0) > 0 for v in col.values())
        dose = pd.DataFrame([v["dose"] for v in col.values()]).astype(float).mean()
        dc = " / ".join(num(dose[k], 2) for k in ["-4", "-2", "-1", "1", "2", "4"])
        rows.append([NAME[j], "full (15 directions)", s["layer"], f"{s['n_own_minus_others_pos']} ({s['n_own_minus_others_t2']}; {P(binom_p(s['n_own_minus_others_pos'], 66, .5))})",
                     f"{s['n_rank1']}/66", f"{cor(s['r_natural_pushed'])} ({cor(s['r_natural_pushed_centred'])})", f"{own_pos}/66", dc])
    for key, j, des in [("V720", "V720_gemma4_26b_a4b", "full (15 directions)"), ("V730", "V730_qwen35_27b", "short (10)"), ("V731", "V731_qwen35_9b", "short (10)"), ("V732", "V732_qwen35_4b", "short (10)"),
                        ("V733", "V733_qwen35_2b", "short (10)"), ("V734", "V734_qwen35_08b", "short (10)"), ("V735", "V735_gemma4_e4b", "short (10)"), ("V736", "V736_olmo7", "short (10)")]:
        r = RDV4A[(key, "V4a")]
        rows.append([NAME[j], des, r["layer"], f"{r['pos']} ({r['t2']}; {P(binom_p(r['pos'], 66, .5))})", f"{r['first']}/66", f"{cor(r['r'])} ({cor(r['rc'])})", "n.a.", "n.a."])
    for j in ("V740_qwen38", "V741_gptoss20", "V742_granite42", "V721_olmo32_base", "V722_gemma4_31b_base"):
        rows.append([NAME[j], "not run", "", "", "", "", "", ""])
    for key, nm in (("V810", "Qwen3.6 27B, discriminant directions"), ("V811", "Gemma 4 31B, discriminant directions"), ("V812", "OLMo 3.1 32B, discriminant directions"), ("V813", "Gemma 4 26B, discriminant directions")):
        if key == "V810":
            t = open(RV / "V810_steer2_qwen36/V6a_dlda/digest.md", encoding="utf-8").read(); m = re.search(r"own - others > 0: (\d+)/66 \(t > 2: (\d+)\); own direction first: (\d+)/66; r\(natural, pushed\) = ([\d.-]+) \(centred ([\d.-]+)\)", t)
            r = dict(pos=int(m.group(1)), t2=int(m.group(2)), first=int(m.group(3)), r=float(m.group(4)), rc=float(m.group(5)), layer=19)
        else: r = RDV4A[(key, "V6a")]
        rows.append([nm, "short (10)", r["layer"], f"{r['pos']} ({r['t2']})", f"{r['first']}/66", f"{cor(r['r'])} ({cor(r['rc'])})", "n.a.", "n.a."])
    A = table(["LLM", "Design (comparison directions)", "Layer", "Own > others, of 66 (t > 2; binomial P)", "Own direction first", "r (induced, natural) (centred)", "Own induction raises own judgment", "Mean effect at doses -4/-2/-1/+1/+2/+4"], rows)
    # panel B answers (52)
    def v6b_local():
        t = open(RV / "V810_steer2_qwen36/V6b_steer2/digest.md", encoding="utf-8").read().splitlines()
        fx = lambda ln: ln.replace("d50|sit", "d50#sit").replace("d70|sit2end", "d70#sit2end")
        return {r[0].replace("#", "|"): r for r in ([c.strip() for c in fx(ln).strip().split("|")[1:-1]] for ln in t if ln.startswith("| d"))}
    def v6b_rd(job):
        return {r[0]: r for r in rd_rows(job, r"d\d0") if len(r) == 7}
    V6B = {"V610_qwen36": v6b_local(), "V611_gemma4_31b": v6b_rd("V811_steer2_gemma4_31b"), "V612_olmo32": v6b_rd("V812_steer2_olmo32"), "V720_gemma4_26b_a4b": v6b_rd("V813_steer2_gemma4_26b_a4b")}
    V4B = {j: jload(RV / j / "V4b_steer/summary.json") for j in MAIN}; V4B["V720_gemma4_26b_a4b"] = {"n_own_pos": 25, "n_own_minus_others_pos": 23, "layer": 9}
    V5A = {"V610_qwen36": jload(RV / "V710_follow_qwen36/V5a_incontext/summary.json"), "V611_gemma4_31b": jload(RV / "V711_follow_gemma4_31b/V5a_incontext/summary.json"),
           "V612_olmo32": {"n_check_positive": 48, "n_own_push_positive": 45, "n_own_minus_others_positive": 39}, "V720_gemma4_26b_a4b": {"n_check_positive": 43, "n_own_push_positive": 40, "n_own_minus_others_positive": 35}}
    rows = []
    for j in ["V610_qwen36", "V611_gemma4_31b", "V612_olmo32", "V720_gemma4_26b_a4b"]:
        b = V4B[j]; d5 = V6B[j]["d50|sit"]; d7 = V6B[j]["d70|sit2end"]; a = V5A[j]
        rows.append([NAME[j], f"{b['n_own_pos']} ({P(binom_p(b['n_own_pos'], 52, .5))}) / {b['n_own_minus_others_pos']}", f"{d5[1].split('/')[0]} ({P(binom_p(int(d5[1].split('/')[0]), 52, .5))}) / {d5[2].split('/')[0]} / {d5[3].split('/')[0]}",
                     f"{d7[1].split('/')[0]} ({P(binom_p(int(d7[1].split('/')[0]), 52, .5))}) / {d7[2].split('/')[0]} / {d7[3].split('/')[0]}", f"{a['n_own_push_positive']} ({P(binom_p(a['n_own_push_positive'], 52, .5))}) / {a['n_own_minus_others_positive']}",
                     f"{a['n_check_positive']}/60"])
    B = table(["LLM", "Identification layer: response changed as predicted (P) / own > other constructs", "Half depth, scenario: predicted (P) / own > others / own > random", "70% depth, scenario to end: predicted (P) / own > others / own > random",
               "Identification layer: in-context judgment of the mediator raised (P) / own > others", "Manipulation raised the in-context judgment (of 60)"], rows)
    # panel C sweep (25)
    def sweep(job):
        if job == "V612_olmo32": R = {(r[0], r[1], r[2]): r for r in rd_rows("V712_follow_olmo32", r"(Lstar|d50|d70)")}; return R
        if job == "V720_gemma4_26b_a4b": R = {(r[0], r[1], r[2]): r for r in rd_rows("V720_gemma4_26b_a4b", r"(Lstar|d50|d70)")}; return R
        src = {"V610_qwen36": "V710_follow_qwen36", "V611_gemma4_31b": "V711_follow_gemma4_31b"}.get(job, S3JOB.get(job))
        s = jload(RV / src / "V5b_sweep/summary.json"); R = {}
        for k, c in s["cells"].items():
            d, t, dose = k.split("|"); R[(d, t, "+-" + dose)] = [d, t, "+-" + dose, f"{c['own_pos']}/{c['n']}", f"{c['own_gt_span']}/{c['n']}", str(c["own_mean"]), str(c["span_mean"])]
        return R
    rows = []
    for j in ["V610_qwen36", "V611_gemma4_31b", "V612_olmo32", "V720_gemma4_26b_a4b", "V730_qwen35_27b", "V731_qwen35_9b", "V732_qwen35_4b", "V733_qwen35_2b", "V734_qwen35_08b", "V735_gemma4_e4b", "V736_olmo7", "V740_qwen38", "V741_gptoss20"]:
        R = sweep(j); cells = []
        for key in (("Lstar", "sit", "+-2"), ("d50", "sit", "+-2"), ("d70", "sit2end", "+-2"), ("d70", "sit", "+-2")):
            r = R[key]; cells.append(f"{r[3].split('/')[0]} / {r[4].split('/')[0]} ({num(float(r[5]), 3)})")
        rows.append([NAME[j]] + cells)
    rows.append(["Granite 4.2 30B", "n.a. (V839 sweep; results not local)", "n.a.", "n.a.", "n.a."])
    C = table(["LLM", "Identification layer, scenario", "Half depth, scenario", "70% depth, scenario to end", "70% depth, scenario only"], rows)
    # panel D removal V4c
    rows = []
    for j in MAIN:
        n_e, med, excl = removal_shares(j)
        for form, lab in (("mean", "mean replacement"), ("zero", "zeroing")):
            g = lambda c: pct(med.get(f"{c}__{form}", np.nan))
            ex = max(excl.get(f"{c}__{form}", 0) for c in ("own_k10", "own_spec_k10", "near_spec_k10", "rand_spec_k10", "unif"))
            rows.append([NAME[j], lab, n_e, g("own_k10"), g("own_spec_k10"), g("near_spec_k10"), g("rand_spec_k10"), g("unif"), g("shared_k3"), pct(ex)])
    rows.append(["Gemma 4 26B", "both", "ran (V720 V4c); no summary in the local files", "", "", "", "", "", "", ""])
    D = table(["LLM", "Form", "Experiments with absolute effect > .3", "Own (10 directions)", "Own, specific", "Most similar constructs, specific", "Other (random) constructs, specific", "Uniform random subspaces", "Three shared directions", "Items excluded (digit mass < .5), largest set"], rows)
    # panel E V5c
    def v5c(job, loc):
        if job == "V612_olmo32": R = {r[0]: float(r[1]) for r in rd_rows("V712_follow_olmo32", r"(fam5|near5|none|own_spec|rand5|unif)") if len(r) == 2}
        elif job == "V720_gemma4_26b_a4b": R = {r[0]: float(r[1]) for r in rd_rows("V720_gemma4_26b_a4b", r"(fam5|near5|none|own_spec|rand5|unif)") if len(r) == 2}
        else:
            t = open(RV / {"V610_qwen36": "V710_follow_qwen36", "V611_gemma4_31b": "V711_follow_gemma4_31b"}[job] / "V5c_remove2/digest.md", encoding="utf-8").read().splitlines()
            R = {c[0]: float(c[1]) for c in ([x.strip() for x in ln.strip().split("|")[1:-1]] for ln in t) if len(c) == 2 and re.match(r"^-?\d", c[1])}
        return [pct(R[f"{c}__{loc}"]) if f"{c}__{loc}" in R else ("invalid responses" if (job, c, loc) == ("V610_qwen36", "unif_k100", "toend") else "n.a.")
                for c in ("own_spec_k10", "own_spec_k30", "own_spec_k100", "unif_k10", "unif_k30", "unif_k50", "unif_k100", "fam5_k50", "near5_k50", "rand5_k50")]
    rows = []
    nexp = {"V610_qwen36": 36, "V611_gemma4_31b": 29, "V612_olmo32": 31, "V720_gemma4_26b_a4b": 39}
    for j in ["V610_qwen36", "V611_gemma4_31b", "V612_olmo32", "V720_gemma4_26b_a4b"]:
        for loc, lab in (("v4c", "layers of panel D"), ("toend", "identification layer to last")):
            vals = v5c(j, loc)
            if all(v == "n.a." for v in vals): vals = ["invalid responses in every condition" if j == "V611_gemma4_31b" else "not reported"] + [""] * 9
            rows.append([NAME[j], lab, nexp[j]] + vals)
    E = table(["LLM", "Layers", "Experiments", "Own, 10", "Own, 30", "Own, 100", "Uniform, 10", "Uniform, 30", "Uniform, 50", "Uniform, 100", "Own + 4 most similar (50)", "5 most similar (50)", "5 random constructs (50)"], rows)
    notes = ("Notes. Panel A. Inducing a construct's representation (added to the situation's tokens at the identification layer, in multiples of the construct's "
             "natural high-minus-low difference) in the low situations of Study 1, judged on the construct's first-person item: count of the 66 judgments for which the induced own "
             "representation (doses +-2) changed the judgment more than the comparison directions (5 most similar and 5 random constructs in the full design; fewer in "
             "the short design); own direction first: rank 1 among the 15 (full) or 10 (short) directions; r: correlation over construct-by-judgment cells (726 in the "
             "full design) between induced and natural (situation) effects. 'Discriminant directions': the same test with diagonal discriminant directions (V6a). "
             "Panel B. 52 experiments in which MVCA induced the mediator and its denial; induction in the treatment condition; 'predicted' = the response changed in the "
             "direction that the theory predicts (M -> Y), chance 26; own > others: more than the 3 most similar and 3 random constructs; own > random: more than "
             "random patterns of the same size. Half depth = layer nearest 50% of depth on the scenario's tokens; 70% depth = from the scenario's first token to the "
             "end of the prompt. In-context judgment: the construct's first-person item presented after the scenario (V5a). Panel C. 25 of the 52 experiments in "
             "which all three main LLMs' responses had followed MVCA in run 1 (v5b_experiments.json): induced response changed as predicted / more than with random patterns "
             "(own mean change of the 1-9 response), dose +-2. Panel D. Removal of a 10-direction subspace at every position across the layers at which the "
             "constructs were identified best (Qwen3.6 19-48, Gemma 4 31B 18-45, OLMo 3.1 32B 13-48): median share of the manipulation's effect on the response "
             "left, over experiments with |effect| > .3; 'specific' = after removing the three directions shared by the 66 constructs; items with digit mass < .5 "
             "excluded (recomputed here with removal_v6.py's rules). Zeroing made Gemma 4 31B's responses invalid (digit mass < .5 in 77-100% of items). Panel E. Mean "
             "replacement by subspace size (V5c; median share left); Gemma 4 31B's responses were invalid in every condition when removal ran to the last layer. Sources: "
             "V61x/V4a_causal, V4b_steer, V4c_remove (remove_v4c.csv), V710/V711 V5a, V5b, V5c, V810 V6a/V6b, V83x V5b summaries; RD for Gemma 4 26B, OLMo 3.1 32B "
             "follow-ups and the short-design LLMs; exact binomial P computed here.")
    return ("**Table S6. Study 4: effects of inducing and removing the mediator's representation on judgments and responses, by LLM.**\n\n*Panel A. Induced judgments (Study 1 situations).*\n\n" + A +
            "\n\n*Panel B. Responses and in-context judgments in the 52 experiments.*\n\n" + B + "\n\n*Panel C. Responses in 25 experiments by depth (all LLMs with the sweep).*\n\n" + C +
            "\n\n*Panel D. Removal of the mediator's representation (10 directions).*\n\n" + D + "\n\n*Panel E. Removal by number of directions (mean replacement).*\n\n" + E + "\n\n" + notes)

# ------------------------------------------------------------------ S7
MOD7 = ["berger_heath_identity_divergence", "griskevicius_going_green", "mead_social_exclusion", "reed_identity_salience", "rucker_galinsky_powerlessness", "schultz_boomerang_norms", "white_argo_identity_threat"]
def s7():
    f8 = pd.read_csv(REL / "figures/fig8A_study3_moderator_interactions.csv")
    mv = lambda e, l: f8[(f8.test_id == e) & (f8.llm == l)].iloc[0]
    # L* aligned
    LST = {}
    for j in MAIN:
        R = pd.read_csv(RV / j / "V4d_moderation/moderation_aligned.csv"); R = R[R.kind == "own"].set_index("exp"); LST[j] = {e: (R.loc[e, "dz"], R.loc[e, "t"]) for e in MOD7}
    g26 = {}
    for r in rd_rows("V720_gemma4_26b_a4b", r"(berger|griskevicius|mead|reed|rucker|schultz|white)"):
        if len(r) == 6:
            e = r[0]; smy = int(EXPS[e]["x_to_m_sign"]) * int(EXPS[e]["x_to_y_sign"]); dz = -float(r[3]) * smy; g26[e] = (dz, dz * np.sqrt(18))
    LST["V720_gemma4_26b_a4b"] = g26
    def v6d_local():
        t = open(RV / "V810_steer2_qwen36/V6d_moderation2/digest.md", encoding="utf-8").read().splitlines(); out = {}
        for ln in t:
            if ln.startswith("| d"):
                ln = ln.replace("d50|sit", "d50#sit").replace("d70|sit2end", "d70#sit2end")
                c = [x.strip() for x in ln.strip().split("|")[1:-1]]; out[(c[0].replace("#", "|"), c[1])] = (float(c[3]), float(c[4]))
        return out
    def v6d_rd(job):
        return {(r[0], r[1]): (float(r[3]), float(r[4])) for r in rd_rows(job, r"d\d0") if len(r) == 5}
    V6D = {"V610_qwen36": v6d_local(), "V611_gemma4_31b": v6d_rd("V811_steer2_gemma4_31b"), "V612_olmo32": v6d_rd("V812_steer2_olmo32"), "V720_gemma4_26b_a4b": v6d_rd("V813_steer2_gemma4_26b_a4b")}
    J4 = ["V610_qwen36", "V611_gemma4_31b", "V612_olmo32", "V720_gemma4_26b_a4b"]
    def cell(dz, t): return f"{num(dz)} ({num(t, 1)})"
    out = []
    for lab, getter in (("Panel A. Identification layer, scenario tokens (V4d).", lambda j, e: LST[j][e]),
                        ("Panel B. Half depth, scenario tokens (V6d).", lambda j, e: V6D[j][("d50|sit", e)]),
                        ("Panel C. 70% depth, scenario to end of prompt (V6d).", lambda j, e: V6D[j][("d70|sit2end", e)])):
        rows = []; cnt = {j: [0, 0, 0] for j in J4}
        for e in MOD7:
            r = [EXP_LABEL[e], EXPS[e]["moderation"]["name"]]
            for j in J4:
                dz, t = getter(j, e); r.append(cell(dz, t)); cnt[j][0] += dz > 0; cnt[j][1] += t > 2; cnt[j][2] += t < -2
            a = mv(e, "Averaged over the three LLMs"); g = mv(e, "Gemma 4 26B")
            r += [f"{num(a.interaction_d_z)} [{num(a.ci_low)}, {num(a.ci_high)}]", num(g.interaction_d_z)]
            rows.append(r)
        mvpos = sum(mv(e, "Averaged over the three LLMs").interaction_d_z > 0 for e in MOD7); mvsig = sum(mv(e, "Averaged over the three LLMs").ci_excludes_zero == "yes" for e in MOD7)
        rows.append(["**Predicted direction (of 7) / t > 2 / t < -2**", ""] + [f"{c[0]} / {c[1]} / {c[2]}" for c in (cnt[j] for j in J4)] + [f"{mvpos} positive, {mvsig} with CI > 0", ""])
        out.append(f"*{lab}*\n\n" + table(["Experiment", "Situational moderator", "Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B", "Gemma 4 26B", "MVCA, averaged over its 3 LLMs [95% CI]", "MVCA, Gemma 4 26B"], rows))
    pooled = pd.read_csv(REL / "figures/fig8_pooled.csv").set_index("panel").loc["A"]
    notes = ("Notes. Each cell: interaction d_z (t) between the situational moderator and the induced mediator representation on the response, aligned so that "
             "positive values are in the direction that the theory predicts (effect of the induced representation at the level at which the larger effect is "
             "predicted minus that at the other level; (push_A - push_B) x X->M sign x X->Y sign; 18 wording-by-profile combinations per experiment, doses +-2). "
             "Gemma 4 26B at the identification layer: computed here from its V4d digest (own interaction d_z, B - A) with the same alignment rule, t = d_z x "
             "sqrt(18); the digest's own 'predicted sign' column uses the superseded rule. MVCA: the corresponding interaction of the moderator and the induced "
             f"mediator statement in MVCA's cognitive agents (release data figures/fig8A_study3_moderator_interactions.csv); MVCA's pooled estimate over all 11 of its "
             f"tests is d_z = {num(pooled.pooled_d_z)} [{num(pooled.ci_low)}, {num(pooled.ci_high)}] (fig8_pooled.csv). Sources: V61x/V4d_moderation/moderation_aligned.csv; "
             "V810 V6d digest; RD (V811-V813 V6d, V720 V4d).")
    return "**Table S7. Situational moderation of the induced mediator representation's effect on the responses, by experiment and LLM.**\n\n" + "\n\n".join(out) + "\n\n" + notes

# ------------------------------------------------------------------ S8
TESTS = [("thaler_costs_justify", "Beer price x belief that higher costs justify higher prices"), ("freedman_consistency", "Foot in the door x preference for consistency"),
         ("mogilner_materialism", "Time versus money x materialism"), ("whiteargo_cse", "Identity threat x collective self-esteem"),
         ("kivetz_guilt", "Effort and indulgence x guilt about luxuries"), ("reed_importance", "Identity salience x importance of family identity"),
         ("thaler_selfmonitoring_null", "Null test: beer price x self-monitoring")]
SHORT = {"thaler_costs_justify": "Costs justify prices", "freedman_consistency": "Consistency", "mogilner_materialism": "Materialism", "whiteargo_cse": "Collective self-esteem",
         "kivetz_guilt": "Luxury guilt", "reed_importance": "Family importance", "thaler_selfmonitoring_null": "Self-monitoring (null)"}
V7I = [("V812a_indiv_qwen36", "V610_qwen36"), ("V812b_indiv_gemma4_31b", "V611_gemma4_31b"), ("V812c_indiv_olmo32", "V612_olmo32"), ("V813b_indiv_gemma4_26b_a4b", "V720_gemma4_26b_a4b"),
       ("V850_indiv_qwen35_27b", "V730_qwen35_27b"), ("V851_indiv_qwen35_9b", "V731_qwen35_9b"), ("V852_indiv_qwen35_4b", "V732_qwen35_4b"), ("V853_indiv_qwen35_2b", "V733_qwen35_2b"),
       ("V854_indiv_qwen35_08b", "V734_qwen35_08b"), ("V855_indiv_gemma4_e4b", "V735_gemma4_e4b"), ("V856_indiv_olmo7", "V736_olmo7"), ("V858_indiv_gptoss20", "V741_gptoss20"),
       ("V859_indiv_granite42", "V742_granite42")]
MEDJ = ["V840_mediate_qwen36", "V841_mediate_gemma4_31b", "V842_mediate_olmo32", "V843_mediate_gemma4_26b_a4b"]
def s8():
    rowsA, rowsB, cnt, cnt211, proj_tot = [], [], {t: [0, 0] for t, _ in TESTS}, {t: [0, 0] for t, _ in TESTS}, 0
    for job, lj in V7I:
        sm = jload(RV / job / "V7i_indiv/summary.json"); L = sm["locations"]["Lstar"]
        pj = [sm["A"][t]["trait_projection_high_minus_low"][f"projdec_{L}"] for t, _ in TESTS]; k = sum(x > 0 for x in pj); proj_tot += k
        cells = []
        for t, _ in TESTS:
            x = sm["A"][t]["interaction"]; cells.append(f"{num(x['dz'])} ({num(x['t'], 1)})")
            cnt[t][0] += x["t"] > 2; cnt[t][1] += x["t"] < -2; cnt211[t][0] += x["t"] > 2.11; cnt211[t][1] += x["t"] < -2.11
        rowsA.append([NAME[lj], pct(sm["reading"]["lowo_acc_Lstar"]), f"{k}/7"] + cells)
        cb = []
        for t, _ in TESTS:
            B = sm["B"][t]; cb.append(f"{num(B['Lstar|own']['dz'])} ({num(B['Lstar|own']['t'], 1)}) / {num(B['d50|own']['dz'])} ({num(B['d50|own']['t'], 1)}) / {num(B['d50|span']['dz'])}")
        rowsB.append([NAME[lj], f"{sm['locations']['Lstar']} / {sm['locations']['d50']}"] + cb)
    rowsA.append(["**Not run**", "Qwen3.8 27B (the job did not run)", "", "", "", "", "", "", "", ""])
    rowsA.append(["**t > 2 in the predicted direction / t < -2 (of 13)**", "", f"{proj_tot}/91"] + [f"{cnt[t][0]} / {cnt[t][1]}" for t, _ in TESTS])
    f8 = pd.read_csv(REL / "figures/fig8B_study4_characteristic_interactions.csv")
    mrow = ["MVCA, averaged over its 3 LLMs [95% CI]", "", ""]; grow = ["MVCA, Gemma 4 26B", "", ""]
    for t, _ in TESTS:
        av = f8[(f8.test_id == t) & (f8.llm == "Averaged over the three LLMs")].iloc[0]; g = f8[(f8.test_id == t) & (f8.llm == "Gemma 4 26B")].iloc[0]
        mrow.append(f"{num(av.interaction_d_z)} [{num(av.ci_low)}, {num(av.ci_high)}]"); grow.append(num(g.interaction_d_z))
    rowsA += [mrow, grow]
    A = table(["LLM", "Characteristic identified (of 7; chance 14%)", "Statement changes the internal state at the response position toward the characteristic (of 7 tests)"] + [SHORT[t] for t, _ in TESTS], rowsA)
    B = table(["LLM", "Layers (identification / half depth)"] + [SHORT[t] for t, _ in TESTS], rowsB)
    M = pd.read_csv(AN / "mediate_summary_v6.csv"); J4n = ["Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B", "Gemma 4 26B"]; rows2 = []
    for i, job in enumerate(MEDJ):
        for t, lab in TESTS:
            g = M[(M.job == job) & (M.test == t)]; a = g[g.kind == "asis"].iloc[0]
            def sh(kinds, loc):
                x = g[(g.kind.isin(kinds)) & (g["loc"] == loc)].sort_values("kind")
                if not len(x): return "invalid responses" if job == "V841_mediate_gemma4_31b" and loc == "after" else "n.a."
                if (x.broken > .5).all(): return "invalid responses"
                return " / ".join(pct(v) for v in x.share_left)
            rows2.append([J4n[i], SHORT[t], f"{num(a['mean'])} ({num(a.dz)})", sh(["own"], "stmt"), sh(["ownadj"], "stmt"), sh(["rand0", "rand1"], "stmt"), sh(["other0", "other1"], "stmt"),
                          sh(["own"], "after"), sh(["rand0", "rand1"], "after"), sh(["other0", "other1"], "after")])
    C = table(["LLM", "Characteristic (test)", "Heterogeneity as described: mean (d_z)", "At the statement: own", "own, adjusted", "random (2)", "other characteristics (2)",
               "From the statement on: own", "random (2)", "other characteristics (2)"], rows2)
    diff = [SHORT[t] for t, _ in TESTS if cnt[t] != cnt211[t]]
    notes = ("Notes. Panel A. Seven characteristics of MVCA's tests of individual heterogeneity; 611 pairs composed by the six writers. Identification: share of "
             "the held-out writer's pairs assigned to the correct one of the seven characteristics (identification layer). 'Statement changes the internal state': the "
             "internal state at the response position (identification layer), projected on the characteristic's representation, is higher after the high-level than "
             "after the low-level statement in the agent profile (7 distinct tests; MVCA's two self-monitoring null tests share their materials and give identical "
             "results). Test columns: interaction of the described characteristic and the manipulation on the response, d_z (t), in the direction that the theory "
             "predicts (18 wording-by-profile combinations; t > 2.11 is P < .05 two-sided at 17 degrees of freedom). Counts use |t| > 2"
             + (f"; with |t| > 2.11 they are the same except for {', '.join(diff)} (" + "; ".join(f"{SHORT[t]} {cnt211[t][0]} / {cnt211[t][1]}" for t, _ in TESTS if cnt[t] != cnt211[t]) + ")" if diff else "")
             + ". MVCA: interaction in MVCA's cognitive agents (release data figures/fig8B_study4_characteristic_interactions.csv). Panel B. The characteristic's "
             "representation induced at the agent profile's tokens (doses +-2) instead of the described statement: own representation at the identification layer, "
             "d_z (t) / own at half depth, d_z (t) / random pattern of the same size at half depth, d_z. Panel C. Share of the described characteristic's heterogeneity "
             "left after removing one direction per layer (layers of the V1 grid from 10% to 85% of depth: Qwen3.6 13-51, Gemma 4 31B 6-48, OLMo 3.1 32B 13-51, "
             "Gemma 4 26B 3-24) at the statement's tokens or from the statement on; own = the characteristic's representation; adjusted = after removing the parts "
             "that go with rated wealth, pleasantness and sociability; random = two random directions; other = the directions of two other characteristics. Shares "
             "are unstable where the as-is heterogeneity is near zero (materialism, collective self-esteem, guilt and the null test in most LLMs). 'invalid responses' "
             "= digit mass < .5 in most items (Gemma 4 31B, from the statement on: 53-100%). Sources: results_v6/V812a-c, V813b, V850-V856, V858, V859 "
             "V7i_indiv/summary.json (Qwen3.8 27B, V857, did not run); analysis_v6/mediate_summary_v6.csv; V84x V7m_mediate digests.")
    return ("**Table S8. Study 5: representations of individual characteristics and the heterogeneity they produce, by LLM.**\n\n*Panel A. Identification, the "
            "described characteristic's representation, and the heterogeneity it produces (thirteen LLMs).*\n\n" + A + "\n\n*Panel B. Inducing the characteristic's "
            "representation instead of describing it.*\n\n" + B + "\n\n*Panel C. Removal of the characteristic's representation (share of the heterogeneity left).*\n\n"
            + C + "\n\n" + notes)

# ------------------------------------------------------------------ S9
EXAMPLES = [("perceived_fit", "claude", "L1"), ("reciprocity_obligation", "deepseek", "L4"), ("ownership", "grok", "L1"), ("felt_activation", "kimi", "L4"),
            ("loss_framing", "minimax", "L1"), ("blame_attribution", "muse", "L4")]
def s9():
    rows = []
    for c, w, lt in EXAMPLES:
        cand = sorted([p for p in PAIRS if p["construct"] == c and p["writer"] == w and p["low_type"] == lt], key=lambda p: p["pid"])
        if not cand: cand = sorted([p for p in PAIRS if p["construct"] == c and p["writer"] == w], key=lambda p: p["pid"])
        p = cand[0]
        esc = lambda s: s.replace("|", "\\|")
        rows.append([f"{PLAIN[c]} ({SPEC[c]['group']}; {WNAME[w]}; low = {'opposite' if p['low_type'] == 'L1' else 'absent'}; domain: {p['domain']}; `{p['pid']}`)", esc(p["high"]), esc(p["low"])])
    A = table(["Construct (process; writer; low pole; domain; pair id)", "High situation", "Low situation"], rows)
    raw = [json.loads(l) for l in open(MAT / "pairs_v6_raw.jsonl", encoding="utf-8")]; kept = [json.loads(l) for l in open(MAT / "pairs_v6.jsonl", encoding="utf-8")]
    exc = jload(MAT / "exclusions_v6.json"); er = collections.Counter((k.split("__")[0], v.split()[0]) for k, v in exc.items())
    qc_pass = {"deepseek": 1503, "kimi": 1362, "muse": 1473, "grok": 1509, "minimax": 1212, "claude": 1511}   # QC_REPORT_v6.md table
    cr = collections.Counter(p["writer"] for p in raw); ck = collections.Counter(p["writer"] for p in kept); cf = collections.Counter(p["writer"] for p in PAIRS)
    nohf = collections.Counter(p["writer"] for p in raw if not p["hard_fail"])
    rows = [[WNAME[w], f"{cr[w]:,}", f"{qc_pass[w]:,} ({nohf[w]:,})", f"{ck[w]:,}", er[(w, "R1")], er[(w, "R2")], f"{cf[w]:,}"] for w in WRITERS]
    rows.append(["**All writers**", f"{len(raw):,}", f"{sum(qc_pass.values()):,} ({sum(nohf.values()):,})", f"{len(kept):,}", sum(v for (w, r), v in er.items() if r == "R1"), sum(v for (w, r), v in er.items() if r == "R2"), f"{len(PAIRS):,}"])
    B = table(["Writer", "Composed", "Passed the automated checks (raw-file flags)", "Kept for rating", "Excluded: high not rated above low", "Excluded: near-duplicate", "Retained"], rows)
    notes = ("Notes. Panel A. One retained pair for each of six constructs from different MVCA processes, one per writer, the first pair (by pair id) of the stated "
             "low-pole type; texts verbatim from materials_v6/upload_v6/pairs_v6.json. Panel B. Composed: all pairs returned by the writers (pairs_v6_raw.jsonl); "
             "passed the automated checks: QC_REPORT_v6.md (in parentheses, pairs without a hard-failure flag in the current raw file, which differs); kept for rating: "
             "pairs_v6.jsonl (within the target number per writer, construct and low pole); exclusions: exclusions_v6.json (rule 1: the high situation not rated "
             "higher than the low situation on the construct's own scale; rule 2: high situations with TF-IDF cosine > .80 within writer and construct); retained: "
             "pairs_v6.json. Reasons for failing the automated checks (all writers, QC_REPORT_v6.md): length difference 873, banned words 297, a mental state named 184, "
             "length outside 40-80 words 182, rated difference too small 149 or too large 76, study frame 24.")
    return "**Table S9. Example pairs of situations and the numbers of composed, excluded and retained pairs.**\n\n*Panel A. Example pairs.*\n\n" + A + "\n\n*Panel B. Pairs by writer.*\n\n" + B + "\n\n" + notes

# ------------------------------------------------------------------ write
def main():
    head = ("# Supplementary Tables S1-S9 (v1)\n\n"
            "Built by draft_2026-10-21/make_si_tables.py from the analysis files of latent_2026-10-17 (materials_v6, analysis_v6, results_v6) and the companion "
            "paper's release data. Percentages are rounded to whole numbers, correlations to two decimals. 'n.a.' = the analysis ran, but its value is not in the "
            "local files (explained in the notes); 'not run' = the analysis was not conducted for that LLM. Minus signs are typographic.\n")
    parts = [head, s1(), s2(), s3(), s4(), s5(), s6(), s7(), s8(), s9()]
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8"); print("wrote", OUT, sum(len(p) for p in parts), "chars")

if __name__ == "__main__":
    main()
