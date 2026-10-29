"""Write the GPU job files for the formal runs on materials v6 (GPU_PLAN_v6.md). One wrapper per model (or checkpoint) and step list;
resumable (a step whose summary.json exists is skipped). Jobs run in file-name order on the Colab runner (runner_v1.ipynb).
Usage: python make_v6_jobs.py smoke | run1 | run2      -> jobs6/<set>/*.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUTD = HERE / "jobs6"
WRAP = '''# {name}: materials v6 (GPU_PLAN_v6.md), steps {steps} on {model}{revtxt}. Resumable: a step whose summary.json exists is skipped.
# When the model changes, the previous one is released and every other cached repo is removed (the next model's files are kept).
exec(open(f"{{ROOT}}/latent_lib.py", encoding="utf-8").read(), globals()); exec(open(f"{{ROOT}}/test_lib3.py", encoding="utf-8").read(), globals())
import hashlib as _hl
# Colab's Drive mount can serve the old copy of a replaced file for minutes: wait (up to 30 min) until every code file this job runs has the
# fingerprint it was written for (sha256 of the text, first 10 hex), so a job never runs a mix of old and new code.
_EXPECT = {expect!r}
def _tsha(_p): return _hl.sha256(open(_p, encoding="utf-8").read().encode()).hexdigest()[:10]
for _w in range(60):
    _bad = [_f for _f, _h in _EXPECT.items() if _tsha(f"{{ROOT}}/code_v6/{{_f}}") != _h]
    if not _bad: break
    beat(f"{name}: waiting for Drive to sync {{_bad}}"); time.sleep(30)
if _bad: log(f"WARNING {name}: code still differs after 30 min: {{_bad}}")
# results this job builds on that another runner may still be producing (paths under runner/results): wait up to 12 h
_NEED = {need!r}
for _w in range(720):
    _miss = [_p for _p in _NEED if not os.path.exists(f"{{ROOT}}/results/{{_p}}")]
    if not _miss: break
    beat(f"{name}: waiting for {{_miss[:2]}}"); time.sleep(60)
exec(open(f"{{ROOT}}/test_lib4.py", encoding="utf-8").read(), globals()); _LIBSRC = open(f"{{ROOT}}/code_v6/lib_v6.py", encoding="utf-8").read(); exec(_LIBSRC, globals())
_MODEL = "{model}"; _REV = {rev!r}; _STEPS = {steps!r}; _CFG = {cfg!r}; MOCK_SMALL = {small!r}
_BASE = OUT; _RES = {{"model": _MODEL, "rev": _REV, "small": MOCK_SMALL, "cfg": _CFG, "code": {{"lib_v6.py": _hl.sha256(_LIBSRC.encode()).hexdigest()[:10]}}}}
if CUR.get("id") != model_key(_MODEL, _REV) and "MOCK_LOAD" not in globals():
    _RES["gpu_gb_after_free"] = _free_gpu(); _RES["disk"] = wipe_hub_except([] if _REV else [_MODEL])
    if _REV: shutil.rmtree(f"{{HUB}}/models--" + _MODEL.replace("/", "--"), ignore_errors=True)      # one checkpoint on disk at a time
for _s in _STEPS:
    V_MODEL, V_REV = _MODEL, _REV
    OUT = f"{{_BASE}}/{{_s}}"; os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{{OUT}}/summary.json"): _RES[_s] = "skip (done)"; continue
    if os.path.exists(f"{{OUT}}/ERROR.txt"): os.remove(f"{{OUT}}/ERROR.txt")
    _t0 = time.time(); _SRC = open(f"{{ROOT}}/code_v6/{{_s}}.py", encoding="utf-8").read(); _RES["code"][_s] = _hl.sha256(_SRC.encode()).hexdigest()[:10]
    try:
        exec(compile(_SRC, _s, "exec"), globals()); _RES[_s] = f"ok {{time.time() - _t0:.0f}}s"
    except Exception:
        open(f"{{OUT}}/ERROR.txt", "w").write(traceback.format_exc()); _RES[_s] = "FAILED: " + traceback.format_exc().splitlines()[-1][:200]
    log(f"{name} {{_s}}: {{_RES[_s]}}"); beat(f"{name} {{_s}} done")
OUT = _BASE; save_json("summary.json", _RES); beat("{name} ok")
'''
CODECHECK = '''# V6c0: what the runner sees on Drive right now (Colab's Drive mount can serve a replaced file's old copy for several minutes):
# sha256 (first 10 hex) of every code and materials file -> results/V6c0_codecheck/summary.json. Compare with python make_v6_jobs.py sha.
import hashlib as _hl
_R = {}
for _d in ("code_v6", "materials/v6"):
    for _f in sorted(os.listdir(f"{ROOT}/{_d}")): _R[f"{_d}/{_f}"] = _hl.sha256(open(f"{ROOT}/{_d}/{_f}", "rb").read()).hexdigest()[:10]
for _f in ("runner_lib.py", "latent_lib.py", "test_lib3.py", "test_lib4.py"): _R[_f] = _hl.sha256(open(f"{ROOT}/{_f}", "rb").read()).hexdigest()[:10]
try:
    import psutil as _ps; _R["ram_total_gb"] = round(_ps.virtual_memory().total / 1e9, 1); _R["ram_avail_gb"] = round(_ps.virtual_memory().available / 1e9, 1)
except Exception: pass
_R["disk_free_gb"] = round(shutil.disk_usage("/").free / 1e9, 1); _R["vram_gb"] = round(VRAM, 1)
save_json("summary.json", _R); beat("V6c0 ok")
'''
FULL = ["V1_read", "V2_jlens", "V3_exp", "V4a_causal", "V4b_steer", "V4c_remove", "V4d_moderation", "V9_pack"]
LITE4A = dict(V4A_NT=1, V4A_RAND=2, V4A_SPAN=2, V4A_OWN=[4, -4], V4A_WRITER=False, V4A_NAT=1, V4A_NAT_WRITERS=["claude", "deepseek", "kimi"])
ORIG = json.loads((HERE / "materials_v6" / "upload_v6" / "origins_v6.json").read_text())
import hashlib
def tsha(f): return hashlib.sha256(open(HERE / "code_v6" / f, encoding="utf-8").read().encode()).hexdigest()[:10]
def job(name, model, steps, cfg=None, rev=None, small=False, need=None):
    expect = {f: tsha(f) for f in ["lib_v6.py"] + [f"{s_}.py" for s_ in steps]}
    return name, WRAP.format(name=name, model=model, rev=rev, steps=steps, cfg=cfg or {}, small=small, expect=expect, need=list(need or []), revtxt=f" (revision {rev})" if rev else "")
SETS = {
    # smoke: every step on small subsets (MOCK_SMALL) in three families, a checkpoint revision, and the full-size main model for timing
    "smoke": [job("V600_smoke_qwen35_08b", "Qwen/Qwen3.5-0.8B", ["V0_embed"] + FULL, dict(V1_FACET=3, V_EMB="Qwen/Qwen3-Embedding-0.6B"), small=True),
              job("V601_smoke_gemma4_e4b", "google/gemma-4-E4B-it", FULL, dict(V1_FACET=2), small=True),
              job("V602_smoke_olmo7_ck", ORIG["repo"], ["V1_read", "V4a_causal", "V9_pack"], dict(LITE4A), rev="stage1-step10000", small=True),
              job("V603_probe_qwen36", "Qwen/Qwen3.6-27B", FULL, dict(V1_FACET=1), small=True),
              job("V604_gemma4_e4b_removal", "google/gemma-4-E4B-it", ["V1_read", "V4c_remove", "V9_pack"], dict(SMALL_N=4, SMALL_EXPS=8, SMALL_NV=3, SMALL_NA=2), small=True),
              job("V605_gemma4_e4b_meanrepl", "google/gemma-4-E4B-it", ["V1_read", "V4c_remove", "V9_pack"], dict(SMALL_N=4, SMALL_EXPS=8, SMALL_NV=3, SMALL_NA=2), small=True),
              job("V606_qwen36_meanrepl", "Qwen/Qwen3.6-27B", ["V1_read", "V4c_remove", "V9_pack"], dict(SMALL_N=4, SMALL_EXPS=8, SMALL_NV=3, SMALL_NA=2), small=True),
              job("V607_qwen36_specific", "Qwen/Qwen3.6-27B", ["V4c_remove", "V9_pack"], dict(SMALL_N=4, SMALL_EXPS=8, SMALL_NV=3, SMALL_NA=2), small=True),
              job("V608_gemma4_e4b_specific", "google/gemma-4-E4B-it", ["V4c_remove", "V9_pack"], dict(SMALL_N=4, SMALL_EXPS=8, SMALL_NV=3, SMALL_NA=2), small=True),
              ("V6c0_codecheck", CODECHECK),
              job("V6c1_gemma4_e4b_removal", "google/gemma-4-E4B-it", ["V1_read", "V4c_remove", "V9_pack"], dict(SMALL_N=4, SMALL_EXPS=8, SMALL_NV=3, SMALL_NA=2), small=True),
              job("V6c2_qwen36_removal", "Qwen/Qwen3.6-27B", ["V1_read", "V4c_remove", "V9_pack"], dict(SMALL_N=4, SMALL_EXPS=8, SMALL_NV=3, SMALL_NA=2), small=True),
              job("V6c3_qwen36_removal_k10", "Qwen/Qwen3.6-27B", ["V4c_remove", "V9_pack"], dict(SMALL_N=4, SMALL_EXPS=8, SMALL_NV=3, SMALL_NA=2), small=True),
              job("V6c4_gemma4_e4b_removal_k10", "google/gemma-4-E4B-it", ["V4c_remove", "V9_pack"], dict(SMALL_N=4, SMALL_EXPS=8, SMALL_NV=3, SMALL_NA=2), small=True)],
    # run 1 (~13.7 h on G4): the three main models, every step
    "run1": [("V609_codecheck", CODECHECK), job("V610_qwen36", "Qwen/Qwen3.6-27B", FULL, dict(V1_FACET=3)),
             job("V611_gemma4_31b", "google/gemma-4-31B-it", FULL, dict(V1_FACET=3)),
             job("V612_olmo32", "allenai/Olmo-3.1-32B-Instruct", FULL, dict(V1_FACET=3)),
             # re-run of Qwen3.6's step 4a with the full 1-9 answer distribution (added after V610's 4a had started; deterministic, so ev must match V610)
             job("V613_qwen36_v4a_p9", "Qwen/Qwen3.6-27B", ["V4a_causal", "V9_pack"])],
    # run 2 (~12 h): MVCA's model with every step; base models; size series (+ short 4a); breadth; training checkpoints; text encoder
    "run2": [("V619_codecheck", CODECHECK), job("V620_gemma4_26b_a4b", "google/gemma-4-26B-A4B-it", FULL, dict(V1_FACET=3)),
             job("V621_olmo32_base", "allenai/Olmo-3-1125-32B", ["V1_read", "V9_pack"]),
             job("V622_gemma4_31b_base", "google/gemma-4-31B", ["V1_read", "V9_pack"])]
            + [job(f"V63{k}_{t}", m, ["V1_read", "V4a_causal", "V9_pack"], dict(LITE4A)) for k, (t, m) in enumerate(
                [("qwen35_27b", "Qwen/Qwen3.5-27B"), ("qwen35_9b", "Qwen/Qwen3.5-9B"), ("qwen35_4b", "Qwen/Qwen3.5-4B"), ("qwen35_2b", "Qwen/Qwen3.5-2B"),
                 ("qwen35_08b", "Qwen/Qwen3.5-0.8B"), ("gemma4_e4b", "google/gemma-4-E4B-it"), ("olmo7", "allenai/Olmo-3-7B-Instruct")])]
            + [job(f"V64{k}_{t}", m, ["V1_read", "V9_pack"]) for k, (t, m) in enumerate([("qwen38", "Qwen/Qwen3.8-27B"), ("gptoss20", "openai/gpt-oss-20b"), ("granite42", "ibm-granite/granite-4.2-30b")])]
            + [job(f"V65{k:02d}_olmo7_{rev.replace('-', '_')}", ORIG["repo"], ["V1_read", "V9_pack"], dict(V1_FRAC=3), rev=rev) for k, rev in enumerate(ORIG["checkpoints"])]
            + [job(f"V66{k}_olmo7_{m.split('-')[-1].lower()}", m, ["V1_read", "V9_pack"], dict(V1_FRAC=3)) for k, m in enumerate(ORIG["post_training"])]
            + [job("V670_embed", "Qwen/Qwen3-Embedding-8B", ["V0_embed", "V9_pack"])],
}
# run 2 as agreed with the PI on 20 Oct (supersedes "run2"): batched forward passes (run_rows6), gated by a check against run 1; three
# follow-ups on the three run-1 models and on Gemma 4 26B-A4B; then run 2's original models with unchanged materials, steps and settings.
# Upload in two parts: V700 + V701 first; the rest only after V701's comparison with run 1 passes.
V5B = json.loads((HERE / "analysis_v6" / "v5b_experiments.json").read_text())
FOLLOW = ["V5a_incontext", "V5b_sweep", "V5c_remove2", "V9_pack"]
VERIFY = dict(V3_EXPS=["aaker_keller_brand_extension", "berger_heath_identity_divergence", "carter_flag_priming", "greene_personal_force", "schultz_boomerang_norms", "thaler_transaction_utility"],
              V4A_JUDGE=["affective_reaction", "ownership", "perceived_fit", "status_desire"], V4B_EXPS=["aaker_keller_brand_extension", "greene_personal_force", "peck_shu_mere_touch", "thaler_transaction_utility"],
              V4C_EXPS=["aaker_keller_brand_extension", "bellezza_red_sneaker", "berger_milkman_virality"], V8_REF="V610_qwen36")
SETS["run2b"] = ([("V700_codecheck", CODECHECK),
                  job("V701_verify_qwen36", "Qwen/Qwen3.6-27B", ["V3_exp", "V4a_causal", "V4b_steer", "V4c_remove", "V8_verify", "V9_pack"], VERIFY),
                  job("V710_follow_qwen36", "Qwen/Qwen3.6-27B", FOLLOW, dict(V5B_EXPS=V5B)),
                  job("V711_follow_gemma4_31b", "google/gemma-4-31B-it", FOLLOW, dict(V5B_EXPS=V5B)),
                  job("V712_follow_olmo32", "allenai/Olmo-3.1-32B-Instruct", FOLLOW, dict(V5B_EXPS=V5B)),
                  job("V720_gemma4_26b_a4b", "google/gemma-4-26B-A4B-it", FULL[:-1] + FOLLOW, dict(V1_FACET=3, V5B_EXPS=V5B)),
                  job("V721_olmo32_base", "allenai/Olmo-3-1125-32B", ["V1_read", "V9_pack"]),
                  job("V722_gemma4_31b_base", "google/gemma-4-31B", ["V1_read", "V9_pack"])]
                 + [job(f"V73{k}_{t}", m, ["V1_read", "V4a_causal", "V9_pack"], dict(LITE4A)) for k, (t, m) in enumerate(
                     [("qwen35_27b", "Qwen/Qwen3.5-27B"), ("qwen35_9b", "Qwen/Qwen3.5-9B"), ("qwen35_4b", "Qwen/Qwen3.5-4B"), ("qwen35_2b", "Qwen/Qwen3.5-2B"),
                      ("qwen35_08b", "Qwen/Qwen3.5-0.8B"), ("gemma4_e4b", "google/gemma-4-E4B-it"), ("olmo7", "allenai/Olmo-3-7B-Instruct")])]
                 + [job(f"V74{k}_{t}", m, ["V1_read", "V9_pack"]) for k, (t, m) in enumerate([("qwen38", "Qwen/Qwen3.8-27B"), ("gptoss20", "openai/gpt-oss-20b"), ("granite42", "ibm-granite/granite-4.2-30b")])]
                 + [job(f"V75{k:02d}_olmo7_{rev.replace('-', '_')}", ORIG["repo"], ["V1_read", "V9_pack"], dict(V1_FRAC=3), rev=rev) for k, rev in enumerate(ORIG["checkpoints"])]
                 + [job(f"V76{k}_olmo7_{m.split('-')[-1].lower()}", m, ["V1_read", "V9_pack"], dict(V1_FRAC=3)) for k, m in enumerate(ORIG["post_training"])]
                 + [job("V770_embed", "Qwen/Qwen3-Embedding-8B", ["V0_embed", "V9_pack"])])
# run 3 (PI 20 Oct: "do everything that can be done"), queued behind run 2. Only new step files (queued run-2 jobs pin the existing ones).
#   V81x: MVCA Study 2 steering and moderation at the two pre-specified mid-depth locations (V6b, V6d) + 4a with diagonal discriminant
#         directions (V6a, short 4a) on the three run-1 models and Gemma 4 26B-A4B;
#   V82x: source analysis, short 4a at OLMo 3 7B checkpoints (early, mid, end of pretraining, final base) and SFT / DPO;
#   V83x: Study 3 (V3) and the push-location sweep (V5b) on the size-series and breadth models.
SIZE_BREADTH = [("qwen35_27b", "Qwen/Qwen3.5-27B"), ("qwen35_9b", "Qwen/Qwen3.5-9B"), ("qwen35_4b", "Qwen/Qwen3.5-4B"), ("qwen35_2b", "Qwen/Qwen3.5-2B"),
                ("qwen35_08b", "Qwen/Qwen3.5-0.8B"), ("gemma4_e4b", "google/gemma-4-E4B-it"), ("olmo7", "allenai/Olmo-3-7B-Instruct"),
                ("qwen38", "Qwen/Qwen3.8-27B"), ("gptoss20", "openai/gpt-oss-20b"), ("granite42", "ibm-granite/granite-4.2-30b")]
STEER2 = ["V6b_steer2", "V6d_moderation2", "V6a_dlda", "V9_pack"]
V1OF = {"qwen35_27b": "V730_qwen35_27b", "qwen35_9b": "V731_qwen35_9b", "qwen35_4b": "V732_qwen35_4b", "qwen35_2b": "V733_qwen35_2b", "qwen35_08b": "V734_qwen35_08b",
        "gemma4_e4b": "V735_gemma4_e4b", "olmo7": "V736_olmo7", "qwen38": "V740_qwen38", "gptoss20": "V741_gptoss20", "granite42": "V742_granite42"}
SRC = [("stage1-step10000", "V7503_olmo7_stage1_step10000"), ("stage1-step300000", "V7506_olmo7_stage1_step300000"),
       ("stage1-step1413814", "V7509_olmo7_stage1_step1413814"), ("main", "V7512_olmo7_main")]
# two runners (20 Oct): queue B = runner_v1b (jobs_pending_b), queue A = runner_v1 after run 2 (jobs_pending); _NEED waits for the other runner
SETS["run3"] = ([("B/V800_codecheck", CODECHECK)]
                + [job(f"B/V81{k}_steer2_{t}", m, STEER2, dict(LITE4A), need=nd) for k, (t, m, nd) in enumerate(
                    [("qwen36", "Qwen/Qwen3.6-27B", []), ("gemma4_31b", "google/gemma-4-31B-it", []), ("olmo32", "allenai/Olmo-3.1-32B-Instruct", []),
                     ("gemma4_26b_a4b", "google/gemma-4-26B-A4B-it", ["V720_gemma4_26b_a4b/V1_read/dirsrc.pt"])])]
                + [job(f"B/{nm}_indiv_{t}", m, ["V7i_indiv", "V9_pack"], {}, need=nd) for nm, (t, m, nd) in zip(["V812a", "V812b", "V812c", "V813b"],
                    [("qwen36", "Qwen/Qwen3.6-27B", []), ("gemma4_31b", "google/gemma-4-31B-it", []), ("olmo32", "allenai/Olmo-3.1-32B-Instruct", []),
                     ("gemma4_26b_a4b", "google/gemma-4-26B-A4B-it", ["V720_gemma4_26b_a4b/V1_read/dirsrc.pt"])])]
                + [job(f"A/V82{k}_src4a_olmo7_{rev.replace('-', '_')}", ORIG["repo"], ["V4a_causal", "V9_pack"], dict(LITE4A), rev=rev, need=[f"{v1}/V1_read/dirsrc.pt"])
                   for k, (rev, v1) in enumerate(SRC)]
                + [job(f"A/V82{4 + k}_src4a_olmo7_{m.split('-')[-1].lower()}", m, ["V4a_causal", "V9_pack"], dict(LITE4A), need=[f"V76{k}_olmo7_{m.split('-')[-1].lower()}/V1_read/dirsrc.pt"])
                   for k, m in enumerate(ORIG["post_training"][:2])]
                + [job(f"{'A' if t in ('gptoss20', 'granite42') else 'B'}/V83{k}_v3sweep_{t}", m, ["V3_exp", "V5b_sweep", "V9_pack"], dict(V5B_EXPS=V5B), need=[f"{V1OF[t]}/V1_read/dirsrc.pt"])
                   for k, (t, m) in enumerate(SIZE_BREADTH)])
SETS["mock3"] = [job("V880_mock_qwen36", "Qwen/Qwen3.6-27B", ["V1_read", "V6b_steer2", "V6d_moderation2", "V6a_dlda", "V9_pack"], dict(LITE4A), small=True),
                 job("V881_mock_gemma4", "google/gemma-4-E4B-it", ["V1_read", "V3_exp", "V5b_sweep", "V9_pack"], dict(V5B_EXPS=V5B), small=True),
                 job("V882_mock_olmo_ck", ORIG["repo"], ["V1_read", "V4a_causal", "V9_pack"], dict(LITE4A), rev="stage1-step10000", small=True),
                 job("V883_mock_indiv", "google/gemma-4-E4B-it", ["V1_read", "V7i_indiv", "V9_pack"], {}, small=True)]
# run 4 (21 Oct night, PI: hard stop 10:00 EDT; one runner): V84x individuality mediation by removal (V7m) on the four run-3 models (first);
# V85x individuality (V7i) on the size-series and breadth models.
SETS["run4"] = ([job(f"A/V85{k}_indiv_{t}", m, ["V7i_indiv", "V9_pack"], {}, need=[f"{V1OF[t]}/V1_read/dirsrc.pt"]) for k, (t, m) in enumerate(SIZE_BREADTH)]
                + [job(f"A/V84{k}_mediate_{t}", m, ["V7m_mediate", "V9_pack"], {}, need=nd) for k, (t, m, nd) in enumerate(
                    [("qwen36", "Qwen/Qwen3.6-27B", []), ("gemma4_31b", "google/gemma-4-31B-it", []), ("olmo32", "allenai/Olmo-3.1-32B-Instruct", []),
                     ("gemma4_26b_a4b", "google/gemma-4-26B-A4B-it", ["V720_gemma4_26b_a4b/V1_read/dirsrc.pt"])])])
# run 4b (21 Oct night): Study 3 (V3) across OLMo 3 7B training - checkpoints 1k, 3k, 10k, 30k, 300k, 1.41M, final base, then SFT and DPO
# (Instruct = V836): when do the experiments' manipulations start moving the constructs and the answers? Named V844-V84c (after the mediation jobs).
_V3CK = ["stage1-step1000", "stage1-step3000", "stage1-step10000", "stage1-step30000", "stage1-step300000", "stage1-step1413814", "main"]
SETS["run4b"] = ([job(f"A/V84{'456789abc'[k]}_v3_olmo7_{rev.replace('-', '_')}", ORIG["repo"], ["V3_exp", "V9_pack"], {}, rev=rev,
                      need=[f"V75{ORIG['checkpoints'].index(rev):02d}_olmo7_{rev.replace('-', '_')}/V1_read/dirsrc.pt"]) for k, rev in enumerate(_V3CK)]
                 + [job(f"A/V84{'456789abc'[len(_V3CK) + k]}_v3_olmo7_{m.split('-')[-1].lower()}", m, ["V3_exp", "V9_pack"], {},
                        need=[f"V76{k}_olmo7_{m.split('-')[-1].lower()}/V1_read/dirsrc.pt"]) for k, m in enumerate(ORIG["post_training"][:2])])
SETS["mock4"] = [job("V884_mock_mediate", "google/gemma-4-E4B-it", ["V1_read", "V7m_mediate", "V9_pack"], {}, small=True)]
# local dry run of the run2b jobs (mock/mock_v6.py): every step on small subsets with the real tokenizers
SETS["mock2b"] = [job("V780_mock_qwen36", "Qwen/Qwen3.6-27B", ["V1_read", "V3_exp", "V4a_causal", "V4b_steer", "V4c_remove", "V4d_moderation", "V5a_incontext", "V5b_sweep", "V5c_remove2", "V8_verify", "V9_pack"],
                      dict(V1_FACET=1, V5B_EXPS=V5B, V8_REF="V780_mock_qwen36"), small=True),
                  job("V781_mock_gemma4", "google/gemma-4-E4B-it", ["V1_read", "V3_exp", "V4b_steer", "V5a_incontext", "V5b_sweep", "V5c_remove2", "V9_pack"], dict(V5B_EXPS=V5B), small=True),
                  job("V782_mock_olmo_ck", ORIG["repo"], ["V1_read", "V4a_causal", "V5c_remove2", "V9_pack"], dict(LITE4A), rev="stage1-step10000", small=True)]
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if which == "sha":                     # local fingerprints to compare with the V6c0 code check
        import hashlib
        R = {f"code_v6/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()[:10] for p in sorted((HERE / "code_v6").glob("*.py"))}
        R.update({f"materials/v6/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()[:10] for p in sorted((HERE / "materials_v6" / "upload_v6").glob("*.json"))})
        R.update({f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:10] for f in ("runner_lib.py", "latent_lib.py", "test_lib3.py", "test_lib4.py")})
        print(json.dumps(R, indent=1)); sys.exit()
    d = OUTD / which; d.mkdir(parents=True, exist_ok=True)
    for f in d.rglob("*.py"): f.unlink()
    for name, src in SETS[which]: (d / f"{name}.py").parent.mkdir(parents=True, exist_ok=True); (d / f"{name}.py").write_text(src, encoding="utf-8")
    print(which, sorted(str(p.relative_to(d)) for p in d.rglob("*.py")))
