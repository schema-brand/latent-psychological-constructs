# V701_verify_qwen36: materials v6 (GPU_PLAN_v6.md), steps ['V3_exp', 'V4a_causal', 'V4b_steer', 'V4c_remove', 'V8_verify', 'V9_pack'] on Qwen/Qwen3.6-27B. Resumable: a step whose summary.json exists is skipped.
# When the model changes, the previous one is released and every other cached repo is removed (the next model's files are kept).
exec(open(f"{ROOT}/latent_lib.py", encoding="utf-8").read(), globals()); exec(open(f"{ROOT}/test_lib3.py", encoding="utf-8").read(), globals())
import hashlib as _hl
# Colab's Drive mount can serve the old copy of a replaced file for minutes: wait (up to 30 min) until every code file this job runs has the
# fingerprint it was written for (sha256 of the text, first 10 hex), so a job never runs a mix of old and new code.
_EXPECT = {'lib_v6.py': 'bee1a13af2', 'V3_exp.py': 'abf8f58fce', 'V4a_causal.py': '98ba598656', 'V4b_steer.py': '24e2bd692a', 'V4c_remove.py': '69c7a746ed', 'V8_verify.py': '7bf1a9d147', 'V9_pack.py': 'cfc83b4fbc'}
def _tsha(_p): return _hl.sha256(open(_p, encoding="utf-8").read().encode()).hexdigest()[:10]
for _w in range(60):
    _bad = [_f for _f, _h in _EXPECT.items() if _tsha(f"{ROOT}/code_v6/{_f}") != _h]
    if not _bad: break
    beat(f"V701_verify_qwen36: waiting for Drive to sync {_bad}"); time.sleep(30)
if _bad: log(f"WARNING V701_verify_qwen36: code still differs after 30 min: {_bad}")
exec(open(f"{ROOT}/test_lib4.py", encoding="utf-8").read(), globals()); _LIBSRC = open(f"{ROOT}/code_v6/lib_v6.py", encoding="utf-8").read(); exec(_LIBSRC, globals())
_MODEL = "Qwen/Qwen3.6-27B"; _REV = None; _STEPS = ['V3_exp', 'V4a_causal', 'V4b_steer', 'V4c_remove', 'V8_verify', 'V9_pack']; _CFG = {'V3_EXPS': ['aaker_keller_brand_extension', 'berger_heath_identity_divergence', 'carter_flag_priming', 'greene_personal_force', 'schultz_boomerang_norms', 'thaler_transaction_utility'], 'V4A_JUDGE': ['affective_reaction', 'ownership', 'perceived_fit', 'status_desire'], 'V4B_EXPS': ['aaker_keller_brand_extension', 'greene_personal_force', 'peck_shu_mere_touch', 'thaler_transaction_utility'], 'V4C_EXPS': ['aaker_keller_brand_extension', 'bellezza_red_sneaker', 'berger_milkman_virality'], 'V8_REF': 'V610_qwen36'}; MOCK_SMALL = False
_BASE = OUT; _RES = {"model": _MODEL, "rev": _REV, "small": MOCK_SMALL, "cfg": _CFG, "code": {"lib_v6.py": _hl.sha256(_LIBSRC.encode()).hexdigest()[:10]}}
if CUR.get("id") != model_key(_MODEL, _REV) and "MOCK_LOAD" not in globals():
    _RES["gpu_gb_after_free"] = _free_gpu(); _RES["disk"] = wipe_hub_except([] if _REV else [_MODEL])
    if _REV: shutil.rmtree(f"{HUB}/models--" + _MODEL.replace("/", "--"), ignore_errors=True)      # one checkpoint on disk at a time
for _s in _STEPS:
    V_MODEL, V_REV = _MODEL, _REV
    OUT = f"{_BASE}/{_s}"; os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/summary.json"): _RES[_s] = "skip (done)"; continue
    if os.path.exists(f"{OUT}/ERROR.txt"): os.remove(f"{OUT}/ERROR.txt")
    _t0 = time.time(); _SRC = open(f"{ROOT}/code_v6/{_s}.py", encoding="utf-8").read(); _RES["code"][_s] = _hl.sha256(_SRC.encode()).hexdigest()[:10]
    try:
        exec(compile(_SRC, _s, "exec"), globals()); _RES[_s] = f"ok {time.time() - _t0:.0f}s"
    except Exception:
        open(f"{OUT}/ERROR.txt", "w").write(traceback.format_exc()); _RES[_s] = "FAILED: " + traceback.format_exc().splitlines()[-1][:200]
    log(f"V701_verify_qwen36 {_s}: {_RES[_s]}"); beat(f"V701_verify_qwen36 {_s} done")
OUT = _BASE; save_json("summary.json", _RES); beat("V701_verify_qwen36 ok")
