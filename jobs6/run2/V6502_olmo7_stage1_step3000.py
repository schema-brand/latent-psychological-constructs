# V6502_olmo7_stage1_step3000: materials v6 (GPU_PLAN_v6.md), steps ['V1_read', 'V9_pack'] on allenai/Olmo-3-1025-7B (revision stage1-step3000). Resumable: a step whose summary.json exists is skipped.
# When the model changes, the previous one is released and every other cached repo is removed (the next model's files are kept).
exec(open(f"{ROOT}/latent_lib.py", encoding="utf-8").read(), globals()); exec(open(f"{ROOT}/test_lib3.py", encoding="utf-8").read(), globals())
import hashlib as _hl
# Colab's Drive mount can serve the old copy of a replaced file for minutes: wait (up to 30 min) until every code file this job runs has the
# fingerprint it was written for (sha256 of the text, first 10 hex), so a job never runs a mix of old and new code.
_EXPECT = {'lib_v6.py': '92bfc9d31d', 'V1_read.py': '5db50bb843', 'V9_pack.py': 'cfc83b4fbc'}
def _tsha(_p): return _hl.sha256(open(_p, encoding="utf-8").read().encode()).hexdigest()[:10]
for _w in range(60):
    _bad = [_f for _f, _h in _EXPECT.items() if _tsha(f"{ROOT}/code_v6/{_f}") != _h]
    if not _bad: break
    beat(f"V6502_olmo7_stage1_step3000: waiting for Drive to sync {_bad}"); time.sleep(30)
if _bad: log(f"WARNING V6502_olmo7_stage1_step3000: code still differs after 30 min: {_bad}")
exec(open(f"{ROOT}/test_lib4.py", encoding="utf-8").read(), globals()); _LIBSRC = open(f"{ROOT}/code_v6/lib_v6.py", encoding="utf-8").read(); exec(_LIBSRC, globals())
_MODEL = "allenai/Olmo-3-1025-7B"; _REV = 'stage1-step3000'; _STEPS = ['V1_read', 'V9_pack']; _CFG = {'V1_FRAC': 3}; MOCK_SMALL = False
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
    log(f"V6502_olmo7_stage1_step3000 {_s}: {_RES[_s]}"); beat(f"V6502_olmo7_stage1_step3000 {_s} done")
OUT = _BASE; save_json("summary.json", _RES); beat("V6502_olmo7_stage1_step3000 ok")
