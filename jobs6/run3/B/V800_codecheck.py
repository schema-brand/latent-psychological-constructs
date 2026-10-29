# V6c0: what the runner sees on Drive right now (Colab's Drive mount can serve a replaced file's old copy for several minutes):
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
