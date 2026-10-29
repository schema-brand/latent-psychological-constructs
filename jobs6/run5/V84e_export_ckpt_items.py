# V84e_export_ckpt_items: no model. Copies the Study 3 files (V3_exp items.csv, digest.md, summary.json) of the OLMo 3 7B checkpoint jobs
# (V844-V84c), whose full packs are about 89 MB each, into one small pack (runner/packs, V9 format) for the training analysis of Study 3.
# Runs after V84d and before V850. Jobs not yet finished are skipped (their files do not exist).
import zipfile, hashlib, glob
_job = "V84e_export_ckpt_items"; PK = f"{ROOT}/packs"; os.makedirs(PK, exist_ok=True); _z = f"/tmp/{_job}.zip"; _have = []
with zipfile.ZipFile(_z, "w", zipfile.ZIP_DEFLATED) as _Z:
    for _d in sorted(glob.glob(f"{ROOT}/results/V84[4-9a-c]_v3_olmo7_*")):
        _j = os.path.basename(_d)
        for _f in ["V3_exp/items.csv", "V3_exp/digest.md", "V3_exp/summary.json"]:
            _p = f"{_d}/{_f}"
            if os.path.exists(_p): _Z.write(_p, f"{_j}/{_f}"); _have.append(f"{_j}/{_f}")
_data = open(_z, "rb").read(); _CH = 9_000_000; _pieces = []
for _k in range(0, len(_data), _CH):
    _nm = f"{_job}__c{_k // _CH:03d}.bin"; open(f"{PK}/{_nm}", "wb").write(_data[_k:_k + _CH]); _pieces.append(dict(name=_nm, bytes=len(_data[_k:_k + _CH])))
json.dump({"job": _job, "zip_bytes": len(_data), "sha256": hashlib.sha256(_data).hexdigest(), "pieces": _pieces, "files": _have}, open(f"{PK}/{_job}__manifest.json", "w"), indent=1)
os.remove(_z); open(f"{OUT}/summary.json", "w").write(json.dumps({"files": _have, "zip_mb": round(len(_data) / 1e6, 2)}))
log(f"V84e exported {len(_have)} files, {len(_data) / 1e6:.1f} MB"); beat("V84e ok")
