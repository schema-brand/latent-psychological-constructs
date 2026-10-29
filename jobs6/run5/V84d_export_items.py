# V84d_export_items: no model. Copies the Study 3 item files (V3_exp items.csv, digest.md, summary.json) of Gemma 4 26B-A4B (V720) and
# Granite 4.2 30B (V839), whose full packs (219 MB, 90 MB) are too large to fetch, into one small pack (runner/packs, V9 format) so that the
# Study 3 analyses can include all fourteen instruction-tuned LLMs. Runs after the checkpoint series (V844-V84c) and before V850.
import zipfile, hashlib
_job = "V84d_export_items"; PK = f"{ROOT}/packs"; os.makedirs(PK, exist_ok=True); _z = f"/tmp/{_job}.zip"; _have = []
with zipfile.ZipFile(_z, "w", zipfile.ZIP_DEFLATED) as _Z:
    for _j in ["V720_gemma4_26b_a4b", "V839_v3sweep_granite42"]:
        for _f in ["V3_exp/items.csv", "V3_exp/digest.md", "V3_exp/summary.json"]:
            _p = f"{ROOT}/results/{_j}/{_f}"
            if os.path.exists(_p): _Z.write(_p, f"{_j}/{_f}"); _have.append(f"{_j}/{_f}")
_data = open(_z, "rb").read(); _CH = 9_000_000; _pieces = []
for _k in range(0, len(_data), _CH):
    _nm = f"{_job}__c{_k // _CH:03d}.bin"; open(f"{PK}/{_nm}", "wb").write(_data[_k:_k + _CH]); _pieces.append(dict(name=_nm, bytes=len(_data[_k:_k + _CH])))
json.dump({"job": _job, "zip_bytes": len(_data), "sha256": hashlib.sha256(_data).hexdigest(), "pieces": _pieces, "files": _have}, open(f"{PK}/{_job}__manifest.json", "w"), indent=1)
os.remove(_z); open(f"{OUT}/summary.json", "w").write(json.dumps({"files": _have, "zip_mb": round(len(_data) / 1e6, 2)}))
log(f"V84d exported {len(_have)} files, {len(_data) / 1e6:.1f} MB"); beat("V84d ok")
