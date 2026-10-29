# V9: pack this job's results for download: one zip of every step's summary, digest, csv and export files (the direction source
# dirsrc.pt, needed only on the GPU, is left out), split into 9 MB pieces (runner/packs/<job>__c000.bin, ...) so that the Drive connector
# (10 MB limit) can fetch each piece; <job>__manifest.json gives the piece names, sizes and the sha256 of the whole zip. fetch_v6.py
# reassembles, checks and unzips.
import hashlib
_job = os.path.basename(_BASE); PK = f"{ROOT}/packs"; os.makedirs(PK, exist_ok=True); TMPZ = f"/tmp/{_job}.zip" if os.path.isdir("/tmp") else f"{_BASE}.zip"
with zipfile.ZipFile(TMPZ, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(_BASE):
        for f in fs:
            if f == "dirsrc.pt" or root.endswith("V9_pack"): continue
            p = os.path.join(root, f); z.write(p, os.path.relpath(p, _BASE))
data = open(TMPZ, "rb").read(); CH = 9_000_000; pieces = []
for k in range(0, len(data), CH):
    nm = f"{_job}__c{k // CH:03d}.bin"; open(f"{PK}/{nm}", "wb").write(data[k:k + CH]); pieces.append(dict(name=nm, bytes=len(data[k:k + CH])))
man = {"job": _job, "zip_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "pieces": pieces}
json.dump(man, open(f"{PK}/{_job}__manifest.json", "w"), indent=1); os.remove(TMPZ)
save_json("summary.json", {"job": _job, "zip_mb": round(len(data) / 1e6, 1), "n_pieces": len(pieces)}); log(f"V9 packed {_job}: {len(data) / 1e6:.1f} MB in {len(pieces)} pieces"); beat("V9 ok")
