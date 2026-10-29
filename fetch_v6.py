"""Reassemble GPU result packs fetched with the Drive connector. Each job's V9 step writes runner/packs/<job>__manifest.json and 9 MB
pieces <job>__c000.bin, ... ; the connector saves each download as tool-results/*download_file_content*.txt (JSON, base64 content).
Usage: python fetch_v6.py      -> decodes every saved piece/manifest into results_v6/packs/, reassembles complete jobs (sha256 checked)
                                  and unzips them into results_v6/<job>/ ; prints what is still missing."""
import json, glob, base64, os, zipfile, hashlib, io, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; RV = HERE / "results_v6"; PK = RV / "packs"; PK.mkdir(parents=True, exist_ok=True)
TR = os.environ.get("TOOL_RESULTS", "tool-results")   # REPOSITORY: folder of Drive downloads (was a local session folder)
for f in glob.glob(f"{TR}/mcp-739c3c66-*download_file_content*.txt"):
    try: d = json.load(open(f, encoding="utf-8"))
    except Exception: continue
    t = d.get("title", "")
    if re.match(r"V[678][0-9]", t) and (t.endswith(".bin") or t.endswith("__manifest.json") or t.endswith(".zip")):
        (PK / t).write_bytes(base64.b64decode(d["content"])); os.remove(f)
for mf in sorted(PK.glob("*__manifest.json")):
    m = json.load(open(mf)); out = RV / m["job"]
    if (out / ".complete").exists(): continue
    miss = [p["name"] for p in m["pieces"] if not (PK / p["name"]).exists()]
    if miss: print(m["job"], "missing", len(miss), "of", len(m["pieces"]), miss[:5]); continue
    data = b"".join((PK / p["name"]).read_bytes() for p in m["pieces"])
    if hashlib.sha256(data).hexdigest() != m["sha256"]: print(m["job"], "SHA MISMATCH"); continue
    out.mkdir(exist_ok=True); zipfile.ZipFile(io.BytesIO(data)).extractall(out); (out / ".complete").write_text("ok")
    print("reassembled", m["job"], f"{len(data) / 1e6:.1f} MB ->", sorted(x.name for x in out.iterdir() if x.is_dir()))
for z in PK.glob("*.zip"):                      # first smoke packs (single zip)
    out = RV / z.name.split("__p")[0]
    if not (out / ".complete").exists(): out.mkdir(exist_ok=True); zipfile.ZipFile(z).extractall(out); (out / ".complete").write_text("ok"); print("unzipped", z.name)
