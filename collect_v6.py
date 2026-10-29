"""Materials v6: collect writer outputs (runner packs downloaded from Drive + local Claude outputs), re-run the quality checks locally and
write flat tables. Usage:
  python collect_v6.py decode '{"<drive id>": "<pack name>.zip", ...}'   # decode saved Drive downloads into materials_v6/packs/
  python collect_v6.py pairs      # -> materials_v6/pairs_v6_raw.jsonl + qc_pairs_v6.json (per writer x construct)
  python collect_v6.py exp        # -> materials_v6/variants_v6_raw.jsonl + qc_exp_v6.json"""
import json, sys, glob, zipfile, base64, os, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"; PK = M6 / "packs"; PK.mkdir(exist_ok=True)
TR = os.environ.get("TOOL_RESULTS", "tool-results")   # REPOSITORY: folder of Drive downloads (was a local session folder)
L = {"OR_ROOT": str(HERE / "or_runner" / "local_root"), "OR_KEY": "x"}
exec(compile((HERE / "or_runner" / "or_runner_lib.py").read_text(encoding="utf-8"), "or_runner_lib.py", "exec"), L)

def decode(m):
    for f in glob.glob(f"{TR}/mcp-739c3c66-*download_file_content*.txt"):
        try: d = json.load(open(f, encoding="utf-8"))
        except Exception: continue
        if d.get("id") in m:
            (PK / m[d["id"]]).write_bytes(base64.b64decode(d["content"])); os.remove(f); print("decoded", m[d["id"]])

def runner_results(prefix):
    """Latest result per request id from all packs whose job starts with prefix."""
    R = {}
    for z in sorted(PK.glob("*.zip")):
        with zipfile.ZipFile(z) as Z:
            for n in Z.namelist():
                if not n.split("/")[0].startswith(prefix): continue
                for line in Z.read(n).decode("utf-8").splitlines():
                    r = json.loads(line)
                    if r["id"] not in R or r["t"] >= R[r["id"]]["t"]: R[r["id"]] = r
    return R

def claude_results(prefix, req_meta):
    R = {}
    for f in sorted((M6 / "claude_out").glob(f"claude__{prefix}*.json")):
        cid = f.stem; meta = req_meta(cid)
        try: obj = L["parse_json"](f.read_text(encoding="utf-8")); st = "ok"
        except Exception as e: obj, st = None, f"parse_error {e}"
        R[cid] = {"id": cid, "model": "claude-sonnet-local", "status": st, "parsed": obj, "meta": meta, "t": "local"}
    return R

import re as _re
_PAIR = _re.compile(r'"domain"\s*:\s*"(.*?)"\s*,\s*"high"\s*:\s*"(.*?)"\s*,\s*"low"\s*:\s*"(.*?)"\s*}', _re.S)
def repair_pairs(txt):
    """Writers sometimes leave unescaped quotes (dialogue) inside the JSON strings; recover the pairs field by field."""
    P = [{"domain": d, "high": h.replace('\\"', '"'), "low": l.replace('\\"', '"')} for d, h, l in _PAIR.findall(txt or "")]
    return {"pairs": P} if P else None

def pairs():
    spec = json.loads((M6 / "spec_v6.json").read_text(encoding="utf-8")); C = spec["constructs"]
    def meta_of(cid):
        w, k, lowi = cid.split("__"); low, i = lowi.split("_"); c = C[k]
        n = 8 if (c.get("v4_format") or c["role"] == "rival") else 6
        return {"writer": w, "construct": k, "role": c["role"], "low": low, "n": n, "banned": c["banned"], "domains": spec["domains"][w],
                "number_exempt": bool(c.get("number_exempt")), "prompt_version": "v6.1" if (k == "uniqueness" and i.startswith("u")) else "v6.0", "call": cid}
    R = runner_results("B") | runner_results("U") | claude_results("", lambda cid: meta_of(cid) if "__exp__" not in cid else None)
    rows, Q = [], collections.defaultdict(lambda: collections.Counter())
    for cid, r in R.items():
        meta = r.get("meta") or {}
        if not meta or "__exp__" in cid: continue
        obj = r.get("parsed")
        if obj is None and r.get("content"): obj = repair_pairs(r["content"]); r["repaired"] = obj is not None
        qc = L["qc_pairs"](obj, meta) if obj else {"items": [], "n": 0}
        key = f"{meta['writer']}|{meta['construct']}"; Q[key]["calls"] += 1; Q[key]["pairs"] += qc.get("n", 0); Q[key]["ok"] += qc.get("ok", 0)
        for j, (p, it) in enumerate(zip((obj or {}).get("pairs", []), qc["items"])):
            rows.append({"pid": f"{cid}__{j}", "writer": meta["writer"], "construct": meta["construct"], "role": meta["role"], "low_type": meta["low"],
                         "domain": p.get("domain"), "high": p.get("high"), "low": p.get("low"), "flags": it["flags"], "diff": it.get("diff"),
                         "hard_fail": any(f in L["HARD_FLAGS"] for f in it["flags"]), "call": cid, "prompt_version": meta.get("prompt_version")})
    with open(M6 / "pairs_v6_raw.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {k: dict(v, ok_rate=round(v["ok"] / v["pairs"], 2) if v["pairs"] else None) for k, v in sorted(Q.items())}
    json.dump(summ, open(M6 / "qc_pairs_v6.json", "w"), indent=1)
    flags = collections.Counter(f for r in rows for f in r["flags"])
    byw = collections.defaultdict(lambda: [0, 0])
    for r in rows: byw[r["writer"]][0] += 1; byw[r["writer"]][1] += not r["hard_fail"]
    print(len(rows), "pairs;", {w: f"{ok}/{n}" for w, (n, ok) in byw.items()}, "| flags", dict(flags.most_common(8)))
    weak = {k: v["ok_rate"] for k, v in summ.items() if v["pairs"] and v["ok_rate"] < 0.8}
    print("writer|construct below 80% ok:", len(weak), dict(sorted(weak.items(), key=lambda x: x[1])[:25]))

def exp():
    R = runner_results("C")
    A = json.loads((M6 / "exp_writer_assignment.json").read_text())
    CM = json.loads((M6 / "claude_exp_meta.json").read_text(encoding="utf-8"))
    def meta_of(cid): return CM[cid]
    R |= {k: v for k, v in claude_results("exp__", meta_of).items()}
    rows, Q = [], {}
    for cid, r in R.items():
        meta = r.get("meta") or {}; obj = r.get("parsed")
        qc = L["qc_variants"](obj, meta) if obj else {"items": [], "n": 0, "ok": 0}
        Q[cid] = {"status": r.get("status"), "n": qc.get("n"), "ok": qc.get("ok"), "flags": qc.get("flags")}
        for v, it in zip((obj or {}).get("variants", []), qc["items"]):
            rows.append(dict(v, experiment=meta["experiment"], writer=meta["writer"], call=cid, flags=it["flags"], hard_fail=bool([f for f in it["flags"] if f != "mental_state"])))
    with open(M6 / "variants_v6_raw.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(Q, open(M6 / "qc_exp_v6.json", "w"), indent=1)
    byw = collections.Counter(r["writer"] for r in rows if not r["hard_fail"]); allw = collections.Counter(r["writer"] for r in rows)
    print(len(rows), "variants; ok by writer", {w: f"{byw[w]}/{allw[w]}" for w in allw}, "| flags", dict(collections.Counter(f for r in rows for f in r["flags"]).most_common(8)))

if __name__ == "__main__":
    m = sys.argv[1]
    if m == "decode": decode(json.loads(sys.argv[2]))
    elif m == "pairs": pairs()
    elif m == "exp": exp()
