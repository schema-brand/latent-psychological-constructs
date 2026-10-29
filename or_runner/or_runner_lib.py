"""OpenRouter job runner for the latent study's materials (OPENROUTER_PLAN_2026-10-19.md). Runs on a Colab CPU runtime, or locally for tests.

Layout under ROOT:
  jobs/<job>.jsonl       one request per line: {"id", "model", "messages" | "prompt", "params", "check", "meta"}
                         check = "pairs" | "ratings" | "json" | "text"; meta carries what the quality checks need (n, banned, ids, scales, ...)
  out/<job>/part-*.jsonl results in small part files, each closed after writing; out/<job>/DONE once every request has a final result
  control.json           {"state": "run" | "pause" | "stop", "budget_usd": 58, "concurrency": {"default": 6, "<model>": n},
                          "skip_jobs": [], "only_jobs": [], "retry_failed": false, "exit_when_idle": false}
  status.json            heartbeat every 30 s (progress, spend, latency, errors, quality); log.txt
The notebook sets OR_ROOT and OR_KEY (from Colab Secrets) before exec. The key is never printed or written."""
import os, re, json, time, glob, random, difflib, threading, queue, hashlib, traceback, urllib.request, urllib.error
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor

ROOT = globals().get("OR_ROOT", os.environ.get("OR_ROOT", "or_runner_local"))
KEY = globals().get("OR_KEY") or os.environ.get("OR_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
LIB_SHA = globals().get("OR_LIB_SHA", "local")
DEFAULT_CONTROL = {"state": "run", "budget_usd": 58.0, "concurrency": {"default": 6}, "skip_jobs": [], "only_jobs": [],
                   "retry_failed": False, "exit_when_idle": False, "max_attempts": 3, "timeout_s": 600}
for d in ("jobs", "out"): os.makedirs(f"{ROOT}/{d}", exist_ok=True)

def now(): return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

_loglock = threading.Lock()
def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with _loglock:
        with open(f"{ROOT}/log.txt", "a", encoding="utf-8") as f: f.write(line + "\n")

def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f: return json.load(f)
    except Exception: return default

def write_json(path, obj):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError:
        with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=1)

# ------------------------------------------------------------------ quality checks (also imported locally for analysis)
MENTAL = re.compile(r"\byou(?:'re| are)? (?:feel|felt|feeling|like|liked|love|loved|think|thought|remember|remembered|notice|noticed|realize|realise|realized|"
                    r"wonder|want|wanted|wish|hope|know|knew|sense|sensed|enjoy|enjoyed|believe|recall|decide)\b|\bit seems\b|\bof course\b|\bnaturally\b|"
                    r"\b(?:imagine|suppose|picture|pretend)\b", re.I)
STUDYW = re.compile(r"\b(?:stud(?:y|ies)|survey|questionnaire|researcher|experiment|laborator(?:y|ies)|lab|participant|market research)\b", re.I)
NUM = re.compile(r"\d|\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|half|dozen)\b", re.I)

def banned_regex(words):
    pats = [r"\b" + re.escape(w.rstrip("*")).replace(r"\ ", r"\s+") + (r"\w*" if w.endswith("*") else r"\b") for w in words if w.strip()]
    return re.compile("|".join(pats), re.I) if pats else None

def parse_json(txt):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (txt or "").strip(), flags=re.S)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i: raise ValueError("no JSON object")
    return json.loads(t[i:j + 1])

def _words(s): return re.findall(r"[A-Za-z0-9'’-]+", s or "")
def _last_sentence(s):
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", (s or "").strip()) if p.strip()]
    return parts[-1] if parts else ""

HARD_FLAGS = ("missing_text", "words_out_of_range", "length_mismatch", "banned", "mental_state", "study_frame", "number_rule", "diff_too_small", "diff_too_large")

def qc_pair(high, low, meta):
    fl = []
    if not (high or "").strip() or not (low or "").strip(): return ["missing_text"], {}
    wh, wl = len(_words(high)), len(_words(low))
    lo, hi = meta.get("words", [35, 95])
    if not (lo <= wh <= hi and lo <= wl <= hi): fl.append("words_out_of_range")
    if abs(wh - wl) / max(wh, wl) > meta.get("len_tol", 0.15): fl.append("length_mismatch")
    br = banned_regex(meta.get("banned", []))
    hits = sorted({m.group(0).lower() for s in (high, low) for m in (br.finditer(s) if br else [])})
    if hits: fl.append("banned")
    if MENTAL.search(high) or MENTAL.search(low): fl.append("mental_state")
    if STUDYW.search(high) or STUDYW.search(low): fl.append("study_frame")
    if not meta.get("number_exempt") and bool(NUM.search(high)) != bool(NUM.search(low)): fl.append("number_rule")
    a, b = [w.lower() for w in _words(high)], [w.lower() for w in _words(low)]
    diff = 1 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    if diff < meta.get("diff_min", 0.03): fl.append("diff_too_small")
    if diff > meta.get("diff_max", 0.65): fl.append("diff_too_large")
    if _last_sentence(high) != _last_sentence(low): fl.append("ending_differs")
    return fl, {"wh": wh, "wl": wl, "diff": round(diff, 3), "banned_hits": hits}

def qc_pairs(obj, meta):
    P = obj.get("pairs") if isinstance(obj, dict) else None
    if not isinstance(P, list): return {"n": 0, "n_expected": meta.get("n"), "ok": 0, "flags": {"no_pairs": 1}, "items": []}
    flags, items, ok = Counter(), [], 0
    doms = [p.get("domain") for p in P if isinstance(p, dict)]
    for p in P:
        if not isinstance(p, dict): flags["bad_item"] += 1; continue
        fl, info = qc_pair(p.get("high"), p.get("low"), meta)
        flags.update(fl); hard = [f for f in fl if f in HARD_FLAGS]; ok += not hard
        items.append({"flags": fl, **info})
    out = {"n": len(P), "n_expected": meta.get("n"), "ok": ok, "flags": dict(flags), "items": items}
    if meta.get("domains"): out["domain_mismatch"] = sum(d not in meta["domains"] for d in doms)
    if len(set(doms)) < len(doms): out["dup_domains"] = len(doms) - len(set(doms))
    return out

def qc_ratings(obj, meta):
    R = obj.get("ratings") if isinstance(obj, dict) else None
    if not isinstance(R, list): return {"n": 0, "missing": len(meta.get("ids", [])), "bad_values": 0}
    ids, scales = meta.get("ids", []), meta.get("scales", [])
    got = {str(r.get("id")): r for r in R if isinstance(r, dict)}
    missing = [i for i in ids if i not in got]; bad = 0; vals = Counter()
    for i in ids:
        r = got.get(i) or {}
        for s in scales:
            v = r.get(s)
            if not isinstance(v, int) or not 1 <= v <= 7: bad += 1
            else: vals[v] += 1
    return {"n": len(R), "missing": len(missing), "bad_values": bad, "values": dict(sorted(vals.items()))}

def qc_variants(obj, meta):
    """Everyday rewordings of a published experiment: each variant has treatment and control texts that differ only in the manipulation."""
    V = obj.get("variants") if isinstance(obj, dict) else None
    if not isinstance(V, list): return {"n": 0, "n_expected": meta.get("n"), "ok": 0, "flags": {"no_variants": 1}, "items": []}
    flags, items, ok = Counter(), [], 0; br = banned_regex(meta.get("banned", []))
    for v in V:
        fl = []
        if not isinstance(v, dict): flags["bad_item"] += 1; continue
        bt, bc, q = v.get("body_treatment") or "", v.get("body_control") or "", v.get("question") or ""
        if not bt.strip() or not bc.strip() or not q.strip(): fl.append("missing_text")
        if meta.get("moderated") and not ((v.get("body_treatment_B") or "").strip() and (v.get("body_control_B") or "").strip()): fl.append("missing_condition_B")
        for key, body in (("x_span_treatment", bt), ("x_span_control", bc)):
            spans = v.get(key) or []
            if not spans or any(sp and sp not in body for sp in spans): fl.append("span_mismatch")
        a, b = [w.lower() for w in _words(bt)], [w.lower() for w in _words(bc)]
        diff = 1 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 1
        if diff > meta.get("diff_max", 0.5): fl.append("diff_too_large")
        if diff < 0.01: fl.append("diff_too_small")
        text = " ".join([bt, bc, q, v.get("body_treatment_B") or "", v.get("body_control_B") or ""])
        body = " ".join([bt, bc, v.get("body_treatment_B") or "", v.get("body_control_B") or ""])
        for sp in (v.get("x_span_treatment") or []) + (v.get("x_span_control") or []):
            if sp: body = body.replace(sp, " ")   # the rules allow manipulation words inside the manipulated passages; the question is the response measure
        if br and br.search(body): fl.append("banned")
        if STUDYW.search(text): fl.append("study_frame")
        if MENTAL.search(bt) or MENTAL.search(bc): fl.append("mental_state")
        if not re.search(r"(?<!\d)1(?!\d)", q) or not re.search(r"(?<!\d)9(?!\d)", q): fl.append("question_not_1_9")
        flags.update(set(fl)); hard = [f for f in fl if f != "mental_state"]; ok += not hard
        items.append({"vid": v.get("vid"), "flags": sorted(set(fl)), "words_t": len(_words(bt)), "diff": round(diff, 3)})
    return {"n": len(V), "n_expected": meta.get("n"), "ok": ok, "flags": dict(flags), "items": items}

def run_qc(check, obj, meta):
    if check == "variants": return qc_variants(obj, meta)
    if check == "pairs": return qc_pairs(obj, meta)
    if check == "ratings": return qc_ratings(obj, meta)
    return {}

def qc_pass(check, qc):
    if check == "variants": return qc.get("n") == qc.get("n_expected") and qc.get("ok", 0) >= max(1, int(0.8 * (qc.get("n_expected") or 1)))
    if check == "pairs": return qc.get("n") == qc.get("n_expected") and qc.get("ok", 0) >= max(1, int(0.8 * (qc.get("n_expected") or 1)))
    if check == "ratings": return qc.get("missing", 1) == 0 and qc.get("bad_values", 1) == 0
    return True

# ------------------------------------------------------------------ one request
RETRY_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}

def _post(body, timeout):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json", "X-Title": "MVCA latent materials"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)

def execute(job, req, ctl):
    t0 = time.time(); params = dict(req.get("params") or {})
    msgs = req.get("messages") or [{"role": "user", "content": req["prompt"]}]
    body = {"model": req["model"], "messages": msgs, "usage": {"include": True}, **params}
    check = req.get("check", "text"); meta = req.get("meta") or {}
    res = {"id": req["id"], "job": job, "model": req["model"], "t": now(), "attempts": 0, "status": "error", "cost": 0.0, "usage_all": []}
    max_att = int(ctl.get("max_attempts", 3)); cap = int(meta.get("max_tokens_cap", 64000)); parse_retry = 1
    while res["attempts"] < max_att + parse_retry:
        res["attempts"] += 1
        try:
            d = _post(body, ctl.get("timeout_s", 900))
            if "error" in d and not d.get("choices"): raise RuntimeError(f"api error: {str(d['error'])[:300]}")
            ch = d["choices"][0]; txt = (ch.get("message") or {}).get("content") or ""; u = d.get("usage") or {}
            c = float(u.get("cost") or 0); res["cost"] += c
            res["usage_all"].append({"in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                                     "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "cost": c,
                                     "finish": ch.get("finish_reason"), "provider": d.get("provider")})
            res["finish"] = ch.get("finish_reason"); res["content"] = txt
            if ch.get("finish_reason") == "length" or not txt.strip():
                res["status"] = "length" if ch.get("finish_reason") == "length" else "empty"
                if body.get("max_tokens"): body["max_tokens"] = min(cap, int(body["max_tokens"]) * 2)
                continue
            if check in ("pairs", "ratings", "json", "variants"):
                try: obj = parse_json(txt)
                except Exception as e:
                    res["status"] = "parse_error"; res["error"] = str(e)[:200]
                    if parse_retry > 0: parse_retry -= 1; continue
                    break
                res["parsed"] = obj; res["qc"] = run_qc(check, obj, meta); res["qc_pass"] = qc_pass(check, res["qc"])
            res["status"] = "ok"; res.pop("error", None); break
        except urllib.error.HTTPError as e:
            msg = ""
            try: msg = e.read().decode("utf-8", "replace")[:300]
            except Exception: pass
            res["status"] = "error"; res["error"] = f"HTTP {e.code} {msg}"
            if e.code not in RETRY_HTTP: break
            time.sleep(min(120, 8 * 2 ** (res["attempts"] - 1)) + random.random() * 3)
        except Exception as e:
            res["status"] = "error"; res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(min(120, 8 * 2 ** (res["attempts"] - 1)) + random.random() * 3)
    u = res["usage_all"]
    res["in"] = sum(x["in"] or 0 for x in u); res["out"] = sum(x["out"] or 0 for x in u); res["reasoning"] = sum(x["reasoning"] or 0 for x in u)
    res["latency"] = round(time.time() - t0, 1); res["meta"] = meta
    return res

# ------------------------------------------------------------------ the runner
class Runner:
    def __init__(self):
        self.ctl = dict(DEFAULT_CONTROL, **read_json(f"{ROOT}/control.json", {}))
        if not os.path.exists(f"{ROOT}/control.json"): write_json(f"{ROOT}/control.json", self.ctl)
        self.jobs = {}                       # job -> {"total", "ids": set, "final": set, "loaded": True}
        self.pending = defaultdict(deque)    # model -> deque[(job, req)]
        self.inflight = Counter(); self.pools = {}; self.pool_size = {}
        self.results = queue.Queue(); self.buf = defaultdict(list); self.buf_t = {}
        self.S = {"jobs": defaultdict(lambda: defaultdict(float)), "models": defaultdict(lambda: defaultdict(float)),
                  "constructs": defaultdict(lambda: defaultdict(float)), "flags": defaultdict(Counter), "finish": defaultdict(Counter),
                  "errors": deque(maxlen=12)}
        self.spent = 0.0; self.t_start = time.time(); self.credits = None; self.t_credit = 0; self.state = self.ctl["state"]

    # -- loading
    def _done_ids(self, job):
        done, final_ok = set(), set()
        for f in glob.glob(f"{ROOT}/out/{job}/part-*.jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try: r = json.loads(line)
                    except Exception: continue
                    done.add(r["id"]); self.spent += float(r.get("cost") or 0); self._stat(r, replay=True)
                    if r.get("status") == "ok": final_ok.add(r["id"])
        return done, final_ok

    def discover(self):
        ctl = self.ctl
        for path in sorted(glob.glob(f"{ROOT}/jobs/*.jsonl")):
            job = os.path.basename(path)[:-6]
            if job in self.jobs or job in ctl.get("skip_jobs", []) or (ctl.get("only_jobs") and job not in ctl["only_jobs"]): continue
            reqs = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip(): reqs.append(json.loads(line))
            os.makedirs(f"{ROOT}/out/{job}", exist_ok=True)
            done, ok = self._done_ids(job)
            redo = done - ok if ctl.get("retry_failed") else set()
            todo = [r for r in reqs if r["id"] not in done or r["id"] in redo]
            self.jobs[job] = {"total": len(reqs), "final": done - redo, "ids": {r["id"] for r in reqs}}
            for r in todo: self.pending[r["model"]].append((job, r))
            log(f"job {job}: {len(reqs)} requests, {len(done)} already done, {len(todo)} queued")

    # -- dispatch
    def _pool(self, model):
        n = int(self.ctl["concurrency"].get(model, self.ctl["concurrency"].get("default", 6)))
        if self.pool_size.get(model) != n:
            self.pools[model] = ThreadPoolExecutor(max_workers=n, thread_name_prefix=model.split("/")[-1][:12]); self.pool_size[model] = n
        return self.pools[model], n

    def dispatch(self):
        if self.state != "run": return
        for model, dq in self.pending.items():
            pool, n = self._pool(model)
            while dq and self.inflight[model] < n:
                if self.spent >= float(self.ctl.get("budget_usd", 58)):
                    if self.state != "budget_hold": log(f"budget reached (${self.spent:.2f}); holding")
                    self.state = "budget_hold"; return
                job, req = dq.popleft(); self.inflight[model] += 1
                fut = pool.submit(execute, job, req, dict(self.ctl))
                fut.add_done_callback(lambda f, m=model, j=job, r=req: self.results.put((m, j, r, f)))

    # -- results
    def _stat(self, r, replay=False):
        J = self.S["jobs"][r["job"]]; M = self.S["models"][r["model"]]
        for X in (J, M):
            X["done"] += 1; X["cost"] += float(r.get("cost") or 0); X["latency"] += float(r.get("latency") or 0)
            X["out_tokens"] += r.get("out") or 0; X["reasoning"] += r.get("reasoning") or 0
            X[f"status_{r.get('status')}"] += 1
            if r.get("status") == "ok" and r.get("qc_pass") is False: X["qc_fail"] += 1
        self.S["finish"][r["model"]][str(r.get("finish"))] += 1
        qc = r.get("qc") or {}
        if "items" in qc:
            J["pairs"] += qc.get("n", 0); J["pairs_ok"] += qc.get("ok", 0)
            self.S["flags"][r["job"]].update(qc.get("flags", {}))
            c = (r.get("meta") or {}).get("construct")
            if c:
                C = self.S["constructs"][f"{(r.get('meta') or {}).get('writer', r['model'])}|{c}"]; C["pairs"] += qc.get("n", 0); C["pairs_ok"] += qc.get("ok", 0)
        if "missing" in qc:
            J["ratings_missing"] += qc.get("missing", 0); J["ratings_bad"] += qc.get("bad_values", 0)
        if r.get("status") != "ok" and not replay: self.S["errors"].append(f"{r['t']} {r['model']} {r['id']}: {r.get('status')} {str(r.get('error'))[:160]}")

    def collect(self):
        while True:
            try: model, job, req, fut = self.results.get_nowait()
            except queue.Empty: break
            self.inflight[model] -= 1
            try: r = fut.result()
            except Exception as e:
                r = {"id": req["id"], "job": job, "model": model, "t": now(), "status": "error", "error": f"runner: {e}", "cost": 0.0, "meta": req.get("meta")}
            self.spent += float(r.get("cost") or 0); self._stat(r)
            self.buf[job].append(r); self.buf_t.setdefault(job, time.time()); self.jobs[job]["final"].add(r["id"])

    def flush(self, force=False):
        for job, rows in list(self.buf.items()):
            if rows and (force or len(rows) >= 50 or time.time() - self.buf_t.get(job, 0) > 120):
                path = f"{ROOT}/out/{job}/part-{int(time.time()*1000)}-{random.randint(0, 9999):04d}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush(); os.fsync(f.fileno())
                self.buf[job] = []; self.buf_t.pop(job, None)
        for job, J in self.jobs.items():
            if not self.buf.get(job) and len(J["final"] & J["ids"]) >= J["total"] and not os.path.exists(f"{ROOT}/out/{job}/DONE"):
                with open(f"{ROOT}/out/{job}/DONE", "w") as f: f.write(now() + "\n")
                log(f"job {job} DONE ({J['total']} requests)")

    # -- status
    def credits_info(self):
        if time.time() - self.t_credit < 300: return self.credits
        self.t_credit = time.time()
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": "Bearer " + KEY})
            d = json.load(urllib.request.urlopen(req, timeout=30)).get("data", {})
            self.credits = {"total_credits": d.get("total_credits"), "total_usage": d.get("total_usage"),
                            "remaining": round((d.get("total_credits") or 0) - (d.get("total_usage") or 0), 3)}
        except Exception as e: self.credits = {"error": str(e)[:120]}
        return self.credits

    def status(self):
        def summ(X):
            n = max(1.0, X["done"]); out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in X.items()}
            out["mean_latency"] = round(X["latency"] / n, 1); out["cost_per_call"] = round(X["cost"] / n, 5)
            return out
        jobs = {}
        for j, J in self.jobs.items():
            s = summ(self.S["jobs"][j]); s.update(total=J["total"], final=len(J["final"] & J["ids"]), flags=dict(self.S["flags"][j]))
            if s.get("pairs"): s["pairs_ok_rate"] = round(s["pairs_ok"] / s["pairs"], 3)
            jobs[j] = s
        weak = {k: round(v["pairs_ok"] / v["pairs"], 2) for k, v in self.S["constructs"].items() if v["pairs"] >= 4 and v["pairs_ok"] / v["pairs"] < 0.8}
        st = {"t": now(), "lib_sha": LIB_SHA, "state": self.state, "uptime_min": round((time.time() - self.t_start) / 60, 1),
              "spent_usd": round(self.spent, 4), "budget_usd": self.ctl.get("budget_usd"), "credits": self.credits_info(),
              "queued": {m: len(q) for m, q in self.pending.items() if q}, "inflight": {m: n for m, n in self.inflight.items() if n},
              "concurrency": self.pool_size, "models": {m: summ(X) for m, X in self.S["models"].items()},
              "finish": {m: dict(c) for m, c in self.S["finish"].items()}, "jobs": jobs, "weak_writer_constructs": weak,
              "recent_errors": list(self.S["errors"])}
        write_json(f"{ROOT}/status.json", st)
        if time.time() - getattr(self, "t_line", 0) > 120:
            self.t_line = time.time()
            js = " | ".join(f"{j}: {v['final']}/{v['total']}" + (f" pairs_ok {v.get('pairs_ok_rate')}" if v.get("pairs") else "") for j, v in jobs.items() if v["final"] < v["total"] or j == max(jobs, default=""))
            ms = " ".join(f"{m.split('/')[-1][:10]}={int(X['done'])}/{X['cost_per_call']:.4f}/{X['mean_latency']}s" for m, X in st["models"].items())
            print(f"[{st['t']}] {self.state} spent ${st['spent_usd']:.3f} credits ${(st['credits'] or {}).get('remaining')} queued {sum(st['queued'].values())} inflight {sum(st['inflight'].values())} || {js} || {ms} || errors {len(st['recent_errors'])}", flush=True)
        return st

    # -- packing results for download (control.json: {"pack": {"tag": "<tag>", "jobs": ["<job>", ...] or "all"}})
    def pack(self):
        P = self.ctl.get("pack") or {}
        tag = P.get("tag")
        if not tag: return
        import zipfile
        os.makedirs(f"{ROOT}/packs", exist_ok=True)
        jobs = sorted(self.jobs) if P.get("jobs") in (None, "all") else P["jobs"]
        for job in jobs:
            path = f"{ROOT}/packs/{tag}__{job}.zip"
            if os.path.exists(path) or not os.path.isdir(f"{ROOT}/out/{job}"): continue
            with zipfile.ZipFile(path + ".tmp", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                for f in sorted(glob.glob(f"{ROOT}/out/{job}/part-*.jsonl")): z.write(f, f"{job}/{os.path.basename(f)}")
            os.replace(path + ".tmp", path); log(f"packed {path} ({os.path.getsize(path)//1024} KB)")

    # -- loop
    def run(self):
        log(f"runner start; lib {LIB_SHA}; root {ROOT}"); self.discover(); t_ctl = t_disc = t_stat = 0
        while True:
            t = time.time()
            if t - t_ctl > 10:
                self.ctl = dict(DEFAULT_CONTROL, **read_json(f"{ROOT}/control.json", {})); t_ctl = t
                want = self.ctl.get("state", "run")
                if want != self.state and not (self.state == "budget_hold" and self.spent >= float(self.ctl.get("budget_usd", 58))):
                    log(f"state {self.state} -> {want}"); self.state = want
            if t - t_disc > 30:
                self.discover(); t_disc = t
                try:
                    if (self.ctl.get("pack") or {}).get("tag"): self.flush(force=True); self.pack()
                except Exception as e: log(f"pack error: {e}")
            self.collect(); self.dispatch(); self.flush()
            if t - t_stat > 30: self.status(); t_stat = t
            busy = sum(self.inflight.values()); queued = sum(len(q) for q in self.pending.values())
            if self.state == "stop" and busy == 0: break
            if self.ctl.get("exit_when_idle") and busy == 0 and queued == 0 and not any(self.buf.values()): break
            time.sleep(1)
        self.collect(); self.flush(force=True); self.status(); log("runner stopped")

def run_forever():
    assert KEY, "OR_KEY is not set"
    Runner().run()
