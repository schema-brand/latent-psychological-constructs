# lib_v6.py: shared code for the formal runs on materials v6 (PAPER_OUTLINE_v1_2026-10-19.md; GPU_PLAN_v6.md). Exec'd by every V job
# after runner_lib, latent_lib, test_lib3 and test_lib4. Adds: the v6 materials (pairs, experiments with per-agent rendering, agents and
# eligibility, judgment items, comparison constructs), a loader that also takes checkpoint revisions, the forward-pass routines that read
# token groups (mean, last, union of spans, decision token) averaged over layer groups and edit the states at chosen token positions
# (run_rows6, batched by token length, from run 2 on; run_marks6_b1 = the run 1 one-at-a-time routine), direction utilities on the v6
# direction source (cells = writer x construct x low-pole type), and small statistics helpers.
import contextlib, difflib, zipfile
M6 = f"{ROOT}/materials/v6"
if "_V6" not in globals(): _V6 = {}
def mat6(name):
    if name not in _V6: _V6[name] = json.load(open(f"{M6}/{name}", encoding="utf-8"))
    return _V6[name]
def C6(k, d=None): return globals().get("_CFG", {}).get(k, d)          # config of the current job only (globals persist across jobs)
def spec6(): return mat6("spec_v6.json")["constructs"]
def cons6(): return sorted(spec6())
def comps6(): return mat6("comparisons_v6.json")
def judge_items6(): return mat6("judgment_items_v6.json")
def agents6(): return mat6("agents_v6.json")["agents"]
def sgn(s):
    try: return int(float(str(s).replace("+", ""))) if str(s).strip() not in ("", "None") else 0
    except ValueError: return 0
def grid6(nL=None):
    nL = nL or CUR["n_layers"]; return sorted({int(round(f * nL)) for f in (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)})

def pairs6(small=False, frac=1, writers=None):
    """Final v6 pairs. frac = k keeps every k-th pair within each writer x construct cell (checkpoint reads); small keeps SMALL_N (1) per cell."""
    P = mat6("pairs_v6.json"); out, seen = [], {}
    for p in P:
        if writers and p["writer"] not in writers: continue
        k = (p["writer"], p["construct"]); n = seen.get(k, 0); seen[k] = n + 1
        if small and n >= C6("SMALL_N", 1): continue
        if frac > 1 and n % frac: continue
        out.append(dict(p, high=p["high"].strip(), low=p["low"].strip()))     # some chat templates trim the turn; spans must match
    return out

def model_key(mid, rev=None): return f"{mid}@{rev}" if rev else mid
def load6(mid, rev=None):
    """load_test_model with an optional revision (training checkpoints). CUR['id'] = model_key(mid, rev)."""
    key = model_key(mid, rev)
    if CUR.get("id") == key: return {"reused": True}
    if rev is None and "MOCK_LOAD" not in globals(): d = load_test_model(mid); return d
    unload(); JL.clear(); diag = {"mid": mid, "rev": rev}; t0 = time.time()
    if "MOCK_LOAD" in globals(): model, tok, how = MOCK_LOAD(mid)
    else:
        tok = AutoTokenizer.from_pretrained(mid, token=_tok, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(mid, token=_tok, revision=rev, device_map="cuda", torch_dtype=torch.bfloat16); how = "causal"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model.eval()
    for p_ in model.parameters(): p_.requires_grad_(False)
    lname, layers = find_layers(model); cfg = getattr(model.config, "text_config", None) or model.config
    CUR.update(id=key, model=model, tok=tok, layers=layers, n_layers=len(layers), d=cfg.hidden_size, prec="bf16")
    CUR["digits"] = [tok.encode(str(k), add_special_tokens=False) for k in range(1, 10)]; CUR.pop("_digits_for", None); calibrate_digits()
    diag.update(how=how, n_layers=len(layers), d=cfg.hidden_size, load_s=round(time.time() - t0), chat_template=bool(getattr(tok, "chat_template", None)))
    log(f"loaded {key} via {how} in {diag['load_s']}s; {len(layers)} layers; d={cfg.hidden_size}"); return diag

def chat6(user, system=None, gen=True):
    """chat() for instruction models; plain text for checkpoints without a chat template (same format at every checkpoint)."""
    if not getattr(CUR["tok"], "chat_template", None):
        return ((system + "\n\n") if system else "") + user + ("\n\nAnswer: " if gen else "")
    return chat(user, system, gen)
JUDGE_FMT6 = " Answer on a scale from 1 (not at all) to 9 (extremely)." + FMT9
def judge6(text, c, persona=None):
    return chat6(text + "\n\n" + judge_items6()[c] + JUDGE_FMT6, (SYS + persona) if persona else None)

# ---------- experiments: rendering per agent, items ----------
class _Keep(dict):
    def __missing__(self, k): return "{" + k + "}"
def fills6(eid, agent):
    el = mat6("agents_v6.json")["eligibility"].get(eid, {}); f = el.get("profile_fills")
    return (f or {}).get(agent.get("gender")) if f and agent else None
def render6(eid, v, field, agent):
    t = v.get(field + "_template") if str(v.get("templated")) == "True" else None
    t = t if t is not None else v.get(field)
    if t is None: return None
    if "{" in t:
        f = fills6(eid, agent)
        if f:
            try: t = t.format_map(_Keep(f))
            except (ValueError, IndexError): pass
    return t
def eligible6(eid):
    ids = mat6("agents_v6.json")["eligibility"].get(eid, {}).get("agents"); A = agents6()
    return [a for a in A if a["id"] in ids] if ids else list(A)
def diff_spans(a, b):
    """Word-level passages of text a that differ from text b (used when a listed manipulation span is not in the rendered text)."""
    wa = [(m.start(), m.end()) for m in re.finditer(r"\S+", a)]; wb = re.findall(r"\S+", b); out = []
    sm = difflib.SequenceMatcher(None, [a[s:e] for s, e in wa], wb, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal" and i2 > i1: out.append(a[wa[i1][0]:wa[i2 - 1][1]])
    return [s for s in out if len(s.strip()) >= 3]
def exp_items6(eids=None, nv=None, na=None, arms=("treatment", "control"), levels=("A", "B")):
    """One item per experiment x variant (v00 first, then rewordings) x condition level x eligible agent x arm. Level B exists only
    for moderated experiments. xs = manipulation spans found in the rendered body (else the passages that differ from the other arm)."""
    E = mat6("experiments_v6.json"); rows = []
    for eid, e in E.items():
        if eids and eid not in eids: continue
        vs = e["variants"][:nv] if nv else e["variants"]; ag = eligible6(eid); ag = ag[:na] if na else ag
        for v in vs:
            for lev in levels:
                suf = "" if lev == "A" else "_B"
                if lev == "B" and "body_treatment_B" not in v: continue
                for a in ag:
                    for arm in arms:
                        oth = "control" if arm == "treatment" else "treatment"
                        body = render6(eid, v, "body_" + arm + suf, a).strip(); other = (render6(eid, v, "body_" + oth + suf, a) or "").strip()
                        xs_raw = v.get("x_span_" + arm + suf) if v.get("x_span_" + arm + suf) is not None else v.get("x_span_" + arm)
                        xs = [render6(eid, {"t": x}, "t", a) for x in as_list(xs_raw)]; xs = [x for x in xs if x and x in body]
                        if not xs and other and other != body: xs = diff_spans(body, other)
                        q = v["question"].strip()
                        rows.append(dict(exp=eid, vid=v["vid"], level=lev, cond_level=v.get("condition_level" + suf), arm=arm, agent=a["id"], body=body, q=q, xs=xs,
                                         writer=v.get("writer", "mvca"), prompt=chat6(body + "\n\n" + q + FMT9, SYS + a["persona"])))
    return rows

# ---------- forward passes ----------
def run_marks6(prompts, marks=None, groups=None, prefix=None, hook_fn=None, tag=None):
    """Unedited reads go through the batched routine run_rows6; a hook_fn (per-prompt context manager) keeps the one-at-a-time path."""
    if hook_fn is None: return run_rows6(prompts, marks, groups, prefix, tag)
    return run_marks6_b1(prompts, marks, groups, prefix, hook_fn, tag)

# ---------- batched forward passes (v6.1, 20 Oct): rows of identical token length run side by side ----------
# No padding: a row in a batch sees exactly the tokens it would see alone, and every edit acts on its own row only. Each row is a prompt
# or (prompt, edits); an edit is ("add", l, vec, where) or ("proj", {l: U}, where, {l: mu} or None), where = None (every position), a
# substring (its tokens), a list of substrings (union) or ("from", substring) (from its first token to the end of the prompt).
TOKBUDGET, MAXROWS = 8192, 128
# Live tuning and load monitor: runner/tune.json ({"TOKBUDGET": ..., "MAXROWS": ...}, re-read every 60 s) overrides the job config, so the
# batch size can be raised or lowered while a job runs; every 30 s runner/gpu.json (and a line of runner/gpu_log.csv) records the job,
# rows and tokens per second, the batch size, peak memory and the GPU's utilization (mean of nvidia-smi samples every 2 s).
_MON = {"t": 0.0, "rows": 0, "tok": 0, "oom": 0, "tune_t": 0.0, "tune": {}, "util": []}
def _tune6():
    if time.time() - _MON["tune_t"] > 60:
        _MON["tune_t"] = time.time()
        try: _MON["tune"] = json.load(open(f"{ROOT}/tune.json"))
        except Exception: _MON["tune"] = {}
    return _MON["tune"]
def _gpu_sampler6():
    while True:
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().split(",")
            _MON["util"].append(int(q[0])); _MON["mem"] = (round(int(q[1]) / 1024, 1), round(int(q[2]) / 1024, 1)); _MON["util"] = _MON["util"][-200:]
        except Exception: return
        time.sleep(2)
if "_GPUTHREAD" not in globals() and "MOCK_LOAD" not in globals():
    import threading; _GPUTHREAD = threading.Thread(target=_gpu_sampler6, daemon=True); _GPUTHREAD.start()
def _mon6(nrows, ntok, tag, bs, budget):
    _MON["rows"] += nrows; _MON["tok"] += ntok; now = time.time()
    if not _MON["t"]: _MON["t"] = now; return
    dt = now - _MON["t"]
    if dt < 30: return
    u = _MON["util"]; _MON["util"] = []
    try: peak = round(torch.cuda.max_memory_allocated() / 1e9, 1); torch.cuda.reset_peak_memory_stats()
    except Exception: peak = None
    rec = dict(time=time.strftime("%H:%M:%S"), job=STATE.get("job") if "STATE" in globals() else None, step=tag or os.path.basename(str(globals().get("OUT", ""))), rows_per_s=round(_MON["rows"] / dt, 2), tokens_per_s=round(_MON["tok"] / dt),
               batch_rows=bs, tokbudget=budget, gpu_util=round(sum(u) / len(u)) if u else None, gpu_util_min=min(u) if u else None, mem_used_gb=(_MON.get("mem") or (None, None))[0],
               mem_total_gb=(_MON.get("mem") or (None, None))[1], peak_alloc_gb=peak, oom_splits=_MON["oom"])
    _MON.update(t=now, rows=0, tok=0)
    try:
        json.dump(rec, open(f"{ROOT}/gpu.json", "w"), indent=1); new = not os.path.exists(f"{ROOT}/gpu_log.csv")
        with open(f"{ROOT}/gpu_log.csv", "a") as f:
            if new: f.write(",".join(rec) + "\n")
            f.write(",".join("" if v is None else str(v) for v in rec.values()) + "\n")
    except Exception: pass
_TOKC = {}
def _tok6(text):
    e = _TOKC.get(text)
    if e is None:
        if len(_TOKC) > 50000: _TOKC.clear()
        t = CUR["tok"](text, add_special_tokens=False, return_offsets_mapping=True); e = _TOKC[text] = (t["input_ids"], t["offset_mapping"])
    return e
def _pos6(text, offs, where, T):
    if where is None: return None
    if isinstance(where, tuple) and len(where) == 2 and where[0] == "from":
        p = _pos6(text, offs, where[1], T); return list(range(p[0], T)) if p else []
    if isinstance(where, (list, tuple)): return sorted({j for s_ in where for j in _pos6(text, offs, s_, T)})
    if not where: return []
    s = text.find(where)
    return [] if s < 0 else [j for j, (a, b) in enumerate(offs) if b > s and a < s + len(where) and b > a]
class BHooks6:
    """Per-row edits of a batch. ops = {l: [(row, 'add', vec, pos) | (row, 'proj', (U, mu), pos)]}; pos = LongTensor or None (every position);
    hidden-state index l = output of block l - 1 (as Hooks6)."""
    def __init__(self, ops): self.ops, self.hs = ops, []
    def __enter__(self):
        for l, ops in self.ops.items():
            def hook(mod, inp, out, ops=ops):
                h = out[0] if isinstance(out, tuple) else out; hf = h.float()
                if hf.data_ptr() == h.data_ptr(): hf = hf.clone()
                T = hf.shape[1]
                for b, kind, x, pos in ops:
                    if pos is not None and (pos.numel() == 0 or int(pos.max()) >= T): continue
                    seg = hf[b] if pos is None else hf[b, pos]
                    if kind == "add": seg = seg + x
                    else: U, mu = x; seg = seg - ((seg - (mu if mu is not None else 0.0)) @ U.T) @ U
                    if pos is None: hf[b] = seg
                    else: hf[b, pos] = seg
                h2 = hf.to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            self.hs.append(CUR["layers"][l - 1].register_forward_hook(hook))
        return self
    def __exit__(self, *e):
        for h in self.hs: h.remove()
@torch.no_grad()
def run_rows6(rows, marks=None, groups=None, prefix=None, tag=None):
    """Batched run_marks6 (same returns, input order). Rows with the same token length (and the same need for states) are run together,
    up to TOKBUDGET tokens and MAXROWS rows per pass (job config can override); a pass that runs out of memory is split in halves."""
    tok, model = CUR["tok"], CUR["model"]; prefix = ans_prefix() if prefix is None else prefix; dev = _dev()
    dig = [d[0] for d in CUR["digits"]]; vals = np.arange(1, 10); G = list((groups or {}).items()); need = sorted({l for _, ls in G for l in ls})
    n = len(rows); t0 = time.time(); R = []
    for r in rows:
        p, ed = (r, []) if isinstance(r, str) else (r[0], list(r[1] or []))
        text = p + prefix; ids, offs = _tok6(text); R.append((text, ids, offs, ed))
    ev, mass, P9, M = np.zeros(n), np.zeros(n), torch.zeros(n, 9, dtype=torch.float16), {}
    budget, maxr = C6("TOKBUDGET", TOKBUDGET), C6("MAXROWS", MAXROWS); bylen = {}; done = [0]
    for i, (text, ids, offs, ed) in enumerate(R):
        mk = (marks[i] if marks else None) or {}; bylen.setdefault((len(ids), bool(mk) and bool(G)), []).append(i)
    def devf(x, cache):
        k = id(x)
        if k not in cache: cache[k] = x.to(dev, torch.float32)
        return cache[k]
    def one(bi, T, want):
        cache, ops = {}, {}
        for b, i in enumerate(bi):
            text, _, offs, ed = R[i]
            for e in ed:
                if e[0] == "add":
                    _, l, v, where = e; pos = _pos6(text, offs, where, T); pt = None if pos is None else torch.tensor(pos, device=dev, dtype=torch.long)
                    ops.setdefault(l, []).append((b, "add", devf(v, cache), pt))
                elif e[0] == "proj":
                    _, Us, where, mus = e; pos = _pos6(text, offs, where, T); pt = None if pos is None else torch.tensor(pos, device=dev, dtype=torch.long)
                    for l, U in Us.items():
                        mu = (mus or {}).get(l); ops.setdefault(l, []).append((b, "proj", (devf(U, cache), None if mu is None else devf(mu, cache)), pt))
        ids = torch.tensor([R[i][1] for i in bi], device=dev)
        with BHooks6(ops):
            try: out = model(input_ids=ids, output_hidden_states=want, logits_to_keep=1, use_cache=False)
            except TypeError: out = model(input_ids=ids, output_hidden_states=want)
        lp = torch.log_softmax(out.logits[:, -1].float(), -1); Pd = lp[:, dig].exp(); m = Pd.sum(-1); pn = (Pd / m[:, None]).cpu()
        for b, i in enumerate(bi):
            ev[i] = float(pn[b].numpy() @ vals); mass[i] = float(m[b]); P9[i] = pn[b].half()
        if want:
            hs = out.hidden_states
            for b, i in enumerate(bi):
                text, _, offs, _ = R[i]
                for name, (kind, x) in ((marks[i] or {}).items()):
                    idx = [T - 1] if kind == "dec" else (_pos6(text, offs, list(x), T) if kind == "union" else _pos6(text, offs, x, T))
                    if not idx: continue
                    if kind == "last": idx = idx[-1:]
                    it = torch.tensor(idx, device=hs[0].device); per = {l: hs[l][b].index_select(0, it).float().mean(0) for l in need}
                    M.setdefault(name, {})[i] = torch.stack([torch.stack([per[l] for l in ls]).mean(0) for _, ls in G]).to(STORE_DTYPE).cpu()
            del hs
        del out
    def safe(bi, T, want):
        try: one(bi, T, want)
        except torch.cuda.OutOfMemoryError:
            if len(bi) == 1: raise
            _MON["oom"] += 1; torch.cuda.empty_cache(); h = len(bi) // 2; safe(bi[:h], T, want); safe(bi[h:], T, want)
    for (T, want), idx in sorted(bylen.items()):
        tu = _tune6(); budget = int(tu.get("TOKBUDGET") or C6("TOKBUDGET", TOKBUDGET)); maxr = int(tu.get("MAXROWS") or C6("MAXROWS", MAXROWS))
        bs = max(1, min(maxr, budget // max(T, 1)))
        for k0 in range(0, len(idx), bs):
            safe(idx[k0:k0 + bs], T, want); done[0] += len(idx[k0:k0 + bs]); _mon6(len(idx[k0:k0 + bs]), T * len(idx[k0:k0 + bs]), tag, bs, budget)
            if tag and done[0] // 500 != (done[0] - len(idx[k0:k0 + bs])) // 500: beat(f"{tag} {done[0]}/{n}")
    res = dict(ev=ev, mass=mass, p9=P9, groups=[g for g, _ in G], sec=time.time() - t0, n=n)
    for name, dct in M.items():
        res[name] = torch.stack([dct.get(i, torch.full((len(G), CUR["d"]), float("nan"), dtype=STORE_DTYPE)) for i in range(n)])
    return res

@torch.no_grad()
def run_marks6_b1(prompts, marks=None, groups=None, prefix=None, hook_fn=None, tag=None):
    """Run 1 routine (kept for the batched-vs-one-at-a-time check). Batch 1, no padding. marks[i] = {name: (kind, x)}, kind 'mean' / 'last' (tokens overlapping substring x), 'union' (x = list of
    substrings) or 'dec' (the last token, after the answer prefix). groups = {gname: [hidden-state indices]}: each mark is averaged over
    its tokens and then over the group's layers. hook_fn(i, pos_of) returns a context manager for prompt i (pos_of(substring) gives the
    token positions overlapping it). Returns ev [n], mass [n], p9 [n, 9] and one [n, G, d] tensor per mark (STORE_DTYPE, CPU)."""
    tok, model = CUR["tok"], CUR["model"]; prefix = ans_prefix() if prefix is None else prefix
    dig = [d[0] for d in CUR["digits"]]; vals = np.arange(1, 10); G = list((groups or {}).items()); need = sorted({l for _, ls in G for l in ls})
    ev, mass, P9, M = [], [], [], {}; t0 = time.time()
    for i, p in enumerate(prompts):
        text = p + prefix; e = tok(text, add_special_tokens=False, return_offsets_mapping=True); offs = e["offset_mapping"]
        def pos_of(sub, _t=text, _o=offs):
            if not sub: return []
            s = _t.find(sub)
            return [] if s < 0 else [j for j, (a, b) in enumerate(_o) if b > s and a < s + len(sub) and b > a]
        mk = (marks[i] if marks else None) or {}; want = bool(mk) and bool(G)
        ids = torch.tensor([e["input_ids"]], device=_dev())
        with (hook_fn(i, pos_of) if hook_fn else contextlib.nullcontext()):
            try: out = model(input_ids=ids, output_hidden_states=want, logits_to_keep=1, use_cache=False)
            except TypeError: out = model(input_ids=ids, output_hidden_states=want)
        lp = torch.log_softmax(out.logits[0, -1].float(), -1); Pd = lp[dig].exp(); m = Pd.sum()
        ev.append(float((Pd / m).cpu().numpy() @ vals)); mass.append(float(m)); P9.append((Pd / m).half().cpu())
        if want:
            hs = out.hidden_states; T = ids.shape[1]
            for name, (kind, x) in mk.items():
                idx = [T - 1] if kind == "dec" else (sorted({j for s_ in x for j in pos_of(s_)}) if kind == "union" else pos_of(x))
                if not idx: continue
                if kind == "last": idx = idx[-1:]
                it = torch.tensor(idx, device=hs[0].device); per = {l: hs[l][0].index_select(0, it).float().mean(0) for l in need}
                M.setdefault(name, {})[i] = torch.stack([torch.stack([per[l] for l in ls]).mean(0) for _, ls in G]).to(STORE_DTYPE).cpu()
            del hs
        del out
        if tag and i % 250 == 0: beat(f"{tag} {i}/{len(prompts)}")
    res = dict(ev=np.array(ev), mass=np.array(mass), p9=torch.stack(P9) if P9 else torch.zeros(0, 9), groups=[g for g, _ in G], sec=time.time() - t0, n=len(prompts))
    for name, dct in M.items():
        res[name] = torch.stack([dct.get(i, torch.full((len(G), CUR["d"]), float("nan"), dtype=STORE_DTYPE)) for i in range(len(prompts))])
    return res

class Hooks6:
    """Edits of the hidden states (hidden-state index l = output of block l - 1). add: (l, vec, pos) adds vec at token positions pos
    (None = every position); proj: (l, U, pos[, mu]) removes the row space of U (orthonormal rows) at those positions; with mu (the mean
    state at that layer) the component is set to the mean's component instead of zero (mean replacement)."""
    def __init__(self, add=(), proj=()): self.add, self.proj, self.hs = list(add), list(proj), []
    def __enter__(self):
        dev = _dev(); by = {}
        for l, v, pos in self.add: by.setdefault(l, []).append(("add", v.to(dev, torch.float32), pos))
        for pr in self.proj:
            l, U, pos = pr[:3]; mu = pr[3] if len(pr) > 3 and pr[3] is not None else None
            by.setdefault(l, []).append(("proj", (U.to(dev, torch.float32), None if mu is None else mu.to(dev, torch.float32)), pos))
        for l, ops in by.items():
            def hook(mod, inp, out, ops=ops):
                h = out[0] if isinstance(out, tuple) else out; hf = h.float()
                for kind, x, pos in ops:
                    if pos is not None and (len(pos) == 0 or int(max(pos)) >= hf.shape[1]): continue
                    ix = slice(None) if pos is None else torch.tensor(pos, device=hf.device)
                    seg = hf[:, ix]
                    if kind == "add": seg = seg + x
                    else: U, mu = x; seg = seg - ((seg - (mu if mu is not None else 0.0)) @ U.T) @ U
                    hf = hf.clone(); hf[:, ix] = seg
                h2 = hf.to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            self.hs.append(CUR["layers"][l - 1].register_forward_hook(hook))
        return self
    def __exit__(self, *e):
        for h in self.hs: h.remove()
def add_at(l, vec, text):
    """hook_fn: add vec at hidden-state index l on the tokens of `text` (the situation) only."""
    return lambda i, pos_of: Hooks6(add=[(l, vec, pos_of(text[i] if isinstance(text, list) else text))])
def proj_at(Us, text=None, mus=None):
    """hook_fn: remove the subspaces Us = {l: U} on the tokens of text[i] (None = at every position); mus = {l: mean state} turns zeroing
    into mean replacement."""
    return lambda i, pos_of: Hooks6(proj=[(l, U, None if text is None else pos_of(text[i]), (mus or {}).get(l)) for l, U in Us.items()])

# ---------- directions (v6 direction source written by V1_read) ----------
def load_v1(mid, rev=None, small=None):
    """Summary and direction source of the latest V1_read of this model (and revision) with the same small flag."""
    small = MOCK_SMALL if small is None else small; best = None
    for f in glob.glob(f"{ROOT}/results/*/V1_read/summary.json"):
        try:
            s = json.load(open(f, encoding="utf-8"))
            if s.get("model") == mid and s.get("rev") == rev and bool(s.get("small")) == bool(small) and os.path.exists(os.path.join(os.path.dirname(f), "dirsrc.pt")):
                if best is None or os.path.getmtime(f) > best[0]: best = (os.path.getmtime(f), f, s)
        except Exception: pass
    if best is None: return None, None
    DS = torch.load(os.path.join(os.path.dirname(best[1]), "dirsrc.pt"), weights_only=False); DS["_byc"] = {}
    for (w, c, lt), (v, k) in DS["cells"].items(): DS["_byc"].setdefault(c, []).append((w, lt, v, k))
    return best[2], DS
def direction6(DS, c, l, exclude_writer=None, writers=None):
    """Mean high - low difference of construct c at hidden-state index l (raw space), pooled over writers and low-pole types."""
    j = DS["layers"].index(l); num, n = None, 0
    for w, lt, v, k in DS["_byc"].get(c, []):
        if w == exclude_writer or (writers and w not in writers): continue
        x = v[j].float() * k; num = x if num is None else num + x; n += k
    return None if num is None else num / n
def unit6(v): return F.normalize(v.float(), dim=0)
def natural6(DS, c, l): v = direction6(DS, c, l); return float(v.norm()) if v is not None else None
def span_random6(DS, l, n, g):
    """Random unit directions in the span of the centred cell differences (all writers, constructs and low poles) at layer l."""
    j = DS["layers"].index(l); M = torch.stack([v[j].float() for (w, c, lt), (v, k) in DS["cells"].items()]); M = M - M.mean(0)
    return [F.normalize(torch.randn(M.shape[0], generator=g) @ M, dim=0) for _ in range(n)]
def ortho6(U): return torch.linalg.qr(U.float().T)[0].T
def sub6(DS, c, k, layers):
    return {l: ortho6(DS["sub"][(c, l)][:k]) for l in layers if (c, l) in DS["sub"]}
def shared6(DS, layers, k=3):
    """The components shared by all constructs: top-k right singular vectors of the 66 construct mean directions at each layer (raw)."""
    out = {}
    for l in layers:
        Dc = torch.stack([direction6(DS, c, l) for c in sorted(DS["_byc"])]); out[l] = torch.linalg.svd(Dc, full_matrices=False)[2][:k]
    return out
def specific6(DS, c, k, layers, SH):
    """Construct-specific subspace: the top-k components of construct c after the shared components SH = {l: [k, d]} are taken out."""
    out = {}
    for l in layers:
        if (c, l) not in DS["sub"]: continue
        U = DS["sub"][(c, l)].float(); S = SH[l].float(); U = U - (U @ S.T) @ S; out[l] = ortho6(U[:k]) if k <= U.shape[0] else ortho6(U)
    return out
def unif_sub6(DS, layers, k, g):
    """Random k-dimensional subspaces drawn uniformly within the span of the centred cell differences (orthonormal basis, random rotation)."""
    out = {}
    for l in layers:
        j = DS["layers"].index(l); M = torch.stack([v[j].float() for (w, c, lt), (v, kk) in DS["cells"].items()]); M = M - M.mean(0)
        Q = torch.linalg.qr(M.T)[0]; out[l] = ortho6((Q @ torch.randn(Q.shape[1], k, generator=g)).T)
    return out
def span_sub6(DS, layers, k, g):
    out = {}
    for l in layers:
        j = DS["layers"].index(l); M = torch.stack([v[j].float() for (w, c, lt), (v, kk) in DS["cells"].items()]); M = M - M.mean(0)
        out[l] = ortho6(torch.randn(k, M.shape[0], generator=g) @ M)
    return out

# ---------- statistics ----------
def dz(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0: return dict(mean=round(float(x.mean()), 4) if len(x) else None, dz=None, t=None, n=int(len(x)))
    m, s = x.mean(), x.std(ddof=1); return dict(mean=round(float(m), 4), dz=round(float(m / s), 3), t=round(float(m / (s / np.sqrt(len(x)))), 2), n=int(len(x)))
def push_of(D, key_cols, dcol="d"):
    """(mean d at +2 - mean d at -2) / 2 per key, from rows with a dose column."""
    a = D[D.dose == 2].groupby(key_cols)[dcol].mean(); b = D[D.dose == -2].groupby(key_cols)[dcol].mean(); return ((a - b.reindex(a.index)) / 2)
def timing(res_list):
    n = sum(r["n"] for r in res_list); s = sum(r["sec"] for r in res_list); return dict(n_forward=int(n), sec=round(s, 1), sec_per_forward=round(s / max(n, 1), 4))
print("lib_v6 ready")
