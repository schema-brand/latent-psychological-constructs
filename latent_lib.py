# latent_lib.py: shared code for the v3 exploration jobs (exec'd by each job after runner_lib; idempotent).
# Materials: ROOT/materials/<MATV>/ (exp_*.json, family_*.json, awareness.json, profiles30.json). States live in the global CACHE
# so later jobs in the same runner session reuse them; every job recomputes what it needs if CACHE is empty.
SYS = "You are the person described below. Answer every question as this person would, in the first person.\n\n"
FMT9 = ' Reply with a number from 1 to 9 in the form {"rating": n}.'
NEG_PAIRS = [
    ("I think this cafe is quiet today.", "I don't think this cafe is quiet today."),
    ("To me, this song sounds familiar.", "To me, this song doesn't sound familiar at all."),
    ("I feel like the meeting went well.", "I don't feel like the meeting went well."),
    ("I think the street outside is busy right now.", "I don't think the street outside is busy right now."),
    ("I am sure I locked the front door.", "I am not sure I locked the front door."),
    ("It seems to me that the coffee is still hot.", "It doesn't seem to me that the coffee is still hot."),
    ("I feel that the chair is comfortable.", "I don't feel that the chair is comfortable at all."),
    ("I think the clock on the wall is accurate.", "I don't think the clock on the wall is accurate."),
    ("I believe the train will be on time.", "I don't believe the train will be on time."),
    ("I find this room bright.", "I don't find this room bright at all."),
]
if "CACHE" not in globals(): CACHE = {}
def calibrate_digits():
    """Choose the answer prefix and the digit tokens that follow it for the loaded tokenizer. With the trailing space kept, the
    digit must be ONE token after the prefix; if a tokenizer merges space and digit, drop the trailing space and use ' k' tokens."""
    if CUR.get("_digits_for") == CUR.get("id"): return
    tok = CUR["tok"]; head = '<|channel|>final<|message|>' if "gpt-oss" in str(CUR.get("id", "")) else ""
    ctx = "Answer.\n" + head
    for pre in ('{"rating": ', '{"rating":'):
        base = tok.encode(ctx + pre, add_special_tokens=False); ids = []
        for k in range(1, 10):
            full = tok.encode(ctx + pre + ("" if pre.endswith(" ") else " ") + str(k), add_special_tokens=False)
            ids.append(full[-1] if full[:len(base)] == base and len(full) == len(base) + 1 else None)
        if all(i is not None for i in ids) and len(set(ids)) == 9:
            CUR["digits"] = [[i] for i in ids]; CUR["ans_prefix"] = head + pre; CUR["_digits_for"] = CUR.get("id")
            log(f"digits calibrated for {CUR.get('id')}: prefix {pre!r}"); return
    CUR["ans_prefix"] = head + '{"rating": '; CUR["_digits_for"] = CUR.get("id"); log(f"digit calibration FAILED for {CUR.get('id')}")
def ans_prefix():
    calibrate_digits(); return CUR["ans_prefix"]
if "_get_model_orig" not in globals(): _get_model_orig = get_model
def get_model(mid, prec="auto"):
    c = _get_model_orig(mid, prec); calibrate_digits(); return c
if "_run_prompts_orig" not in globals(): _run_prompts_orig = run_prompts
STORE_DTYPE = torch.bfloat16      # runner_lib stores states as fp16, which overflows for Gemma (massive activations); bf16 keeps fp32 range
@torch.no_grad()
def _run_prompts_bf16(prompts, prefix, spans=None, layers=None, want_states=True, bs=8, hook=None):
    tok, model = CUR["tok"], CUR["model"]
    layers = list(range(CUR["n_layers"] + 1)) if layers is None else list(layers)
    dig = [d[0] if len(d) == 1 else None for d in CUR["digits"]]; vals = np.arange(1, 10)
    ev, mass, dec, sp = [], [], [], {}
    for k in range(0, len(prompts), bs):
        texts = [p + prefix for p in prompts[k:k + bs]]
        encs = [tok(t, add_special_tokens=False, return_offsets_mapping=True) for t in texts]
        T = max(len(e["input_ids"]) for e in encs)
        ids = torch.full((len(texts), T), tok.pad_token_id); att = torch.zeros((len(texts), T), dtype=torch.long)
        for i, e in enumerate(encs):
            n = len(e["input_ids"]); ids[i, T - n:] = torch.tensor(e["input_ids"]); att[i, T - n:] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        kw = dict(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda(), output_hidden_states=want_states)
        with (hook if hook is not None else Hook()):
            try: out = model(**kw, logits_to_keep=1)
            except TypeError: out = model(**kw)
        lp = torch.log_softmax(out.logits[:, -1].float(), -1)
        P = torch.stack([lp[:, i].exp() if i is not None else torch.zeros(len(texts), device=lp.device) for i in dig], 1)
        m = P.sum(1); ev += ((P / m.unsqueeze(1)).cpu().numpy() @ vals).tolist(); mass += m.cpu().tolist()
        if want_states:
            hs = torch.stack([out.hidden_states[l] for l in layers], 1)
            dec.append(hs[:, :, -1].to(STORE_DTYPE).cpu())
            if spans is not None:
                for i, e in enumerate(encs):
                    n = len(e["input_ids"]); offs = e["offset_mapping"]
                    for name, sub in (spans[k + i] or {}).items():
                        s = texts[i].find(sub)
                        if s < 0: continue
                        idx = [T - n + j for j, (a, b) in enumerate(offs) if b > s and a < s + len(sub) and b > a]
                        if idx: sp.setdefault(name, {})[k + i] = hs[i][:, idx].float().mean(1).to(STORE_DTYPE).cpu()
        del out
    res = dict(ev=np.array(ev), mass=np.array(mass), layers=layers)
    if want_states:
        res["dec"] = torch.cat(dec)
        for name, dct in sp.items():
            res["span_" + name] = torch.stack([dct.get(i, torch.full((len(layers), CUR["d"]), float("nan"), dtype=STORE_DTYPE)) for i in range(len(prompts))])
    return res
def run_prompts(prompts, prefix=None, **kw):
    prefix = ans_prefix() if prefix is None else prefix
    if isinstance(CUR.get("model"), torch.nn.Module): return _run_prompts_bf16(prompts, prefix, **kw)
    return _run_prompts_orig(prompts, prefix=prefix, **kw)

def load_v3(matv):
    d = f"{ROOT}/materials/{matv}"; E, FA = {}, {}
    for f in sorted(glob.glob(f"{d}/exp_*.json")):
        x = json.load(open(f, encoding="utf-8")); E[x["id"]] = x
    for f in sorted(glob.glob(f"{d}/family_*.json")):
        x = json.load(open(f, encoding="utf-8")); FA[x["family"]] = x
    AW = json.load(open(f"{d}/awareness.json", encoding="utf-8")) if os.path.exists(f"{d}/awareness.json") else None
    PR = json.load(open(f"{d}/profiles30.json", encoding="utf-8"))
    return E, FA, AW, PR

def as_list(x): return [x] if isinstance(x, str) else list(x or [])
def q_of(v, arm): return v.get("question") or v.get("question_" + arm)

def build_items(E, PR, prefix="", vids=None, profiles=None, arms=("treatment", "control"), exps=None):
    rows = []
    for eid, e in E.items():
        if exps and eid not in exps: continue
        for v in e["variants"]:
            if vids and v["vid"] not in vids: continue
            for arm in arms:
                body, q = v["body_" + arm], q_of(v, arm); xs = as_list(v["x_span_" + arm])
                for p in (profiles or PR):
                    rows.append(dict(exp=eid, fam=e["family"], vid=v["vid"], arm=arm, profile=p["id"], length=v["length"],
                                     prompt=chat(prefix + body + "\n\n" + q + FMT9, SYS + p["persona"]),
                                     spans={"x1": xs[0], "q": q}))
    return pd.DataFrame(rows)

def run_items(D, layers, spans=True, bs=1, hook=None, states=True):
    r = run_prompts(list(D.prompt), spans=list(D.spans) if (spans and states) else None, layers=layers, want_states=states, bs=bs, hook=hook)
    return r

def text_states(texts, layers, context=None, as_thought=False, bs=1):
    """States of free texts (statements or contrast scenarios) read as a user turn, no generation prompt.
    Returns span-mean states over the text [n, L, d] and the state at the turn's last token [n, L, d]."""
    prompts, spans = [], []
    for t in texts:
        u = t if not as_thought else ((context + "\n\n") if context else "") + 'A thought on your mind right now: "' + t + '"'
        prompts.append(chat(u, None, gen=False)); spans.append({"s": t})
    r = run_prompts(prompts, prefix="", spans=spans, layers=layers, want_states=True, bs=bs)
    return r["span_s"].float(), r["dec"].float()

NEUTRAL_Q = "How common is a situation like this in everyday life? Answer on a scale from 1 (very unusual) to 9 (very common)."
def answer_states(texts, layers, question=NEUTRAL_Q, bs=1):
    """Decision-token states for texts followed by a neutral 1-9 question and the answer prefix (the same position type as the items)."""
    prompts = [chat(t + "\n\n" + question + FMT9, None) for t in texts]
    r = run_prompts(prompts, layers=layers, want_states=True, bs=bs)
    return r["dec"].float(), r["ev"]

def unit(x): return F.normalize(x.float(), dim=-1)

def paired_t(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0: return dict(mean=float(np.mean(x)) if len(x) else None, t=None, n=int(len(x)))
    return dict(mean=round(float(x.mean()), 4), t=round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2), n=int(len(x)))

def arm_diffs(D, col):
    """Treatment - control per (exp, vid, profile) for a column of D."""
    T = D[D.arm == "treatment"].set_index(["exp", "vid", "profile"])[col]; C = D[D.arm == "control"].set_index(["exp", "vid", "profile"])[col]
    return (T - C.reindex(T.index)).rename(col + "_diff")

def shift_tensor(D, S):
    """S: [n, L, d] states aligned with D rows. Returns dict (exp, vid) -> [n_profiles, L, d] treatment - control (paired by profile)."""
    idx = {(r.exp, r.vid, r.arm, r.profile): i for i, r in enumerate(D.itertuples())}
    out = {}
    for (eid, vid), g in D[D.arm == "treatment"].groupby(["exp", "vid"]):
        ti = [idx[(eid, vid, "treatment", p)] for p in g.profile]; ci = [idx[(eid, vid, "control", p)] for p in g.profile]
        out[(eid, vid)] = S[ti].float() - S[ci].float()
    return out

def beat_every(i, n, tag, k=50):
    if i % k == 0: beat(f"{tag} {i}/{n}")
print("latent_lib ready")
