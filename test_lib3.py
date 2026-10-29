# test_lib.py: shared code for the 18 Oct 2026 test run (TEST_PLAN_2026-10-18.md v2). Exec'd by each TT job after runner_lib and
# latent_lib. Adds: a loader that checks for missing weights (new multimodal-wrapped checkpoints), forward passes that read several
# positions at once (state at the last token of a text, mean over a text, the decision token), band layers, direction utilities, and
# the cross-set (multitrait-multimethod) and band-test analyses. States of the test run live in CACHE and in LOC (local disk).
import copy
LOC = "/content/latent_cache" if os.path.isdir("/content") else os.path.join(ROOT, "_loc"); os.makedirs(LOC, exist_ok=True)
HUB = os.path.expanduser("~/.cache/huggingface/hub")
QUESTIONS = {
    "common": "How common is a situation like this in everyday life? Answer on a scale from 1 (very unusual) to 9 (very common).",
    "time": "At what time of day does a situation like this most likely take place? Answer on a scale from 1 (early in the morning) to 9 (late at night).",
    "indoors": "How likely is it that a situation like this takes place indoors? Answer on a scale from 1 (certainly outdoors) to 9 (certainly indoors).",
    "season": "In which season does a situation like this most likely take place? Answer on a scale from 1 (the middle of winter) to 9 (the middle of summer)."}
QKEYS = list(QUESTIONS)
LENS_T = {"Qwen/Qwen3.6-27B": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
          "google/gemma-4-31B-it": "gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt"}

def test_profiles(n=10):
    PR = json.load(open(f"{ROOT}/materials/v3a/profiles30.json", encoding="utf-8"))
    return sorted(PR, key=lambda p: p["id"])[:n]          # p_151 ... in order (MVCA paper50 test sample)

def wipe_hub():
    """Remove the whole Hugging Face cache (D8: all repos share one blobs folder)."""
    free0 = shutil.disk_usage("/").free / 1e9; shutil.rmtree(HUB, ignore_errors=True); free1 = shutil.disk_usage("/").free / 1e9
    log(f"hub wiped: free {free0:.0f} -> {free1:.0f} GB"); return round(free0, 1), round(free1, 1)

def load_test_model(mid):
    """Load bf16 on the GPU. Text-only causal class first; if it raises OR reports missing weights, use the image-text class.
    Fills CUR like runner_lib.get_model and calibrates the digit tokens. Returns a diagnostics dict."""
    if CUR.get("id") == mid: return {"reused": True}
    unload(); JL.clear(); diag = {"mid": mid}; t0 = time.time()
    if "MOCK_LOAD" in globals(): model, tok, how = MOCK_LOAD(mid)
    else:
        from transformers import AutoModelForImageTextToText
        tok = AutoTokenizer.from_pretrained(mid, token=_tok); kw = dict(token=_tok, device_map="cuda", torch_dtype=torch.bfloat16)
        model, how = None, None
        try:
            model, info = AutoModelForCausalLM.from_pretrained(mid, output_loading_info=True, **kw)
            miss = [k for k in info.get("missing_keys", []) if "vision" not in k and "visual" not in k]
            diag["causal_missing_keys"] = miss[:20]; diag["causal_n_missing"] = len(miss)
            if miss: del model; gc.collect(); torch.cuda.empty_cache(); model = None
            else: how = "causal"
        except Exception as ex:
            diag["causal_error"] = repr(ex)[:400]; model = None
        if model is None:
            model, info = AutoModelForImageTextToText.from_pretrained(mid, output_loading_info=True, **kw)
            diag["it_n_missing"] = len(info.get("missing_keys", [])); diag["it_missing_keys"] = info.get("missing_keys", [])[:20]; how = "image-text"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    lname, layers = find_layers(model)
    cfg = getattr(model.config, "text_config", None) or model.config
    CUR.update(id=mid, model=model, tok=tok, layers=layers, n_layers=len(layers), d=cfg.hidden_size, prec="bf16")
    CUR["digits"] = [tok.encode(str(k), add_special_tokens=False) for k in range(1, 10)]; CUR.pop("_digits_for", None); calibrate_digits()
    diag.update(how=how, cls=type(model).__name__, layers_path=lname, n_layers=len(layers), d=cfg.hidden_size, load_s=round(time.time() - t0),
                gpu_gb=round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else None,
                layer_types=sorted(set(getattr(cfg, "layer_types", None) or [])), ans_prefix=CUR.get("ans_prefix"))
    log(f"loaded {mid} via {how} ({diag['cls']}) in {diag['load_s']}s; {len(layers)} layers at {lname}; d={cfg.hidden_size}")
    return diag

def bands(nL=None):
    """Hidden-state indices (0 = embeddings, l = output of block l). B1 = 30-50% of depth (preregistered band of 18 Oct),
    B2 = 60-75% (where the shared-suffix patch effects entered). STORE = B1 + B2."""
    nL = nL or CUR["n_layers"]
    B1 = [l for l in range(nL + 1) if 0.30 * nL <= l <= 0.50 * nL]; B2 = [l for l in range(nL + 1) if 0.60 * nL <= l <= 0.75 * nL]
    return B1, B2, B1 + B2

def _dev(): return next(CUR["model"].parameters()).device

@torch.no_grad()
def run_marks(prompts, marks=None, layers=None, prefix=None, hook=None):
    """One forward pass per prompt (batch 1, no padding). marks: list (one per prompt) of {name: (kind, substring)}, kind 'last'
    (state at the last token overlapping the substring) or 'mean' (mean over its tokens). Returns ev [n], mass [n],
    dec [n, L, d] (last token, after the answer prefix) and one [n, L, d] tensor per mark name (bf16, CPU)."""
    tok, model = CUR["tok"], CUR["model"]; prefix = ans_prefix() if prefix is None else prefix
    layers = list(range(CUR["n_layers"] + 1)) if layers is None else list(layers)
    dig = [d[0] for d in CUR["digits"]]; vals = np.arange(1, 10); ev, mass, dec = [], [], []; M = {}
    for i, p in enumerate(prompts):
        text = p + prefix; e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor([e["input_ids"]], device=_dev())
        with (hook if hook is not None else contextlib.nullcontext()):
            try: out = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1)
            except TypeError: out = model(input_ids=ids, output_hidden_states=True)
        lp = torch.log_softmax(out.logits[0, -1].float(), -1); P = lp[dig].exp(); m = P.sum()
        ev.append(float((P / m).cpu().numpy() @ vals)); mass.append(float(m))
        hs = torch.stack([out.hidden_states[l][0] for l in layers], 0)            # [L, T, d]
        dec.append(hs[:, -1].to(STORE_DTYPE).cpu())
        for name, (kind, sub) in ((marks[i] if marks else None) or {}).items():
            s = text.find(sub)
            if s < 0: continue
            idx = [j for j, (a, b) in enumerate(e["offset_mapping"]) if b > s and a < s + len(sub) and b > a]
            if not idx: continue
            v = hs[:, idx[-1]] if kind == "last" else hs[:, idx].float().mean(1)
            M.setdefault(name, {})[i] = v.to(STORE_DTYPE).cpu()
        del out, hs
        if i % 200 == 0: beat(f"{STATE.get('job')} {i}/{len(prompts)}")
    res = dict(ev=np.array(ev), mass=np.array(mass), dec=torch.stack(dec), layers=layers)
    for name, dct in M.items():
        res[name] = torch.stack([dct.get(i, torch.full((len(layers), CUR["d"]), float("nan"), dtype=STORE_DTYPE)) for i in range(len(prompts))])
    return res

import contextlib
class AddHook:
    """Adds vec (on the model's device) to the output of block l at every position."""
    def __init__(self, add): self.add, self.hs = add, []
    def __enter__(self):
        for l, a in self.add.items():
            a = a.to(_dev(), torch.bfloat16)
            def hook(mod, inp, out, a=a):
                h = out[0] if isinstance(out, tuple) else out; h = h + a
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.hs.append(CUR["layers"][l].register_forward_hook(hook))
        return self
    def __exit__(self, *e):
        for h in self.hs: h.remove()

def contrast_pass(texts, profiles, qkeys=QKEYS, tag="cp", include_none=True):
    """texts: list of str. profiles: list of profile dicts or None (no profile). For every text x profile x question, one forward
    pass; returns EV [t, p, q], DEC [t, p, q, L, d], END [t, p, L, d] and MEAN [t, p, L, d] (END/MEAN from the first question; the
    model reads left to right, so the text's own states do not depend on the question that follows; L0 checks this)."""
    _, _, ST = bands(); P = ([None] if include_none else []) + list(profiles or [])
    prompts, marks, keys = [], [], []
    for ti, t in enumerate(texts):
        for pi, pr in enumerate(P):
            for qi, q in enumerate(qkeys):
                prompts.append(chat(t + "\n\n" + QUESTIONS[q] + FMT9, (SYS + pr["persona"]) if pr else None))
                marks.append({"end": ("last", t), "mean": ("mean", t)} if qi == 0 else None); keys.append((ti, pi, qi))
    r = run_marks(prompts, marks, layers=ST)
    nT, nP, nQ = len(texts), len(P), len(qkeys); L, d = len(ST), CUR["d"]
    EV = np.zeros((nT, nP, nQ)); DEC = torch.zeros(nT, nP, nQ, L, d, dtype=STORE_DTYPE)
    END = torch.zeros(nT, nP, L, d, dtype=STORE_DTYPE); MEAN = torch.zeros(nT, nP, L, d, dtype=STORE_DTYPE)
    for k, (ti, pi, qi) in enumerate(keys):
        EV[ti, pi, qi] = r["ev"][k]; DEC[ti, pi, qi] = r["dec"][k]
        if qi == 0: END[ti, pi] = r["end"][k]; MEAN[ti, pi] = r["mean"][k]
    return dict(EV=EV, MASS=r["mass"].reshape(nT, nP, nQ), DEC=DEC, END=END, MEAN=MEAN, layers=ST, profiles=[p["id"] if p else None for p in P], qkeys=list(qkeys))

# ---------- analysis utilities (CPU) ----------
def cosL(a, b):
    """Cosine per layer between [L, d] tensors, returned as a list."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1).tolist()

def band_mean_cos(a, b, idx):
    c = F.cosine_similarity(a.float()[idx], b.float()[idx], dim=-1); return float(c.mean())

def split_half(D, idx, n=50, seed=0):
    """D: per-pair differences [n_pairs, L, d]. Mean band cosine between the mean directions of random halves (n splits)."""
    g = np.random.default_rng(seed); k = D.shape[0]; out = []
    for _ in range(n):
        p = g.permutation(k); a, b = D[p[:k // 2]].float().mean(0), D[p[k // 2:]].float().mean(0); out.append(band_mean_cos(a, b, idx))
    return float(np.mean(out))

def sb(r): return 2 * r / (1 + r) if r > -1 else float("nan")    # Spearman-Brown to the full set

def std_scale(X):
    """Per-layer, per-dimension SD over all rows of X [n, L, d] (for standardized analyses; D11)."""
    return torch.nan_to_num(X.float(), nan=0.0).std(0) + 1e-6

def mtmm(DIRS, idx, rel=None):
    """DIRS: {(method, construct): direction [L, d]}. Returns the matrix of band cosines and the summary cells:
    mono-trait hetero-method (convergent), hetero-trait mono-method (method variance), hetero-trait hetero-method (baseline)."""
    keys = sorted(DIRS); C = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]: C[(a, b)] = band_mean_cos(DIRS[a], DIRS[b], idx)
    mt_hm, ht_mm, ht_hm = [], [], []
    for (a, b), c in C.items():
        if a[1] == b[1] and a[0] != b[0]:
            if rel:      # correct for unreliability only when both split-half reliabilities are usable (>= .1); else NaN
                ra, rb = rel.get(a, 1), rel.get(b, 1); c = c / np.sqrt(ra * rb) if min(ra, rb) >= .1 else float("nan")
            mt_hm.append((a[1], c))
        elif a[0] == b[0]: ht_mm.append(c)
        else: ht_hm.append(c)
    conv = {}
    for t, c in mt_hm: conv.setdefault(t, []).append(c)
    return dict(convergent={t: round(float(np.nanmean(v)), 3) if not np.all(np.isnan(v)) else None for t, v in conv.items()}, convergent_mean=round(float(np.nanmean([c for _, c in mt_hm])), 3) if mt_hm and not np.all(np.isnan([c for _, c in mt_hm])) else None,
                method_shared=round(float(np.mean(ht_mm)), 3) if ht_mm else None, baseline=round(float(np.mean(ht_hm)), 3) if ht_hm else None,
                pairs={f"{a[0]}:{a[1]}|{b[0]}:{b[1]}": round(c, 3) for (a, b), c in C.items()})

def band_test(MS_by_exp, fam_of, DIRS, idx, n_rand=1000, seed=7, scale=None):
    """MS_by_exp: {exp: mean shift per variant [V, L, d]}; DIRS: {family: direction [L, d]}. Projection of each variant's shift on the
    unit direction, averaged over band layers idx; t over variants; two-sided p against n_rand random directions; rank of the own family."""
    torch.manual_seed(seed); d = next(iter(DIRS.values())).shape[-1]; RND = F.normalize(torch.randn(n_rand, d), dim=-1); out = {}
    for e, MS in MS_by_exp.items():
        X = MS.float()[:, idx]
        if scale is not None: X = X / scale[idx]
        def tv(x): x = np.asarray(x, float); s = x.std(ddof=1); return float(x.mean() / (s / np.sqrt(len(x)))) if s > 0 else 0.0
        U = {f: F.normalize((DIRS[f].float()[idx] / (scale[idx] if scale is not None else 1.0)), dim=-1) for f in DIRS}
        tf = {f: round(tv((X * U[f][None]).sum(-1).mean(1).numpy()), 2) for f in DIRS}
        pr = torch.einsum("vld,rd->vlr", X, RND).mean(1).numpy(); trand = pr.mean(0) / (pr.std(0, ddof=1) / np.sqrt(X.shape[0]) + 1e-9)
        own = tf.get(fam_of[e]);
        if own is None: continue
        out[e] = dict(t_own=own, p_rand=round(float((np.abs(trand) >= abs(own)).mean()), 4), rank=1 + sum(v > own for k, v in tf.items() if k != fam_of[e]), t_by=tf)
    return out

def save_local(name, obj): torch.save(obj, f"{LOC}/{name}")
def load_local(name): return torch.load(f"{LOC}/{name}", weights_only=False) if os.path.exists(f"{LOC}/{name}") else None   # our own files
print("test_lib ready")

# ---------------- test plan 2 additions (TEST_PLAN_2_2026-10-18_night.md) ----------------
F2T = {"ownership": "ownership", "loss": "loss_framing", "reciprocity": "reciprocity_obligation", "self_perception": "helper_self_view", "anchoring": "high_starting_value"}
COMPS = ["affective_reaction", "favorable_evaluation", "pleasantness", "request_contrast", "setting_appeal"]
def layer_set(nL=None):
    """Hidden-state indices read in test plan 2: 11 depth bins (0, 10%, ..., 100%), both bands, and the patching layers."""
    nL = nL or CUR["n_layers"]; B1, B2, _ = bands(nL)
    bins = {int(round(f * nL)) for f in (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)}; pl = {int(round(f * nL)) for f in (.625, .6875, .75)}
    return sorted(bins | set(B1) | set(B2) | pl)
def patch_layers(nL=None):
    nL = nL or CUR["n_layers"]; return sorted({int(round(f * nL)) for f in (.625, .6875, .75)})

def text_pass(texts, profiles=None, layers=None):
    """Each text read as a user turn (no question, no generation prompt), with no profile or under each profile.
    Returns MEAN [t, P, L, d] (mean over the text's tokens) and END [t, P, L, d] (last token of the text), bf16 on CPU."""
    layers = layers or layer_set(); P = list(profiles) if profiles else [None]
    prompts, marks, keys = [], [], []
    for ti, t in enumerate(texts):
        for pi, pr in enumerate(P):
            prompts.append(chat(t, (SYS + pr["persona"]) if pr else None, gen=False)); marks.append({"mean": ("mean", t), "end": ("last", t)}); keys.append((ti, pi))
    r = run_marks(prompts, marks, layers=layers, prefix="")
    nT, nP = len(texts), len(P); MEAN = torch.zeros(nT, nP, len(layers), CUR["d"], dtype=STORE_DTYPE); END = torch.zeros_like(MEAN)
    for k, (ti, pi) in enumerate(keys): MEAN[ti, pi] = r["mean"][k]; END[ti, pi] = r["end"][k]
    return dict(MEAN=MEAN, END=END, layers=layers)

def lowo(DIFF, PAIRS, writers, cons):
    """Leave-one-writer-out nearest-centroid classification of single pair differences [n, d] among `cons`; returns per-construct accuracy."""
    acc = {c: [] for c in cons}; DIR = {}
    for i, p in enumerate(PAIRS): DIR.setdefault((p["writer"], p["construct"]), []).append(i)
    DIR = {k: DIFF[v].float().mean(0) for k, v in DIR.items()}
    for w in writers:
        cl = [c for c in cons if any(w2 != w and (w2, c) in DIR for w2 in writers)]
        if len(cl) < 2: continue
        C = torch.stack([F.normalize(torch.stack([DIR[(w2, c)] for w2 in writers if w2 != w and (w2, c) in DIR]).mean(0), dim=-1) for c in cl])
        for i, p in enumerate(PAIRS):
            if p["writer"] == w and p["construct"] in cl: acc[p["construct"]].append(cl[int((F.normalize(DIFF[i].float(), dim=-1) @ C.T).argmax())] == p["construct"])
    return {c: round(float(np.mean(v)), 3) for c, v in acc.items() if v}

def _free_gpu():
    """Release every reference to the loaded model (CUR, J-lens wrapper, any nn.Module or CUDA tensor in globals)."""
    import torch.nn as _nn
    for _k in list(globals()):
        if _k.startswith("_") or _k in ("CUR", "JL", "CACHE", "STATE", "GPU", "VRAM", "LENS", "PARAMS", "LENS_T", "QUESTIONS", "QKEYS", "F2T", "COMPS"): continue
        _v = globals()[_k]
        if isinstance(_v, _nn.Module) or (isinstance(_v, torch.Tensor) and _v.is_cuda) or _k in ("jm", "lens", "model", "r", "ref", "new", "cp", "cp0", "cpP", "S", "S3", "L3", "ST_M"):
            del globals()[_k]
    unload(); JL.clear(); CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
    return round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0.0
print("test_lib3 additions ready")
