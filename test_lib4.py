# test_lib4.py: additions for test plan 3 (TEST_PLAN_3_2026-10-19.md). Exec'd after latent_lib.py and test_lib3.py.
# Adds: a hub wipe that keeps the next model's files, the full text set of step R (v4 pairs + v5 triplets + loss framing v2 pairs),
# the final-norm finder for the logit-lens readout, the direction source saved by step R (and its loader for C1/C2), and the
# first-person judgment items built from the MVCA scale definitions (spec v4: definition + anchors 1 / 4 / 7).

# xet downloads broke after the xet cache was deleted mid-session (19 Oct, "Background writer channel closed"): use plain HTTP downloads
os.environ["HF_HUB_DISABLE_XET"] = "1"
try:
    import huggingface_hub.constants as _hc; _hc.HF_HUB_DISABLE_XET = True
except Exception: pass
def wipe_hub_except(keep=()):
    """Remove every cached model repo except those in `keep` (repo ids), and every blob in the shared hub/blobs folder that no kept
    repo links to (19 Oct: the blobs of deleted repos stayed in hub/blobs and filled the disk). The xet cache is left alone."""
    free0 = shutil.disk_usage("/").free / 1e9
    for d in glob.glob(f"{HUB}/models--*"):
        if os.path.basename(d)[len("models--"):].replace("--", "/") not in keep: shutil.rmtree(d, ignore_errors=True)
    used = set()
    for d in glob.glob(f"{HUB}/models--*"):
        for root, _, files in os.walk(d):
            for f in files:
                p = os.path.join(root, f)
                if os.path.islink(p): used.add(os.path.realpath(p))
    for root, _, files in os.walk(f"{HUB}/blobs"):
        for f in files:
            p = os.path.realpath(os.path.join(root, f))
            if p not in used:
                try: os.remove(p)
                except OSError: pass
    free1 = shutil.disk_usage("/").free / 1e9; log(f"hub wiped except {list(keep)}: free {free0:.0f} -> {free1:.0f} GB"); return round(free0, 1), round(free1, 1)

def r_texts(small=False):
    """All texts read in step R, with metadata. v4: high/low of each pair; v5t: neutral/weak/strong; v5l: loss framing v2 high/low."""
    P4 = json.load(open(f"{ROOT}/materials/v4/pairs_v4.json", encoding="utf-8"))
    TRI = json.load(open(f"{ROOT}/materials/v5/triplets_v5.json", encoding="utf-8")); L2 = json.load(open(f"{ROOT}/materials/v5/loss2_v5.json", encoding="utf-8"))
    if small: P4 = [p for i, p in enumerate(P4) if i % 12 == 0]; TRI = TRI[::12]; L2 = L2[::12]
    texts, meta = [], []
    for p in P4:
        for side in ("high", "low"):
            meta.append(dict(set="v4", id=p["pid"], side=side, writer=p["writer"], construct=p["construct"], kind=p["kind"], low_type=p.get("low_type"), domain=p.get("domain"))); texts.append(p[side])
    for t in TRI:
        for lev in ("neutral", "weak", "strong"):
            meta.append(dict(set="v5t", id=t["tid"], side=lev, writer=t["writer"], construct=t["construct"], kind="target", low_type=None, domain=t.get("domain"))); texts.append(t[lev])
    for p in L2:
        for side in ("high", "low"):
            meta.append(dict(set="v5l", id=p["pid"], side=side, writer=p["writer"], construct="loss_framing_v2", kind="target", low_type="L4", domain=p.get("domain"))); texts.append(p[side])
    return texts, meta

def final_norm():
    """The model's final norm module (applied before the unembedding), or None."""
    best = None
    for n, m in CUR["model"].named_modules():
        if re.search(r"(^|\.)(norm|final_layernorm|ln_f)$", n) and ".layers." not in n and "vision" not in n and "visual" not in n:
            if best is None or len(n) < len(best[0]): best = (n, m)
    return best[1] if best else None

@torch.no_grad()
def logit_lens_contrast(m_hi, m_lo, k=30):
    """Tokens raised by the high mean state over the low mean state when both are read through the final norm and the unembedding."""
    W = CUR["model"].get_output_embeddings().weight; nrm = final_norm(); dev = W.device
    def lp(x):
        h = x.to(dev, W.dtype)[None]; h = nrm(h) if nrm is not None else h
        return torch.log_softmax((h @ W.T).float()[0], -1)
    a, b = lp(m_hi), lp(m_lo); d = (a - b) * (a > -20).float() - 99 * (a <= -20).float()
    top = torch.topk(d, k); tk = CUR["tok"]
    return [(tk.decode([int(t)]), round(float(d[t]), 2)) for t in top.indices]

def load_dirsrc(mid):
    """Direction source written by step R (TT31) for this model: cell means per layer, std per layer, natural sizes, subspaces."""
    for f in sorted(glob.glob(f"{ROOT}/results/*/TT31_read/summary.json")):
        try:
            if json.load(open(f, encoding="utf-8")).get("model") == mid: return torch.load(os.path.join(os.path.dirname(f), "dirsrc.pt"), weights_only=False)
        except Exception: pass
    return None

def direction(DS, c, layer, exclude_writer=None, writers=None, std=False):
    """Mean high - low difference of construct c at a layer (pooled over pairs of the new writers, all low poles), optionally leaving
    one writer out or using only `writers`. With std=True the direction is estimated on standardized dimensions and mapped back to
    raw space (multiplied by the per-dimension std), so it can be added to the model's states."""
    num, n = None, 0
    for (w, cc, lt), (v, k) in DS["cells"].items():
        if cc != c or w in ("A", "B") or w == exclude_writer or (writers and w not in writers): continue
        x = v[DS["layers"].index(layer)].float() * k; num = x if num is None else num + x; n += k
    if num is None: return None
    d = num / n
    if std: sc = DS["std"][DS["layers"].index(layer)].float(); d = (d / sc) * sc        # identical in raw space; kept explicit
    return d

# first-person judgment items from the MVCA scale definitions (spec v4); 1-9 response format as in the experiments (FMT9)
JUDGE = {
    "ownership": "How much do you feel that the object in this situation is yours, as if it were already your own?",
    "loss_framing": "If you went without the thing in this situation, how much would that feel like losing something you already have, rather than simply not gaining something?",
    "reciprocity_obligation": "How much do you feel that you owe the person making the request a yes, or should meet them halfway, because they first did something for you or gave ground?",
    "helper_self_view": "How much does this situation make you see yourself as the kind of person who takes action for good causes and helps people in need?",
    "high_starting_value": "If you had to estimate the quantity in this situation, how much would your estimate start from a specific figure that is at hand?",
    "favorable_evaluation": "How good do you find the object, cause or request itself (its look, quality, usefulness or merit)?",
    "affective_reaction": "How much does the object itself interest or delight you?",
    "request_contrast": "How small or easy does the request in this situation seem to you?",
    "pleasantness": "How pleasant do you feel in this situation?",
    "setting_appeal": "How attractive do you find the incidental surroundings described in this situation (for example a window, a lamp, a path or a room)?"}
JUDGE_FMT = " Answer on a scale from 1 (not at all) to 9 (extremely)." + FMT9
def judge_prompt(text, c, persona=None):
    return chat(text + "\n\n" + JUDGE[c] + JUDGE_FMT, (SYS + persona) if persona else None)
print("test_lib4 ready")
