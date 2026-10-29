"""Build runner_lib.py (helpers + job loop) and runner_v1.ipynb (18 Oct 2026). The notebook installs packages, mounts Drive,
checks the sha256 of runner_lib.py on Drive, execs it and runs the job loop, which executes job scripts that Claude places in
Drive MyDrive/MVCA_latent/runner/jobs_pending/. Jobs run in the same Python process, so a loaded model stays in memory between
jobs. Each job writes to runner/results/<job>/ (small summary JSON for Claude to read, CSVs, tensors). The runner writes a
heartbeat to runner/status.json; a job that fails leaves ERROR.txt and moves to jobs_failed/. Stop: put {"stop": true} in
runner/control.json.
Helpers available to jobs: get_model, chat, run_prompts (one forward pass: 1-9 rating expected value + residual states at the
decision token and at named text spans, all or selected layers, optional steering/ablation hooks), Hook, gen_text, jlens_read,
embed_texts, save_json/save_df/save_pt, load_mat, beat.
Usage: python build_runner.py -> runner_lib.py + runner_v1.ipynb (upload both to Drive MVCA_latent/runner/)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
cells = []
def md(s): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')})
def code(s): cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')})

md(r'''
# MVCA latent：整夜任务执行器（runner v1）

1. Runtime → Change runtime type → **A100 GPU + High-RAM**。
2. Runtime → **Run all**，允许挂载 Drive。第二格会一直运行（这是正常的），它会自动执行 Claude 放进 `MVCA_latent/runner/jobs_pending/` 的任务。
3. 标签页保持打开，电脑插电不要休眠。要停止：点停止按钮即可。
''')

code(r'''
# ── 1. 环境 ────────────────────────────────────────────────────────────────
import os, sys, time, json, shutil, traceback, subprocess, gc, re, math, glob, itertools
!pip install -q -U "transformers>=4.51" accelerate huggingface_hub bitsandbytes scikit-learn sentence-transformers
!pip install -q --no-deps git+https://github.com/anthropics/jacobian-lens.git
import torch, numpy as np, pandas as pd, torch.nn.functional as F
from google.colab import userdata, drive
from huggingface_hub import login, whoami
_tok = userdata.get("HF_TOKEN"); login(token=_tok, add_to_git_credential=False)
drive.mount("/content/drive")
ROOT = "/content/drive/MyDrive/MVCA_latent/runner"
for d in ("jobs_pending", "jobs_running", "jobs_done", "jobs_failed", "results", "materials"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
GPU = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
print(GPU, f"| {VRAM:.0f} GB | HF:", whoami(token=_tok)["name"], "| disk free", shutil.disk_usage("/content").free // 10**9, "GB")
import hashlib
_lib = open(f"{ROOT}/runner_lib.py", "rb").read()
assert hashlib.sha256(_lib).hexdigest() == "@@SHA@@", "runner_lib.py 校验不符：请告诉 Claude"
exec(compile(_lib.decode("utf-8"), "runner_lib.py", "exec"), globals())
''')

code(r'''
# ── 2. 执行器主循环（会一直运行；这是正常的）
run_loop()
''')

import hashlib
lib = r'''
# runner_lib.py: helpers + job loop for the MVCA latent runner (exec'd by runner_v1.ipynb after a sha256 check).
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
STATE = {"gpu": GPU, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "job": None, "msg": "setup", "done": [], "failed": []}
CUR = {}
def beat(msg=None):
    if msg is not None: STATE["msg"] = msg
    STATE["time"] = time.strftime("%Y-%m-%d %H:%M:%S"); STATE["model"] = CUR.get("id")
    STATE["gpu_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)
    try:
        with open(f"{ROOT}/status.json", "w") as f: json.dump(STATE, f, indent=1)
    except Exception: pass
def log(s):
    with open(f"{ROOT}/log.txt", "a") as f: f.write(time.strftime("%H:%M:%S ") + s + "\n")
import threading
def _pulse():                       # heartbeat every 60 s, also while a job runs
    while True:
        time.sleep(60); beat()
threading.Thread(target=_pulse, daemon=True).start()

PARAMS = {"Qwen/Qwen3-8B": 8.2, "Qwen/Qwen3-14B": 14.8, "Qwen/Qwen3-32B": 32.8, "google/gemma-3-27b-it": 27.4,
          "google/gemma-3-12b-it": 12.2, "allenai/Olmo-3.1-32B-Instruct": 32.2, "allenai/Olmo-3-7B-Instruct": 7.3, "openai/gpt-oss-20b": 21.0}
LENS = {"Qwen/Qwen3-8B": "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
        "Qwen/Qwen3-14B": "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt",
        "Qwen/Qwen3-32B": "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt",
        "google/gemma-3-27b-it": "gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt",
        "openai/gpt-oss-20b": "gpt-oss-20b/jlens/Salesforce-wikitext/gpt-oss-20b_jacobian_lens.pt"}

def find_layers(m):
    best = None
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0 and name.endswith("layers"):
            if best is None or len(mod) > len(best[1]): best = (name, mod)
    return best

def unload():
    old = CUR.get("id")
    for k in list(CUR): CUR.pop(k)
    JL.clear(); gc.collect(); torch.cuda.empty_cache(); return old

def get_model(mid, prec="auto"):
    if CUR.get("id") == mid: return CUR
    old = unload()
    if old: shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub/models--" + old.replace("/", "--")), ignore_errors=True)
    if prec == "auto": prec = "bf16" if VRAM >= PARAMS.get(mid, 30) * 2 * 1.12 + 6 else "4bit"
    beat(f"loading {mid} ({prec})"); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid, token=_tok)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = dict(token=_tok, device_map="cuda", torch_dtype=torch.bfloat16)
    if prec == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(mid, **kw).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(mid, **kw).eval()
    for p in model.parameters(): p.requires_grad_(False)
    _, layers = find_layers(model)
    cfg = getattr(model.config, "text_config", None) or model.config
    CUR.update(id=mid, model=model, tok=tok, layers=layers, n_layers=len(layers), d=cfg.hidden_size, prec=prec)
    CUR["digits"] = [tok.encode(str(k), add_special_tokens=False) for k in range(1, 10)]
    log(f"loaded {mid} {prec} in {time.time()-t0:.0f}s; {len(layers)} layers; digits single-token: {all(len(x)==1 for x in CUR['digits'])}")
    return CUR

def chat(user, system=None, gen=True):
    tok = CUR["tok"]; msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    except Exception:   # templates without a system role: fold it into the user turn
        msgs = [{"role": "user", "content": (system + "\n\n" if system else "") + user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)

class Hook:
    """Context manager. add: {layer: vector} added to the output of decoder layer `layer` (0-based) at every position;
    proj: {layer: unit vector} projected out of that output (ablation). Vectors are float tensors of size d."""
    def __init__(self, add=None, proj=None): self.add, self.proj, self.hs = add or {}, proj or {}, []
    def __enter__(self):
        for l in sorted(set(self.add) | set(self.proj)):
            a = self.add.get(l); u = self.proj.get(l)
            a = None if a is None else a.to("cuda", torch.bfloat16); u = None if u is None else F.normalize(u.float(), dim=-1).to("cuda", torch.bfloat16)
            def hook(mod, inp, out, a=a, u=u):
                h = out[0] if isinstance(out, tuple) else out
                if u is not None: h = h - (h @ u).unsqueeze(-1) * u
                if a is not None: h = h + a
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.hs.append(CUR["layers"][l].register_forward_hook(hook))
        return self
    def __exit__(self, *e):
        for h in self.hs: h.remove()

@torch.no_grad()
def run_prompts(prompts, prefix='{"rating": ', spans=None, layers=None, want_states=True, bs=8, hook=None):
    """One forward pass per prompt (batched). prompts: chat-formatted strings; prefix is appended (the assistant starts its answer)
    and the next-token distribution over the digits 1-9 gives the rating expected value (ev) and the probability mass on digits.
    spans: list (one per prompt) of {name: substring}; states are averaged over each substring's tokens.
    Returns dict(ev, mass, dec=[n, L, d] states at the last token, span_<name>=[n, L, d]); L = hidden-state indices in `layers`."""
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
            hs = torch.stack([out.hidden_states[l] for l in layers], 1)          # [B, L, T, d]
            dec.append(hs[:, :, -1].half().cpu())
            if spans is not None:
                for i, e in enumerate(encs):
                    n = len(e["input_ids"]); offs = e["offset_mapping"]
                    for name, sub in (spans[k + i] or {}).items():
                        s = texts[i].find(sub)
                        if s < 0: continue
                        idx = [T - n + j for j, (a, b) in enumerate(offs) if b > s and a < s + len(sub) and b > a]   # tokens overlapping the span
                        if idx: sp.setdefault(name, {})[k + i] = hs[i][:, idx].float().mean(1).half().cpu()
        del out
    res = dict(ev=np.array(ev), mass=np.array(mass), layers=layers)
    if want_states:
        res["dec"] = torch.cat(dec)
        for name, dct in sp.items():
            res["span_" + name] = torch.stack([dct.get(i, torch.full((len(layers), CUR["d"]), float("nan"), dtype=torch.float16)) for i in range(len(prompts))])
    return res

@torch.no_grad()
def gen_text(prompts, n=1, max_new_tokens=48, sample=True, temp=0.7, top_p=0.8, top_k=20, bs=8, hook=None):
    tok, model = CUR["tok"], CUR["model"]; tok.padding_side = "left"; outs = []
    for k in range(0, len(prompts), bs):
        enc = tok(prompts[k:k + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, num_return_sequences=n)
        kw.update(dict(do_sample=True, temperature=temp, top_p=top_p, top_k=top_k) if sample else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
        with (hook if hook is not None else Hook()):
            o = model.generate(**enc, **kw)
        outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs

JL = {}
def jlens_read(prompts, words, positions=(-1,)):
    """Jacobian-lens log-probabilities of single-token words at the given positions: array [n, L, W] (mean over positions)."""
    import jlens
    if "lens" not in JL:
        JL["jm"] = jlens.from_hf(CUR["model"], CUR["tok"])
        JL["lens"] = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=LENS[CUR["id"]])
    tok = CUR["tok"]; wid = [tok.encode(w, add_special_tokens=False) for w in words]
    keep = [i for i, w in enumerate(wid) if len(w) == 1]; ids = [wid[i][0] for i in keep]
    out = []
    for p in prompts:
        ll, _, _ = JL["lens"].apply(JL["jm"], p, positions=list(positions))
        Ls = sorted(ll, key=lambda x: int(x))
        out.append(np.stack([torch.log_softmax(ll[l].float(), -1)[:, ids].mean(0).cpu().numpy() for l in Ls]))
    return np.stack(out), [words[i] for i in keep], [int(l) for l in Ls]

_EMB = {}
def embed_texts(texts, name="sentence-transformers/all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer
    if name not in _EMB: _EMB[name] = SentenceTransformer(name, device="cuda")
    return _EMB[name].encode(texts, normalize_embeddings=True, batch_size=64)

def load_mat(name): return json.load(open(f"{ROOT}/materials/{name}", encoding="utf-8"))
def save_json(name, obj):
    with open(f"{OUT}/{name}", "w", encoding="utf-8") as f: json.dump(obj, f, indent=1, ensure_ascii=False, default=float)
def save_df(name, df): df.to_csv(f"{OUT}/{name}", index=False)
def save_pt(name, obj): torch.save(obj, f"{OUT}/{name}")
beat("helpers ready"); print("helpers ready")

def run_loop():
    global OUT
    log("runner started on " + GPU)
    while True:
        beat()
        try:
            if json.load(open(f"{ROOT}/control.json")).get("stop"): beat("stopped by control.json"); break
        except Exception: pass
        jobs = sorted(f for f in os.listdir(f"{ROOT}/jobs_pending") if f.endswith(".py"))
        if not jobs:
            time.sleep(20); continue
        jf = jobs[0]; name = jf[:-3]
        try:
            src = open(f"{ROOT}/jobs_pending/{jf}", encoding="utf-8").read()
            shutil.move(f"{ROOT}/jobs_pending/{jf}", f"{ROOT}/jobs_running/{jf}")
        except Exception as ex:
            log(f"could not take {jf}: {ex}"); time.sleep(10); continue
        OUT = f"{ROOT}/results/{name}"; os.makedirs(OUT, exist_ok=True)
        STATE.update(job=name, job_started=time.strftime("%H:%M:%S")); beat("running"); log("start " + name); t0 = time.time()
        try:
            exec(compile(src, jf, "exec"), globals())
            shutil.move(f"{ROOT}/jobs_running/{jf}", f"{ROOT}/jobs_done/{jf}")
            STATE["done"].append(f"{name} ({time.time()-t0:.0f}s)"); log(f"done {name} in {time.time()-t0:.0f}s")
        except Exception:
            tb = traceback.format_exc()
            with open(f"{OUT}/ERROR.txt", "w") as f: f.write(tb)
            try: shutil.move(f"{ROOT}/jobs_running/{jf}", f"{ROOT}/jobs_failed/{jf}")
            except Exception: pass
            STATE["failed"].append(name); STATE["last_error"] = tb[-1500:]; log(f"FAILED {name}: {tb.splitlines()[-1]}")
        STATE["job"] = None; beat("idle"); gc.collect(); torch.cuda.empty_cache()
'''
(HERE / 'runner_lib.py').write_text(lib.strip('\n') + '\n', encoding='utf-8', newline='\n')
sha = hashlib.sha256((HERE / 'runner_lib.py').read_bytes()).hexdigest()
for c in cells: c['source'] = c['source'].replace('@@SHA@@', sha)
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'machine_shape': 'hm', 'provenance': []},
      'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 0}
for c in nb['cells']:
    c['source'] = [l + '\n' for l in c['source'].split('\n')]; c['source'][-1] = c['source'][-1].rstrip('\n')
(HERE / 'runner_v1.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')
print('wrote runner_v1.ipynb')
