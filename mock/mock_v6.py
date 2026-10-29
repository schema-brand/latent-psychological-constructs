"""Local dry run of the v6 GPU jobs (make_v6_jobs.py) on CPU: the real tokenizer and chat template of each job's model, with a tiny
randomly initialised decoder (Qwen3 architecture, 10 layers, d = 64) in place of the weights. Checks that every step runs end to end on
the v6 materials (spans found, hooks applied, summaries written). Fake jlens and sentence-transformers modules.
Usage: python mock/mock_v6.py [job-name-prefix ...]     (default: every job in jobs6/smoke)"""
import os, sys, json, time, glob, shutil, itertools, math, re, types, tempfile, traceback, subprocess, gc
import numpy as np, pandas as pd, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
ROOT = tempfile.mkdtemp(prefix="mock6_"); os.makedirs(f"{ROOT}/materials/v6"); os.makedirs(f"{ROOT}/results")
for f in glob.glob(os.path.join(BASE, "materials_v6", "upload_v6", "*.json")): shutil.copy(f, f"{ROOT}/materials/v6/")
for f in ("latent_lib.py", "test_lib3.py", "test_lib4.py", "runner_lib.py"): shutil.copy(os.path.join(BASE, f), f"{ROOT}/{f}")
shutil.copytree(os.path.join(BASE, "code_v6"), f"{ROOT}/code_v6")
torch.cuda.memory_allocated = lambda *a, **k: 0
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
fake_jl = types.ModuleType("jlens")
class _FL:
    def apply(self, jm, p, positions=(-1,)): return ({str(l): torch.randn(1, 2000) for l in range(10)}, None, None)
fake_jl.from_hf = lambda m, t: None; fake_jl.JacobianLens = types.SimpleNamespace(from_pretrained=lambda *a, **k: _FL()); sys.modules["jlens"] = fake_jl
fake_st = types.ModuleType("sentence_transformers")
class _ST:
    def __init__(self, *a, **k): pass
    def encode(self, texts, **k): x = np.random.default_rng(len(texts)).normal(size=(len(texts), 32)); return x / np.linalg.norm(x, axis=1, keepdims=True)
fake_st.SentenceTransformer = _ST; sys.modules["sentence_transformers"] = fake_st
G = dict(__name__="__main__", os=os, sys=sys, json=json, time=time, glob=glob, shutil=shutil, itertools=itertools, math=math, re=re, traceback=traceback,
         subprocess=subprocess, gc=gc, np=np, pd=pd, torch=torch, F=F, ROOT=ROOT, GPU="mock-cpu", VRAM=0.0, _tok=None)
exec(compile(open(f"{ROOT}/runner_lib.py", encoding="utf-8").read(), "runner_lib.py", "exec"), G)
_TOKS = {}
def MOCK_LOAD(mid):
    if mid not in _TOKS: _TOKS[mid] = AutoTokenizer.from_pretrained(mid)
    tok = _TOKS[mid]; torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok) + 8, hidden_size=64, intermediate_size=128, num_hidden_layers=10, num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=8192)
    return Qwen3ForCausalLM(cfg).eval(), tok, "mock"
G["MOCK_LOAD"] = MOCK_LOAD
jobs = sorted(glob.glob(os.path.join(BASE, "jobs6", os.environ.get("MOCK_SET", "smoke"), "*.py")))      # MOCK_SET=mock2b for the run 2 code
if len(sys.argv) > 1: jobs = [j for j in jobs if any(os.path.basename(j).startswith(a) for a in sys.argv[1:])]
ok = True
for jf in jobs:
    name = os.path.basename(jf)[:-3]; G["OUT"] = f"{ROOT}/results/{name}"; os.makedirs(G["OUT"], exist_ok=True); t0 = time.time()
    try:
        exec(compile(open(jf, encoding="utf-8").read(), jf, "exec"), G)
        R = json.load(open(f"{G['OUT']}/summary.json", encoding="utf-8")); bad = {k: v for k, v in R.items() if isinstance(v, str) and v.startswith("FAILED")}
        print(f"{'OK  ' if not bad else 'FAIL'} {name} ({time.time() - t0:.0f}s): {json.dumps({k: v for k, v in R.items() if k in R.get('cfg', {}) or k.startswith('V')})[:900]}")
        for k in bad:
            print(open(f"{G['OUT']}/{k}/ERROR.txt").read()[-2500:]); ok = False
    except Exception:
        traceback.print_exc(); ok = False; print("FAIL", name)
print("ROOT", ROOT)
sys.exit(0 if ok else 1)
