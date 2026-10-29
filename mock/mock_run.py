"""Local dry run of the v3 job scripts with a FAKE model (random states, random digit probabilities). No model is loaded;
this only checks that the job code runs end to end on materials in the v3 format.
Usage: python mock/mock_run.py <materials_dir or 'synthetic'> T02 T03 T04 T05 T06 T07"""
import os, sys, json, time, glob, shutil, itertools, math, re, types, tempfile
import numpy as np, pandas as pd, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
ROOT = tempfile.mkdtemp(prefix="mockroot_"); os.makedirs(f"{ROOT}/materials/v3a"); os.makedirs(f"{ROOT}/results")
src = sys.argv[1]
if src == "synthetic":
    rng = np.random.default_rng(0); W = "alpha beta gamma delta table chair window street river cup lamp coat shop friend road".split()
    def txt(n): return " ".join(rng.choice(W, n)) + "."
    fams = ["ownership", "loss", "reciprocity", "self_perception", "anchoring"]
    for k in range(10):
        fam = fams[k // 2]; vs = []
        for j in range(12):
            L = ["short", "medium", "long"][j // 4]; a = txt(8); b = txt(8); r1 = txt(5); r2 = txt(5); pre = txt(20); mid = txt(10)
            vs.append(dict(vid=f"v{j+1:02d}", setting="s", length=L, body_treatment=pre + " " + a + " " + mid + " " + r1, body_control=pre + " " + b + " " + mid + " " + r2,
                           x_span_treatment=[a, r1], x_span_control=[b, r2], question="How much? 1 = none, 9 = a lot."))
        st = {k2: [txt(10) for _ in range(4)] for k2 in ("orig_mediator", "orig_denial", "kf_mediator", "kf_denial", "kf_matched_control")}
        json.dump(dict(id=f"exp{k}", family=fam, mediator="m", x="x", predicted_sign=1, original_dv="d", leakage_terms=[], statements=st, variants=vs),
                  open(f"{ROOT}/materials/v3a/exp_exp{k}.json", "w"))
    for fam in fams:
        json.dump(dict(family=fam, mediator="m", pairs=[dict(pid=f"c{i}", domain="d", high=txt(40), low=txt(40)) for i in range(20)]), open(f"{ROOT}/materials/v3a/family_{fam}.json", "w"))
    aw = dict(study_prefixes=[txt(15) for _ in range(6)], neutral_prefixes=[txt(15) for _ in range(6)],
              study_vs_everyday=[dict(pid=f"a{i}", everyday=txt(60), study=txt(60), question="Q? 1 = no, 9 = yes.") for i in range(30)],
              hypothetical_vs_real=[dict(pid=f"h{i}", real=txt(60), hypothetical=txt(60), question="Q? 1 = no, 9 = yes.") for i in range(30)],
              fiction_vs_self=[dict(pid=f"f{i}", self=txt(60), fiction=txt(60), question_self="Q? 1 = no, 9 = yes.", question_fiction="Q? 1 = no, 9 = yes.") for i in range(20)],
              awareness_questions=["Is this a study? 1 = no, 9 = yes."] * 5)
    json.dump(aw, open(f"{ROOT}/materials/v3a/awareness.json", "w"))
else:
    for f in glob.glob(os.path.join(src, "*.json")): shutil.copy(f, f"{ROOT}/materials/v3a/")
if not os.path.exists(f"{ROOT}/materials/v3a/profiles30.json"):
    pr = json.load(open(os.path.join(BASE, "profiles30.json"), encoding="utf-8")) if os.path.exists(os.path.join(BASE, "profiles30.json")) else [dict(id=f"p{i}", persona=f"Person {i} is a test persona.") for i in range(30)]
    json.dump(pr, open(f"{ROOT}/materials/v3a/profiles30.json", "w"))
shutil.copy(os.path.join(BASE, "latent_lib.py"), f"{ROOT}/latent_lib.py")
if os.path.isdir(os.path.join(BASE, "materials_v3B")):
    os.makedirs(f"{ROOT}/materials/v3b", exist_ok=True)
    for f_ in glob.glob(os.path.join(BASE, "materials_v3B", "*.json")): shutil.copy(f_, f"{ROOT}/materials/v3b/")

# ---- fake runner environment ----
G = dict(__name__="__main__", os=os, sys=sys, json=json, time=time, glob=glob, shutil=shutil, itertools=itertools, math=math, re=re,
         np=np, pd=pd, torch=torch, F=F, ROOT=ROOT, GPU="mock", VRAM=80.0)
NL, DM = 16, 32
class Tok:
    def encode(self, w, add_special_tokens=False): return [abs(hash(w)) % 1000] if len(w.strip()) < 12 else [1, 2]
    def decode(self, ids): return f"<{ids[0]}>"
    def __call__(self, t, **kw): return {"input_ids": [ord(c) % 997 for c in t]}
class FakeModel:
    def __call__(self, input_ids=None, output_hidden_states=True, **kw):
        T = input_ids.shape[1]; return types.SimpleNamespace(hidden_states=[torch.randn(1, T, DM) for _ in range(NL + 1)])
CUR = dict(id="mock/model", n_layers=NL, d=DM, tok=Tok(), layers=[None] * NL, model=FakeModel(), digits=[[i] for i in range(9)])
def chat(user, system=None, gen=True): return f"<sys>{system or ''}</sys><user>{user}</user>" + ("<asst>" if gen else "")
def run_prompts(prompts, prefix='{"rating": ', spans=None, layers=None, want_states=True, bs=8, hook=None):
    n = len(prompts); layers = list(range(NL + 1)) if layers is None else list(layers)
    g = torch.Generator().manual_seed(abs(hash(prompts[0] if prompts else "")) % 2**31)
    res = dict(ev=np.random.default_rng(n).uniform(1, 9, n), mass=np.ones(n), layers=layers)
    if want_states:
        res["dec"] = torch.randn(n, len(layers), DM, generator=g).half()
        if spans is not None:
            names = set(k for s in spans if s for k in s)
            for nm in names: res["span_" + nm] = torch.randn(n, len(layers), DM, generator=g).half()
    return res
class Hook:
    def __init__(self, add=None, proj=None): self.add, self.proj = add or {}, proj or {}
    def __enter__(self):
        for l in list(self.add) + list(self.proj): assert 0 <= l < NL, f"hook layer {l} out of range"
        return self
    def __exit__(self, *e): pass
def gen_text(prompts, n=1, **kw): return ['{"rating": 5}'] * (len(prompts) * n)
def embed_texts(texts, name=None): x = np.random.default_rng(len(texts)).normal(size=(len(texts), 24)); return x / np.linalg.norm(x, axis=1, keepdims=True)
class FakeLens:
    def apply(self, jm, p, positions=(-1,)): return ({str(l): torch.randn(1, 1000) for l in range(NL)}, None, None)
JL = {}
fake_jlens = types.ModuleType("jlens"); fake_jlens.from_hf = lambda m, t: None
fake_jlens.JacobianLens = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeLens()); sys.modules["jlens"] = fake_jlens
LENS = {"mock/model": "x"}
def get_model(mid, prec="auto"): return CUR
def beat(msg=None): pass
def log(s): pass
def load_mat(name): return json.load(open(f"{ROOT}/materials/{name}"))
def save_json(name, obj): json.dump(obj, open(f"{G['OUT']}/{name}", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
def save_df(name, df): df.to_csv(f"{G['OUT']}/{name}", index=False)
def save_pt(name, obj): torch.save(obj, f"{G['OUT']}/{name}")
G.update(CUR=CUR, chat=chat, run_prompts=run_prompts, Hook=Hook, gen_text=gen_text, embed_texts=embed_texts, JL=JL, LENS=LENS,
         get_model=get_model, beat=beat, log=log, load_mat=load_mat, save_json=save_json, save_df=save_df, save_pt=save_pt)
G["save_json"].__globals__.update(G); G["save_df"].__globals__.update(G); G["save_pt"].__globals__.update(G)
ok = True
for job in sys.argv[2:]:
    if job == "QWEN":
        CUR["id"] = "Qwen/Qwen3-32B"; print("mock id = Qwen/Qwen3-32B"); continue
    if job == "CLEAR":
        G.get("CACHE", {}).clear(); print("CACHE cleared"); continue
    f = sorted(glob.glob(os.path.join(BASE, "jobs", job + "_*.py")))[0]
    G["OUT"] = f"{ROOT}/results/{os.path.basename(f)[:-3]}"; os.makedirs(G["OUT"], exist_ok=True)
    t0 = time.time()
    try:
        exec(compile(open(f, encoding="utf-8").read(), f, "exec"), G)
        r = json.load(open(f"{G['OUT']}/summary.json", encoding="utf-8"))
        print(f"OK {job} ({time.time()-t0:.1f}s): summary keys {list(r)[:12]}; size {os.path.getsize(G['OUT'] + '/summary.json')} bytes")
    except Exception as ex:
        import traceback; traceback.print_exc(); ok = False; print(f"FAIL {job}")
print("ROOT", ROOT)
sys.exit(0 if ok else 1)
