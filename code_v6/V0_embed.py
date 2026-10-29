# V0 (Study 1-2 text baseline): Qwen3-Embedding-8B embeddings of both texts of every v6 pair, in the order of V1's pairs_meta.csv (high,
# low, high, low, ...), for the text-baseline ladder and for partialling the encoder out of the cross-model geometry.
SMALL = MOCK_SMALL; EMB = C6("V_EMB", "Qwen/Qwen3-Embedding-8B"); T0 = time.time()
_free_gpu()
from sentence_transformers import SentenceTransformer
ST_M = SentenceTransformer(EMB, device="cuda" if torch.cuda.is_available() else "cpu", model_kwargs={"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {})
P = pairs6(SMALL); texts = [t for p in P for t in (p["high"], p["low"])]
pd.DataFrame([{k: p[k] for k in ("pid", "writer", "construct", "role", "low_type", "domain")} for p in P]).to_csv(f"{OUT}/pairs_meta.csv", index=False)
Em = torch.tensor(ST_M.encode(texts, normalize_embeddings=True, batch_size=32)).half()
torch.save({"emb": Em, "encoder": EMB, "order": "high,low per pair"}, f"{OUT}/emb_v6.pt"); save_json("summary.json", {"encoder": EMB, "n": len(texts), "dim": int(Em.shape[1]), "wall_s": round(time.time() - T0)})
open(f"{OUT}/digest.md", "w", encoding="utf-8").write(f"# V0 embeddings ({EMB}): {len(texts)} texts, dim {Em.shape[1]}\n")
del ST_M; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None; log("V0 done"); beat("V0 ok")
