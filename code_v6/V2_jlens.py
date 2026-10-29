# V2 (Study 1, content): J-lens readout of every construct contrast, for models with a fitted lens (LENS_T: Qwen3.6-27B, Gemma 4 31B).
# Per construct, 2 pairs per writer (one with each low-pole type); the lens reads the state at the last token of the high and of the low
# text (text as a user turn, no question) at 50-100% of depth; tokens raised by the high text over the low text, averaged over pairs.
MID, REV, SMALL = V_MODEL, V_REV, MOCK_SMALL; T0 = time.time()
load6(MID, REV)
if MID not in LENS_T:
    save_json("summary.json", {"model": MID, "skipped": "no fitted J-lens for this model"})
else:
    import jlens
    if "lens" not in JL:
        JL["jm"] = jlens.from_hf(CUR["model"], CUR["tok"]); JL["lens"] = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=LENS_T[MID])
    P = pairs6(); CONS = cons6()[:3] if SMALL else cons6(); WR = sorted({p["writer"] for p in P}); nL = CUR["n_layers"]
    SEL = sorted({int(round(f * (nL - 1))) for f in (0.5, 0.625, 0.75, 0.875, 0.94, 1.0)})
    def cut(t): p = chat6(t, None, gen=False); return p[:p.rfind(t) + len(t)]
    R = {"model": MID, "layers": SEL, "top": {}, "n_pairs": {}}; tk = CUR["tok"]; keep = None
    for c in CONS:
        Sx = {"high": None, "low": None}; n = 0
        for w in (WR[:1] if SMALL else WR):
            pp = [p for p in P if p["construct"] == c and p["writer"] == w]
            pick = ([p for p in pp if p["low_type"] == "L1"][:1] + [p for p in pp if p["low_type"] == "L4"][:1]) or pp[:2]
            for p in pick[:1] if SMALL else pick:
                for side in ("high", "low"):
                    ll, _, _ = JL["lens"].apply(JL["jm"], cut(p[side]), positions=[-1])
                    keep = [l for l in sorted(ll, key=lambda x: int(x)) if int(l) in SEL]
                    lp = torch.stack([torch.log_softmax(ll[l][0].float(), -1) for l in keep]).cpu()
                    Sx[side] = lp if Sx[side] is None else Sx[side] + lp
                n += 1
        if not n: continue
        H, Lo = Sx["high"] / n, Sx["low"] / n; Dd = H - Lo; R["top"][c] = {}; R["n_pairs"][c] = n
        for jj, l in enumerate(keep):
            sc = Dd[jj] * (H[jj] > -18).float() - 99 * (H[jj] <= -18).float(); top = torch.topk(sc, 50)
            R["top"][c][str(l)] = [(tk.decode([int(t)]), round(float(sc[t]), 2)) for t in top.indices]
        beat(f"V2 {c}")
    R["wall_s"] = round(time.time() - T0); save_json("summary.json", R)
    L = [f"# V2 J-lens readout of the construct contrasts ({MID})", ""]
    for c, v in R["top"].items():
        l = sorted(v, key=int)[len(v) // 2]; L.append(f"- {c} (layer {l}): {[t for t, _ in v[l][:12]]}")
    open(f"{OUT}/digest.md", "w", encoding="utf-8").write("\n".join(L)); log(f"V2 {MID} done"); beat("V2 ok")
