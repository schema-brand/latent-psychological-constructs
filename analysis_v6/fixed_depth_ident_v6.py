"""Study 1 identification at a fixed depth (BR4: best-layer selection).
For each of the 16 LLMs, reads the per-layer table of V1_read/digest.md and reports identification
(standardized mean, first of 66) at the best layer and at the recorded layer closest to 50% of the depth,
plus the range over recorded layers between 30% and 70% of the depth."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results_v6"
JOBS = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32",
        "Gemma 4 26B-A4B": "V720_gemma4_26b_a4b", "OLMo 3.1 32B base": "V721_olmo32_base", "Gemma 4 31B base": "V722_gemma4_31b_base",
        "Qwen3.5 27B": "V730_qwen35_27b", "Qwen3.5 9B": "V731_qwen35_9b", "Qwen3.5 4B": "V732_qwen35_4b", "Qwen3.5 2B": "V733_qwen35_2b",
        "Qwen3.5 0.8B": "V734_qwen35_08b", "Gemma 4 E4B": "V735_gemma4_e4b", "OLMo 3 7B Instruct": "V736_olmo7",
        "Qwen3.8 27B": "V740_qwen38", "gpt-oss 20B": "V741_gptoss20", "Granite 4.2 30B": "V742_granite42"}
out = {}
for name, job in JOBS.items():
    f = ROOT / job / "V1_read" / "digest.md"
    txt = f.read_text(encoding="utf-8")
    nl = int(re.search(r"L\* = \d+ of (\d+)", txt).group(1))
    rows = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"^\| (\d+) \| ([\d.]+) / ", txt, re.M)]
    best_l, best = max(rows, key=lambda r: r[1])
    mid_l, mid = min(rows, key=lambda r: abs(r[0] / nl - 0.5))
    band = [v for l, v in rows if 0.3 <= l / nl <= 0.7]
    out[name] = dict(n_layers=nl, best_layer=best_l, best=round(best, 4), mid_layer=mid_l, mid_depth=round(mid_l / nl, 2), mid=round(mid, 4),
                     band30_70_min=round(min(band), 4), band30_70_max=round(max(band), 4), n_band=len(band))
    print(name, out[name])
vals = [v["mid"] for v in out.values()]; gaps = [v["best"] - v["mid"] for v in out.values()]
print("fixed ~50% depth: %.3f to %.3f; best minus fixed: %.3f to %.3f" % (min(vals), max(vals), min(gaps), max(gaps)))
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
