"""Materials v6 QC report (Chinese summary for the PI + tables): writes materials_v6/QC_REPORT_v6.md from pairs_v6_raw.jsonl, pairs_v6.jsonl,
variants_v6_raw.jsonl, variants_v6.jsonl, spec_v6.json and, when present, the rating summary. Usage: python qc_report_v6.py"""
import json, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; M6 = HERE / "materials_v6"
SOFT = {"ending_differs", "number_rule"}
C = json.loads((M6 / "spec_v6.json").read_text(encoding="utf-8"))["constructs"]
raw = [json.loads(l) for l in open(M6 / "pairs_v6_raw.jsonl", encoding="utf-8")]
fin = [json.loads(l) for l in open(M6 / "pairs_v6.jsonl", encoding="utf-8")]
vraw = [json.loads(l) for l in open(M6 / "variants_v6_raw.jsonl", encoding="utf-8")]
vfin = [json.loads(l) for l in open(M6 / "variants_v6.jsonl", encoding="utf-8")]
W = ["deepseek", "kimi", "muse", "grok", "minimax", "claude"]
L = ["# 材料 v6 质检报告", "", f"构念 {len(C)} 个（中介 {sum(c['role']=='mediator' for c in C.values())}，竞争解释 {sum(c['role']=='rival' for c in C.values())}，对照 {sum(c['role']=='comparison' for c in C.values())}）。", ""]
L += ["## 对比情境", "", "| 写作者 | 写出的对数 | 通过质检 | 通过率 | 最终保留 |", "|---|---|---|---|---|"]
for w in W:
    a = [r for r in raw if r["writer"] == w]; ok = [r for r in a if not [f for f in r["flags"] if f not in SOFT]]
    L.append(f"| {w} | {len(a)} | {len(ok)} | {len(ok)/max(1,len(a)):.2f} | {sum(r['writer']==w for r in fin)} |")
L += ["", f"合计保留 {len(fin)} 对。", "", "不合格原因（全部写作者）：" + "，".join(f"{k} {v}" for k, v in collections.Counter(f for r in raw for f in r["flags"] if f not in SOFT).most_common()), ""]
per = collections.Counter(r["construct"] for r in fin)
short = sorted(((per[c], c) for c in C), key=lambda x: x[0])[:12]
L += ["每个构念的保留对数最少的 12 个：" + "，".join(f"{c} {n}" for n, c in short), ""]
L += ["## 实验改写", "", "| 写作者 | 写出的版本 | 通过 |", "|---|---|---|"]
for w in W:
    a = [r for r in vraw if r["writer"] == w]; L.append(f"| {w} | {len(a)} | {sum(not r['hard_fail'] for r in a)} |")
cnt = collections.Counter(r["experiment"] for r in vfin)
L += ["", f"保留 {len(vfin)} 个改写版本；每个实验的版本数：最少 {min(cnt.values()) if cnt else 0}，最多 {max(cnt.values()) if cnt else 0}；没有改写的实验 {60-len(cnt)} 个。", ""]
(M6 / "QC_REPORT_v6.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
