"""Build or_runner.ipynb (Colab CPU notebook for the OpenRouter runner). Usage: python or_runner/build_or_notebook.py"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent

MD = """# MVCA latent：OpenRouter 材料执行器（or_runner）

1. **Runtime → Change runtime type → CPU**（不需要 GPU）。
2. 左侧钥匙图标 **Secrets**：添加 `OPENROUTER_API_KEY`（值为 OpenRouter 密钥），并打开 *Notebook access*。只需做一次。
3. **Runtime → Run all**。允许挂载 Drive 后，第二格会一直运行，执行 `MVCA_latent/or_runner/jobs/` 里的任务，结果实时写到 `out/`，状态写到 `status.json`。
4. 暂停 / 停止 / 改并发 / 改预算：修改 Drive 上的 `control.json`（Claude 负责），不用动这个笔记本。"""

SETUP = r'''# 1. Drive and key (the key is read from Colab Secrets and never printed)
from google.colab import drive, userdata
drive.mount("/content/drive")
import os, hashlib
OR_ROOT = "/content/drive/MyDrive/MVCA_latent/or_runner"
os.makedirs(OR_ROOT, exist_ok=True)
try:
    OR_KEY = userdata.get("OPENROUTER_API_KEY")
except Exception as e:
    OR_KEY = None
print("key present:", bool(OR_KEY))
assert OR_KEY, "Add OPENROUTER_API_KEY in Secrets (key icon on the left) and enable notebook access, then rerun."'''

RUN = r'''# 2. Run the runner (re-reads the newest or_runner_lib.py from Drive each time this cell starts)
_src = open(f"{OR_ROOT}/or_runner_lib.py", "rb").read()
OR_LIB_SHA = hashlib.sha256(_src).hexdigest()[:16]
print("or_runner_lib sha256:", OR_LIB_SHA)
exec(compile(_src.decode("utf-8"), "or_runner_lib.py", "exec"), globals())
run_forever()'''

def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code": c.update(execution_count=None, outputs=[])
    return c

nb = {"nbformat": 4, "nbformat_minor": 0, "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}}, "cells": [cell("markdown", MD), cell("code", SETUP), cell("code", RUN)]}
(HERE / "or_runner.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("written", HERE / "or_runner.ipynb")
