"""Run the OpenRouter runner locally (tests): reads the key from the environment variable OPENROUTER_API_KEY, else from ../MVCA_X_app_v4/.env (never printed), uses or_runner/local_root as ROOT.
Usage: python or_runner/local_run.py <job.jsonl> [...]   (copies the jobs in, runs until idle, then exits)"""
import os, re, sys, json, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent; ENV = HERE.parent.parent / "MVCA_X_app_v4" / ".env"   # REPOSITORY: the environment variable OPENROUTER_API_KEY is read first (next line)
key = os.environ.get("OPENROUTER_API_KEY") or next(m.group(1) for l in open(ENV, encoding="utf-8") if (m := re.match(r'\s*OPENROUTER_API_KEY\s*=\s*"?([^"\s]+)"?', l)))
root = HERE / "local_root"; (root / "jobs").mkdir(parents=True, exist_ok=True)
for j in sys.argv[1:]: shutil.copy(j, root / "jobs" / Path(j).name)
(root / "control.json").write_text(json.dumps({"state": "run", "budget_usd": float(os.environ.get("OR_BUDGET", "1.0")), "concurrency": {"default": 10}, "exit_when_idle": True}), encoding="utf-8")
g = {"OR_ROOT": str(root), "OR_KEY": key, "OR_LIB_SHA": "local", "__name__": "or_runner_lib"}
exec(compile((HERE / "or_runner_lib.py").read_text(encoding="utf-8"), "or_runner_lib.py", "exec"), g)
g["run_forever"]()
