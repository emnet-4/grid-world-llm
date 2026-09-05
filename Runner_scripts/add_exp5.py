"""
Add the fine-tuning condition to run folders that already exist.

Runs step10_exp5_oneshot.py in every folder listed in runs_index.json, skipping
any that already have a result. Nothing else in the folder is touched.

This is kept separate from run_all_experiments.py because it has requirements
the main runner cannot assume:

  - transformers, peft and accelerate, rather than Ollama
  - a HuggingFace login, since both model repositories are gated
  - a GPU in practice. Llama 3.2 3B will train on an M-series Mac, slowly.
    Mistral 7B will not, in any reasonable time.

Because of the last point, this usually runs for one teaching direction rather
than both, and the resulting condition should be reported as covering fewer
runs than the rest.

Usage
-----
    python3 add_exp5.py                          # every run
    MAX_RUNS=5 python3 add_exp5.py               # first five per set
    ONLY_STUDENT=llama3.2 python3 add_exp5.py    # one direction only
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STEP = "step10_exp5_oneshot.py"
SUPPORT = ["probe_questions.py", "benchmarks.py"]
RESULT_FILE = "RL_Exp5_Time3.json"

MAX_RUNS = int(os.environ.get("MAX_RUNS", 0))          # 0 means no limit
ONLY_STUDENT = os.environ.get("ONLY_STUDENT", "").strip()

# The official Llama and Mistral repositories are gated and need a HuggingFace
# account. This mirror is the same model without the gate, so the condition can
# be run without one. Override with HF_MODEL_ID to use something else.
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "unsloth/Llama-3.2-3B-Instruct")


def main() -> None:
    index_path = os.path.join(HERE, "runs_index.json")
    if not os.path.exists(index_path):
        raise SystemExit("runs_index.json not found. Run the main batch first.")

    with open(index_path) as fh:
        index = json.load(fh)

    if ONLY_STUDENT:
        index = [e for e in index if e["student"] == ONLY_STUDENT]
        print(f"Restricted to runs where the Student is {ONLY_STUDENT}")

    if MAX_RUNS:
        by_set: dict[str, list[dict]] = {}
        for entry in index:
            by_set.setdefault(entry["set"], []).append(entry)
        index = [e for entries in by_set.values() for e in entries[:MAX_RUNS]]
        print(f"Restricted to the first {MAX_RUNS} runs per set")

    print(f"Model: {HF_MODEL_ID}")
    print(f"{len(index)} runs to process\n")

    done = skipped = failed = 0
    started = time.time()

    for entry in index:
        folder = os.path.join(HERE, entry["folder"])
        label = f"{entry['set']} run{entry['run']}"

        if not os.path.isdir(folder):
            print(f"[skip] {label}: folder missing")
            skipped += 1
            continue

        if os.path.exists(os.path.join(folder, RESULT_FILE)):
            print(f"[skip] {label}: already done")
            skipped += 1
            continue

        if not os.path.exists(os.path.join(folder, "grid_world.json")):
            print(f"[skip] {label}: no grid_world.json")
            skipped += 1
            continue

        for name in SUPPORT:
            source = os.path.join(HERE, name)
            if os.path.exists(source):
                shutil.copy(source, os.path.join(folder, name))

        env = os.environ.copy()
        env["GRID_SEED"] = str(entry["seed"])
        env["STUDENT_MODEL"] = entry["student"]
        env["HF_MODEL_ID"] = HF_MODEL_ID

        print(f"\n--- {label}  (Student = {entry['student']}) ---")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, STEP)], cwd=folder, env=env)

        if result.returncode == 0:
            done += 1
            print(f"    done in {time.time() - t0:.0f}s")
        else:
            failed += 1
            print(f"    FAILED")

    print(f"\n{'=' * 58}")
    print(f"{done} completed, {skipped} skipped, {failed} failed")
    print(f"total {(time.time() - started) / 60:.0f} min")
    if done:
        print("Now run analyze_all.py")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()
