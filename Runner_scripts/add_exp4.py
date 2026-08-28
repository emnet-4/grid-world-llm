"""
Adds Exp 4 to run folders that already exist.

Only runs the three Exp 4 steps. Everything else -- the grid, the filler, the
Exp 1/2/3 measurements -- is already on disk and is left alone.

Skips any folder that already has RL_Exp4_Time3.json, so it is safe to re-run
after an interruption.
"""

import subprocess
import sys
import os
import json
import shutil
import time

STEPS = [
    "step3b_teaching_map.py",
    "step4b_exp4_time2.py",
    "step7_time3_exp4.py",
]

SUPPORT = ["probe_questions.py", "benchmarks.py"]

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SOURCE_DIR, "runs_index.json")) as f:
    index = json.load(f)

print(f"Found {len(index)} existing runs\n")

done = skipped = failed = 0
t_start = time.time()

for entry in index:
    folder = os.path.join(SOURCE_DIR, entry["folder"])
    label = f"{entry['set']} run{entry['run']}"

    if not os.path.isdir(folder):
        print(f"[skip] {label} -- folder missing")
        skipped += 1
        continue

    if os.path.exists(os.path.join(folder, "RL_Exp4_Time3.json")):
        print(f"[skip] {label} -- Exp 4 already done")
        skipped += 1
        continue

    # these must exist for Exp 4 to run
    missing = [f for f in ["grid_world.json", "filler_conversation.json"]
               if not os.path.exists(os.path.join(folder, f))]
    if missing:
        print(f"[skip] {label} -- missing {missing}")
        skipped += 1
        continue

    # refresh the shared modules in case they have changed
    for f in SUPPORT:
        shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(folder, f))

    env = os.environ.copy()
    env["GRID_SEED"] = str(entry["seed"])
    env["TEACHER_MODEL"] = entry["teacher"]
    env["STUDENT_MODEL"] = entry["student"]

    print(f"\n--- {label} ---")
    t0 = time.time()
    ok = True
    for script in STEPS:
        r = subprocess.run([sys.executable, os.path.join(SOURCE_DIR, script)],
                           cwd=folder, env=env)
        if r.returncode != 0:
            print(f"  !! {script} failed")
            ok = False
            break

    if ok:
        done += 1
        print(f"    done in {time.time() - t0:.0f}s")
    else:
        failed += 1

print(f"\n{'=' * 55}")
print(f"{done} runs got Exp 4, {skipped} skipped, {failed} failed")
print(f"total {(time.time() - t_start) / 60:.0f} min")
print("Now run analyze_all.py")
print(f"{'=' * 55}")
