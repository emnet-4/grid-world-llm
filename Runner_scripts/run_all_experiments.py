"""
Runs the full experiment N times in each model direction.

Two sets:
    Set A -- Llama teaches, Mistral learns
    Set B -- Mistral teaches, Llama learns

Each set gets N_RUNS independent runs with a fresh room layout, so results
can be reported with error bars instead of as single numbers.

The filler conversation is generated ONCE and reused across every run. It is
unrelated to the task by construction, so regenerating it per run would cost
hundreds of model calls for nothing.

Each run writes its own folder containing all five measurement files.
"""

import subprocess
import sys
import os
import json
import shutil
import time

N_RUNS = 30  #change for testing, but 30 is the number used in the paper

MODEL_SETS = [
    {"name": "A_llama_teaches",   "teacher": "llama3.2", "student": "mistral"},
    {"name": "B_mistral_teaches", "teacher": "mistral",  "student": "llama3.2"},
]

STEPS = [
    "step1_generate_grid.py",
    "step2_before_probe.py",
    "step3_teaching.py",
    "step4_after_probe.py",
    "step6b_save_representation.py",
    "step7_time3_exp1.py",
    "step7_time3_exp2.py",
    "step9_exp3_probe.py",
    "step3b_teaching_map.py",
    "step4b_exp4_time2.py",
    "step7_time3_exp4.py",
]

SUPPORT = ["probe_questions.py", "benchmarks.py"]

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
FILLER = os.path.join(SOURCE_DIR, "filler_conversation.json")


def ensure_filler():
    if os.path.exists(FILLER):
        print("Reusing existing filler_conversation.json\n")
        return
    print("Generating filler conversation (once, reused for all runs)...\n")
    r = subprocess.run([sys.executable,
                        os.path.join(SOURCE_DIR, "step6_generate_filler.py")],
                       cwd=SOURCE_DIR)
    if r.returncode != 0:
        sys.exit("Filler generation failed.")


def run_one(folder, seed, teacher, student):
    os.makedirs(folder, exist_ok=True)
    for f in SUPPORT + ["filler_conversation.json"]:
        shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(folder, f))

    env = os.environ.copy()
    env["GRID_SEED"] = str(seed)
    env["TEACHER_MODEL"] = teacher
    env["STUDENT_MODEL"] = student

    for script in STEPS:
        r = subprocess.run([sys.executable, os.path.join(SOURCE_DIR, script)],
                           cwd=folder, env=env)
        if r.returncode != 0:
            print(f"  !! {script} failed")
            return False
    return True


if __name__ == "__main__":
    ensure_filler()

    index = []
    t_start = time.time()

    for mset in MODEL_SETS:
        print(f"\n{'#' * 62}")
        print(f"SET {mset['name']}   teacher={mset['teacher']}   student={mset['student']}")
        print(f"{'#' * 62}")

        for run in range(1, N_RUNS + 1):
            folder = os.path.join(SOURCE_DIR, f"runs_{mset['name']}_run{run}")
            seed = run * 17

            print(f"\n--- {mset['name']} run {run}/{N_RUNS} (seed {seed}) ---")
            t0 = time.time()
            ok = run_one(folder, seed, mset["teacher"], mset["student"])
            print(f"    {'done' if ok else 'FAILED'} in {time.time() - t0:.0f}s")

            if ok:
                index.append({
                    "set": mset["name"],
                    "teacher": mset["teacher"],
                    "student": mset["student"],
                    "run": run,
                    "seed": seed,
                    "folder": os.path.basename(folder),
                })

            # save as we go, so a crash doesn't lose the index
            with open(os.path.join(SOURCE_DIR, "runs_index.json"), "w") as f:
                json.dump(index, f, indent=2)

    print(f"\n{'#' * 62}")
    print(f"{len(index)} runs completed in {(time.time() - t_start) / 60:.0f} min")
    print("Saved runs_index.json -- now run analyze_all.py")
    print(f"{'#' * 62}")
