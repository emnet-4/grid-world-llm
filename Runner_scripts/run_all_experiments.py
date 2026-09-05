"""
Run the core experiment N times in each teaching direction.

Two sets, so that each model takes a turn as Teacher and as Student:

    Set A   Llama teaches, Mistral learns
    Set B   Mistral teaches, Llama learns

Every run gets a fresh room layout. runs_index.json is written after each
completed run, so a batch that is interrupted still leaves usable results and
the analysis scripts will work on whatever finished.

WHAT THIS INCLUDES

The core eight steps: build the world, probe, teach, probe again, save the
Student's representation, then the three flushed conditions and the map
condition.

WHAT THIS DOES NOT INCLUDE

Two extensions are deliberately left out, because they have requirements this
runner cannot assume:

    add_teaching_formats.py   the coordinate and trail teaching formats.
                              Roughly two thirds of a batch each, so running
                              all three would triple the time.

    add_exp5.py               LoRA fine-tuning. Needs a GPU in practice, plus
                              peft, accelerate and a HuggingFace login.

Both operate on folders this runner has already created, so the core batch is
never at risk if an extension fails.

RUNTIME

About seven hours per direction at N_RUNS = 30, on a laptop. Lower N_RUNS to
shorten it; N_RUNS = 1 runs the whole pipeline once in about ten minutes, which
is the quickest way to check the setup works.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

N_RUNS = 30

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

# copied into every run folder so the step scripts can import them
SUPPORT = ["probe_questions.py", "benchmarks.py"]

HERE = os.path.dirname(os.path.abspath(__file__))
FILLER = os.path.join(HERE, "filler_conversation.json")


def ensure_filler() -> None:
    """
    The filler is unrelated to the task by construction, so one is generated
    once and reused. Regenerating it per run would cost hundreds of model calls
    for nothing.
    """
    if os.path.exists(FILLER):
        print("Reusing existing filler_conversation.json\n")
        return

    print("Generating filler conversation (once, reused for every run)\n")
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "step6_generate_filler.py")],
        cwd=HERE)
    if result.returncode != 0:
        sys.exit("Filler generation failed.")


def run_once(folder: str, seed: int, teacher: str, student: str) -> bool:
    os.makedirs(folder, exist_ok=True)

    for name in SUPPORT + ["filler_conversation.json"]:
        shutil.copy(os.path.join(HERE, name), os.path.join(folder, name))

    env = os.environ.copy()
    env["GRID_SEED"] = str(seed)
    env["TEACHER_MODEL"] = teacher
    env["STUDENT_MODEL"] = student

    for script in STEPS:
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, script)], cwd=folder, env=env)
        if result.returncode != 0:
            print(f"  !! {script} failed")
            return False
    return True


def main() -> None:
    ensure_filler()

    index: list[dict] = []
    started = time.time()

    for model_set in MODEL_SETS:
        print(f"\n{'#' * 62}")
        print(f"SET {model_set['name']}   "
              f"teacher {model_set['teacher']}, student {model_set['student']}")
        print(f"{'#' * 62}")

        for run in range(1, N_RUNS + 1):
            folder = os.path.join(HERE, f"runs_{model_set['name']}_run{run}")
            seed = run * 17

            print(f"\n--- {model_set['name']} run {run}/{N_RUNS} (seed {seed}) ---")
            t0 = time.time()
            ok = run_once(folder, seed, model_set["teacher"], model_set["student"])
            print(f"    {'done' if ok else 'FAILED'} in {time.time() - t0:.0f}s")

            if ok:
                index.append({
                    "set": model_set["name"],
                    "teacher": model_set["teacher"],
                    "student": model_set["student"],
                    "run": run,
                    "seed": seed,
                    "folder": os.path.basename(folder),
                })

            # written after every run, so an interrupted batch stays usable
            with open(os.path.join(HERE, "runs_index.json"), "w") as fh:
                json.dump(index, fh, indent=2)

    minutes = (time.time() - started) / 60
    print(f"\n{'#' * 62}")
    print(f"{len(index)} runs completed in {minutes:.0f} min")
    print("Saved runs_index.json")
    print("\nNext:")
    print("  python3 baseline_agents.py")
    print("  python3 analyze_all.py")
    print("\nOptional extensions:")
    print("  FORMATS=trail python3 add_teaching_formats.py")
    print("  python3 add_exp5.py                    (needs a GPU)")
    print(f"{'#' * 62}")


if __name__ == "__main__":
    main()
