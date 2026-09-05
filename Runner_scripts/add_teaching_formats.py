"""
Add the three teaching formats to run folders that already exist.

Runs, for every folder in runs_index.json and for each of the three formats,
the teaching and both probes. Skips any format already completed, so it is safe
to re-run after an interruption or to add one format at a time.

The grid, the filler and the earlier conditions are left alone.

Usage
-----
    python3 add_teaching_formats.py                 # all three formats
    FORMATS=trail python3 add_teaching_formats.py   # just one

Runtime is roughly two thirds of a full batch per format, since the probes are
the expensive part and there are two of them rather than three.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

ALL_FORMATS = ["coordinates", "trail", "both"]
FORMATS = os.environ.get("FORMATS", ",".join(ALL_FORMATS)).split(",")

STEPS = ["step3_teaching_formats.py", "step4_probe_formats.py"]
SUPPORT = ["probe_questions.py", "benchmarks.py"]

LABELS = {"coordinates": "Coord", "trail": "Trail", "both": "Both"}


def already_done(folder: str, fmt: str) -> bool:
    label = LABELS[fmt]
    return os.path.exists(os.path.join(folder, f"RL_{label}_Time3.json"))


def main() -> None:
    with open(os.path.join(HERE, "runs_index.json")) as fh:
        index = json.load(fh)

    print(f"{len(index)} runs, formats: {', '.join(FORMATS)}\n")

    done = skipped = failed = 0
    started = time.time()

    for entry in index:
        folder = os.path.join(HERE, entry["folder"])
        if not os.path.isdir(folder):
            print(f"[skip] {entry['folder']} missing")
            skipped += 1
            continue

        for support in SUPPORT:
            source = os.path.join(HERE, support)
            if os.path.exists(source):
                shutil.copy(source, os.path.join(folder, support))

        for fmt in FORMATS:
            fmt = fmt.strip()
            if fmt not in ALL_FORMATS:
                print(f"[skip] unknown format {fmt!r}")
                continue

            label = f"{entry['set']} run{entry['run']} [{fmt}]"

            if already_done(folder, fmt):
                print(f"[skip] {label} already done")
                skipped += 1
                continue

            env = os.environ.copy()
            env["GRID_SEED"] = str(entry["seed"])
            env["TEACHER_MODEL"] = entry["teacher"]
            env["STUDENT_MODEL"] = entry["student"]
            env["TEACHING_FORMAT"] = fmt

            print(f"\n--- {label} ---")
            t0 = time.time()

            ok = True
            for script in STEPS:
                result = subprocess.run(
                    [sys.executable, os.path.join(HERE, script)],
                    cwd=folder, env=env)
                if result.returncode != 0:
                    print(f"  !! {script} failed")
                    ok = False
                    break

            if ok:
                done += 1
                print(f"    done in {time.time() - t0:.0f}s")
            else:
                failed += 1

    print(f"\n{'=' * 58}")
    print(f"{done} completed, {skipped} skipped, {failed} failed")
    print(f"total {(time.time() - started) / 60:.0f} min")
    print("Now run analyze_all.py")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()
