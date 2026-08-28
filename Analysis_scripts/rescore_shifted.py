"""
Re-score existing runs with translation-corrected position error.

WHY THIS IS NEEDED

The teaching gives only relative offsets. "R0 is 2 units east of R3" fixes the
distance between rooms but says nothing about where the origin is. So a Student
that reconstructs the layout perfectly and anchors it at a different origin has
given a CORRECT answer, and the raw scorer marks it as a total failure.

This is not hypothetical. Gemini, given the same teaching, answered:

    R0 (2,3)  R1 (1,0)  R2 (-1,5)  R3 (0,3)  R4 (-1,4)
    truth:
    R0 (6,6)  R1 (5,3)  R2 (3,8)   R3 (4,6)  R4 (3,7)

Every room off by exactly (+4, +3). Raw error 7.00, worse than random guessing.
After removing the shift, error 0.00. A perfect answer scored as the worst
possible one.

WHAT THIS DOES

For each set of position answers, finds the single (dx, dy) shift that best
aligns the model's layout with the truth, applies it, and rescores. Reports
both numbers so the difference is visible.

Adds to each condition file:
  mean_position_error_raw        what was reported before
  mean_position_error_shifted    after best-fit translation
  best_fit_shift                 the (dx, dy) that was removed
  shift_improvement              how much the correction recovered

Same for the map question. Nothing is overwritten -- the raw values stay.
"""

import json
import os
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = [
    "RL_Exp1_Time1.json", "RL_Exp1_Time2.json", "RL_Exp1_Time3.json",
    "RL_Exp2_Time3.json", "RL_Exp3_Time3.json",
    "RL_Exp4_Time2.json", "RL_Exp4_Time3.json",
]

SEARCH = range(-9, 10)   # candidate shifts on each axis

MIN_ROOMS = 3
# Fitting a translation to fewer than three points is meaningless. With one
# room the shift lands it exactly on target and the error is zero by
# construction -- that is what produced the spurious 1.29 in Exp1_Time3,
# where four of five rooms had refused and only one answer remained.


def manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def best_fit_shift(pred, truth):
    """
    Find the (dx, dy) that minimises total Manhattan error when added to every
    predicted coordinate. Brute force over the grid -- only 361 combinations
    and at most 5 rooms, so speed is not a concern.

    Returns (dx, dy, mean_error_after_shift).
    """
    common = [k for k in pred if k in truth]
    if len(common) < MIN_ROOMS:
        return 0, 0, None

    best = (0, 0, float("inf"))
    for dx in SEARCH:
        for dy in SEARCH:
            total = sum(
                manhattan(pred[k][0] + dx, pred[k][1] + dy, truth[k][0], truth[k][1])
                for k in common
            )
            if total < best[2]:
                best = (dx, dy, total)

    return best[0], best[1], best[2] / len(common)


def raw_error(pred, truth):
    common = [k for k in pred if k in truth]
    if len(common) < MIN_ROOMS:
        return None
    return sum(manhattan(pred[k][0], pred[k][1], truth[k][0], truth[k][1])
               for k in common) / len(common)


def rescore_file(path, truth):
    with open(path) as f:
        d = json.load(f)

    changed = False

    # ---- position answers ----
    pred = {}
    for r in d.get("position_answers", []):
        if r.get("parsed"):
            pred[r["room"]] = (r["parsed"][0], r["parsed"][1])

    if pred:
        raw = raw_error(pred, truth)
        dx, dy, shifted = best_fit_shift(pred, truth)
        d["mean_position_error_raw"] = raw
        d["mean_position_error_shifted"] = shifted
        d["position_best_fit_shift"] = [dx, dy]
        d["position_shift_improvement"] = (raw - shifted) if raw is not None else None
        changed = True

    # ---- map answer ----
    m = d.get("map_answer")
    if m and m.get("parsed_coords"):
        mpred = {k: (v[0], v[1]) for k, v in m["parsed_coords"].items()}
        raw = raw_error(mpred, truth)
        dx, dy, shifted = best_fit_shift(mpred, truth)
        d["map_mean_error_raw"] = raw
        d["map_mean_error_shifted"] = shifted
        d["map_best_fit_shift"] = [dx, dy]
        d["map_shift_improvement"] = (raw - shifted) if raw is not None else None
        changed = True

    # ---- Exp3 probe: predictions live elsewhere ----
    pd = d.get("probe_details")
    if pd and pd.get("predictions"):
        ppred = {k: (v[0], v[1]) for k, v in pd["predictions"].items()}
        raw = raw_error(ppred, truth)
        dx, dy, shifted = best_fit_shift(ppred, truth)
        d["mean_position_error_raw"] = raw
        d["mean_position_error_shifted"] = shifted
        d["position_best_fit_shift"] = [dx, dy]
        d["position_shift_improvement"] = raw - shifted
        changed = True

    if changed:
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    return d


if __name__ == "__main__":
    with open(os.path.join(HERE, "runs_index.json")) as f:
        index = json.load(f)

    print(f"Re-scoring {len(index)} runs\n")

    tally = {}

    for entry in index:
        folder = os.path.join(HERE, entry["folder"])
        gw = os.path.join(folder, "grid_world.json")
        if not os.path.exists(gw):
            print(f"[skip] {entry['folder']} -- no grid_world.json")
            continue

        with open(gw) as f:
            world = json.load(f)
        truth = {n: (r["x"], r["y"]) for n, r in world["rooms"].items()}

        for cond in CONDITIONS:
            path = os.path.join(folder, cond)
            if not os.path.exists(path):
                continue
            d = rescore_file(path, truth)

            key = (entry["set"], cond)
            tally.setdefault(key, {"raw": [], "shifted": [], "shifts": []})
            if d.get("mean_position_error_shifted") is not None:
                tally[key]["raw"].append(d["mean_position_error_raw"])
                tally[key]["shifted"].append(d["mean_position_error_shifted"])
                tally[key]["shifts"].append(tuple(d["position_best_fit_shift"]))

    # ---- report ----
    print("=" * 92)
    print("POSITION ERROR: raw vs translation-corrected")
    print("=" * 92)
    print(f"{'set':<20}{'condition':<22}{'raw':<10}{'shifted':<10}{'gained':<10}{'typical shift'}")
    print("-" * 92)

    for (s, cond), v in sorted(tally.items()):
        if not v["raw"]:
            continue
        raw = sum(v["raw"]) / len(v["raw"])
        sh = sum(v["shifted"]) / len(v["shifted"])
        common_shift = max(set(v["shifts"]), key=v["shifts"].count)
        print(f"{s:<20}{cond.replace('.json',''):<22}"
              f"{raw:<10.2f}{sh:<10.2f}{raw - sh:<10.2f}{common_shift}")

    print("\nBenchmarks for RAW error:      random guess 6.60, centre guess 5.00")
    print("Benchmarks for SHIFTED error:  random guess 5.47, degenerate answer ~4.00")
    print("\nThe shifted benchmarks are higher than you would expect because the")
    print("best-fit shift also helps bad answers. Putting all rooms on one point")
    print("and letting the optimiser drop that blob onto the true layout scores")
    print("about 4.00. So ~4.00, not 5.00, is the line a real layout has to beat.")
    print(f"\nCells with fewer than {MIN_ROOMS} answered rooms are excluded.")
    print("\n'gained' is how much error was pure origin offset rather than a wrong layout.")
    print("A large gain means the layout was better than the raw number suggested.")
