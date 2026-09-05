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
    # pair-fact teaching and the recovery conditions
    "RL_Exp1_Time1.json", "RL_Exp1_Time2.json", "RL_Exp1_Time3.json",
    "RL_Exp2_Time3.json", "RL_Exp3_Time3.json", "RL_Exp5_Time3.json",
    # the alternative teaching formats
    "RL_Exp4_Time2.json", "RL_Exp4_Time3.json",
    "RL_Coord_Time2.json", "RL_Coord_Time3.json",
    "RL_Trail_Time2.json", "RL_Trail_Time3.json",
    "RL_Both_Time2.json", "RL_Both_Time3.json",
]

GRID_SIZE = 10
SEARCH = range(-9, 10)   # candidate shifts on each axis

# Whether to also try flipping each axis before shifting.
#
# The teaching gives relative offsets, so a Student can get the shape right and
# the orientation wrong. One response chained the trail as "from R1 (0,0),
# moving 2 west and 3 south leads to R2 (2,3)", adding where it should have
# subtracted on both axes. That is a reflection, and no amount of translation
# corrects it, so a correct-but-mirrored layout scores as a total failure.
#
# Reporting both numbers separates "the layout is wrong" from "the layout is
# right and the conventions are inverted", which are different failures.
TRY_REFLECTIONS = True
REFLECTIONS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

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
    predicted coordinate. Brute force -- 361 combinations over at most 5 rooms,
    so speed is not a concern.

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


def best_fit_rigid(pred, truth):
    """
    As above, but also try flipping each axis before shifting.

    A model that consistently treats west as increasing x has the layout right
    and the convention backwards. Translation cannot fix that; a reflection
    can. Returns (flip_x, flip_y, dx, dy, mean_error).
    """
    common = [k for k in pred if k in truth]
    if len(common) < MIN_ROOMS:
        return 1, 1, 0, 0, None

    centre = (GRID_SIZE - 1) / 2 if "GRID_SIZE" in globals() else 4.5

    best = (1, 1, 0, 0, float("inf"))
    for fx, fy in (REFLECTIONS if TRY_REFLECTIONS else [(1, 1)]):
        flipped = {
            k: (centre + fx * (v[0] - centre), centre + fy * (v[1] - centre))
            for k, v in pred.items()
        }
        for dx in SEARCH:
            for dy in SEARCH:
                total = sum(
                    manhattan(flipped[k][0] + dx, flipped[k][1] + dy,
                              truth[k][0], truth[k][1])
                    for k in common
                )
                if total < best[4]:
                    best = (fx, fy, dx, dy, total)

    return best[0], best[1], best[2], best[3], best[4] / len(common)


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
        fx, fy, rdx, rdy, rigid = best_fit_rigid(pred, truth)

        d["mean_position_error_raw"] = raw
        d["mean_position_error_shifted"] = shifted
        d["position_best_fit_shift"] = [dx, dy]
        d["position_shift_improvement"] = (raw - shifted) if raw is not None else None

        d["mean_position_error_rigid"] = rigid
        d["position_best_fit_reflection"] = [fx, fy]
        d["position_best_fit_rigid_shift"] = [rdx, rdy]
        # a large gain here that shift alone did not capture means the layout
        # was mirrored rather than misplaced
        d["position_reflection_gain"] = (
            (shifted - rigid) if (shifted is not None and rigid is not None) else None
        )
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
            tally.setdefault(key, {"raw": [], "shifted": [], "rigid": [],
                                   "shifts": [], "flips": []})
            if d.get("mean_position_error_shifted") is not None:
                tally[key]["raw"].append(d["mean_position_error_raw"])
                tally[key]["shifted"].append(d["mean_position_error_shifted"])
                tally[key]["shifts"].append(tuple(d["position_best_fit_shift"]))
            if d.get("mean_position_error_rigid") is not None:
                tally[key]["rigid"].append(d["mean_position_error_rigid"])
                tally[key]["flips"].append(tuple(d["position_best_fit_reflection"]))

    # ---- report ----
    print("=" * 104)
    print("POSITION ERROR: raw, translation-corrected, and reflection-corrected")
    print("=" * 104)
    print(f"{'set':<20}{'condition':<22}{'raw':<9}{'shifted':<10}"
          f"{'reflected':<11}{'shift gain':<12}{'reflect gain':<14}{'mirrored?'}")
    print("-" * 104)

    for (s, cond), v in sorted(tally.items()):
        if not v["raw"]:
            continue
        raw = sum(v["raw"]) / len(v["raw"])
        sh = sum(v["shifted"]) / len(v["shifted"])
        rig = sum(v["rigid"]) / len(v["rigid"]) if v["rigid"] else None

        # how often the best fit needed an axis flipped. A layout that is
        # correctly shaped but mirrored cannot be repaired by translation, so
        # a high rate here means the Student inverted a convention rather than
        # reconstructing the wrong layout.
        flipped = sum(1 for f in v["flips"] if f != (1, 1))
        flip_rate = flipped / len(v["flips"]) if v["flips"] else 0.0

        rig_txt = f"{rig:.2f}" if rig is not None else "—"
        gain_txt = f"{sh - rig:.2f}" if rig is not None else "—"

        print(f"{s:<20}{cond.replace('.json',''):<22}"
              f"{raw:<9.2f}{sh:<10.2f}{rig_txt:<11}"
              f"{raw - sh:<12.2f}{gain_txt:<14}{flip_rate:.0%}")

    print("\nBenchmarks for RAW error:      random guess 6.60, centre guess 5.00")
    print("Benchmarks for SHIFTED error:  random guess 5.47, degenerate answer ~4.00")
    print("\nThe shifted benchmarks are higher than you would expect because the")
    print("best-fit shift also helps bad answers. Putting all rooms on one point")
    print("and letting the optimiser drop that blob onto the true layout scores")
    print("about 4.00. So ~4.00, not 5.00, is the line a real layout has to beat.")
    print(f"\nCells with fewer than {MIN_ROOMS} answered rooms are excluded.")
    print("\n'shift gain' is error that was pure origin offset rather than a wrong")
    print("layout. 'reflect gain' is the further improvement from allowing an axis to")
    print("be flipped: a Student can reconstruct the correct shape and invert a")
    print("convention, which translation alone cannot repair. 'mirrored?' is how often")
    print("the best fit needed a flip.")
