"""
Pulls everything needed to evaluate the experiment into one report.

Prints, and writes to diagnostic_report.txt:

  1. Sample sizes    -- how many rooms/questions actually got answered per cell.
                        Bars built on 2 rooms should not be read like bars built
                        on 50.
  2. Answer rate     -- the thing that actually varied between the two models in
                        the flushed conditions.
  3. Full metrics    -- accuracy, error raw and shift-corrected, retry rate.
  4. Refusal check   -- how often the model refused even after 12 forced attempts.
  5. Raw samples     -- a few actual responses per condition, so the numbers can
                        be sanity-checked against what the model really said.
"""

import json
import os
import glob
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def p(line=""):
    print(line)
    OUT.append(line)


CONDITIONS = [
    ("RL_Exp1_Time1", "before teaching"),
    ("RL_Exp1_Time2", "teaching in context"),
    ("RL_Exp1_Time3", "teaching flushed"),
    ("RL_Exp2_Time3", "own notes returned"),
    ("RL_Exp3_Time3", "stored in weights"),
    ("RL_Exp4_Time2", "map in context"),
    ("RL_Exp4_Time3", "map flushed"),
]

with open(os.path.join(HERE, "runs_index.json")) as f:
    index = json.load(f)

sets = sorted({e["set"] for e in index})


def load(entry, cond):
    path = os.path.join(HERE, entry["folder"], cond + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    m = statistics.mean(vals)
    sem = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
    return m, sem, len(vals)


# ---------------------------------------------------------------------
# 1. sample sizes -- the thing that invalidates bars
# ---------------------------------------------------------------------

p("=" * 96)
p("1. SAMPLE SIZES")
p("=" * 96)
p("How many position questions actually produced an answer. A cell with 2/50")
p("cannot be compared to a cell with 50/50, however similar the bars look.")
p()
p(f"{'condition':<18}{'set':<20}{'rooms answered':<18}{'runs with 0':<14}{'usable?'}")
p("-" * 96)

sample_flags = {}

for cond, _ in CONDITIONS:
    for s in sets:
        entries = [e for e in index if e["set"] == s]
        ans = tot = zero_runs = 0
        for e in entries:
            d = load(e, cond)
            if not d:
                continue
            a = d.get("n_position_answered", 0)
            t = d.get("n_position_total", 0)
            ans += a
            tot += t
            if a == 0:
                zero_runs += 1
        if tot == 0:
            continue
        rate = ans / tot
        flag = "yes" if rate > 0.8 else ("THIN" if rate > 0.3 else "NO -- do not plot")
        sample_flags[(cond, s)] = (ans, tot, flag)
        p(f"{cond:<18}{s:<20}{f'{ans}/{tot}  ({rate:.0%})':<18}{f'{zero_runs}':<14}{flag}")


# ---------------------------------------------------------------------
# 2. answer rate as its own measure
# ---------------------------------------------------------------------

p()
p("=" * 96)
p("2. ANSWER RATE  (this is a result, not just a caveat)")
p("=" * 96)
p("Fraction of position questions answered, after up to 12 forced attempts.")
p("Large differences here mean the models behave differently when they have")
p("nothing to go on, which is worth reporting in its own right.")
p()
p(f"{'condition':<18}" + "".join(f"{s:<22}" for s in sets))
p("-" * 96)

for cond, _ in CONDITIONS:
    row = f"{cond:<18}"
    for s in sets:
        if (cond, s) in sample_flags:
            a, t, _ = sample_flags[(cond, s)]
            row += f"{f'{a/t:.0%}':<22}"
        else:
            row += f"{'-':<22}"
    p(row)


# ---------------------------------------------------------------------
# 3. metrics
# ---------------------------------------------------------------------

METRICS = [
    ("triplet_accuracy",             "triplet acc",   "chance 0.50"),
    ("direction_accuracy",           "direction acc", "chance 0.20"),
    ("mean_position_error",          "pos err raw",   "rand 6.60 / centre 5.00"),
    ("mean_position_error_shifted",  "pos err shift", "rand 5.47 / degenerate 4.00"),
    ("map_mean_error",               "map err raw",   "rand 6.60 / centre 5.00"),
    ("map_mean_error_shifted",       "map err shift", "rand 5.47 / degenerate 4.00"),
    ("pct_needed_retry",             "retry rate",    ""),
]

p()
p("=" * 96)
p("3. METRICS  (mean +/- SEM [n runs])")
p("=" * 96)

for s in sets:
    teacher = next(e["teacher"] for e in index if e["set"] == s)
    student = next(e["student"] for e in index if e["set"] == s)
    p()
    p(f"SET {s}   teacher={teacher}  student={student}")
    p("-" * 96)
    p(f"{'condition':<18}" + "".join(f"{lab:<20}" for _, lab, _ in METRICS))

    for cond, _ in CONDITIONS:
        row = f"{cond:<18}"
        for key, _, _ in METRICS:
            vals = [load(e, cond).get(key) if load(e, cond) else None
                    for e in index if e["set"] == s]
            m, sem, n = agg(vals)
            row += f"{(f'{m:.2f}+-{sem:.2f}[{n}]' if m is not None else '—'):<20}"
        p(row)

p()
p("Benchmarks:")
for _, lab, bench in METRICS:
    if bench:
        p(f"  {lab:<16} {bench}")


# ---------------------------------------------------------------------
# 4. refusals surviving the forcing
# ---------------------------------------------------------------------

p()
p("=" * 96)
p("4. REFUSALS THAT SURVIVED 12 FORCED ATTEMPTS")
p("=" * 96)
p(f"{'condition':<18}{'set':<20}{'unanswered':<16}{'max attempts seen'}")
p("-" * 96)

for cond, _ in CONDITIONS:
    for s in sets:
        unans = []
        maxatt = 0
        for e in [x for x in index if x["set"] == s]:
            d = load(e, cond)
            if not d:
                continue
            unans.append(d.get("n_still_unanswered", 0))
            maxatt = max(maxatt, d.get("max_attempts", 0))
        if unans:
            p(f"{cond:<18}{s:<20}{f'{sum(unans)}':<16}{maxatt}")


# ---------------------------------------------------------------------
# 5. raw responses
# ---------------------------------------------------------------------

p()
p("=" * 96)
p("5. RAW RESPONSES  (first run of each set, so numbers can be checked)")
p("=" * 96)

for s in sets:
    first = next((e for e in index if e["set"] == s), None)
    if not first:
        continue
    p()
    p(f"### SET {s}, {first['folder']}")

    gw = os.path.join(HERE, first["folder"], "grid_world.json")
    if os.path.exists(gw):
        with open(gw) as f:
            w = json.load(f)
        p("  true layout: " + ", ".join(
            f"{n}({r['x']},{r['y']})" for n, r in sorted(w["rooms"].items())))

    for cond, desc in CONDITIONS:
        d = load(first, cond)
        if not d:
            continue
        p()
        p(f"  --- {cond} ({desc}) ---")

        for r in d.get("position_answers", [])[:3]:
            got = r.get("parsed")
            p(f"    {r['room']} true({r['true_x']},{r['true_y']}) got {got} "
              f"attempts {r.get('attempts')}")
            p(f"      {r['raw_response'][:150]!r}")

        m = d.get("map_answer")
        if m:
            p(f"    MAP placed {m.get('n_rooms_placed')}/{m.get('n_rooms_total')} "
              f"attempts {m.get('attempts')}")
            p(f"      {m['raw_response'][:200]!r}")


with open(os.path.join(HERE, "diagnostic_report.txt"), "w") as f:
    f.write("\n".join(OUT))

p()
p("=" * 96)
p("Saved diagnostic_report.txt")
p("=" * 96)
