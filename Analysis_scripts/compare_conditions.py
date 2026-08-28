"""
Compare two conditions on one metric, with the spread needed to say whether a
difference is meaningful.

Written for a specific question: with Llama as Student the probe condition
scores higher on direction accuracy than in-context teaching, and the write-up
needs to say whether that gap is larger than run-to-run noise.

Prints, for each pair: both means, both standard deviations across runs, both
standard errors, the difference, and Welch's t on the difference. Also reports
how often the probe beat teaching run by run, which is easier to interpret than
a t statistic when the samples are small.

Usage
-----
    python3 compare_conditions.py
"""

from __future__ import annotations

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

# (metric key, condition A, condition B) -- A is the one being questioned
COMPARISONS = [
    ("direction_accuracy", "RL_Exp3_Time3", "RL_Exp1_Time2"),
    ("direction_accuracy", "RL_Exp2_Time3", "RL_Exp1_Time2"),
    ("mean_position_error", "RL_Exp3_Time3", "RL_Exp1_Time2"),
]


def load_index() -> list[dict]:
    with open(os.path.join(HERE, "runs_index.json")) as fh:
        return json.load(fh)


def values_for(entries: list[dict], condition: str, metric: str) -> list[float]:
    out = []
    for entry in entries:
        path = os.path.join(HERE, entry["folder"], condition + ".json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            value = json.load(fh).get(metric)
        if value is not None:
            out.append(value)
    return out


def describe(values: list[float]) -> tuple[float, float, float, int]:
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    sem = sd / n ** 0.5 if n else 0.0
    return mean, sd, sem, n


def welch_t(a: list[float], b: list[float]) -> float | None:
    """Welch's t. Not a p-value, but enough to see whether a gap is small."""
    if len(a) < 2 or len(b) < 2:
        return None
    va = statistics.variance(a) / len(a)
    vb = statistics.variance(b) / len(b)
    denom = (va + vb) ** 0.5
    if denom == 0:
        return None
    return (statistics.mean(a) - statistics.mean(b)) / denom


def paired_wins(entries: list[dict], cond_a: str, cond_b: str,
                metric: str, lower_is_better: bool) -> tuple[int, int]:
    """How often cond_a beat cond_b within the same run."""
    wins = total = 0
    for entry in entries:
        pa = os.path.join(HERE, entry["folder"], cond_a + ".json")
        pb = os.path.join(HERE, entry["folder"], cond_b + ".json")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        with open(pa) as fh:
            va = json.load(fh).get(metric)
        with open(pb) as fh:
            vb = json.load(fh).get(metric)
        if va is None or vb is None:
            continue
        total += 1
        if (va < vb) if lower_is_better else (va > vb):
            wins += 1
    return wins, total


def main() -> None:
    index = load_index()
    sets = sorted({entry["set"] for entry in index})

    for metric, cond_a, cond_b in COMPARISONS:
        lower_better = "error" in metric
        print("\n" + "=" * 82)
        print(f"{metric}   {cond_a}  vs  {cond_b}")
        print(f"({'lower' if lower_better else 'higher'} is better)")
        print("=" * 82)

        for s in sets:
            entries = [e for e in index if e["set"] == s]
            student = next(e["student"] for e in entries)

            a = values_for(entries, cond_a, metric)
            b = values_for(entries, cond_b, metric)
            if not a or not b:
                print(f"\n  {s}: no data")
                continue

            ma, sda, sema, na = describe(a)
            mb, sdb, semb, nb = describe(b)
            t = welch_t(a, b)
            wins, total = paired_wins(entries, cond_a, cond_b, metric, lower_better)

            print(f"\n  Student = {student}")
            print(f"    {cond_a:<16} {ma:.3f}   sd {sda:.3f}   sem {sema:.3f}   n={na}")
            print(f"    {cond_b:<16} {mb:.3f}   sd {sdb:.3f}   sem {semb:.3f}   n={nb}")
            print(f"    difference       {ma - mb:+.3f}")
            if t is not None:
                print(f"    Welch t          {t:+.2f}")
            if total:
                print(f"    {cond_a} beat {cond_b} in {wins} of {total} runs "
                      f"({wins/total:.0%})")

            if t is not None and abs(t) < 2:
                print("    -> the gap is within run-to-run noise")
            elif t is not None:
                print("    -> the gap is larger than run-to-run noise")


if __name__ == "__main__":
    main()
