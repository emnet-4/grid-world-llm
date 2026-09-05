"""
Aggregate the experiment runs and produce one figure per metric.

Reads every run listed in runs_index.json, averages each metric across runs
within a teaching direction, and draws a separate bar chart for each measure
with the baseline agents overlaid as horizontal lines.

Bars computed from an unusually small number of answered questions are hatched
and labelled with their sample size, because in the flushed conditions the
Student sometimes refuses almost every question and the resulting mean is not
comparable to a bar built from a full sample.

Inputs
------
runs_index.json        written by run_all_experiments.py
<run folder>/*.json    one file per condition per run
agent_baselines.json   optional, written by baseline_agents.py

Outputs
-------
fig_triplet.png        triplet accuracy
fig_direction.png      direction accuracy
fig_position.png       position error
fig_map.png            whole-layout error
fig_retries.png        how often the Student had to be pushed to answer
aggregate_results.json all numbers behind the figures
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Bars built from fewer than this fraction of the expected answers are marked
# as unreliable on the figure.
LOW_SAMPLE_THRESHOLD = 0.5

DPI = 150
FIG_SIZE = (10, 5.5)


# ---------------------------------------------------------------------
# what to plot
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Metric:
    key: str            # field name in the condition JSON
    filename: str       # output figure
    axis_label: str
    title: str
    lower_is_better: bool
    chance_note: str = ""


METRICS = [
    Metric("triplet_accuracy", "fig_triplet.png",
           "Triplet accuracy",
           "Is room A closer to B or to C?",
           lower_is_better=False,
           chance_note="two options, so 0.50 is chance"),

    Metric("direction_accuracy", "fig_direction.png",
           "Direction accuracy",
           "Is room A east/west and north/south of B?",
           lower_is_better=False,
           chance_note="both axes must be right, so chance is about 0.20"),

    Metric("mean_position_error", "fig_position.png",
           "Position error (Manhattan)",
           "What are room A's coordinates?",
           lower_is_better=True),

    Metric("map_mean_error", "fig_map.png",
           "Whole-layout error (Manhattan)",
           "Give the coordinates of every room at once",
           lower_is_better=True),

    Metric("pct_needed_retry", "fig_retries.png",
           "Fraction of questions needing a retry",
           "How often the Student had to be pushed to answer",
           lower_is_better=True),
]

# (json filename, short label for the x axis, longer description)
CONDITIONS = [
    ("RL_Exp1_Time1", "Time 1\nbefore teaching",
     "no teaching has happened yet"),
    ("RL_Exp1_Time2", "Time 2\nteaching in context",
     "the teaching transcript is in the prompt"),
    ("RL_Exp1_Time3", "Exp 1 Time 3\nteaching flushed",
     "teaching pushed out of the context window"),
    ("RL_Exp2_Time3", "Exp 2 Time 3\nown notes returned",
     "flushed, but handed back its own coordinate list"),
    ("RL_Exp3_Time3", "Exp 3 Time 3\nstored in weights",
     "flushed, answered by a probe trained on that list"),
    ("RL_Exp4_Time2", "Exp 4 Time 2\nmap in context",
     "taught with a drawn map instead of sentences"),
    ("RL_Exp4_Time3", "Exp 4 Time 3\nmap flushed",
     "map pushed out of the context window"),
    ("RL_Exp5_Time3", "Exp 5 Time 3\nweights fine-tuned",
     "LoRA-trained on ground truth, model answers with no context"),

    # the three teaching formats, each with the transcript readable and flushed
    ("RL_Coord_Time2", "Coordinates\nin context",
     "taught absolute positions, transcript readable"),
    ("RL_Coord_Time3", "Coordinates\nflushed",
     "taught absolute positions, transcript flushed"),
    ("RL_Trail_Time2", "Trail\nin context",
     "taught as a walk visiting every room, transcript readable"),
    ("RL_Trail_Time3", "Trail\nflushed",
     "taught as a walk visiting every room, transcript flushed"),
    ("RL_Both_Time2", "Trail + coords\nin context",
     "taught both ways, transcript readable"),
    ("RL_Both_Time3", "Trail + coords\nflushed",
     "taught both ways, transcript flushed"),
]

SET_LABELS = {
    "A_llama_teaches":   "Llama teaches, Mistral learns",
    "B_mistral_teaches": "Mistral teaches, Llama learns",
}

SET_COLOURS = ["#4C72B0", "#DD8452"]

AGENT_STYLE = {
    "random":   dict(colour="black",      linestyle="--", width=1.6),
    "line":     dict(colour="seagreen",   linestyle="-.", width=1.6),
    "quadrant": dict(colour="darkorange", linestyle=":",  width=2.2),
    # kept so older baseline files still plot
    "fixed":    dict(colour="darkorange", linestyle=":",  width=2.2),
}


# ---------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------

def load_index() -> list[dict]:
    with open(os.path.join(HERE, "runs_index.json")) as fh:
        return json.load(fh)


def load_baselines() -> dict | None:
    path = os.path.join(HERE, "agent_baselines.json")
    if not os.path.exists(path):
        print("No agent_baselines.json found. Run baseline_agents.py to add "
              "baseline lines to the figures.")
        return None
    with open(path) as fh:
        return json.load(fh)["agents"]


def load_condition(folder: str, condition: str) -> dict | None:
    path = os.path.join(HERE, folder, condition + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------

@dataclass
class Cell:
    """One condition in one teaching direction, averaged across runs."""
    mean: float | None
    sem: float | None
    n_runs: int
    answered: int
    expected: int

    @property
    def answered_fraction(self) -> float:
        return self.answered / self.expected if self.expected else 1.0

    @property
    def is_low_sample(self) -> bool:
        return self.expected > 0 and self.answered_fraction < LOW_SAMPLE_THRESHOLD


def summarise(values: list[float]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    mean = statistics.mean(clean)
    sem = statistics.stdev(clean) / len(clean) ** 0.5 if len(clean) > 1 else 0.0
    return mean, sem


def build_table(index: list[dict]) -> dict:
    """table[set][condition][metric_key] -> Cell"""
    sets = sorted({entry["set"] for entry in index})
    table: dict = {s: {c: {} for c, _, _ in CONDITIONS} for s in sets}

    for s in sets:
        entries = [e for e in index if e["set"] == s]
        for condition, _, _ in CONDITIONS:
            loaded = [load_condition(e["folder"], condition) for e in entries]
            loaded = [d for d in loaded if d is not None]

            answered = sum(d.get("n_position_answered", 0) for d in loaded)
            expected = sum(d.get("n_position_total", 0) for d in loaded)

            for metric in METRICS:
                values = [d.get(metric.key) for d in loaded]
                mean, sem = summarise(values)
                table[s][condition][metric.key] = Cell(
                    mean=mean, sem=sem,
                    n_runs=len([v for v in values if v is not None]),
                    answered=answered, expected=expected,
                )
    return table


# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------

def draw_metric(metric: Metric, table: dict, sets: list[str],
                baselines: dict | None) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    x = np.arange(len(CONDITIONS))
    width = 0.8 / max(len(sets), 1)
    any_low_sample = False
    annotations: list[tuple[float, float, str]] = []

    for i, s in enumerate(sets):
        means, sems, hatches = [], [], []
        for condition, _, _ in CONDITIONS:
            cell = table[s][condition][metric.key]
            means.append(cell.mean if cell.mean is not None else 0.0)
            sems.append(cell.sem if cell.sem is not None else 0.0)
            hatches.append(cell.is_low_sample and cell.mean is not None)

        offset = i * width - 0.4 + width / 2
        bars = ax.bar(x + offset, means, width, yerr=sems, capsize=3,
                      color=SET_COLOURS[i % len(SET_COLOURS)],
                      label=SET_LABELS.get(s, s.replace("_", " ")))

        # mark bars built from very few answered questions
        for j, (bar, (condition, _, _)) in enumerate(zip(bars, CONDITIONS)):
            if not hatches[j]:
                continue
            any_low_sample = True
            cell = table[s][condition][metric.key]
            bar.set_hatch("////")
            bar.set_edgecolor("black")
            bar.set_linewidth(1.0)
            annotations.append((bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + sems[j],
                                f"n={cell.answered}"))

    if baselines:
        for agent, stats in baselines.items():
            entry = stats.get(metric.key)
            if not entry:
                continue
            style = AGENT_STYLE.get(agent, dict(colour="grey",
                                                linestyle="--", width=1.5))
            ax.axhline(entry["mean"], ls=style["linestyle"],
                       c=style["colour"], lw=style["width"], zorder=5,
                       label=f"{agent} agent: {entry['mean']:.2f} "
                             f"(sd {entry['sd']:.2f})")

    if not metric.lower_is_better:
        ax.set_ylim(0, 1)

    pad = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for xpos, ypos, text in annotations:
        ax.text(xpos, ypos + pad, text, ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in CONDITIONS],
                       fontsize=7.5)
    ax.set_ylabel(metric.axis_label)
    ax.grid(axis="y", alpha=0.25)

    direction = "lower is better" if metric.lower_is_better else "higher is better"
    subtitle = f"{direction}"
    if metric.chance_note:
        subtitle += f"   |   {metric.chance_note}"
    if any_low_sample:
        subtitle += ("\nhatched bars are computed from few answered questions, "
                     "shown as n, and are not comparable to the rest")

    ax.set_title(f"{metric.title}\n{subtitle}", fontsize=10)
    ax.legend(fontsize=7.5, loc="best")

    plt.tight_layout()
    out = os.path.join(HERE, metric.filename)
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {metric.filename}")


# ---------------------------------------------------------------------
# text output
# ---------------------------------------------------------------------

def print_table(table: dict, sets: list[str], index: list[dict],
                baselines: dict | None) -> None:
    print("\n" + "=" * 100)
    print("RESULTS   mean +/- standard error across runs")
    print("=" * 100)

    for s in sets:
        teacher = next(e["teacher"] for e in index if e["set"] == s)
        student = next(e["student"] for e in index if e["set"] == s)
        n_runs = sum(1 for e in index if e["set"] == s)

        print(f"\n{SET_LABELS.get(s, s)}   "
              f"(teacher {teacher}, student {student}, {n_runs} runs)")
        print("-" * 100)
        header = f"{'condition':<22}"
        header += "".join(f"{m.axis_label.split('(')[0].strip():<24}"
                          for m in METRICS[:4])
        print(header)

        for condition, label, _ in CONDITIONS:
            row = f"{condition:<22}"
            for metric in METRICS[:4]:
                cell = table[s][condition][metric.key]
                if cell.mean is None:
                    row += f"{'—':<24}"
                else:
                    flag = "*" if cell.is_low_sample else ""
                    row += f"{f'{cell.mean:.3f}±{cell.sem:.3f}{flag}':<24}"
            print(row)

        low = [c for c, _, _ in CONDITIONS
               if table[s][c][METRICS[2].key].is_low_sample]
        if low:
            print(f"\n  * few answered questions, not comparable: {', '.join(low)}")
            for c in low:
                cell = table[s][c][METRICS[2].key]
                print(f"      {c}: {cell.answered} of {cell.expected} answered")

    if baselines:
        print("\n" + "-" * 100)
        print("Baseline agents")
        for agent, stats in baselines.items():
            parts = []
            for metric in METRICS[:4]:
                entry = stats.get(metric.key)
                if entry:
                    parts.append(f"{metric.key.split('_')[0]} {entry['mean']:.3f}")
            print(f"  {agent:<10} " + "   ".join(parts))

    print("\nWhat each condition is:")
    for condition, _, description in CONDITIONS:
        print(f"  {condition:<16} {description}")


def save_results(table: dict, sets: list[str], index: list[dict],
                 baselines: dict | None) -> None:
    out = {
        "runs_per_set": {s: sum(1 for e in index if e["set"] == s) for s in sets},
        "low_sample_threshold": LOW_SAMPLE_THRESHOLD,
        "agents": baselines,
        "results": {
            s: {
                condition: {
                    metric.key: {
                        "mean": table[s][condition][metric.key].mean,
                        "sem": table[s][condition][metric.key].sem,
                        "n_runs": table[s][condition][metric.key].n_runs,
                        "answered": table[s][condition][metric.key].answered,
                        "expected": table[s][condition][metric.key].expected,
                        "low_sample": table[s][condition][metric.key].is_low_sample,
                    }
                    for metric in METRICS
                }
                for condition, _, _ in CONDITIONS
            }
            for s in sets
        },
    }
    path = os.path.join(HERE, "aggregate_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved aggregate_results.json")


# ---------------------------------------------------------------------

def main() -> None:
    index = load_index()
    baselines = load_baselines()
    sets = sorted({entry["set"] for entry in index})
    table = build_table(index)

    print_table(table, sets, index, baselines)

    print("\nFigures:")
    for metric in METRICS:
        has_data = any(
            table[s][c][metric.key].mean is not None
            for s in sets for c, _, _ in CONDITIONS
        )
        if has_data:
            draw_metric(metric, table, sets, baselines)
        else:
            print(f"  ({metric.filename} skipped, no data)")

    save_results(table, sets, index, baselines)


if __name__ == "__main__":
    main()
