"""
Prints a short results summary that fits in a chat message.

Run this instead of pasting whole files. Reads runs_index.json plus whatever
condition files exist, so it works on a partial batch.
"""

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

CONDS = [
    ("RL_Exp1_Time1", "before teaching"),
    ("RL_Exp1_Time2", "teaching in ctx"),
    ("RL_Exp1_Time3", "flushed"),
    ("RL_Exp2_Time3", "own notes"),
    ("RL_Exp3_Time3", "weights"),
    ("RL_Exp4_Time2", "map in ctx"),
    ("RL_Exp4_Time3", "map flushed"),
]

with open(os.path.join(HERE, "runs_index.json")) as f:
    index = json.load(f)

sets = sorted({e["set"] for e in index})

# how many runs actually completed
print("RUNS COMPLETED")
for s in sets:
    n = sum(1 for e in index if e["set"] == s)
    print(f"  {s}: {n}")

# agents
bp = os.path.join(HERE, "agent_baselines.json")
if os.path.exists(bp):
    with open(bp) as f:
        agents = json.load(f)["agents"]
    print("\nAGENTS")
    for a, m in agents.items():
        d = m.get("direction_accuracy", {}).get("mean")
        p = m.get("mean_position_error", {}).get("mean")
        t = m.get("triplet_accuracy", {}).get("mean")
        print(f"  {a:<10} dir {d:.3f}  pos {p:.2f}  trip {t:.3f}")

# models
def get(entry, cond, key):
    path = os.path.join(HERE, entry["folder"], cond + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get(key)

print("\nMODELS")
for s in sets:
    print(f"\n{s}")
    entries = [e for e in index if e["set"] == s]
    for cond, desc in CONDS:
        d = [v for v in (get(e, cond, "direction_accuracy") for e in entries) if v is not None]
        p = [v for v in (get(e, cond, "mean_position_error") for e in entries) if v is not None]
        t = [v for v in (get(e, cond, "triplet_accuracy") for e in entries) if v is not None]
        ans = [get(e, cond, "n_position_answered") for e in entries]
        tot = [get(e, cond, "n_position_total") for e in entries]
        ans = sum(a for a in ans if a is not None)
        tot = sum(t2 for t2 in tot if t2 is not None)

        if not d:
            continue
        print(f"  {desc:<16} dir {statistics.mean(d):.3f}  "
              f"pos {statistics.mean(p):.2f}  "
              f"trip {statistics.mean(t):.3f}  "
              f"[{len(d)} runs, {ans}/{tot} rooms answered]")
