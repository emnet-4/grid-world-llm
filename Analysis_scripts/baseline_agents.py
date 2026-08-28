"""
Baseline agents.

Instead of drawing a single calculated line for chance, this runs two dumb
agents over the SAME grids the models saw, many times, so every metric gets an
empirical mean and standard deviation. A result can then be compared against
the range a baseline actually produces rather than against a point.

  RANDOM AGENT   picks uniformly at random
                   triplet   -> one of the two options
                   direction -> one of east/west crossed with one of north/south
                   position  -> a random cell
                   map       -> a random cell per room

  LINE AGENT     believes every room sits on a single straight line, either
                 a row or a column, and answers every question from that
                 internal layout.

                 This is the most informative of the three, because unlike the
                 other two it has a coherent world model. It is just the wrong
                 one. If a model scores no better than this, then having any
                 self-consistent layout is enough to reach that score and the
                 layout does not have to be correct.

  QUADRANT AGENT answers the centre of each room's quadrant. It knows which
                 region a room belongs to, which is fixed by convention, but
                 nothing about where it sits inside that region. So it stands
                 for "knows the neighbourhood, not the position", and unlike
                 the old fixed-centre agent it respects the no-overlap rule and
                 can be beaten by a model that is accurate to a cell or two.

Both agents emit TEXT, which is then run through the same parsers the models'
answers go through. That way any quirk of the parsing applies equally to the
baselines and to the models, and the comparison stays fair.

Note on the centre: on a 0-9 axis the mean distance from 4 to a random cell is
25/10 = 2.5, and from 4.5 it is also 2.5. So an integer centre and a half-cell
centre score identically, and the agent answers an integer as a model would.
"""

import json
import os
import random
import statistics

from probe_questions import (load_world, build_direction_questions,
                             build_position_questions, manhattan,
                             parse_triplet, parse_direction, parse_position,
                             parse_map, GRID_SIZE)

N_REPEATS = 30
SEED = 20250101
CENTRE = GRID_SIZE // 2 - 1   # 4 on a 10-wide grid

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------
# the agents -- both return TEXT, exactly as a model would
# ---------------------------------------------------------------------

class RandomAgent:
    name = "random"

    def __init__(self, rng):
        self.rng = rng

    def triplet(self, anchor, opt1, opt2):
        return self.rng.choice([opt1, opt2])

    def direction(self, p, q):
        return f"{self.rng.choice(['east', 'west'])} and {self.rng.choice(['north', 'south'])}"

    def position(self, room):
        return f"{self.rng.randrange(GRID_SIZE)},{self.rng.randrange(GRID_SIZE)}"

    def map(self, room_names):
        return "\n".join(
            f"{n}: {self.rng.randrange(GRID_SIZE)},{self.rng.randrange(GRID_SIZE)}"
            for n in room_names
        )


class LineAgent:
    """
    Assumes the rooms are collinear. Picks an orientation and a line position
    at random, spreads the rooms along it in a random order, then answers every
    question from that layout. Because it has a real layout it can compute
    distances and directions rather than guessing them.
    """
    name = "line"

    def __init__(self, rng, room_names=None):
        self.rng = rng
        self.layout = None
        self._names = room_names

    def _build(self, room_names):
        if self.layout is not None:
            return
        rng = self.rng
        horizontal = rng.random() < 0.5
        fixed_coord = rng.randrange(GRID_SIZE)

        # distinct positions along the line, since rooms cannot overlap
        slots = rng.sample(range(GRID_SIZE), len(room_names))
        order = list(room_names)
        rng.shuffle(order)

        self.layout = {}
        for name, s in zip(order, slots):
            self.layout[name] = (s, fixed_coord) if horizontal else (fixed_coord, s)

    def _d(self, a, b):
        pa, pb = self.layout[a], self.layout[b]
        return manhattan(pa[0], pa[1], pb[0], pb[1])

    def triplet(self, anchor, opt1, opt2):
        self._build([anchor, opt1, opt2] if self._names is None else self._names)
        for n in (anchor, opt1, opt2):
            if n not in self.layout:
                return self.rng.choice([opt1, opt2])
        return opt1 if self._d(anchor, opt1) <= self._d(anchor, opt2) else opt2

    def direction(self, p, q):
        self._build(self._names or [p, q])
        if p not in self.layout or q not in self.layout:
            return f"{self.rng.choice(['east','west'])} and {self.rng.choice(['north','south'])}"
        (px, py), (qx, qy) = self.layout[p], self.layout[q]
        parts = []
        if px != qx:
            parts.append("east" if px > qx else "west")
        if py != qy:
            parts.append("north" if py > qy else "south")
        return " and ".join(parts) if parts else "east"

    def position(self, room):
        self._build(self._names or [room])
        x, y = self.layout.get(room, (CENTRE, CENTRE))
        return f"{x},{y}"

    def map(self, room_names):
        self._build(room_names)
        return "\n".join(f"{n}: {self.layout[n][0]},{self.layout[n][1]}"
                          for n in room_names)


class QuadrantAgent:
    """
    Answers the centre of each room's quadrant.

    Replaces the old fixed-centre agent, which answered (4,4) for every room.
    That one broke the no-overlap rule and its score depended on where the
    rooms happened to land rather than on anything meaningful, and a model
    that was accurate to within a cell could not beat it in a way that showed.

    This agent knows which quadrant each room belongs to, which is fixed by
    convention, but nothing about where it sits inside that quadrant. So it
    represents "knows the neighbourhood, not the position". It scores about
    2.40, and a model accurate to a cell or two beats it.

    For triplets and direction it answers from its quadrant-centre layout, the
    same way the line agent answers from its own layout.
    """
    name = "quadrant"

    def __init__(self, rng, room_names=None):
        self.rng = rng
        self.centres = None
        self._names = room_names

    def _build(self, world_quadrant_centres):
        if self.centres is None:
            self.centres = world_quadrant_centres

    def _pos(self, room):
        return tuple(self.centres.get(room, (CENTRE, CENTRE)))

    def triplet(self, anchor, opt1, opt2):
        a, o1, o2 = self._pos(anchor), self._pos(opt1), self._pos(opt2)
        d1 = manhattan(a[0], a[1], o1[0], o1[1])
        d2 = manhattan(a[0], a[1], o2[0], o2[1])
        if d1 == d2:
            return self.rng.choice([opt1, opt2])
        return opt1 if d1 < d2 else opt2

    def direction(self, p, q):
        (px, py), (qx, qy) = self._pos(p), self._pos(q)
        parts = []
        if px != qx:
            parts.append("east" if px > qx else "west")
        if py != qy:
            parts.append("north" if py > qy else "south")
        return " and ".join(parts) if parts else "east"

    def position(self, room):
        x, y = self._pos(room)
        return f"{x},{y}"

    def map(self, room_names):
        return "\n".join(f"{n}: {self._pos(n)[0]},{self._pos(n)[1]}"
                          for n in room_names)


# ---------------------------------------------------------------------
# scoring, using the same parsers the models go through
# ---------------------------------------------------------------------

def score_agent(agent, world):
    # the line agent builds one layout and answers everything from it, so it
    # needs the full room list before the first question
    if hasattr(agent, "_names"):
        agent._names = world["room_names"]
    if hasattr(agent, "centres"):
        agent._build(world.get("quadrant_centres", {}))

    rooms = world["rooms"]
    room_names = world["room_names"]
    triplets = world["all_triplets"]

    # triplets
    correct = 0
    for t in triplets:
        raw = agent.triplet(t["anchor"], t["opt1"], t["opt2"])
        parsed = parse_triplet(raw, t["opt1"], t["opt2"])
        correct += (parsed == t["correct"])
    triplet_acc = correct / len(triplets)

    # directions
    dqs = build_direction_questions(rooms, room_names)
    correct = 0
    for d in dqs:
        raw = agent.direction(d["p"], d["q"])
        parsed = parse_direction(raw)
        correct += (parsed is not None and set(parsed) == set(d["correct_parts"]))
    direction_acc = correct / len(dqs)

    # positions
    errs = []
    for pq in build_position_questions(room_names):
        r = rooms[pq["room"]]
        parsed = parse_position(agent.position(pq["room"]))
        if parsed:
            errs.append(manhattan(parsed[0], parsed[1], r["x"], r["y"]))
    pos_err = statistics.mean(errs) if errs else None

    # map
    parsed_map = parse_map(agent.map(room_names), room_names)
    map_errs = []
    if parsed_map:
        for n, (x, y) in parsed_map.items():
            map_errs.append(manhattan(x, y, rooms[n]["x"], rooms[n]["y"]))
    map_err = statistics.mean(map_errs) if map_errs else None

    return {
        "triplet_accuracy": triplet_acc,
        "direction_accuracy": direction_acc,
        "mean_position_error": pos_err,
        "map_mean_error": map_err,
    }


# ---------------------------------------------------------------------
# run
# ---------------------------------------------------------------------

def summarise(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "mean": statistics.mean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "p5": sorted(vals)[max(0, int(0.05 * len(vals)) - 1)],
        "p95": sorted(vals)[min(len(vals) - 1, int(0.95 * len(vals)))],
        "n": len(vals),
    }


if __name__ == "__main__":
    with open(os.path.join(HERE, "runs_index.json")) as f:
        index = json.load(f)

    # every distinct grid the models were tested on
    grids = []
    seen = set()
    for e in index:
        gw = os.path.join(HERE, e["folder"], "grid_world.json")
        if not os.path.exists(gw):
            continue
        if e["seed"] in seen:
            continue
        seen.add(e["seed"])
        with open(gw) as f:
            grids.append(json.load(f))

    print(f"Found {len(grids)} distinct grids from the model runs")
    print(f"Running each agent {N_REPEATS} times per grid\n")

    results = {}

    for AgentClass in (RandomAgent, LineAgent, QuadrantAgent):
        per_metric = {m: [] for m in
                      ["triplet_accuracy", "direction_accuracy",
                       "mean_position_error", "map_mean_error"]}

        for gi, world in enumerate(grids):
            for rep in range(N_REPEATS):
                rng = random.Random(SEED + gi * 1000 + rep)
                agent = AgentClass(rng)
                s = score_agent(agent, world)
                for m in per_metric:
                    per_metric[m].append(s[m])

        results[AgentClass.name] = {m: summarise(v) for m, v in per_metric.items()}

        n_obs = len(per_metric["triplet_accuracy"])
        print(f"{AgentClass.name.upper()} AGENT   ({n_obs} observations: "
              f"{len(grids)} grids x {N_REPEATS} repeats)")
        print(f"  {'metric':<24}{'mean':<10}{'sd':<10}{'5th-95th pct':<20}{'min-max'}")
        print("  " + "-" * 74)
        for m, s in results[AgentClass.name].items():
            if s is None:
                continue
            print(f"  {m:<24}{s['mean']:<10.3f}{s['sd']:<10.3f}"
                  f"{f'{s[chr(112)+chr(53)]:.2f} - {s[chr(112)+chr(57)+chr(53)]:.2f}':<20}"
                  f"{s['min']:.2f} - {s['max']:.2f}")
        print()

    with open(os.path.join(HERE, "agent_baselines.json"), "w") as f:
        json.dump({
            "n_repeats": N_REPEATS,
            "n_grids": len(grids),
            "seed": SEED,
            "agents": results,
        }, f, indent=2)

    print("Saved agent_baselines.json")
    print("\nThe sd and percentile columns are the point of this. A model result")
    print("only means something if it falls outside the range these agents cover.")
