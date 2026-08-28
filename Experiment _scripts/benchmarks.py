"""
Chance benchmarks for position error.

Triplet questions have an obvious 50% chance line. Position error doesn't, so
a reader can't tell whether an error of 5.0 is good or meaningless. These give
two reference points.

Computed by enumerating every cell pair on the grid, not approximated.

    Euclidean (current):   random 5.19   centre 3.81
    Manhattan (planned):   random 6.60   centre 5.00

  random  -- pick a cell uniformly at random. Above this line the model is
             doing no better than guessing.
  centre  -- always answer the middle of the grid. A trivial constant strategy.
             Above this line the model is beaten by a fixed answer, which is
             the line that actually matters.
"""

import math

GRID_SIZE = 10


def _cells(n):
    return [(x, y) for x in range(n) for y in range(n)]


def random_guess_benchmark(grid_size=GRID_SIZE, metric="euclidean"):
    cells = _cells(grid_size)
    if metric == "manhattan":
        total = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in cells for b in cells)
    else:
        total = sum(math.dist(a, b) for a in cells for b in cells)
    return total / (len(cells) ** 2)


def centre_guess_benchmark(grid_size=GRID_SIZE, metric="euclidean"):
    cells = _cells(grid_size)
    c = ((grid_size - 1) / 2, (grid_size - 1) / 2)
    if metric == "manhattan":
        total = sum(abs(c[0] - b[0]) + abs(c[1] - b[1]) for b in cells)
    else:
        total = sum(math.dist(c, b) for b in cells)
    return total / len(cells)


if __name__ == "__main__":
    for metric in ("euclidean", "manhattan"):
        print(f"{metric}:")
        print(f"  random guess: {random_guess_benchmark(metric=metric):.2f}")
        print(f"  centre guess: {centre_guess_benchmark(metric=metric):.2f}")


# ---------------------------------------------------------------------
# reader-facing maps
# ---------------------------------------------------------------------

def render_map(coords, grid_size=GRID_SIZE, cell=4):
    """
    Draw a coordinate set as a text map, north at the top. This is for the
    READER of the report -- it is not what we ask the model for.
    """
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]
    for name, (x, y) in coords.items():
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < grid_size and 0 <= yi < grid_size:
            grid[(grid_size - 1) - yi][xi] = name

    out = []
    for r, row in enumerate(grid):
        out.append(f"y={(grid_size - 1) - r} | " + "".join(c.ljust(cell) for c in row))
    out.append("     +" + "-" * (grid_size * cell))
    out.append("      " + "".join(str(x).ljust(cell) for x in range(grid_size)))
    return "\n".join(out)


def render_side_by_side(true_coords, model_coords, grid_size=GRID_SIZE):
    """True layout, the model's layout, and per-room error."""
    out = ["TRUE LAYOUT", render_map(true_coords, grid_size), "",
           "MODEL'S LAYOUT", render_map(model_coords, grid_size), "",
           "PER-ROOM ERROR (Manhattan)"]

    total, n = 0, 0
    for name in sorted(true_coords):
        tx, ty = true_coords[name]
        if name in model_coords:
            mx, my = (int(round(v)) for v in model_coords[name])
            err = abs(mx - tx) + abs(my - ty)
            total += err
            n += 1
            out.append(f"  {name}: true ({tx},{ty})   model ({mx},{my})   error {err}")
        else:
            out.append(f"  {name}: true ({tx},{ty})   model —   not placed")

    if n:
        out.append(f"\n  Mean error {total / n:.2f}   "
                   f"(random guess {random_guess_benchmark(metric='manhattan'):.2f}, "
                   f"centre guess {centre_guess_benchmark(metric='manhattan'):.2f})")
    return "\n".join(out)


def direction_chance(grid_size=GRID_SIZE, n_rooms=5, trials=20000, seed=0):
    """
    Chance level for the direction question.

    Not 50%. The answer needs BOTH axes right -- one of east/west and one of
    north/south -- so a guesser picking uniformly from the four combinations
    is right about a quarter of the time, minus the pairs whose true answer
    has only one direction (dx or dy is zero), where a two-part guess is
    automatically wrong.

    Estimated by simulation: ~0.20 for 5 rooms on a 10x10 grid.
    """
    import random, itertools
    rng = random.Random(seed)
    options = [{"east", "north"}, {"east", "south"},
               {"west", "north"}, {"west", "south"}]

    hits = total = 0
    cells_all = [(x, y) for x in range(grid_size) for y in range(grid_size)]

    for _ in range(trials):
        cells = rng.sample(cells_all, n_rooms)
        for i, j in itertools.permutations(range(n_rooms), 2):
            dx = cells[i][0] - cells[j][0]
            dy = cells[i][1] - cells[j][1]
            truth = set()
            if dx:
                truth.add("east" if dx > 0 else "west")
            if dy:
                truth.add("north" if dy > 0 else "south")
            total += 1
            if rng.choice(options) == truth:
                hits += 1

    return hits / total
