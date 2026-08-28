"""
Step 1: Generate the ground-truth grid world.

QUADRANT DESIGN. Each room is confined to a fixed region of the grid:

    R0  top left      x 0-4   y 5-9
    R1  top right     x 5-9   y 5-9
    R2  centre        x 3-7   y 3-7
    R3  bottom left   x 0-4   y 0-4
    R4  bottom right  x 5-9   y 0-4

Position within the quadrant is still random, and no two rooms share a cell.

Why: the old fixed-centre benchmark answered (4,4) for every room, which
broke the no-overlap rule and scored well or badly depending on where the
rooms happened to land. With rooms pinned to quadrants, a benchmark that
answers each room's quadrant centre represents "knows the neighbourhood but
not the position", scores about 2.40, and can be beaten by a model that is
accurate to within a cell or two. That makes it a meaningful reference rather
than an artefact.

North is up: y=9 is the top of the grid, y=0 the bottom.

Features (colour, windows, doors) remain independent of position.
"""

import json
import random
import itertools
import os

GRID_SIZE = 10
RANDOM_SEED = int(os.environ.get("GRID_SEED", 42))

# name -> (x_min, x_max, y_min, y_max), inclusive. y=9 is north.
QUADRANTS = {
    "R0": (0, 4, 5, 9),   # top left
    "R1": (5, 9, 5, 9),   # top right
    "R2": (3, 7, 3, 7),   # centre
    "R3": (0, 4, 0, 4),   # bottom left
    "R4": (5, 9, 0, 4),   # bottom right
}

COLORS = ["red", "green", "blue", "yellow", "white"]
WINDOW_COUNTS = [0, 1, 2, 3]
DOOR_COUNTS = [1, 2]


def quadrant_centre(name):
    x0, x1, y0, y1 = QUADRANTS[name]
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def manhattan(r1, r2):
    return abs(r1["x"] - r2["x"]) + abs(r1["y"] - r2["y"])


def generate_rooms(seed=RANDOM_SEED):
    rng = random.Random(seed)

    while True:
        placed = {}
        for name, (x0, x1, y0, y1) in QUADRANTS.items():
            placed[name] = (rng.randint(x0, x1), rng.randint(y0, y1))
        # quadrants overlap slightly at the centre, so re-draw on a collision
        if len(set(placed.values())) == len(placed):
            break

    rooms = {}
    for name, (x, y) in placed.items():
        rooms[name] = {
            "id": name, "x": x, "y": y,
            "color": rng.choice(COLORS),
            "num_windows": rng.choice(WINDOW_COUNTS),
            "num_doors": rng.choice(DOOR_COUNTS),
        }
    return rooms


def describe_features(room):
    return (f"{room['color']} colored walls, {room['num_windows']} windows, "
            f"and {room['num_doors']} doors")


def describe_direction(room_p, room_q):
    """Includes magnitude, so distance questions are answerable in principle."""
    dx = room_p["x"] - room_q["x"]
    dy = room_p["y"] - room_q["y"]
    parts = []
    if dx != 0:
        parts.append(f"{abs(dx)} units {'east' if dx > 0 else 'west'}")
    if dy != 0:
        parts.append(f"{abs(dy)} units {'north' if dy > 0 else 'south'}")
    return " and ".join(parts) if parts else "at the same position as"


def generate_all_triplets(rooms):
    names = sorted(rooms.keys())
    triplets = []
    for combo in itertools.combinations(names, 3):
        for anchor_idx in range(3):
            anchor = combo[anchor_idx]
            others = [combo[j] for j in range(3) if j != anchor_idx]
            d1 = manhattan(rooms[anchor], rooms[others[0]])
            d2 = manhattan(rooms[anchor], rooms[others[1]])
            if d1 == d2:
                continue
            triplets.append({
                "anchor": anchor, "opt1": others[0], "opt2": others[1],
                "correct": others[0] if d1 < d2 else others[1],
            })
    return triplets


if __name__ == "__main__":
    rooms = generate_rooms()
    room_names = sorted(rooms.keys())
    triplets = generate_all_triplets(rooms)

    gt = {}
    for a in room_names:
        for b in room_names:
            gt[f"{a}-{b}"] = manhattan(rooms[a], rooms[b])

    with open("grid_world.json", "w") as f:
        json.dump({
            "rooms": rooms,
            "room_names": room_names,
            "ground_truth_distances": gt,
            "all_triplets": triplets,
            "design": "quadrant",
            "quadrants": {k: list(v) for k, v in QUADRANTS.items()},
            "quadrant_centres": {k: list(quadrant_centre(k)) for k in QUADRANTS},
        }, f, indent=2)

    print(f"Generated {len(rooms)} rooms (quadrant design, seed {RANDOM_SEED}):")
    for name in room_names:
        r = rooms[name]
        qx0, qx1, qy0, qy1 = QUADRANTS[name]
        print(f"  {name}: ({r['x']}, {r['y']})  in quadrant x{qx0}-{qx1} y{qy0}-{qy1}"
              f"  centre {quadrant_centre(name)}  -  {describe_features(r)}")

    print(f"\nTriplet questions: {len(triplets)}")
    print(f"Teaching pairs C({len(rooms)},2): "
          f"{len(list(itertools.combinations(room_names, 2)))}")
    print("\nSaved grid_world.json")
