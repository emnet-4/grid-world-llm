"""
Step 6b (Exp 2): Ask the Student to save what it learned as a compact JSON.

Now validates that every coordinate falls inside the grid (0-9) and re-prompts
if not. Without this the Student can write impossible coordinates (e.g. x=15 on
a 10-wide grid), which then get faithfully read back at Time 3 and wreck the
position error -- measuring the Student's arithmetic rather than whether the
saved-notes mechanism works.
"""

import json
import os
import re
from langchain_ollama import ChatOllama
from probe_questions import load_world

STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "mistral")
NUM_CTX = 4096
MAX_ATTEMPTS = 4
GRID_MIN, GRID_MAX = 0, 9

world = load_world()
rooms = world["rooms"]
room_names = world["room_names"]

with open("teaching_transcript.json", "r") as f:
    teaching = json.load(f)

student = ChatOllama(model=STUDENT_MODEL, num_ctx=NUM_CTX)

room_descriptions = "\n".join(
    f"- {r['id']}: {r['color']} colored walls, {r['num_windows']} windows, {r['num_doors']} doors"
    for r in rooms.values()
)

BASE_PROMPT = (
    f"You know about the following rooms:\n{room_descriptions}\n\n"
    f"You just learned the following from a teacher about their layout:\n"
    f"{teaching['full_context']}\n\n"
    f"Based on what you learned, write down your best estimate of each room's "
    f"grid coordinates.\n\n"
    f"IMPORTANT: both x and y must be whole numbers between {GRID_MIN} and {GRID_MAX} "
    f"inclusive. Coordinates outside that range are invalid.\n\n"
    f"Reply with ONLY a JSON object in exactly this format, nothing else:\n"
    f'{{"R0": [x, y], "R1": [x, y], ...}}\n'
    f"Include all {len(room_names)} rooms: {', '.join(room_names)}."
)


def extract_json(text):
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def validate(parsed):
    """
    Returns (clean_dict, problems). A room is only kept if it has two integers
    inside the grid. problems lists what went wrong, for the retry prompt.
    """
    clean, problems = {}, []

    if parsed is None:
        return clean, ["response was not valid JSON"]

    for name in room_names:
        val = parsed.get(name)

        if val is None:
            problems.append(f"{name} missing")
            continue
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            problems.append(f"{name} not in [x, y] form")
            continue

        try:
            x, y = int(val[0]), int(val[1])
        except (TypeError, ValueError):
            problems.append(f"{name} coordinates not numbers")
            continue

        if not (GRID_MIN <= x <= GRID_MAX and GRID_MIN <= y <= GRID_MAX):
            problems.append(f"{name} = [{x}, {y}] is outside {GRID_MIN}-{GRID_MAX}")
            continue

        clean[name] = [x, y]

    return clean, problems


# ---------------- ask, validate, retry ----------------

attempts = []
saved_repr, problems = {}, ["no attempt yet"]

for attempt in range(1, MAX_ATTEMPTS + 1):
    if attempt == 1:
        prompt = BASE_PROMPT
    else:
        prompt = (
            BASE_PROMPT
            + f"\n\nYour previous answer had these problems: {'; '.join(problems)}.\n"
            + f"Please try again, making sure every coordinate is between "
              f"{GRID_MIN} and {GRID_MAX}."
        )

    raw = student.invoke(prompt).content.strip()
    parsed = extract_json(raw)
    saved_repr, problems = validate(parsed)

    attempts.append({
        "attempt": attempt,
        "raw_response": raw,
        "n_valid": len(saved_repr),
        "problems": problems,
    })

    print(f"Attempt {attempt}: {len(saved_repr)}/{len(room_names)} valid")
    print(f"  raw: {raw[:150]}")
    if problems:
        print(f"  problems: {'; '.join(problems)}")
    print()

    if len(saved_repr) == len(room_names):
        break

if len(saved_repr) < len(room_names):
    print(f"WARNING: after {MAX_ATTEMPTS} attempts only "
          f"{len(saved_repr)}/{len(room_names)} rooms are valid.\n")


# ---------------- score against ground truth ----------------

errors = {}
for name, coords in saved_repr.items():
    true_x, true_y = rooms[name]["x"], rooms[name]["y"]
    err = ((coords[0] - true_x) ** 2 + (coords[1] - true_y) ** 2) ** 0.5
    errors[name] = err
    print(f"  {name}: saved {tuple(coords)}, true ({true_x}, {true_y}), error {err:.2f}")

mean_error = sum(errors.values()) / len(errors) if errors else None
if mean_error is not None:
    print(f"\nMean coordinate error in saved representation: {mean_error:.2f}")

json_text = json.dumps(saved_repr)

with open("student_saved_representation.json", "w") as f:
    json.dump({
        "attempts": attempts,
        "n_attempts_used": len(attempts),
        "parsed_representation": saved_repr,
        "n_valid_rooms": len(saved_repr),
        "n_rooms_total": len(room_names),
        "json_text": json_text,
        "per_room_error": errors,
        "mean_error": mean_error,
    }, f, indent=2)

print("\nSaved student_saved_representation.json")
