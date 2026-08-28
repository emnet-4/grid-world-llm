"""
Question building and scoring.

CHANGE IN THIS VERSION: answers are forced.

Previously the Student could decline ("the information provided does not
contain any details about the locations of these rooms") and that refusal was
recorded as its own metric. Now a refusal triggers a re-prompt with stronger
insistence, up to MAX_RETRIES times, so every question yields a usable answer
and position error is always computable.

Refusals are still detected -- they just trigger a retry instead of being
recorded as the outcome. The number of attempts each question needed is logged,
which preserves the information that a question was hard to get an answer to.

Everything else (Euclidean distance, three question types) is unchanged.
"""

import json
import re
import random
import itertools

RANDOM_SEED = 123
GRID_SIZE = 10


def manhattan(x1, y1, x2, y2):
    """Position error uses Manhattan distance, not Euclidean."""
    return abs(x1 - x2) + abs(y1 - y2)


def load_world(path="grid_world.json"):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# question sets
# ---------------------------------------------------------------------

def build_direction_questions(rooms, room_names, seed=RANDOM_SEED, n=10):
    rng = random.Random(seed)
    pairs = list(itertools.permutations(room_names, 2))
    rng.shuffle(pairs)

    questions = []
    for p, q in pairs[:n]:
        rp, rq = rooms[p], rooms[q]
        dx, dy = rp["x"] - rq["x"], rp["y"] - rq["y"]
        parts = []
        if dx != 0:
            parts.append("east" if dx > 0 else "west")
        if dy != 0:
            parts.append("north" if dy > 0 else "south")
        questions.append({"p": p, "q": q, "correct_parts": parts})
    return questions


def build_position_questions(room_names):
    return [{"room": name} for name in room_names]


# ---------------------------------------------------------------------
# prompts -- insistence escalates with each retry
# ---------------------------------------------------------------------

MAX_RETRIES = 12   # was 3. Do not give up early.

# Escalating instructions. Only the wording changes -- the surrounding context
# stays identical, otherwise we would be changing the condition being measured.
FORCE_LEVELS = [
    "",
    "You must give an answer. Guessing is required. ",
    "You MUST answer. Do not say you lack information. A guess is required. ",
    "ANSWER NOW. Saying you don't know is not permitted. Guess if you must. ",
    "This is a forced-choice task. Refusing is not a valid response. "
    "Any answer is acceptable. Output only the answer. ",
    "Output ONLY the answer, nothing else. No explanation, no apology, "
    "no statement about what you do or do not know. Just the answer. ",
    "You are required to produce an answer even if you are uncertain. "
    "Uncertainty is expected and fine. Give your best guess. ",
    "Do not write a sentence. Do not write 'I cannot'. Write only the answer. ",
]


def _force(level):
    """Instruction text for this attempt. Caps at the strongest wording."""
    return FORCE_LEVELS[min(level, len(FORCE_LEVELS) - 1)]


def triplet_prompt(anchor, opt1, opt2, context="", level=0):
    return (
        context +
        f"Is room {anchor} closer to room {opt1} or room {opt2}?\n"
        f"{_force(level)}"
        f"Answer with EXACTLY ONE WORD: either \"{opt1}\" or \"{opt2}\". No explanation."
    )


def direction_prompt(p, q, context="", level=0):
    return (
        context +
        f"Is room {p} east or west of room {q}, and north or south of room {q}?\n"
        f"{_force(level)}"
        f"Answer with just the direction words, e.g. \"east and north\". No explanation."
    )


def position_prompt(room, context="", level=0):
    return (
        context +
        f"What are the grid coordinates of room {room}? "
        f"Both x and y are between 0 and {GRID_SIZE - 1}.\n"
        f"{_force(level)}"
        f"Answer with EXACTLY the format \"x,y\" (for example \"3,7\"). No explanation."
    )


def map_prompt(room_names, context="", level=0):
    """
    Ask for the WHOLE layout in one response, rather than one room at a time.

    The Student is told rooms occupy distinct cells. That is true of the world
    by construction (positions are sampled without replacement) but it was
    never stated, so the Student had no reason not to put every room in the
    same place. Stating it also makes this question genuinely different from
    five independent position questions, because the answer now has to be
    internally consistent.
    """
    lines = "\n".join(f"{n}: x,y" for n in room_names)
    return (
        context +
        f"List the grid coordinates of every room.\n\n"
        f"Use exactly this format, one room per line, nothing else:\n"
        f"{lines}\n\n"
        f"Both x and y are whole numbers between 0 and {GRID_SIZE - 1}.\n"
        f"Each room is in a different cell. No two rooms share the same coordinates.\n"
        f"{_force(level)}"
        f"List all {len(room_names)} rooms."
    )


# ---------------------------------------------------------------------
# refusal detection -- now only used to trigger a retry
# ---------------------------------------------------------------------

REFUSAL_MARKERS = [
    "impossible to determine", "cannot determine", "can't determine",
    "unable to determine", "not possible to determine",
    "insufficient information", "not enough information",
    "no information", "does not contain", "doesn't contain",
    "without additional", "without specific", "without more",
    "cannot answer", "can't answer", "unable to answer",
    "does not specify", "doesn't specify", "not specified",
    "i'm sorry", "i am sorry", "not given", "unanswerable",
    # phrases seen in actual runs that the earlier list missed
    "cannot provide", "can't provide", "unable to provide",
    "cannot find", "can't find", "cannot be determined",
    "no context", "without context", "without knowing",
    "as a responsible ai", "as an ai", "i don't have",
    "do not have the", "don't have the", "more information",
    "additional context", "not enough context", "isn't enough",
    "is not enough", "cannot say", "can't say", "unclear",
    "does not include", "doesn't include", "no details",
]


# Adverbs the model slips in mid-phrase, e.g. "can't DEFINITIVELY determine",
# which defeats plain substring matching.
_HEDGE_PATTERN = re.compile(
    r"\b(can ?not|can't|cannot|unable to|won't be able to|not able to)\b"
    r"[^.]{0,40}?"
    r"\b(determine|provide|say|tell|know|find|answer|give|specify)\b",
    re.IGNORECASE,
)


def looks_like_refusal(text):
    """
    True if the response reads as a refusal or a hedge rather than an answer.

    Two passes. First a regex for "cannot ... determine" style phrases, which
    tolerates adverbs in between -- "can't definitively determine" slipped past
    plain substring matching. Then the explicit marker list.
    """
    if _HEDGE_PATTERN.search(text):
        return True
    t = text.lower()
    return any(m in t for m in REFUSAL_MARKERS)


def ask_forced(model, prompt_fn, parse_fn):
    """
    Keep asking with escalating insistence until we get a real answer.

    A response is only accepted if it parses AND does not read as a refusal.
    Previously a refusal that happened to contain a number was accepted --
    e.g. "I cannot provide the coordinates of room R2 without knowing the
    specific grid system" had [5,3] pulled out of it and scored as an answer.
    That is now rejected and retried.

    If MAX_RETRIES is exhausted the question is recorded as UNANSWERED, not
    as wrong. Unanswered questions are excluded from accuracy and error, and
    counted separately, because scoring a refusal as an incorrect answer
    silently biases every metric.
    """
    raw = ""
    for level in range(MAX_RETRIES + 1):
        raw = model.invoke(prompt_fn(level)).content.strip()
        parsed = parse_fn(raw)

        if parsed is not None and not looks_like_refusal(raw):
            return raw, parsed, level + 1

    return raw, None, MAX_RETRIES + 1


# ---------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------

def parse_triplet(response, opt1, opt2):
    t = response.lower()
    p1, p2 = t.find(opt1.lower()), t.find(opt2.lower())
    if p1 == -1 and p2 == -1:
        return None
    return opt1 if (p2 == -1 or (p1 != -1 and p1 < p2)) else opt2


def parse_direction(response):
    t = response.lower()
    found = {d for d in ["east", "west", "north", "south"] if d in t}
    # naming all four is hedging, not answering
    return sorted(found) if 0 < len(found) < 4 else None


def parse_position(response):
    m = re.search(r"(\d+)\s*,\s*(\d+)", response)
    if not m:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
        return None
    return (x, y)


def parse_map(response, room_names):
    """
    Pull "R0: 8,1" style lines out of the response. Tolerant of extra prose,
    bullets, parentheses, and "=" instead of ":".

    Returns None if two or more rooms are placed in the same cell. The world
    never does that and the Student is told so in the prompt, so a duplicate
    is a malformed answer rather than a wrong one, and it gets re-prompted
    like any other malformed response.
    """
    coords = {}
    for name in room_names:
        m = re.search(
            rf"\b{re.escape(name)}\b\s*[:=]?\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?",
            response,
        )
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                coords[name] = (x, y)

    if not coords:
        return None

    # reject overlapping rooms
    if len(set(coords.values())) < len(coords):
        return None

    return coords


# ---------------------------------------------------------------------
# main probe
# ---------------------------------------------------------------------

def run_all_probes(model, world, context=""):
    rooms = world["rooms"]
    room_names = world["room_names"]
    triplets = world["all_triplets"]

    direction_qs = build_direction_questions(rooms, room_names)
    position_qs = build_position_questions(room_names)

    all_attempts = []

    # --- triplets ---
    triplet_results = []
    print(f"  Triplet questions ({len(triplets)})...")
    for t in triplets:
        raw, parsed, n = ask_forced(
            model,
            lambda lvl, t=t: triplet_prompt(t["anchor"], t["opt1"], t["opt2"], context, lvl),
            lambda r, t=t: parse_triplet(r, t["opt1"], t["opt2"]),
        )
        all_attempts.append(n)
        triplet_results.append({
            "anchor": t["anchor"], "opt1": t["opt1"], "opt2": t["opt2"],
            "correct": t["correct"], "raw_response": raw, "parsed": parsed,
            "is_correct": parsed == t["correct"], "attempts": n,
        })

    # --- directions ---
    direction_results = []
    print(f"  Direction questions ({len(direction_qs)})...")
    for d in direction_qs:
        raw, parsed, n = ask_forced(
            model,
            lambda lvl, d=d: direction_prompt(d["p"], d["q"], context, lvl),
            parse_direction,
        )
        all_attempts.append(n)
        direction_results.append({
            "p": d["p"], "q": d["q"], "correct_parts": d["correct_parts"],
            "raw_response": raw, "parsed": parsed,
            "is_correct": parsed is not None and set(parsed) == set(d["correct_parts"]),
            "attempts": n,
        })

    # --- positions ---
    position_results = []
    print(f"  Position questions ({len(position_qs)})...")
    for pq in position_qs:
        room = rooms[pq["room"]]
        raw, parsed, n = ask_forced(
            model,
            lambda lvl, pq=pq: position_prompt(pq["room"], context, lvl),
            parse_position,
        )
        all_attempts.append(n)
        err = None
        if parsed:
            err = manhattan(parsed[0], parsed[1], room["x"], room["y"])
        position_results.append({
            "room": pq["room"], "true_x": room["x"], "true_y": room["y"],
            "raw_response": raw, "parsed": list(parsed) if parsed else None,
            "error": err, "attempts": n,
        })

    # --- whole-grid map ---
    print("  Whole-layout question...")
    raw, parsed, n = ask_forced(
        model,
        lambda lvl: map_prompt(room_names, context, lvl),
        lambda r: parse_map(r, room_names),
    )
    all_attempts.append(n)
    per_room = {}
    if parsed:
        for name, (gx, gy) in parsed.items():
            per_room[name] = manhattan(gx, gy, rooms[name]["x"], rooms[name]["y"])
    map_result = {
        "raw_response": raw,
        "rejected_for_overlap": parsed is None and bool(
            re.search(r"\bR\d+\b\s*[:=]?\s*\(?\s*\d+\s*,\s*\d+", raw)
        ),
        "parsed_coords": {k: list(v) for k, v in (parsed or {}).items()},
        "n_rooms_placed": len(parsed or {}),
        "n_rooms_total": len(room_names),
        "per_room_error": per_room,
        "mean_error": (sum(per_room.values()) / len(per_room)) if per_room else None,
        "attempts": n,
    }

    # --- summary ---
    valid_errs = [r["error"] for r in position_results if r["error"] is not None]
    all_q = triplet_results + direction_results + position_results
    n_unanswered = sum(1 for r in all_q if r["parsed"] is None)

    # accuracy over ANSWERED questions only -- an unanswered question is not
    # a wrong answer, and counting it as one drags every condition toward zero
    ans_trip = [r for r in triplet_results if r["parsed"] is not None]
    ans_dir = [r for r in direction_results if r["parsed"] is not None]

    return {
        "triplet_answers": triplet_results,
        "direction_answers": direction_results,
        "position_answers": position_results,
        "map_answer": map_result,

        "triplet_accuracy": (sum(1 for r in ans_trip if r["is_correct"]) / len(ans_trip)) if ans_trip else None,
        "direction_accuracy": (sum(1 for r in ans_dir if r["is_correct"]) / len(ans_dir)) if ans_dir else None,
        "n_triplet_answered": len(ans_trip),
        "n_triplet_total": len(triplet_results),
        "n_direction_answered": len(ans_dir),
        "n_direction_total": len(direction_results),
        "mean_position_error": (sum(valid_errs) / len(valid_errs)) if valid_errs else None,
        "map_mean_error": map_result["mean_error"],
        "map_rooms_placed": map_result["n_rooms_placed"],

        "n_position_answered": len(valid_errs),
        "n_position_total": len(position_results),

        # how hard it was to get answers -- replaces refusal rate
        "mean_attempts": sum(all_attempts) / len(all_attempts),
        "max_attempts": max(all_attempts),
        "pct_needed_retry": sum(1 for a in all_attempts if a > 1) / len(all_attempts),
        "n_still_unanswered": n_unanswered,
    }
