"""
Teaching, in three formats.

The original teaching states one relative fact per room pair, in a random
order, and never gives a starting point. That is one way to present a layout,
and not obviously the natural one. This module implements three alternatives so
the format can be varied while everything downstream stays the same.

  coordinates   Absolute positions, stated directly.
                "R0 is at position (2, 7)."
                Metric information, no route. Nothing has to be assembled, so
                this is the easiest format and acts as a ceiling on what the
                teaching can convey.

  trail         A walk that visits every room in turn, each step given
                relative to the room before it.
                "Start at R0. From R0, walk 5 units east to reach R1."
                Connectivity and sequence, no absolute anchor. This is the
                format closest to how spatial knowledge is ordinarily
                acquired, and the one that should favour a graph-like
                representation if the model has one.

  both          The trail, followed by the coordinate list.
                If a model does better here than with either alone, the two
                formats are contributing different things.

Selected with the TEACHING_FORMAT environment variable:

    TEACHING_FORMAT=coordinates python3 step3_teaching_formats.py
    TEACHING_FORMAT=trail       python3 step3_teaching_formats.py
    TEACHING_FORMAT=both        python3 step3_teaching_formats.py

Teaching stays scripted in all three. The facts are computed from ground truth
and the Teacher only rephrases them, so it cannot invent a fact or omit a room.
The output file is named after the format, so the three can coexist in one run
folder.
"""

from __future__ import annotations

import json
import os
import random

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from probe_questions import load_world

TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "llama3.2")
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "mistral")
NUM_CTX = 4096

FORMAT = os.environ.get("TEACHING_FORMAT", "trail").lower()
TRAIL_SEED = int(os.environ.get("TRAIL_SEED", 99))

VALID_FORMATS = {"coordinates", "trail", "both"}
if FORMAT not in VALID_FORMATS:
    raise SystemExit(f"TEACHING_FORMAT must be one of {sorted(VALID_FORMATS)}, "
                     f"got {FORMAT!r}")

OUTPUT_FILE = f"teaching_{FORMAT}.json"


# ---------------------------------------------------------------------
# building the facts
# ---------------------------------------------------------------------

def describe_features(room: dict) -> str:
    return (f"{room['color']} colored walls, {room['num_windows']} windows, "
            f"and {room['num_doors']} doors")


def describe_step(frm: dict, to: dict) -> str:
    """How to walk from one room to another, as a single instruction."""
    dx, dy = to["x"] - frm["x"], to["y"] - frm["y"]
    parts = []
    if dx:
        parts.append(f"{abs(dx)} units {'east' if dx > 0 else 'west'}")
    if dy:
        parts.append(f"{abs(dy)} units {'north' if dy > 0 else 'south'}")
    return " and then ".join(parts) if parts else "nowhere, you are already there"


def coordinate_facts(world: dict) -> list[str]:
    """One statement per room, giving its absolute position."""
    rooms, names = world["rooms"], world["room_names"]
    return [
        f"{name} is at position ({rooms[name]['x']}, {rooms[name]['y']}). "
        f"It has {describe_features(rooms[name])}."
        for name in names
    ]


def trail_facts(world: dict, seed: int = TRAIL_SEED) -> list[str]:
    """
    A walk visiting every room once. The order is shuffled per run so the
    trail is not always the same sequence, and each step is given relative to
    the previous room rather than to any fixed origin.
    """
    rooms, names = world["rooms"], world["room_names"]
    order = list(names)
    random.Random(seed).shuffle(order)

    facts = [
        f"Start at {order[0]}, which has {describe_features(rooms[order[0]])}."
    ]
    for previous, nxt in zip(order, order[1:]):
        facts.append(
            f"From {previous}, walk {describe_step(rooms[previous], rooms[nxt])} "
            f"to reach {nxt}, which has {describe_features(rooms[nxt])}."
        )
    return facts


def taught_pairs(world: dict, seed: int = TRAIL_SEED) -> list[list[str]]:
    """
    Which room pairs are stated directly by this teaching format.

    This is what makes memorisation detectable. A question about a pair the
    Student was told about can be answered by recall; a question about a pair
    it was not told about requires combining facts. If performance holds on
    the first and collapses on the second, the Student memorised rather than
    inferred.

    coordinates  no pair is stated. Every relational answer must be computed
                 from two positions, so all pairs are held out.
    trail        the consecutive steps of the walk, four of the ten pairs.
                 The other six require chaining steps together.
    both         the same four as the trail.
    """
    if FORMAT == "coordinates":
        return []

    order = list(world["room_names"])
    random.Random(seed).shuffle(order)
    return [sorted([a, b]) for a, b in zip(order, order[1:])]


def build_facts(world: dict) -> list[str]:
    if FORMAT == "coordinates":
        return coordinate_facts(world)
    if FORMAT == "trail":
        return trail_facts(world)
    return trail_facts(world) + coordinate_facts(world)


# ---------------------------------------------------------------------
# the conversation
# ---------------------------------------------------------------------

TEACHER_SYSTEM = (
    "You are teaching a student about the layout of a set of rooms. "
    "State each fact clearly in your own words. Do not add anything false. "
    "Do not narrate what you are doing, just state the fact."
)

STUDENT_SYSTEM = (
    "You are learning about the layout of a set of rooms from a teacher. "
    "When told a fact, confirm it by repeating it back in your own words. "
    "Do not add anything false. Do not narrate, just confirm."
)


def main() -> None:
    world = load_world()
    facts = build_facts(world)

    teacher = ChatOllama(model=TEACHER_MODEL, num_ctx=NUM_CTX)
    student = ChatOllama(model=STUDENT_MODEL, num_ctx=NUM_CTX)

    teacher_history = [SystemMessage(content=TEACHER_SYSTEM)]
    student_history = [SystemMessage(content=STUDENT_SYSTEM)]
    transcript = []

    print(f"Teaching format: {FORMAT}")
    print(f"{len(facts)} facts to state\n")

    for turn, fact in enumerate(facts, start=1):
        teacher_history.append(
            HumanMessage(content=f'State this fact: "{fact}"'))
        said = teacher.invoke(teacher_history).content.strip()
        teacher_history.append(AIMessage(content=said))

        student_history.append(HumanMessage(content=said))
        confirmed = student.invoke(student_history).content.strip()
        student_history.append(AIMessage(content=confirmed))

        transcript.append({
            "turn": turn,
            "ground_truth_fact": fact,
            "teacher_said": said,
            "student_confirmed": confirmed,
        })
        print(f"[{turn}/{len(facts)}] Teacher: {said}")
        print(f"           Student: {confirmed}\n")

    full_context = "\n".join(
        f"Teacher: {t['teacher_said']}\nStudent: {t['student_confirmed']}"
        for t in transcript
    )

    with open(OUTPUT_FILE, "w") as fh:
        json.dump({
            "format": FORMAT,
            "n_facts": len(facts),
            "taught_pairs": taught_pairs(world),
            "ground_truth_facts": facts,
            "transcript": transcript,
            "full_context": full_context,
        }, fh, indent=2)

    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
