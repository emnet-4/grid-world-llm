"""
Step 3: Teaching conversation.
Teacher (who knows the grid) tells Student directional facts about all
C(5,2) = 10 room pairs. Student confirms each fact by paraphrasing.
Saves the full transcript for use as context in the after-probe.
"""

import json
import os
import random
import itertools
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "llama3.2")
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "mistral")
TEACHING_ORDER_SEED = 99

with open("grid_world.json", "r") as f:
    world = json.load(f)

rooms = world["rooms"]
room_names = world["room_names"]

# All C(5,2) = 10 pairs
pairs = list(itertools.combinations(room_names, 2))
rng = random.Random(TEACHING_ORDER_SEED)
rng.shuffle(pairs)

teacher = ChatOllama(model=TEACHER_MODEL)
student = ChatOllama(model=STUDENT_MODEL)


def describe_features(room):
    return f"{room['color']} colored walls, {room['num_windows']} windows, and {room['num_doors']} doors"


def describe_direction(room_p, room_q):
    """
    Now includes MAGNITUDE, not just direction.
    Directional-only facts ("R0 is east of R1") don't determine metric distance,
    so distance-comparison test questions were unanswerable in principle.
    Including unit counts makes the taught information sufficient for the test.
    """
    dx = room_p["x"] - room_q["x"]
    dy = room_p["y"] - room_q["y"]

    parts = []
    if dx != 0:
        parts.append(f"{abs(dx)} units {'east' if dx > 0 else 'west'}")
    if dy != 0:
        parts.append(f"{abs(dy)} units {'north' if dy > 0 else 'south'}")

    return " and ".join(parts) if parts else "at the same position as"


TEACHER_SYSTEM = (
    "You are teaching a student about the layout of a set of rooms. "
    "State each fact clearly in your own words. Don't add false information. "
    "Don't narrate what you're doing, just state the fact."
)
STUDENT_SYSTEM = (
    "You are learning about the layout of rooms from a teacher. "
    "When told a fact, confirm by repeating it back in your own words. "
    "Don't add anything false. Don't narrate, just confirm."
)

teacher_history = [SystemMessage(content=TEACHER_SYSTEM)]
student_history = [SystemMessage(content=STUDENT_SYSTEM)]
transcript = []

print(f"Teaching {len(pairs)} room-pair facts...\n")

for turn_num, (p_name, q_name) in enumerate(pairs, start=1):
    rp, rq = rooms[p_name], rooms[q_name]
    direction = describe_direction(rp, rq)
    fact = (
        f"{p_name}, which has {describe_features(rp)}, is {direction} "
        f"of {q_name}, which has {describe_features(rq)}."
    )

    teacher_history.append(HumanMessage(content=f"State this fact: \"{fact}\""))
    teacher_reply = teacher.invoke(teacher_history).content.strip()
    teacher_history.append(AIMessage(content=teacher_reply))

    student_history.append(HumanMessage(content=teacher_reply))
    student_reply = student.invoke(student_history).content.strip()
    student_history.append(AIMessage(content=student_reply))

    transcript.append({
        "turn": turn_num,
        "pair": [p_name, q_name],
        "ground_truth_fact": fact,
        "teacher_said": teacher_reply,
        "student_confirmed": student_reply,
    })
    print(f"[{turn_num}/{len(pairs)}] Teacher: {teacher_reply}")
    print(f"          Student: {student_reply}\n")

# Build the full text context for the after-probe
full_context = "\n".join(
    f"Teacher: {t['teacher_said']}\nStudent: {t['student_confirmed']}"
    for t in transcript
)

with open("teaching_transcript.json", "w") as f:
    json.dump({
        "format": "pairs",
        # every pair is stated, so no question is held out and
        # memorisation cannot be distinguished from inference here
        "taught_pairs": [sorted(list(p)) for p in pairs],
        "transcript": transcript,
        "full_context": full_context,
    }, f, indent=2)

print(f"\nSaved teaching_transcript.json ({len(transcript)} facts taught)")
