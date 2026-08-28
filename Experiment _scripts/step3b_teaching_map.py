"""
Exp 4 teaching: hand the Student a textual map instead of a conversation.

Exp 1 teaches by stating ten relative facts across ten turns -- "R2 is 9 units
west and 1 unit south of R3" -- and the Student has to stitch those fragments
into a layout itself. That assembly step is where the sign errors were showing
up in the saved JSONs.

Exp 4 delivers the same information as a finished picture. Nothing to assemble.

NOTE: this changes two things at once, format AND turn count. A map is
inherently one-shot, so there is no way to hold turns constant while switching
to a map. Worth stating in the writeup rather than pretending it is a clean
single-variable comparison.

Teaching stays fully scripted -- the map is drawn from ground truth and the
Teacher only presents it.
"""

import json
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from probe_questions import load_world
from benchmarks import render_map

TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "llama3.2")
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "mistral")
NUM_CTX = 4096

world = load_world()
rooms = world["rooms"]
room_names = world["room_names"]

teacher = ChatOllama(model=TEACHER_MODEL, num_ctx=NUM_CTX)
student = ChatOllama(model=STUDENT_MODEL, num_ctx=NUM_CTX)

# --- draw the true layout ---
true_coords = {n: (r["x"], r["y"]) for n, r in rooms.items()}
the_map = render_map(true_coords)

print("Map being taught:\n")
print(the_map)
print()

TEACHER_SYSTEM = (
    "You are showing a student a map of some rooms. Present the map clearly "
    "and say nothing false. Do not narrate what you are doing."
)
STUDENT_SYSTEM = (
    "You are being shown a map of some rooms. Confirm what you can see in it. "
    "Do not add anything that is not on the map. Do not narrate."
)

fact = (
    f"Here is a map of the {len(room_names)} rooms on a 10x10 grid. "
    f"North is up, east is right. Each room is marked in its cell.\n\n"
    f"{the_map}"
)

teacher_history = [SystemMessage(content=TEACHER_SYSTEM),
                   HumanMessage(content=f"Show the student this map:\n\n{fact}")]
teacher_said = teacher.invoke(teacher_history).content.strip()

student_history = [SystemMessage(content=STUDENT_SYSTEM),
                   HumanMessage(content=teacher_said)]
student_said = student.invoke(student_history).content.strip()

print(f"Teacher: {teacher_said[:400]}\n")
print(f"Student: {student_said[:400]}\n")

transcript = [{
    "turn": 1,
    "ground_truth_map": the_map,
    "teacher_said": teacher_said,
    "student_confirmed": student_said,
}]

full_context = f"Teacher: {teacher_said}\nStudent: {student_said}"

with open("teaching_map.json", "w") as f:
    json.dump({
        "transcript": transcript,
        "full_context": full_context,
        "the_map": the_map,
        "true_coords": {k: list(v) for k, v in true_coords.items()},
    }, f, indent=2)

print("Saved teaching_map.json")
