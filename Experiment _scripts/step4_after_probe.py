"""
Step 4: AFTER probe — same three question types, now WITH the teaching
conversation as context.
"""

import json
import os
from langchain_ollama import ChatOllama
from probe_questions import load_world, run_all_probes

STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "mistral")

world = load_world()
rooms = world["rooms"]

with open("teaching_transcript.json", "r") as f:
    teaching = json.load(f)

student = ChatOllama(model=STUDENT_MODEL, num_ctx=4096)

room_descriptions = "\n".join(
    f"- {r['id']}: {r['color']} colored walls, {r['num_windows']} windows, {r['num_doors']} doors"
    for r in rooms.values()
)

CONTEXT = (
    f"You know about the following rooms:\n{room_descriptions}\n\n"
    f"You just learned the following from a teacher about their layout:\n"
    f"{teaching['full_context']}\n\n"
    f"Using what you learned, answer this question directly.\n\n"
)

print("AFTER probe:")
results = run_all_probes(student, world, context=CONTEXT)

def _pct(v):
    return f"{v:.1%}" if v is not None else "n/a (all unanswered)"

print(f"\n  Triplet accuracy:    {_pct(results['triplet_accuracy'])}  "
      f"({results['n_triplet_answered']}/{results['n_triplet_total']} answered)")
print(f"  Direction accuracy:  {_pct(results['direction_accuracy'])}  "
      f"({results['n_direction_answered']}/{results['n_direction_total']} answered)")
print(f"  Mean position error: {results['mean_position_error']}  "
      f"({results['n_position_answered']}/{results['n_position_total']} answered)")
print(f"  Map error:           {results['map_mean_error']}  "
      f"({results['map_rooms_placed']}/{len(world['room_names'])} rooms placed)")
print(f"  Retries -- mean attempts: {results['mean_attempts']:.2f}, "
      f"max: {results['max_attempts']}, "
      f"{results['pct_needed_retry']:.0%} needed more than one")

with open("RL_Exp1_Time2.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved RL_Exp1_Time2.json")
