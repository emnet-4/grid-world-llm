"""
RL_Exp4_Time2 -- probe with the map still in context.

Direct counterpart to RL_Exp1_Time2, which has the ten-fact conversation in
context instead. Same questions, same scoring.
"""

import json
import os
from langchain_ollama import ChatOllama
from probe_questions import load_world, run_all_probes

STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "mistral")
NUM_CTX = 4096

world = load_world()
rooms = world["rooms"]

with open("teaching_map.json", "r") as f:
    teaching = json.load(f)

student = ChatOllama(model=STUDENT_MODEL, num_ctx=NUM_CTX)

room_descriptions = "\n".join(
    f"- {r['id']}: {r['color']} colored walls, {r['num_windows']} windows, {r['num_doors']} doors"
    for r in rooms.values()
)

CONTEXT = (
    f"You know about the following rooms:\n{room_descriptions}\n\n"
    f"You were just shown this map of their layout:\n"
    f"{teaching['full_context']}\n\n"
    f"Using the map, answer this question directly.\n\n"
)

print("RL_Exp4_Time2 probe (map in context):")
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

with open("RL_Exp4_Time2.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved RL_Exp4_Time2.json")
