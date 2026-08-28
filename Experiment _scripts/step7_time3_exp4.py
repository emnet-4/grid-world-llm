"""
RL_Exp4_Time3 -- the map is in the prompt but flushed out by filler.

Direct counterpart to RL_Exp1_Time3. The map goes first so it is what gets cut
when the prompt overflows the context window, and nothing is given in its place.

Tests whether a layout delivered as a picture survives any better than one
delivered as ten sentences. Expectation is no -- neither lives in the weights --
but if Exp 4 holds up better it would suggest the format affects how well the
information is encoded, not just how easily it is read.
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

with open("filler_conversation.json", "r") as f:
    filler = json.load(f)

student = ChatOllama(model=STUDENT_MODEL, num_ctx=NUM_CTX)

room_descriptions = "\n".join(
    f"- {r['id']}: {r['color']} colored walls, {r['num_windows']} windows, {r['num_doors']} doors"
    for r in rooms.values()
)

# Map first so it is what gets truncated when the prompt overflows.
CONTEXT = (
    f"You know about the following rooms:\n{room_descriptions}\n\n"
    f"Earlier you were shown this map:\n"
    f"{teaching['full_context']}\n\n"
    f"Since then you have had this conversation:\n"
    f"{filler['full_text']}\n\n"
    f"Now answer this question directly.\n\n"
)

approx_tokens = len(CONTEXT) // 4
print(f"Context is ~{approx_tokens} tokens before the question is appended.\n")

print("RL_Exp4_Time3 probe (map flushed out):")
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

results["context_approx_tokens"] = approx_tokens

with open("RL_Exp4_Time3.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved RL_Exp4_Time3.json")
