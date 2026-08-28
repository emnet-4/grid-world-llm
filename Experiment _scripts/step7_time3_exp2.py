"""
Step 7 (Exp 2): TIME 3 probe with the Student's saved representation.

Prompt structure:
    [teaching transcript]      <- truncated out by the context limit
    [long filler conversation]
    [Student's own JSON]       <- short, so it survives at the end
    [test question]

Identical to Exp 1 except the Student's self-authored JSON is appended after
the filler. Tests whether a compressed, self-authored representation is enough
to reason from once the original teaching is gone.
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

with open("filler_conversation.json", "r") as f:
    filler = json.load(f)

with open("student_saved_representation.json", "r") as f:
    saved = json.load(f)

student = ChatOllama(model=STUDENT_MODEL, num_ctx=4096)

room_descriptions = "\n".join(
    f"- {r['id']}: {r['color']} colored walls, {r['num_windows']} windows, {r['num_doors']} doors"
    for r in rooms.values()
)

CONTEXT = (
    f"You know about the following rooms:\n{room_descriptions}\n\n"
    f"Earlier you learned the following from a teacher:\n"
    f"{teaching['full_context']}\n\n"
    f"Since then you have had this conversation:\n"
    f"{filler['full_text']}\n\n"
    f"Here are the notes you saved earlier about the room coordinates:\n"
    f"{saved['json_text']}\n\n"
    f"Using your saved notes, answer this question directly.\n\n"
)

approx_tokens = len(CONTEXT) // 4
print(f"Time 3 context is ~{approx_tokens} tokens "
      f"({len(CONTEXT)} chars) before the question is appended.")
print(f"Saved notes: {saved['json_text']}\n")

print("EXP 2 — TIME 3 probe (teaching flushed, saved JSON provided):")
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
results["saved_representation_mean_error"] = saved["mean_error"]

with open("RL_Exp2_Time3.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved RL_Exp2_Time3.json")
