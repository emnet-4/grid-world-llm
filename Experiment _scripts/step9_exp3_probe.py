"""
Step 9 (Exp 3): RL_Exp3_Time3 -- knowledge stored in weights.

DESIGN NOTES (judgment calls made in the absence of a spec):

Following Ilya's paper, this is a linear probe on a FROZEN model. The LLM's
own weights are never modified -- embeddings are extracted from the penultimate
layer and a small head is trained on top. The paper notes linear probing
performs on par with fine-tuning and is a *better* measure of representation
quality, since fine-tuning distorts the representations you're trying to
measure.

What goes in:  the Student model's hidden-state embedding for each room.
What comes out: that room's (x, y) coordinates.
Training data:  the Student's own saved representation from step6b -- the same
                JSON that Exp2 reads back from a file.

Why train on the saved JSON rather than ground truth: it makes Exp3 an exact
parallel to Exp2. Same information, two substrates.

    Exp2 -> representation stored in a FILE, read back at Time3
    Exp3 -> representation stored in WEIGHTS, read out at Time3

Training on ground truth would give the probe information the Student never
had, which isn't a fair comparison.

TWO LIMITATIONS TO REPORT:

1. The readout differs from Exp1/Exp2. Those end with "ask the LLM the
   question." Here the probe produces coordinates and the questions are
   answered from those. So Exp3 measures whether the representation survives
   in weights, but not via the same channel. The alternative -- LoRA
   fine-tuning so the LLM itself answers -- keeps the channel identical but
   is much heavier and needs a GPU.

2. At 5 rooms the probe cannot generalize, only memorize. All C(5,2)=10 pairs
   are taught, so there is no held-out structure to test. Scaling to more
   rooms with partial teaching would make this a real generalization test.

There is no "flush" here. That's the point -- weights don't live in the
context window, so there is nothing to flush.
"""

import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import Ridge

from probe_questions import (load_world, build_direction_questions,
                              build_position_questions, manhattan)

# Smaller than Mistral-7B and runs comfortably on a laptop. Swap if you have
# the memory -- "mistralai/Mistral-7B-v0.1" needs ~14GB in fp16.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RIDGE_ALPHA = 1.0

world = load_world()
rooms = world["rooms"]
room_names = world["room_names"]
triplets = world["all_triplets"]

with open("student_saved_representation.json", "r") as f:
    saved = json.load(f)

training_coords = saved["parsed_representation"]

if len(training_coords) < len(room_names):
    raise SystemExit(
        f"Only {len(training_coords)}/{len(room_names)} rooms in the saved "
        f"representation. Re-run step6b before this."
    )


# ---------------------------------------------------------------------
# 1. Extract embeddings from the frozen model
# ---------------------------------------------------------------------

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def room_text(room):
    return (
        f"Room {room['id']}: {room['color']} colored walls, "
        f"{room['num_windows']} windows, {room['num_doors']} doors."
    )


def embed(text):
    """Mean-pooled last hidden state. Model stays frozen throughout."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs)
    hidden = out.last_hidden_state          # (1, seq_len, dim)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.squeeze(0).numpy()


print("Extracting room embeddings...")
embeddings = {name: embed(room_text(rooms[name])) for name in room_names}
dim = len(next(iter(embeddings.values())))
print(f"  {len(embeddings)} rooms, {dim}-dim embeddings\n")


# ---------------------------------------------------------------------
# 2. Train the probe:  embedding -> (x, y)
# ---------------------------------------------------------------------

X = np.array([embeddings[name] for name in room_names])
y = np.array([training_coords[name] for name in room_names], dtype=float)

print("Training probe on the Student's saved representation...")
print("  Training targets (what the Student believed):")
for name in room_names:
    print(f"    {name}: {tuple(training_coords[name])}")

probe = Ridge(alpha=RIDGE_ALPHA)
probe.fit(X, y)

predicted = {name: probe.predict(embeddings[name].reshape(1, -1))[0] for name in room_names}

print("\n  Probe predictions vs training targets vs ground truth:")
print(f"  {'Room':<6} {'predicted':<16} {'trained on':<14} {'true':<10} {'err vs true':<12}")
recon_errors = {}
for name in room_names:
    px, py = predicted[name]
    tx, ty = training_coords[name]
    gx, gy = rooms[name]["x"], rooms[name]["y"]
    err = manhattan(px, py, gx, gy)
    recon_errors[name] = err
    print(f"  {name:<6} ({px:5.2f}, {py:5.2f})   ({tx}, {ty})        ({gx}, {gy})    {err:.2f}")

mean_recon_error = sum(recon_errors.values()) / len(recon_errors)
print(f"\n  Mean position error: {mean_recon_error:.2f}")


# ---------------------------------------------------------------------
# 3. Answer the questions from the probe's predictions
# ---------------------------------------------------------------------

def dist(a, b):
    pa, pb = predicted[a], predicted[b]
    return manhattan(pa[0], pa[1], pb[0], pb[1])


# --- triplets ---
triplet_results = []
for t in triplets:
    d1 = dist(t["anchor"], t["opt1"])
    d2 = dist(t["anchor"], t["opt2"])
    choice = t["opt1"] if d1 < d2 else t["opt2"]
    triplet_results.append({
        "anchor": t["anchor"], "opt1": t["opt1"], "opt2": t["opt2"],
        "correct": t["correct"], "raw_response": f"{choice} (probe)",
        "parsed": choice, "is_correct": choice == t["correct"],
        "refused": False,
    })

# --- directions ---
direction_qs = build_direction_questions(rooms, room_names)
direction_results = []
for d in direction_qs:
    pp, pq = predicted[d["p"]], predicted[d["q"]]
    dx, dy = pp[0] - pq[0], pp[1] - pq[1]
    parts = []
    if abs(dx) > 1e-6:
        parts.append("east" if dx > 0 else "west")
    if abs(dy) > 1e-6:
        parts.append("north" if dy > 0 else "south")
    direction_results.append({
        "p": d["p"], "q": d["q"], "correct_parts": d["correct_parts"],
        "raw_response": " and ".join(parts) + " (probe)",
        "parsed": sorted(parts),
        "is_correct": set(parts) == set(d["correct_parts"]),
        "refused": False,
    })

# --- positions ---
position_results = []
for pq in build_position_questions(room_names):
    name = pq["room"]
    px, py = predicted[name]
    gx, gy = rooms[name]["x"], rooms[name]["y"]
    err = manhattan(px, py, gx, gy)
    position_results.append({
        "room": name, "true_x": gx, "true_y": gy,
        "raw_response": f"{px:.2f},{py:.2f} (probe)",
        "parsed": [float(px), float(py)], "error": err,
        "refused": False,
    })


# ---------------------------------------------------------------------
# 4. Score, matching the other conditions
# ---------------------------------------------------------------------

triplet_acc = sum(1 for r in triplet_results if r["is_correct"]) / len(triplet_results)
direction_acc = sum(1 for r in direction_results if r["is_correct"]) / len(direction_results)
mean_pos_error = sum(r["error"] for r in position_results) / len(position_results)

results = {
    "triplet_answers": triplet_results,
    "direction_answers": direction_results,
    "position_answers": position_results,
    "map_answer": None,
    "map_mean_error": None,
    "map_rooms_placed": None,

    "triplet_accuracy": triplet_acc,
    "direction_accuracy": direction_acc,
    "triplet_accuracy_answered": triplet_acc,
    "direction_accuracy_answered": direction_acc,

    "mean_position_error": mean_pos_error,
    "n_position_answered": len(position_results),
    "n_position_total": len(position_results),

    # Zero by construction -- a probe always outputs a number. This is a real
    # asymmetry with Exp1/Exp2, where refusal was the informative signal.
    "triplet_refusal_rate": 0.0,
    "direction_refusal_rate": 0.0,
    "position_refusal_rate": 0.0,

    "probe_details": {
        "embedding_model": MODEL_NAME,
        "embedding_dim": int(dim),
        "ridge_alpha": RIDGE_ALPHA,
        "trained_on": "student_saved_representation.json",
        "training_targets": {k: list(v) for k, v in training_coords.items()},
        "predictions": {k: [float(v[0]), float(v[1])] for k, v in predicted.items()},
        "mean_error_vs_ground_truth": mean_pos_error,
        "saved_representation_mean_error": saved["mean_error"],
    },
}

print("\n" + "=" * 60)
print("RL_Exp3_Time3 RESULTS")
print("=" * 60)
print(f"  Triplet accuracy:    {triplet_acc:.1%}")
print(f"  Direction accuracy:  {direction_acc:.1%}")
print(f"  Position error:      {mean_pos_error:.2f}")
print(f"  Retries:             n/a (probe always outputs a value)")
print(f"\n  For comparison, the JSON it was trained on had error "
      f"{saved['mean_error']:.2f}")

def _to_native(o):
    """numpy scalars/arrays aren't JSON serializable -- convert on the way out."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o)} is not JSON serializable")


with open("RL_Exp3_Time3.json", "w") as f:
    json.dump(results, f, indent=2, default=_to_native)

print("\nSaved RL_Exp3_Time3.json")
