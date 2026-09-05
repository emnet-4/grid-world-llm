"""
RL_Exp5_Time3 -- one-shot transfer learning.

The three earlier conditions keep the layout in the prompt, in a file, or in a
separate probe's weights. This one puts it in the language model's own weights
and then asks the model itself, so the answering channel is identical to
Exp1, Exp2 and Exp4.

  Exp1  prompt    dies when the context overflows
  Exp2  file      handed back, read as text
  Exp3  probe     a regression answers, not the model
  Exp5  weights   the model answers, with no context at all   <- this file

WHAT IT TRAINS ON

Ground truth, not the Student's saved coordinates. That is a deliberate
difference from Exp3 and it changes what the number means. Exp3 asks what
happens when the Student's own extraction is stored in weights, errors and all.
Exp5 asks a different question: if the layout had been absorbed perfectly and
written into the weights, how well would the model answer? It is a ceiling
rather than a like-for-like comparison with Exp2, and should be reported as one.

"ONE-SHOT"

One pass over the ten teaching facts, no repetition. EPOCHS is exposed because
a single pass over ten short examples often moves a LoRA adapter very little,
and it is worth knowing whether the condition fails because nothing was learned
or because nothing was retained.

HARDWARE

This is the only part of the pipeline that does not run comfortably on a
laptop. Ollama cannot fine-tune, so the model is loaded through HuggingFace
instead, and a LoRA adapter is trained on the attention projections.

  Llama 3.2 3B    workable on an M-series Mac, slow
  Mistral 7B      needs a GPU in practice

Set MODEL_ID to a checkpoint you can actually load. Both official repositories
are gated on HuggingFace and need `huggingface-cli login`; ungated mirrors
exist if that is a problem.

Requires: transformers, peft, accelerate, torch
"""

from __future__ import annotations

import json
import os
import itertools

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# one generation call per question, 43 questions, two probes: without this the
# per-call notices bury the actual output
transformers.logging.set_verbosity_error()

from probe_questions import (load_world, build_direction_questions,
                             build_position_questions, manhattan,
                             parse_triplet, parse_direction, parse_position,
                             parse_map, triplet_prompt, direction_prompt,
                             position_prompt, map_prompt)

# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------

MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")

EPOCHS = int(os.environ.get("LORA_EPOCHS", 1))   # 1 = one shot
LEARNING_RATE = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

MAX_NEW_TOKENS = 24
OUTPUT_FILE = "RL_Exp5_Time3.json"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


# ---------------------------------------------------------------------
# training data: the same facts the Teacher states, from ground truth
# ---------------------------------------------------------------------

def describe_direction(a: dict, b: dict) -> str:
    dx, dy = a["x"] - b["x"], a["y"] - b["y"]
    parts = []
    if dx:
        parts.append(f"{abs(dx)} units {'east' if dx > 0 else 'west'}")
    if dy:
        parts.append(f"{abs(dy)} units {'north' if dy > 0 else 'south'}")
    return " and ".join(parts) if parts else "at the same position as"


def build_training_text(world: dict) -> list[str]:
    """One string per taught fact, matching the wording the Teacher uses."""
    rooms = world["rooms"]
    names = world["room_names"]
    lines = []
    for p, q in itertools.combinations(names, 2):
        rp, rq = rooms[p], rooms[q]
        lines.append(
            f"{p}, which has {rp['color']} colored walls, "
            f"{rp['num_windows']} windows, and {rp['num_doors']} doors, "
            f"is {describe_direction(rp, rq)} of {q}, which has "
            f"{rq['color']} colored walls, {rq['num_windows']} windows, "
            f"and {rq['num_doors']} doors."
        )
    return lines


# ---------------------------------------------------------------------
# training
# ---------------------------------------------------------------------

def train_adapter(model, tokenizer, texts: list[str]):
    """
    One pass over the facts by default. Deliberately minimal: no batching, no
    scheduler, no evaluation split. With ten short examples anything more
    elaborate would be overfitting machinery to a dataset that cannot support
    it.
    """
    model.train()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    # If the model has a chat template, train on text in the same shape it will
    # be asked in. Training on bare sentences and then querying through a chat
    # template puts the adapter and the inference path in different formats.
    if getattr(tokenizer, "chat_template", None):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Tell me about the room layout."},
                 {"role": "assistant", "content": fact}],
                tokenize=False,
            )
            for fact in texts
        ]

    losses = []
    for epoch in range(EPOCHS):
        for i, text in enumerate(texts):
            batch = tokenizer(text, return_tensors="pt",
                              truncation=True, max_length=128).to(DEVICE)
            batch["labels"] = batch["input_ids"].clone()

            out = model(**batch)
            out.loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            # detach before converting, otherwise torch warns about pulling a
            # scalar out of a tensor that still carries gradients
            loss_value = out.loss.detach().item()
            losses.append(loss_value)
            print(f"  epoch {epoch + 1}, fact {i + 1}/{len(texts)}, "
                  f"loss {loss_value:.4f}")

    model.eval()
    return losses


# ---------------------------------------------------------------------
# asking the model, with no context at all
# ---------------------------------------------------------------------

def ask(model, tokenizer, prompt: str) -> str:
    """
    Put the question to the model as a chat turn, not as raw text.

    Applying the chat template matters here. Without it the model continues the
    prompt rather than responding to it, and since our prompts end with
    instructions ("Answer with exactly one word. No explanation.") it continues
    by generating more instructions. The first run of this script produced
    "No elaboration. No exceptions. Just one word." as an answer to every
    triplet question, which is the model finishing our sentence rather than
    refusing or guessing.
    """
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            # the model's config carries a very large max_length; setting it to
            # None here stops transformers warning on every single call that
            # max_new_tokens takes precedence, which it does and which is what
            # we want
            max_length=None,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # decode only what was generated, so the prompt is never mistaken for a reply
    return tokenizer.decode(output[0][prompt_length:],
                            skip_special_tokens=True).strip()


def run_probe(model, tokenizer, world: dict) -> dict:
    """
    The same 43 questions the other conditions use, asked with an empty
    context. Whatever the model knows has to come from its weights.
    """
    rooms = world["rooms"]
    names = world["room_names"]

    triplet_results = []
    print(f"\n  Triplet questions ({len(world['all_triplets'])})")
    for t in world["all_triplets"]:
        raw = ask(model, tokenizer,
                  triplet_prompt(t["anchor"], t["opt1"], t["opt2"]))
        parsed = parse_triplet(raw, t["opt1"], t["opt2"])
        triplet_results.append({
            "anchor": t["anchor"], "opt1": t["opt1"], "opt2": t["opt2"],
            "correct": t["correct"], "raw_response": raw, "parsed": parsed,
            "is_correct": parsed == t["correct"],
        })

    direction_qs = build_direction_questions(rooms, names)
    direction_results = []
    print(f"  Direction questions ({len(direction_qs)})")
    for d in direction_qs:
        raw = ask(model, tokenizer, direction_prompt(d["p"], d["q"]))
        parsed = parse_direction(raw)
        direction_results.append({
            "p": d["p"], "q": d["q"], "correct_parts": d["correct_parts"],
            "raw_response": raw, "parsed": parsed,
            "is_correct": parsed is not None
                          and set(parsed) == set(d["correct_parts"]),
        })

    position_results = []
    print(f"  Position questions ({len(names)})")
    for pq in build_position_questions(names):
        room = rooms[pq["room"]]
        raw = ask(model, tokenizer, position_prompt(pq["room"]))
        parsed = parse_position(raw)
        error = manhattan(parsed[0], parsed[1], room["x"], room["y"]) if parsed else None
        position_results.append({
            "room": pq["room"], "true_x": room["x"], "true_y": room["y"],
            "raw_response": raw, "parsed": list(parsed) if parsed else None,
            "error": error,
        })

    print("  Whole-layout question")
    raw_map = ask(model, tokenizer, map_prompt(names))
    parsed_map = parse_map(raw_map, names)
    map_errors = {}
    if parsed_map:
        for name, (x, y) in parsed_map.items():
            map_errors[name] = manhattan(x, y, rooms[name]["x"], rooms[name]["y"])

    # how many different cells were named across the five questions. The world
    # always uses five, so fewer means the model repeated itself, and one means
    # it is emitting a constant regardless of which room was asked about
    given = [tuple(r["parsed"]) for r in position_results if r["parsed"]]
    n_distinct = len(set(given))

    answered_triplets = [r for r in triplet_results if r["parsed"] is not None]
    answered_directions = [r for r in direction_results if r["parsed"] is not None]
    valid_errors = [r["error"] for r in position_results if r["error"] is not None]

    return {
        "triplet_answers": triplet_results,
        "direction_answers": direction_results,
        "position_answers": position_results,
        "map_answer": {
            "raw_response": raw_map,
            "parsed_coords": {k: list(v) for k, v in (parsed_map or {}).items()},
            "n_rooms_placed": len(parsed_map or {}),
            "n_rooms_total": len(names),
            "per_room_error": map_errors,
            "mean_error": (sum(map_errors.values()) / len(map_errors))
                          if map_errors else None,
        },

        "triplet_accuracy": (sum(1 for r in answered_triplets if r["is_correct"])
                             / len(answered_triplets)) if answered_triplets else None,
        "direction_accuracy": (sum(1 for r in answered_directions if r["is_correct"])
                               / len(answered_directions)) if answered_directions else None,
        "mean_position_error": (sum(valid_errors) / len(valid_errors))
                               if valid_errors else None,
        "map_mean_error": (sum(map_errors.values()) / len(map_errors))
                          if map_errors else None,
        "map_rooms_placed": len(parsed_map or {}),

        "n_position_answered": len(valid_errors),
        "n_position_total": len(position_results),
        "n_distinct_positions": n_distinct,
        "n_triplet_answered": len(answered_triplets),
        "n_triplet_total": len(triplet_results),
        "n_direction_answered": len(answered_directions),
        "n_direction_total": len(direction_results),
    }


# ---------------------------------------------------------------------

def main() -> None:
    world = load_world()
    facts = build_training_text(world)

    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_ID}")
    print(f"Training on {len(facts)} facts, {EPOCHS} epoch(s)\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
    ).to(DEVICE)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Probe before training as a control. Without it, a fine-tuned model that
    # answers the same thing for every room is ambiguous: the adapter may have
    # collapsed, or the base model may behave that way regardless. Comparing
    # the two separates "training broke it" from "training did nothing".
    print("\nProbing BEFORE training (control):")
    before = run_probe(model, tokenizer, world)

    print("\nTraining:")
    losses = train_adapter(model, tokenizer, facts)

    print("\nProbing AFTER training, with no context:")
    results = run_probe(model, tokenizer, world)

    results["before_training"] = {
        "triplet_accuracy": before["triplet_accuracy"],
        "direction_accuracy": before["direction_accuracy"],
        "mean_position_error": before["mean_position_error"],
        "n_distinct_positions": before.get("n_distinct_positions"),
        "position_answers": before["position_answers"],
    }

    results["training"] = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "trained_on": "ground truth facts",
        "n_facts": len(facts),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": TARGET_MODULES,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
    }

    with open(OUTPUT_FILE, "w") as fh:
        json.dump(results, fh, indent=2)

    def show(label, value, fmt="{:.3f}"):
        print(f"  {label:<22}" + (fmt.format(value) if value is not None else "n/a"))

    print("\nRL_Exp5_Time3 results")
    print(f"  {'':<22}{'before':<12}{'after'}")

    def pair(label, key, fmt="{:.3f}"):
        b, a = before.get(key), results.get(key)
        fb = fmt.format(b) if b is not None else "n/a"
        fa = fmt.format(a) if a is not None else "n/a"
        print(f"  {label:<22}{fb:<12}{fa}")

    pair("triplet accuracy", "triplet_accuracy")
    pair("direction accuracy", "direction_accuracy")
    pair("position error", "mean_position_error", "{:.2f}")
    pair("distinct positions", "n_distinct_positions", "{:.0f}")
    print(f"  {'loss':<22}{losses[0]:.4f} -> {losses[-1]:.4f}")

    if results.get("n_distinct_positions") == 1:
        print("\n  Note: every position answer was identical after training. "
              "Compare\n  against the before column: if that was also 1, the "
              "base model does this\n  regardless and the adapter is not the "
              "cause.")
    print(f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
