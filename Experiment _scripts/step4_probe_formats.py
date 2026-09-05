"""
Probe the Student after teaching in one of the three formats.

Runs both timepoints for whichever format was taught:

  Time2   the teaching transcript is in the prompt
  Time3   the transcript is followed by filler and falls out of the context

Selected the same way as the teaching:

    TEACHING_FORMAT=trail python3 step4_probe_formats.py

Reads teaching_<format>.json and writes RL_<Format>_Time2.json and
RL_<Format>_Time3.json, so the three formats can sit side by side in one run
folder and be compared directly.
"""

from __future__ import annotations

import json
import os

from langchain_ollama import ChatOllama

from probe_questions import load_world, run_all_probes

STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "mistral")
NUM_CTX = 4096

FORMAT = os.environ.get("TEACHING_FORMAT", "trail").lower()
TEACHING_FILE = f"teaching_{FORMAT}.json"

LABELS = {
    "coordinates": "Coord",
    "trail": "Trail",
    "both": "Both",
}
LABEL = LABELS.get(FORMAT, FORMAT.capitalize())


def room_descriptions(world: dict) -> str:
    return "\n".join(
        f"- {r['id']}: {r['color']} colored walls, {r['num_windows']} windows, "
        f"{r['num_doors']} doors"
        for r in world["rooms"].values()
    )


def probe(student, world: dict, context: str, label: str, filename: str) -> dict:
    print(f"\n{label}")
    results = run_all_probes(student, world, context=context)

    def pct(value):
        return f"{value:.1%}" if value is not None else "n/a"

    print(f"\n  triplet   {pct(results['triplet_accuracy'])}"
          f"   ({results['n_triplet_answered']}/{results['n_triplet_total']} answered)")
    print(f"  direction {pct(results['direction_accuracy'])}"
          f"   ({results['n_direction_answered']}/{results['n_direction_total']} answered)")
    print(f"  position  {results['mean_position_error']}"
          f"   ({results['n_position_answered']}/{results['n_position_total']} answered)")
    print(f"  map       {results['map_mean_error']}"
          f"   ({results['map_rooms_placed']}/{len(world['room_names'])} rooms placed)")
    print(f"  retries   mean {results['mean_attempts']:.2f}, "
          f"max {results['max_attempts']}, "
          f"{results['pct_needed_retry']:.0%} needed more than one")

    results["teaching_format"] = FORMAT
    with open(filename, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  saved {filename}")
    return results


def main() -> None:
    world = load_world()

    if not os.path.exists(TEACHING_FILE):
        raise SystemExit(
            f"{TEACHING_FILE} not found. Run step3_teaching_formats.py with "
            f"TEACHING_FORMAT={FORMAT} first."
        )

    with open(TEACHING_FILE) as fh:
        teaching = json.load(fh)

    student = ChatOllama(model=STUDENT_MODEL, num_ctx=NUM_CTX)
    rooms_text = room_descriptions(world)

    # --- Time 2: the teaching is readable ---
    context_time2 = (
        f"You know about the following rooms:\n{rooms_text}\n\n"
        f"You just learned the following from a teacher about their layout:\n"
        f"{teaching['full_context']}\n\n"
        f"Using what you learned, answer this question directly.\n\n"
    )
    probe(student, world, context_time2,
          f"{LABEL} Time2, teaching in context",
          f"RL_{LABEL}_Time2.json")

    # --- Time 3: the teaching is pushed out ---
    if not os.path.exists("filler_conversation.json"):
        print("\nfiller_conversation.json not found, skipping Time3.")
        return

    with open("filler_conversation.json") as fh:
        filler = json.load(fh)

    # teaching first, so it is what gets truncated when the prompt overflows
    context_time3 = (
        f"You know about the following rooms:\n{rooms_text}\n\n"
        f"Earlier you learned the following from a teacher:\n"
        f"{teaching['full_context']}\n\n"
        f"Since then you have had this conversation:\n"
        f"{filler['full_text']}\n\n"
        f"Now answer this question directly.\n\n"
    )
    print(f"\n  Time3 context is roughly {len(context_time3) // 4} tokens "
          f"before the question is appended.")
    probe(student, world, context_time3,
          f"{LABEL} Time3, teaching flushed out",
          f"RL_{LABEL}_Time3.json")


if __name__ == "__main__":
    main()
