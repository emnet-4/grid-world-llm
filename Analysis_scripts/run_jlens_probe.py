"""
J-Lens starter script for the spatial-teaching paper -- overnight-run version.

WHAT THIS DOES
---------------
For a handful of saved Student conversations, this pulls the residual-stream
activation at the token position right before the Student emits its answer
to a position or direction question, and reads out what that activation is
"disposed to say" at several layers using a fitted Jacobian lens. The idea:
even if the Student's *stated* answer is wrong or format-locked, its
*internal* state at that position might already lean toward coordinate-like
content (digits, brackets) or direction-like content (north/south/east/west,
left/right) -- independent of what actually gets verbalized. Comparing that
lean across teaching-format conditions is the "bias in the mind" probe.

BEFORE RUNNING THIS
---------------------
1. git clone https://github.com/anthropics/jacobian-lens && pip install -e jacobian-lens
2. Run prepare_fit_prompts.py first:
     python prepare_fit_prompts.py --model-name unsloth/Llama-3.2-3B-Instruct --n 150 --out fit_prompts.json
3. Set RUNS_ROOT inside load_transcripts() to your actual runs directory
   (the one TODO left in this file -- everything else is filled in).
4. Run this script with `nohup python run_jlens_probe.py > run.log 2>&1 &` or
   similar so it survives a disconnect overnight.

CONFIRMED SETUP (resolved from conversation -- see git history / chat log
if you need the reasoning again)
------------------------------------------------------------------------
- Model: unsloth/Llama-3.2-3B-Instruct, an UNGATED mirror -- no
  huggingface-cli login needed.
- Scope: Llama-as-Student runs only (runs_B_mistral_teaches_run*), since
  Mistral only exists as an Ollama GGUF build with no HF checkpoint. This
  is the same constraint noted in the paper's Limitations for the
  fine-tuning condition -- not a new problem, the same one.
- Caveat that belongs in the paper if this result gets used: this probes
  the HF mirror, not the literal Ollama-served GGUF that generated the
  actual transcripts. Same model family and instruction-tuning, different
  quantization -- closest available match, not bit-identical.

OVERNIGHT-RUN SAFETY
----------------------
- Results are written to disk after EVERY transcript, not just at the end,
  so a crash partway through does not lose completed work.
- Each transcript is wrapped in try/except; one bad transcript is logged and
  skipped rather than killing the run.
- Re-running the script skips transcripts already present in the output
  file (matched by run_id), so it's safe to resume after an interruption.

CRITICAL COMPATIBILITY NOTE (background -- already resolved above)
----------------------------------------------------------------------
J-Lens loads models via transformers.AutoModelForCausalLM and needs actual
weight/gradient access, which is why this only works on the Llama HF
mirror and not on the Ollama-served models directly -- see "Confirmed
setup" above.
"""

import json
import os
import time
import traceback

import torch
import transformers
import jlens

# ---------------------------------------------------------------------------
# 1. CONFIG -- fill these in
# ---------------------------------------------------------------------------

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"  # confirmed real checkpoint from step10_exp5_oneshot.py
FIT_PROMPTS_PATH = "fit_prompts.json"      # from prepare_fit_prompts.py
LENS_CACHE_PATH = "out/jacobian_lens.pt"   # reused across runs if present
RESULTS_PATH = "jlens_results.jsonl"       # appended to incrementally

COORD_MARKERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")", ",", "coord"]
DIRECTION_MARKERS = ["north", "south", "east", "west", "left", "right",
                      "N", "S", "E", "W", "of"]

LAYERS_TO_PROBE = None  # None = probe all layers; or e.g. list(range(0, 32, 4))
TOP_K = 10


# ---------------------------------------------------------------------------
# 2. LOAD MODEL + LENS (fit once, cache to disk)
# ---------------------------------------------------------------------------

def get_device_and_dtype():
    """
    Auto-detects the best available device. .cuda() only works with NVIDIA
    GPUs -- on a Mac (no NVIDIA GPU), this picks Apple Silicon's MPS backend
    if available, otherwise falls back to CPU. CPU works fine for this,
    just slower.
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    print("No GPU detected (no CUDA, no MPS) -- running on CPU. This will "
          "be noticeably slower but will still work.")
    return "cpu", torch.float32


def load_model_and_lens():
    device, dtype = get_device_and_dtype()
    print(f"Using device: {device}")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=dtype
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)

    if os.path.exists(LENS_CACHE_PATH):
        print(f"Loading cached lens from {LENS_CACHE_PATH}")
        lens = jlens.JacobianLens.from_pretrained(LENS_CACHE_PATH)
    else:
        print(f"No cached lens found, fitting a new one from {FIT_PROMPTS_PATH}")
        with open(FIT_PROMPTS_PATH) as f:
            fit_prompts = json.load(f)["prompts"]
        os.makedirs(os.path.dirname(LENS_CACHE_PATH) or ".", exist_ok=True)
        lens = jlens.fit(model, prompts=fit_prompts, checkpoint_path=LENS_CACHE_PATH + ".ckpt")
        lens.save(LENS_CACHE_PATH)
        print(f"Fit complete, saved to {LENS_CACHE_PATH}")

    return model, tokenizer, lens


# ---------------------------------------------------------------------------
# 3. LOAD A HANDFUL OF SAVED CONVERSATIONS
# ---------------------------------------------------------------------------

def load_transcripts():
    """
    Real implementation, built from probe_questions.py's own prompt
    functions rather than reimplementing the wording by hand -- this stays
    correct even if the wording ever changes, and each run folder carries
    its own copy of probe_questions.py, so we import from that exact copy
    rather than one shared version.

    Each probe question turned out to be a SINGLE STATELESS CALL: a text
    block (room list + teaching['full_context']) prepended fresh to each
    question, not an accumulating multi-turn conversation. That means
    reconstructing the exact prompt is: rebuild the context block, call the
    matching *_prompt() function from that run's probe_questions.py at
    level=0 (first-attempt wording), and that IS the full prompt text.

    We only use questions where attempts == 1, so we know for certain which
    wording (retries escalate the "force" instructions) produced the
    recorded answer -- no ambiguity about which level was actually used.

    We only use questions the Student got WRONG, per the actual point of
    this probe: does the internal state lean toward the other format's
    content even while the stated answer fails?

    IMPORTANT: restricted to runs_B_mistral_teaches_run* only. Those are the
    runs where Llama was Student -- the only model we have a real HF
    checkpoint for (see conversation history: Mistral only exists as an
    Ollama GGUF build, incompatible with J-Lens). Also note: this uses the
    HF mirror unsloth/Llama-3.2-3B-Instruct, not the literal Ollama-served
    GGUF that generated these transcripts -- closest available match, not
    bit-identical.

    Handles the known data-quality issue: teaching_coordinates.json
    sometimes has the Teacher fail to state a room's actual (x, y)
    position. Broken runs are detected and skipped automatically.
    """
    import glob
    import importlib.util

    RUNS_ROOT = "TODO: path to the directory containing runs_A_llama_teaches_run*/ etc."
    RUN_GLOB = "runs_B_mistral_teaches_run*"  # Llama-as-Student runs only
    N_PER_CONDITION = 5

    CONDITIONS = {
        "coordinates": {
            "teaching_file": "teaching_coordinates.json",
            "probe_file": "RL_Coord_Time2.json",
            "question_type": "direction",
        },
        "pair_facts": {
            "teaching_file": "teaching_transcript.json",
            "probe_file": "RL_Exp1_Time2.json",
            "question_type": "position",
        },
    }

    def load_module(run_dir, filename, unique_name):
        path = os.path.join(run_dir, filename)
        spec = importlib.util.spec_from_file_location(unique_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def room_descriptions_text(world):
        return "\n".join(
            f"- {r['id']}: {r['color']} colored walls, {r['num_windows']} "
            f"windows, {r['num_doors']} doors"
            for r in world["rooms"].values()
        )

    def build_time2_context(world, teaching):
        rooms_text = room_descriptions_text(world)
        return (
            f"You know about the following rooms:\n{rooms_text}\n\n"
            f"You just learned the following from a teacher about their "
            f"layout:\n{teaching['full_context']}\n\n"
            f"Using what you learned, answer this question directly.\n\n"
        )

    def coordinates_teaching_is_broken(teaching_path):
        with open(teaching_path) as f:
            data = json.load(f)
        stated = sum(t["teacher_said"].count("(") for t in data.get("transcript", []))
        return stated == 0

    def pick_wrong_question(probe_data, question_type):
        """Returns one wrong, single-attempt question entry, or None."""
        if question_type == "direction":
            candidates = [q for q in probe_data.get("direction_answers", [])
                          if not q["is_correct"] and q["attempts"] == 1]
        else:  # position
            candidates = [q for q in probe_data.get("position_answers", [])
                          if q["error"] is not None and q["error"] > 0
                          and q["attempts"] == 1]
        return candidates[0] if candidates else None

    transcripts = []
    for condition, spec in CONDITIONS.items():
        n_collected = 0
        for run_dir in sorted(glob.glob(os.path.join(RUNS_ROOT, RUN_GLOB))):
            if n_collected >= N_PER_CONDITION:
                break

            teaching_path = os.path.join(run_dir, spec["teaching_file"])
            probe_path = os.path.join(run_dir, spec["probe_file"])
            pq_path = os.path.join(run_dir, "probe_questions.py")
            world_path = os.path.join(run_dir, "grid_world.json")
            if not all(os.path.exists(p) for p in
                       [teaching_path, probe_path, pq_path, world_path]):
                continue

            if condition == "coordinates" and coordinates_teaching_is_broken(teaching_path):
                print(f"Skipping {run_dir}: coordinate teaching looks broken")
                continue

            with open(teaching_path) as f:
                teaching = json.load(f)
            with open(probe_path) as f:
                probe_data = json.load(f)

            picked = pick_wrong_question(probe_data, spec["question_type"])
            if picked is None:
                continue  # no clean wrong single-attempt question in this run

            run_id = os.path.basename(run_dir)
            pq = load_module(run_dir, "probe_questions.py", f"pq_{run_id}_{condition}")
            world = pq.load_world(world_path)
            context = build_time2_context(world, teaching)

            if spec["question_type"] == "direction":
                prompt = pq.direction_prompt(picked["p"], picked["q"], context, level=0)
            else:
                prompt = pq.position_prompt(picked["room"], context, level=0)

            transcripts.append({
                "run_id": f"{run_id}_{condition}",
                "prompt": prompt,
                "condition": f"{condition}_in_context",
                "question_type": spec["question_type"],
                "stated_answer": picked.get("raw_response"),
                "was_wrong": True,
            })
            n_collected += 1

        print(f"{condition}: collected {n_collected} transcripts")

    return transcripts


# ---------------------------------------------------------------------------
# 4. APPLY THE LENS AT THE ANSWER POSITION
# ---------------------------------------------------------------------------

def probe_transcript(model, tokenizer, lens, prompt):
    lens_logits, model_logits, _ = lens.apply(model, prompt, positions=[-1])
    per_layer_top_tokens = {}
    layers = LAYERS_TO_PROBE or sorted(lens_logits.keys())
    for layer in layers:
        topk = lens_logits[layer][0].topk(TOP_K).indices
        per_layer_top_tokens[layer] = [tokenizer.decode([t]) for t in topk]
    return per_layer_top_tokens


def analyze_lens_output(per_layer_top_tokens):
    """
    Simple substring-match scoring -- a starting point. Tighten the marker
    lists against your tokenizer's actual vocabulary pieces if this is too
    coarse, and consider weighting by rank within top-K rather than a flat
    count.
    """
    coord_hits, direction_hits = 0, 0
    for tokens in per_layer_top_tokens.values():
        for tok in tokens:
            tok_lower = tok.strip().lower()
            if any(m in tok_lower for m in COORD_MARKERS):
                coord_hits += 1
            if any(m in tok_lower for m in DIRECTION_MARKERS):
                direction_hits += 1
    return {"coord_hits": coord_hits, "direction_hits": direction_hits}


# ---------------------------------------------------------------------------
# 5. MAIN -- resumable, crash-safe, incremental
# ---------------------------------------------------------------------------

def already_done_ids(results_path):
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["run_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    model, tokenizer, lens = load_model_and_lens()
    transcripts = load_transcripts()
    done = already_done_ids(RESULTS_PATH)
    print(f"{len(done)} transcripts already done, {len(transcripts)} total, "
          f"{len(transcripts) - len(done)} remaining.")

    with open(RESULTS_PATH, "a") as out_f:
        for i, t in enumerate(transcripts):
            if t["run_id"] in done:
                continue
            start = time.time()
            try:
                top_tokens = probe_transcript(model, tokenizer, lens, t["prompt"])
                scores = analyze_lens_output(top_tokens)
                record = {**t, **scores, "top_tokens_by_layer": top_tokens,
                          "elapsed_s": round(time.time() - start, 2)}
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                print(f"[{i+1}/{len(transcripts)}] {t['run_id']} "
                      f"{t['condition']}/{t['question_type']} -> {scores}")
            except Exception:
                print(f"[{i+1}/{len(transcripts)}] FAILED on {t['run_id']}:")
                traceback.print_exc()
                out_f.write(json.dumps({"run_id": t["run_id"], "error": traceback.format_exc()}) + "\n")
                out_f.flush()
                continue

    print(f"Done. Results in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
