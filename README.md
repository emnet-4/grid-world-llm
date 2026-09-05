# What kind of spatial model does a language model build?

A teaching task where the ground truth is known, used to test whether a
language model retains a spatial layout it has been taught, and what kind of
representation that layout takes.

Five rooms sit at known coordinates on a grid. A Teacher model states their
positions to a Student model. The Student is then probed with questions that
separate metric knowledge (coordinates) from relational knowledge (which room
is closer, which direction). The teaching is then pushed out of the Student's
context window, and the same questions are asked again.

Three baseline agents run the same task, so every result reads against a range
rather than a single chance line.

---

## Setup

Requires [Ollama](https://ollama.com) running locally with two models.

```bash
ollama pull llama3.2
ollama pull mistral
```

Then:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Check it works:

```bash
python3 -c "from langchain_ollama import ChatOllama; print(ChatOllama(model='llama3.2').invoke('say hi').content)"
```

Two conditions need more than the base install. The LoRA fine-tuning condition
needs `peft` and `accelerate`, both in `requirements.txt`, and loads the model
through HuggingFace rather than Ollama because Ollama cannot fine-tune. The
layer-wise probe needs a lens package installed from source; see
[Layer-wise probing](#layer-wise-probing) below.

---

## Running it

Everything, in order, with one command:

```bash
python3 Runner_scripts/run_everything.py
```

Each stage is skipped if its output already exists, so this can be re-run after
an interruption. To add only the extensions to a batch that already exists:

```bash
SKIP_CORE=1 python3 Runner_scripts/run_everything.py
```

Or run the stages separately:

```bash
python3 Runner_scripts/run_all_experiments.py                      # ~7h per direction
FORMATS=coordinates,trail,both python3 Runner_scripts/add_teaching_formats.py
python3 Runner_scripts/add_exp5.py                                 # needs a GPU
python3 Analysis_scripts/baseline_agents.py
python3 Analysis_scripts/rescore_shifted.py
python3 Analysis_scripts/analyze_all.py
python3 Analysis_scripts/classify_representation.py
```

To check the setup before committing to a long batch, set `N_RUNS = 1` at the
top of `run_all_experiments.py`. That runs the whole pipeline once in about ten
minutes.

`runs_index.json` is written after every completed run, so an interrupted batch
is still usable and the analysis works on whatever finished.

---

## What happens in one run

| Step | Script | What happens |
|---|---|---|
| 1 | `step1_generate_grid.py` | Build the world. Five rooms, one per quadrant, random position inside it. Features assigned independently of position. |
| 2 | `step2_before_probe.py` | Probe before teaching. 43 questions. |
| 3 | `step3_teaching.py` | Teacher states one fact per room pair, all ten pairs. Student paraphrases each back. |
| 4 | `step4_after_probe.py` | The same 43 questions, transcript in context. |
| 5 | `step6b_save_representation.py` | Student writes down the layout it learned, as coordinates. |
| 6 | `step6_generate_filler.py` | About 15,000 tokens of unrelated text. Generated once, reused. |
| 7 | `step7_time3_exp1.py` | Probe with the teaching flushed out, nothing given in its place. |
| 8 | `step7_time3_exp2.py` | The same, but the saved coordinate list is handed back. |
| 9 | `step9_exp3_probe.py` | The same, answered by a probe trained on that list. |
| 10 | `step3b_teaching_map.py` | Alternative teaching: a drawn map. |
| 11 | `step4b_exp4_time2.py` | Probe with the map in context. |
| 12 | `step7_time3_exp4.py` | Probe with the map flushed out. |

---

## The conditions

All share the first two probes and differ only in what the Student has at the
third.

**Experiment 1** gives it nothing. The control, establishing what survives when
the transcript is gone.

**Experiment 2** hands back the coordinate list the Student wrote for itself.
The knowledge has moved from the conversation into a file.

**Experiment 3** answers from a ridge regression trained to map each room's
frozen embedding to those saved coordinates. The language model is never
modified. The knowledge has moved into a separate set of weights.

**Experiment 4** teaches with a drawn map rather than sentences.

**Experiment 5** (`add_exp5.py`) LoRA fine-tunes the model itself on the ground
truth facts, then asks it with an empty context. Unlike Experiment 3 the model
answers directly, so the channel matches the others. This condition runs on
Llama only, since Mistral is served through Ollama and has no accessible
HuggingFace checkpoint.

Experiments 2 and 3 deliberately hold identical information in two substrates,
which leaves the substrate as the only difference between them. Experiment 5
trains on ground truth rather than the Student's beliefs, so it is a ceiling
rather than a like-for-like comparison.

---

## Teaching formats

`add_teaching_formats.py` runs three alternatives to the default pair-fact
teaching, each with the transcript in context and flushed.

**coordinates** states each room's absolute position.

**trail** describes a walk visiting every room, each step relative to the last:
*"From R1, walk 2 units west and then 3 units south to reach R2."*

**both** is the trail followed by the coordinate list.

The trail format matters for more than variety. It states only four of the ten
room pairs, so the other six are held out. Comparing accuracy on stated pairs
against held-out pairs separates a model that memorised the sentences from one
that built a layout. With pair-fact teaching every pair is stated, so that
distinction cannot be tested.

---

## Baseline agents

`baseline_agents.py` runs three agents on the same questions, thirty times on
every grid. Each produces text, which passes through the same parsers as the
models' answers, so any parsing quirk affects both equally.

**Random** guesses uniformly.

**Line** assumes all rooms lie on one row or column, chosen at random, and
answers from that layout. It has a coherent world model of the wrong shape,
which makes it a useful floor for direction: a collinear layout can only ever
give one direction word, while the true answer needs two in roughly four of
five pairs.

**Quadrant** answers the centre of each room's quadrant. It knows the
neighbourhood and nothing more, and it is the reference that matters. A model
beating it has learned something beyond which corner a room sits in.

The framework is scale-invariant. The centre strategy is always 75% of random
at any grid size, direction chance stays near 0.20 regardless of room count,
and the quadrant agent holds at about 37% of random, so results at different
scales remain comparable.

---

## Analysis

`analyze_all.py` produces one figure per measure. Bars computed from fewer than
half the expected answers are hatched and labelled with their sample size,
which matters because in the flushed conditions the Student sometimes refuses
nearly every question and those means are not comparable to the rest.

`classify_representation.py` scores each condition on a metric and a relational
axis, normalised so 0.00 is the random agent and 1.00 is the quadrant agent,
and reports whether the pattern looks Euclidean, graphical, mixed, or
memorisation. It also checks whether stated and held-out pairs were comparable
in difficulty, since direction is easier to answer when rooms are far apart.

`rescore_shifted.py` recomputes position error after removing the best-fitting
translation, and separately after allowing an axis to be flipped. The teaching
gives only relative offsets, so a Student that reconstructs the layout
correctly but anchors it elsewhere would otherwise score as a total failure,
and one that inverts a convention produces a mirrored layout that no
translation repairs.

`compare_conditions.py` reports the spread between two conditions, so a
difference can be checked against run-to-run variation before it is claimed.

---

## Utilities

| Script | Purpose |
|---|---|
| `summary.py` | Short results summary that fits in a message |
| `diagnostic_report.py` | Sample sizes, answer rates, raw responses |
| `probe_questions.py` | Question construction, forcing, parsing, scoring |
| `benchmarks.py` | Chance levels computed by enumeration |

---

## Layer-wise probing

`run_jlens_probe.py` reads the residual stream at the token position
immediately before the Student answers, and decodes what that state is disposed
to say at each layer. The question it asks is whether a Student that fails a
question nonetheless leans internally toward the content of the other format.
A coordinate-taught Student that gets a direction question wrong may still show
direction-like tokens internally, in which case the information is present and
only its expression is format-locked. If it does not, the information is
genuinely absent.

Only questions the Student got **wrong** are used, since that is the case the
probe is designed to distinguish.

```bash
pip install git+https://github.com/anthropics/jacobian-lens
python3 prepare_fit_prompts.py --model-name unsloth/Llama-3.2-3B-Instruct \
    --n 150 --out fit_prompts.json
python3 run_jlens_probe.py
```

Set `RUNS_ROOT` in `run_jlens_probe.py` to your runs directory before running.
The lens is fitted once and cached, and results are appended after every
transcript, so an interrupted run resumes rather than restarting.

Two constraints apply. The probe needs weight and gradient access, so it runs
on the HuggingFace mirror `unsloth/Llama-3.2-3B-Instruct` rather than the
Ollama-served build that generated the transcripts: the same model family and
instruction tuning, a different quantisation, and the closest available match
rather than a bit-identical one. And it is restricted to the runs where Llama
was the Student, for the same reason Experiment 5 is.

---

## Notes on the design

**Teaching is scripted.** The facts are computed from ground truth and the
Teacher only rephrases them, so it cannot invent a fact or skip a pair.
Experiment 4 is the exception: the Teacher redraws the map in its own words and
sometimes corrupts it, so that condition measures transmission through a lossy
channel rather than the effect of format.

**The flush is verified, not assumed.** Planting a word at the start of a long
prompt and asking for it at the end, the model cannot recall it at a
4,096-token cap but can at 32,768.

**Answers are forced.** A refusal triggers a re-prompt with escalating
insistence, up to twelve attempts. Questions still unanswered are excluded from
the mean rather than scored as wrong, because a refusal and a wrong answer are
different failures.

**Position is underdetermined.** The teaching gives relative offsets and never
an origin, so the layout is fixed only up to translation. That is what
`rescore_shifted.py` exists for.

---

## Reproducing a single condition

Every run folder contains the world it used, the teaching transcript, and every
raw model response. To re-analyse without re-running any models, delete the
figures and run `analyze_all.py` again.

To regenerate one condition:

```bash
cd runs_A_llama_teaches_run1
GRID_SEED=17 TEACHER_MODEL=llama3.2 STUDENT_MODEL=mistral \
  python3 ../Experiment\ _scripts/step4_after_probe.py
```
