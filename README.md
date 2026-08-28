# What kind of spatial model does a language model build?

A teaching task where the ground truth is known, used to test whether a language
model retains a spatial layout it has been taught, and what kind of
representation that layout takes.

Five rooms sit at known coordinates on a grid. A Teacher model states their
relative positions to a Student model. The Student is then probed with questions
that separate metric knowledge (coordinates) from relational knowledge (which
room is closer, which direction). The teaching is then pushed out of the
Student's context window, and the same questions are asked again.

Three baseline agents run the same task, so every result can be read against a
range rather than a single chance line.

---

## Setup

Requires [Ollama](https://ollama.com) running locally with two models.

```bash
ollama pull llama3.2
ollama pull mistral
```

Then, in a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Check it works:

```bash
python3 -c "from langchain_ollama import ChatOllama; print(ChatOllama(model='llama3.2').invoke('say hi').content)"
```

---

## Running the experiment

```bash
python3 run_all_experiments.py     # ~6-8 hours for 30 runs per direction
python3 baseline_agents.py         # seconds
python3 rescore_shifted.py         # seconds
python3 analyze_all.py             # seconds
```

`run_all_experiments.py` writes `runs_index.json` after every completed run, so
a partial batch can be analysed at any point. To shorten it, lower `N_RUNS` at
the top of that file.

The remaining three scripts read what is on disk and can be re-run freely.

### Adding Experiment 4 to an existing batch

```bash
python3 add_exp4.py
```

Runs only the Experiment 4 steps in every folder listed in `runs_index.json`,
skipping any that already have them. Useful if a batch was run before
Experiment 4 existed.

---

## What each run does

| Step | Script | What happens |
|---|---|---|
| 1 | `step1_generate_grid.py` | Build the world. Five rooms, one per quadrant, random inside it. Features assigned independently of position. |
| 2 | `step2_before_probe.py` | Probe before teaching. 43 questions. |
| 3 | `step3_teaching.py` | Teacher states one fact per room pair, all ten pairs. Student paraphrases each back. |
| 4 | `step4_after_probe.py` | Same 43 questions, transcript in context. |
| 5 | `step6b_save_representation.py` | Student writes down the layout it learned, as coordinates. |
| 6 | `step6_generate_filler.py` | Generate ~15,000 tokens of unrelated text. Run once, reused. |
| 7 | `step7_time3_exp1.py` | Probe with the teaching flushed out. Nothing given in its place. |
| 8 | `step7_time3_exp2.py` | Same, but the saved coordinate list is handed back. |
| 9 | `step9_exp3_probe.py` | Same, answered by a probe trained on that list. |
| 10 | `step3b_teaching_map.py` | Alternative teaching: a drawn map instead of sentences. |
| 11 | `step4b_exp4_time2.py` | Probe with the map in context. |
| 12 | `step7_time3_exp4.py` | Probe with the map flushed out. |

---

## The four experiments

All share the first two probes. They differ only in what the Student has at the
third.

**Experiment 1** gives it nothing. This is the control, and it establishes what
survives when the transcript is gone.

**Experiment 2** hands back the coordinate list the Student wrote for itself.
The knowledge has moved from the conversation into a file.

**Experiment 3** answers from a ridge regression trained to map each room's
frozen embedding to those same saved coordinates. The language model is never
modified. The knowledge has moved into a set of weights.

**Experiment 4** teaches with a drawn map rather than sentences.

Experiments 2 and 3 deliberately hold identical information in two substrates,
which leaves the substrate as the only difference between them.

---

## Baseline agents

`baseline_agents.py` runs three agents on the same questions, thirty times on
every grid. Each produces text, which passes through the same parsers as the
models' answers, so any parsing quirk affects both equally.

- **Random** guesses uniformly.
- **Line** assumes all rooms lie on one row or column, chosen at random, and
  answers from that layout. It has a coherent world model of the wrong shape.
- **Quadrant** answers the centre of each room's quadrant. It knows the
  neighbourhood and nothing more.

The quadrant agent is the reference that matters. It represents partial
knowledge rather than no knowledge, so a model that beats it has learned
something beyond which corner a room sits in.

---

## Analysis

`analyze_all.py` produces one figure per measure:

- `fig_triplet.png` — is room A closer to B or C
- `fig_direction.png` — is A east/west and north/south of B
- `fig_position.png` — what are A's coordinates
- `fig_map.png` — all five sets of coordinates at once
- `fig_retries.png` — how often the Student had to be pushed to answer

Bars computed from fewer than half the expected answers are hatched and labelled
with their sample size. This matters: in the flushed conditions the Student
sometimes refuses nearly every question, and those means are not comparable to
the rest.

`rescore_shifted.py` recomputes position error after removing the best-fitting
translation. The teaching gives only relative offsets, so a Student that
reconstructs the layout correctly but anchors it elsewhere would otherwise score
as a total failure.

---

## Utilities

| Script | Purpose |
|---|---|
| `summary.py` | Short results summary that fits in a message |
| `diagnostic_report.py` | Sample sizes, answer rates, raw responses |
| `probe_questions.py` | Question construction, forcing, parsing, scoring |
| `benchmarks.py` | Chance levels computed by enumeration|

---

## Notes on the design

**Teaching is scripted.** The facts are computed from ground truth and the
Teacher only rephrases them, so it cannot hallucinate a fact or skip a pair.
Experiment 4 is the exception: the Teacher redraws the map in its own words and
sometimes corrupts it, so that experiment measures transmission through a lossy
channel rather than the effect of format.

**The flush is verified, not assumed.** Planting a word at the start of a long
prompt and asking for it at the end, the model cannot recall it at a
4,096-token cap but can at 32,768.

**Answers are forced.** A refusal triggers a re-prompt with escalating
insistence, up to twelve attempts. Questions still unanswered are excluded from
the mean rather than scored as wrong, because a refusal and a wrong answer are
different failures.

**Position is underdetermined.** The teaching gives relative offsets and never
an origin, so the layout is fixed only up to translation. `rescore_shifted.py`
exists for this reason.

---

## Reproducing a specific run

Every run folder contains the world it used, the teaching transcript, and every
raw model response. To re-analyse without re-running any models, delete the
figures and run `analyze_all.py` again.

To regenerate one condition:

```bash
cd runs_A_llama_teaches_run1
GRID_SEED=17 TEACHER_MODEL=llama3.2 STUDENT_MODEL=mistral \
  python3 ../step4_after_probe.py
```
## Note
Please make sure to have all scripts in the same folder