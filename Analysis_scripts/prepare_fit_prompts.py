"""
Prepares a fit-prompt corpus for jlens.fit(). Run this once, before the
overnight probe run, and it writes fit_prompts.json to disk.

Tries to pull real pretraining-like text from a small standard corpus via
the `datasets` library (WikiText-103), which is the honest way to do this --
the lens is meant to be fit on generic text, not on anything related to your
experiment. Falls back to a small built-in set of diverse passages if
`datasets` isn't available or there's no internet access at fit time (e.g.
if you're on a cluster node without egress), so the overnight run is never
blocked on this step.

Usage:
    python prepare_fit_prompts.py --n 150 --tokens-per-prompt 128 \
        --model-name "TODO: your Student's HF checkpoint ID" \
        --out fit_prompts.json
"""

import argparse
import json


FALLBACK_PASSAGES = [
    "The history of cartography stretches back thousands of years, from "
    "clay tablets marking irrigation canals to the satellite imagery used "
    "in modern navigation systems. Early maps were often schematic rather "
    "than metrically accurate, prioritizing the order in which landmarks "
    "would be encountered over their true distances.",
    "In economics, a market is considered efficient when prices fully "
    "reflect all available information. This idea, developed most notably "
    "by Eugene Fama in the 1960s and 1970s, has shaped decades of research "
    "into asset pricing, though its strongest forms have been repeatedly "
    "challenged by evidence of persistent anomalies.",
    "Photosynthesis converts light energy into chemical energy stored in "
    "glucose. The process occurs in two main stages: the light-dependent "
    "reactions, which take place in the thylakoid membranes, and the "
    "Calvin cycle, which occurs in the stroma of the chloroplast.",
    "The Roman road network eventually spanned over 400,000 kilometers, "
    "connecting the empire's furthest provinces to its capital. Roads were "
    "built in layers, with a foundation of large stones topped by gravel "
    "and then a surface layer, engineered to drain water away from the "
    "roadbed.",
    "A sorting algorithm's time complexity describes how its running time "
    "grows with input size. Quicksort has an average-case complexity of "
    "O(n log n) but a worst-case complexity of O(n squared), which occurs "
    "when the chosen pivot repeatedly splits the array into highly unequal "
    "partitions.",
    "Coral reefs, though they cover less than one percent of the ocean "
    "floor, are home to roughly a quarter of all known marine species. "
    "They form when colonies of tiny animals called polyps secrete "
    "calcium carbonate skeletons over many generations, building "
    "structures that can persist for thousands of years.",
    "The printing press, developed by Johannes Gutenberg around 1440, is "
    "widely credited with enabling the rapid spread of ideas across "
    "Europe. Before its invention, books were copied by hand, a process "
    "that could take a single scribe the better part of a year to "
    "complete for a single volume.",
    "Working memory is typically described as having a limited capacity, "
    "often cited as around four chunks of information for adults under "
    "typical conditions. This capacity can be effectively expanded through "
    "chunking, in which multiple related items are grouped and stored as a "
    "single unit.",
    "Migratory birds rely on a combination of cues to navigate over vast "
    "distances, including the position of the sun, the pattern of "
    "polarized light, and a magnetic sense that may involve "
    "light-sensitive proteins in the retina. Some species can also "
    "recognize and remember the outlines of coastlines and mountain "
    "ranges.",
    "The stock of a bond typically falls when prevailing interest rates "
    "rise, since newly issued bonds offer more competitive yields, making "
    "existing bonds with lower fixed rates less attractive to buyers on "
    "the secondary market.",
]


def build_fallback_prompts(n):
    prompts = []
    i = 0
    while len(prompts) < n:
        prompts.append(FALLBACK_PASSAGES[i % len(FALLBACK_PASSAGES)])
        i += 1
    return prompts[:n]


def build_wikitext_prompts(n, tokens_per_prompt, model_name):
    from datasets import load_dataset
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    prompts = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < 200:  # skip empty lines / section headers
            continue
        ids = tokenizer.encode(text)
        if len(ids) < tokens_per_prompt:
            continue
        truncated = tokenizer.decode(ids[:tokens_per_prompt])
        prompts.append(truncated)
        if len(prompts) >= n:
            break
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--tokens-per-prompt", type=int, default=128)
    ap.add_argument("--model-name", type=str, required=True,
                     help="Your Student's HF checkpoint ID (needed to tokenize consistently)")
    ap.add_argument("--out", type=str, default="fit_prompts.json")
    args = ap.parse_args()

    try:
        prompts = build_wikitext_prompts(args.n, args.tokens_per_prompt, args.model_name)
        if len(prompts) < args.n:
            raise RuntimeError(f"Only got {len(prompts)} prompts, wanted {args.n}")
        source = "wikitext-103"
    except Exception as e:
        print(f"Falling back to built-in passages (reason: {e})")
        prompts = build_fallback_prompts(args.n)
        source = "fallback"

    with open(args.out, "w") as f:
        json.dump({"source": source, "prompts": prompts}, f, indent=2)

    print(f"Wrote {len(prompts)} fit prompts from source={source} to {args.out}")


if __name__ == "__main__":
    main()
