"""
Generates a long, irrelevant conversation used to push the teaching transcript
out of the model's context window.

The content is deliberately unrelated to rooms, positions, or spatial reasoning
so it cannot accidentally help (or hurt) the Student on the test questions.
"""

import json
from langchain_ollama import ChatOllama

FILLER_MODEL = "llama3.2"
N_EXCHANGES = 40          # how many back-and-forth turns of filler to generate
OUTPUT_FILE = "filler_conversation.json"

TOPICS = [
    "the history of bread making",
    "how tides work",
    "the rules of cricket",
    "why leaves change color",
    "the invention of the printing press",
    "how coffee is roasted",
    "the migration patterns of birds",
    "the development of written musical notation",
]


def generate_filler():
    model = ChatOllama(model=FILLER_MODEL)
    exchanges = []

    for i in range(N_EXCHANGES):
        topic = TOPICS[i % len(TOPICS)]
        prompt = (
            f"Write a detailed paragraph about {topic}. "
            f"Be thorough and specific. Do not mention rooms, locations, "
            f"directions, coordinates, or spatial relationships."
        )
        text = model.invoke(prompt).content.strip()
        exchanges.append({"turn": i + 1, "topic": topic, "text": text})
        print(f"[{i+1}/{N_EXCHANGES}] {topic} ({len(text)} chars)")

    return exchanges


if __name__ == "__main__":
    exchanges = generate_filler()
    full_text = "\n\n".join(e["text"] for e in exchanges)

    # rough token estimate: ~4 chars per token
    approx_tokens = len(full_text) // 4

    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "exchanges": exchanges,
            "full_text": full_text,
            "char_count": len(full_text),
            "approx_tokens": approx_tokens,
        }, f, indent=2)

    print(f"\nGenerated {len(full_text)} chars (~{approx_tokens} tokens)")
    print(f"Saved {OUTPUT_FILE}")
    print("\nNOTE: check this against your model's context window.")
    print("Llama 3.2 and Mistral default to ~8k tokens in Ollama.")
    print("The filler needs to be large enough that teaching + filler exceeds it.")
