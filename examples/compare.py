"""Live comparison: every decision model in its `auto` shape (or the styles you name) on one query.

    python examples/compare.py                 # each model in its auto style
    python examples/compare.py pair batch choice

Needs the env keys of the models you want; models without keys are reported and skipped.
"""
import sys
import time

from hev_rerank import MODELS, MissingAPIKeyError, rerank

QUERY = "does vitamin D supplementation improve bone density in older adults"
DOCS = [
    "Meta-analysis of 23 trials: vitamin D alone did not meaningfully increase bone density in older adults.",
    "Dietary sources of vitamin D include oily fish, egg yolk and fortified milk. Many older adults fall short.",
    "Sunscreen reduces vitamin D synthesis in the skin only marginally in practice.",
    "Vitamin D increases intestinal calcium absorption. Deficiency impairs bone mineralization.",
    "Randomised trial: daily vitamin D3 with calcium raised hip bone density in adults over 65 versus placebo.",
    "Astronauts lose 1 to 2 percent of bone density per month in microgravity.",
]

print("| model | style | latency | D00 | D01 | D02 | D03 | D04 | D05 | order |")
print("|---|---|---|---|---|---|---|---|---|---|")
for model in MODELS:
    for style in sys.argv[1:] or ["auto"]:
        shown = MODELS[model][2] if style == "auto" else style
        t = time.perf_counter()
        try:
            hits = rerank(QUERY, DOCS, model=model, style=style)
        except Exception as e:   # report and carry on with the other models
            print(f"| {model} | {shown} | | {type(e).__name__}: {e} |||||||", flush=True)
            continue
        ms = (time.perf_counter() - t) * 1000
        s = {h.index: h.score for h in hits}
        order = " > ".join(f"D{h.index:02d}" for h in hits)
        print(f"| {model} | {shown} | {ms:.0f} ms | " + " | ".join(f"{s[i]:.2f}" for i in range(6)) + f" | {order} |", flush=True)
