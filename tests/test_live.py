"""Live smoke test: one real call per provider in its `auto` style. Skipped unless that provider's keys
are in the environment. Run: pytest tests/test_live.py -s"""
import os

import pytest

from hev_rerank import MODELS, rerank

KEYS = {"jev": ["TYPESAFE_API_KEY"], "perplexity": ["PERPLEXITY_API_KEY"],
        "clef": ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"], "clef-flash": ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"],
        "openai": ["OPENAI_API_KEY"]}
QUERY = "does vitamin D supplementation improve bone density in older adults"
DOCS = [
    "Meta-analysis of 23 trials: vitamin D alone did not meaningfully increase bone density in older adults.",
    "Dietary sources of vitamin D include oily fish, egg yolk and fortified milk. Many older adults fall short.",
    "Sunscreen reduces vitamin D synthesis in the skin only marginally in practice.",
    "Vitamin D increases intestinal calcium absorption. Deficiency impairs bone mineralization.",
    "Randomised trial: daily vitamin D3 with calcium raised hip bone density in adults over 65 versus placebo.",
    "Astronauts lose 1 to 2 percent of bone density per month in microgravity.",
]


@pytest.mark.live
@pytest.mark.parametrize("model", list(MODELS))
def test_live_auto(model):
    missing = [k for k in KEYS[model] if not os.environ.get(k)]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")
    hits = rerank(QUERY, DOCS, model=model)
    assert sorted(h.index for h in hits) == list(range(6))
    assert all(0 <= h.score <= 1 for h in hits)
    assert hits[0].index in (0, 4)       # the two trials are the passages that address the question
    print(model, MODELS[model][2], [(h.index, round(h.score, 2)) for h in hits])
