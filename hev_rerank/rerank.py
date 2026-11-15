"""Decision models as rerankers: one call, any model, any request shape.

Every model answers the same question per document, *is this document relevant to the query?*, in one of
three shapes:

- pair:   one call per (query, document), one yes/no probability (a Noul).
- batch:  the query and up to `max_docs_per_call` documents in one state, one Noul per document, all
          evaluated in parallel by the model. Longer lists are split into concurrent calls.
- choice: one Choice over the document ids; the scores are the options' probabilities and sum to one.

Noul scores are calibrated probabilities, so they work as a sort key (rerank) and as an absolute
threshold (prune). Choice scores are relative to the list they came from, so they only sort.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from . import clef, jev, openai, perplexity

PROMPT_PATH = Path(__file__).with_name("prompt.yaml")
# TypeSafe documents a ~32k-token budget per request (state + questions), ~150k characters of
# English. Keep headroom for the questions and for non-English text.
DEFAULT_MAX_STATE_CHARS = 100_000

PROVIDERS = {"jev": jev, "perplexity": perplexity, "clef": clef, "openai": openai}
STYLES = ("pair", "batch", "choice")

# model alias -> (provider, vendor model id, the shape `style="auto"` picks).
# `auto` is each model's best measured shape: nDCG@10 on BM25 top-30 shortlists of SciFact, NFCorpus and
# FiQA (test), generic prompts, in hev/rerank-bench (https://hevmind.com/research/rerankers).
MODELS = {
    "jev":        ("jev", "jev-latest", "batch"),                    # batch ties pair at ~half the calls
    "perplexity": ("perplexity", "pplx-decider-v1-27b", "choice"),   # choice best on SciFact and FiQA, 4x faster than batch
    "clef":       ("clef", "clef", "pair"),                          # pair is the shape that works (only shape run on full clef)
    "clef-flash": ("clef", "clef-flash", "pair"),                    # pair 0.77 vs batch 0.44 / choice 0.51 on SciFact
    "openai":     ("openai", "gpt-6-luna", "choice"),                # choice 0.75 vs batch 0.68 on SciFact
}


@dataclass(frozen=True)
class RerankResult:
    index: int          # position in the input `documents`
    score: float        # pair/batch: probability the document is relevant, 0..1; choice: option probability
    document: Any       # the input document, unchanged


def load_prompt(path: str | os.PathLike[str] | None = None) -> dict:
    with open(path or PROMPT_PATH) as f:
        return yaml.safe_load(f)


def resolve_model(model: str) -> tuple[str, str, str]:
    """(provider, vendor model id, auto style) for an alias ("clef-flash") or "vendor:model-id"."""
    if model in MODELS:
        return MODELS[model]
    vendor, sep, model_id = model.partition(":")
    if sep and vendor in PROVIDERS and model_id:
        return vendor, model_id, MODELS[vendor][2]
    if model.startswith("jev"):          # 0.1.0 took a Jev model id here, e.g. "jev-latest"
        return "jev", model, MODELS["jev"][2]
    raise ValueError(f"unknown model {model!r}: use one of {sorted(MODELS)} or 'vendor:model-id' "
                     f"with vendor in {sorted(PROVIDERS)}")


def _doc_state(doc: Any) -> Mapping[str, Any] | str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, Mapping):
        return {k: doc[k] for k in ("title", "text", "content", "body") if k in doc} or dict(doc)
    return str(doc)


def _chunk(docs: Sequence[Any], max_docs: int, max_chars: int) -> Iterable[list[int]]:
    batch: list[int] = []
    chars = 0
    for i, d in enumerate(docs):
        n = len(str(_doc_state(d)))
        if batch and (len(batch) >= max_docs or chars + n > max_chars):
            yield batch
            batch, chars = [], 0
        batch.append(i)
        chars += n
    if batch:
        yield batch


def rerank(
    query: str,
    documents: Sequence[Any],
    *,
    model: str = "jev",
    style: str = "auto",
    top_n: int | None = None,
    threshold: float | None = None,
    client: Any = None,
    prompt: Mapping[str, Any] | None = None,
    max_docs_per_call: int = 30,
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
    concurrency: int = 8,
) -> list[RerankResult]:
    """Score every document for relevance to `query` and return them best-first.

    model: "jev", "perplexity", "clef", "clef-flash", "openai", or "vendor:model-id"
      (e.g. "perplexity:pplx-decider-v1-27b"). Keys come from the environment.
    style: "pair", "batch", "choice", or "auto" (the model's best measured shape, see MODELS).
    documents: strings, or mappings with `title` / `text` (extra keys are passed through
      unchanged in the result but not sent to the model unless they are title/text/content/body).
    top_n: keep only the best N. threshold: drop documents scoring below it (pair/batch only: the Noul
      is calibrated; on BEIR shortlists, Jev documents under 0.1 were judged relevant ~0.5% of the time).
    client: a ready client for the provider (TypeSafeClient for Jev, an authenticated httpx.Client with
      base_url for the others); default: built from the environment.
    prompt: overrides any of `question` (batch), `pair_question`, `choice_question`, `criteria`; keys
      left out keep the model's default from prompt.yaml / prompt_decisions.yaml.
    Batch splits lists over `max_docs_per_call` / `max_state_chars` into concurrent calls; pair sends one
    call per document, `concurrency` at a time. Choice is always one call: probabilities from different
    calls aren't comparable, so a list over the limits raises ValueError.
    """
    provider_name, model_id, auto = resolve_model(model)
    style = auto if style == "auto" else style
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}: use one of {list(STYLES)} or 'auto'")
    if style == "choice" and threshold is not None:
        raise ValueError("threshold can't be used with style='choice': choice probabilities sum to one across "
                         "the list, so they rank documents but can't be thresholded. Use style='pair' or 'batch'.")
    if not documents:
        return []
    provider = PROVIDERS[provider_name]
    p = {**load_prompt(provider.PROMPT), **(prompt or {})}
    states = [_doc_state(d) for d in documents]

    if style == "pair":
        groups = [[i] for i in range(len(documents))]
    else:
        groups = list(_chunk(documents, max_docs_per_call, max_state_chars))
        if style == "choice" and len(groups) > 1:
            raise ValueError(f"style='choice' ranks the whole list in one call and {len(documents)} documents exceed "
                             f"max_docs_per_call={max_docs_per_call} / max_state_chars={max_state_chars}; "
                             "shorten the list, raise the limits, or use style='batch'")

    if style == "choice" and len(documents) == 1:
        scored = [(0, 1.0)]     # one option: probability 1, no call needed
    else:
        client = client or provider.connect()

        def score(idx: list[int]) -> list[tuple[int, float]]:
            return list(zip(idx, provider.score(client, model_id, query, [states[i] for i in idx], style, p)))

        if len(groups) == 1:
            scored = score(groups[0])
        else:
            with ThreadPoolExecutor(min(concurrency, len(groups))) as ex:
                scored = [s for part in ex.map(score, groups) for s in part]

    results = [RerankResult(i, s, documents[i]) for i, s in scored]
    if threshold is not None:
        results = [r for r in results if r.score >= threshold]
    results.sort(key=lambda r: -r.score)
    return results[:top_n] if top_n else results
