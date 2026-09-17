"""Jev (TypeSafe's System One model) as a reranker.

Shape: the query and up to `max_docs_per_call` candidate documents go into one Jev *state*;
one Noul ("is this document relevant?") question per document; every question is evaluated
in parallel by the model, so a 30-document call costs about the same wall time as a 5-document
call. The Noul is a calibrated probability, so the score is usable as an absolute relevance
threshold (prune) as well as a sort key (rerank).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient

PROMPT_PATH = Path(__file__).with_name("prompt.yaml")
# TypeSafe documents a ~32k-token budget per request (state + questions), ~150k characters of
# English. Keep headroom for the questions and for non-English text.
DEFAULT_MAX_STATE_CHARS = 100_000


@dataclass(frozen=True)
class RerankResult:
    index: int          # position in the input `documents`
    score: float        # Noul probability the document is relevant, 0..1
    document: Any       # the input document, unchanged


def load_prompt(path: str | os.PathLike[str] | None = None) -> dict:
    with open(path or PROMPT_PATH) as f:
        return yaml.safe_load(f)


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
    top_n: int | None = None,
    threshold: float | None = None,
    client: TypeSafeClient | None = None,
    model: str = "jev-latest",
    prompt: Mapping[str, Any] | None = None,
    max_docs_per_call: int = 30,
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
    concurrency: int = 4,
) -> list[RerankResult]:
    """Score every document for relevance to `query` and return them best-first.

    documents: strings, or mappings with `title` / `text` (extra keys are passed through
    unchanged in the result but not sent to the model unless they are title/text/content/body).
    top_n: keep only the best N. threshold: drop documents scoring below it (the Noul is
    calibrated: on BEIR shortlists, documents under 0.1 were judged relevant ~0.5% of the time).
    Long candidate lists are split into several calls, issued concurrently.
    """
    if not documents:
        return []
    p = prompt or load_prompt()
    client = client or TypeSafeClient(retry=RetryPolicy(max_retries=6, backoff_max=30.0, timeout=90.0))

    def score_batch(idx: list[int]) -> list[tuple[int, float]]:
        ids = {f"D{j:02d}": i for j, i in enumerate(idx)}
        state = {"query": query, "documents": {k: _doc_state(documents[i]) for k, i in ids.items()}}
        questions = {k: Noul(instructions=p["question"].format(id=k), criteria=p["criteria"]) for k in ids}
        r = client.system_one(state=state, questions=questions, model=model)
        return [(ids[k], float(r.answers[k].noul)) for k in ids]

    batches = list(_chunk(documents, max_docs_per_call, max_state_chars))
    if len(batches) == 1:
        scored = score_batch(batches[0])
    else:
        with ThreadPoolExecutor(concurrency) as ex:
            scored = [s for part in ex.map(score_batch, batches) for s in part]

    results = [RerankResult(i, s, documents[i]) for i, s in scored]
    if threshold is not None:
        results = [r for r in results if r.score >= threshold]
    results.sort(key=lambda r: -r.score)
    return results[:top_n] if top_n else results
