"""What the providers share: env keys, one retrying HTTP POST, and the decision-state request shape."""
from __future__ import annotations

import os
import time
from typing import Any, Mapping, Sequence

import httpx


class MissingAPIKeyError(RuntimeError):
    """A provider's API key (or account id) is not in the environment."""


class ProviderError(RuntimeError):
    """The provider returned an error or a response we can't read."""


def require_env(name: str, model: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MissingAPIKeyError(f"model={model!r} needs the {name} environment variable (or pass client=...)")
    return value


def doc_id(i: int) -> str:
    return f"D{i:02d}"


def post(client: httpx.Client, url: str, body: Mapping[str, Any], provider: str, retries: int = 4) -> Any:
    """POST JSON; retry 429, 5xx and connection errors with backoff (Retry-After honored, capped at 30 s)."""
    for attempt in range(retries + 1):
        try:
            r = client.post(url, json=body)
        except httpx.TransportError as e:
            if attempt == retries:
                raise ProviderError(f"{provider}: {type(e).__name__}: {e}") from None
            time.sleep(min(30, 2 ** attempt))
            continue
        if (r.status_code == 429 or r.status_code >= 500) and attempt < retries:
            try:
                delay = float(r.headers.get("retry-after", 2 ** attempt))
            except ValueError:
                delay = 2 ** attempt
            time.sleep(min(30.0, delay))
            continue
        if r.is_error:
            raise ProviderError(f"{provider} HTTP {r.status_code}: {r.text[:500]}")
        return r.json()


def decision_request(query: str, docs: Sequence[Any], style: str, p: Mapping[str, Any], pair_key: str = "D00"):
    """(state, questions) in the System One wire shape shared by Jev, Perplexity and Clef.

    pair:   {"query", "document"}, one Noul.
    batch:  {"query", "documents": {D00: ..., D01: ...}}, one Noul per document id.
    choice: same state, one Choice whose options are the document ids.
    """
    crit = dict(p["criteria"])
    if style == "pair":
        return ({"query": query, "document": docs[0]},
                {pair_key: {"type": "noul", "instructions": p["pair_question"], "criteria": crit}})
    ids = [doc_id(i) for i in range(len(docs))]
    state = {"query": query, "documents": dict(zip(ids, docs))}
    if style == "batch":
        return state, {k: {"type": "noul", "instructions": p["question"].format(id=k), "criteria": crit} for k in ids}
    return state, {"best": {"type": "choice", "instructions": p["choice_question"], "criteria": {k: f"documents.{k}" for k in ids}}}


def read_decision(answers: Mapping[str, Any], n: int, style: str, pair_key: str = "D00") -> list[float]:
    """Scores in document order from System One-style answers (plain dicts)."""
    try:
        if style == "pair":
            return [float(answers[pair_key]["noul"])]
        if style == "batch":
            return [float(answers[doc_id(i)]["noul"]) for i in range(n)]
        probs = answers["best"]["probabilities"]
        return [float(probs.get(doc_id(i), 0.0)) for i in range(n)]
    except (KeyError, TypeError, ValueError) as e:
        raise ProviderError(f"unexpected answer shape: {e!r}") from None
