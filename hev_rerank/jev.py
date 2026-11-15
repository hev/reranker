"""Jev, TypeSafe's System One model, through typesafe_sdk. Port of rerank-bench jev_rerank.py."""
from __future__ import annotations

from pathlib import Path

from typesafe_sdk import RetryPolicy, TypeSafeClient

from ._common import decision_request, require_env

PROMPT = Path(__file__).with_name("prompt.yaml")


def connect(transport=None) -> TypeSafeClient:
    key = require_env("TYPESAFE_API_KEY", "jev")
    return TypeSafeClient(api_key=key, transport=transport,
                          retry=RetryPolicy(max_retries=6, backoff_max=30.0, timeout=90.0))


def score(client: TypeSafeClient, model: str, query, docs, style, p) -> list[float]:
    state, questions = decision_request(query, docs, style, p, pair_key="rel")
    r = client.system_one(state=state, questions=questions, model=model)
    if style == "pair":
        return [float(r.answers["rel"].noul)]
    if style == "batch":
        return [float(r.answers[k].noul) for k in questions]
    probs = r.answers["best"].probabilities
    return [float(probs.get(k, 0.0)) for k in state["documents"]]
