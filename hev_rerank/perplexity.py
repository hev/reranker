"""Perplexity pplx-decider: POST https://api.perplexity.ai/v1/decisions. Port of rerank-bench pplx_rerank.py
(System One wire shape: {model, state, questions} -> {answers})."""
from __future__ import annotations

from pathlib import Path

import httpx

from ._common import decision_request, post, read_decision, require_env

PROMPT = Path(__file__).with_name("prompt_decisions.yaml")


def connect(transport=None) -> httpx.Client:
    key = require_env("PERPLEXITY_API_KEY", "perplexity")
    return httpx.Client(base_url="https://api.perplexity.ai", headers={"Authorization": f"Bearer {key}"},
                        timeout=90, transport=transport)


def score(client: httpx.Client, model: str, query, docs, style, p) -> list[float]:
    state, questions = decision_request(query, docs, style, p)
    raw = post(client, "/v1/decisions", {"model": model, "state": state, "questions": questions}, "Perplexity")
    return read_decision(raw.get("answers", {}), len(docs), style)
