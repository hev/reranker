"""Cloudflare Clef / Clef-flash on Workers AI:
POST https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}.
Port of rerank-bench clef_rerank.py (System One wire shape, wrapped in Cloudflare's {success, result})."""
from __future__ import annotations

from pathlib import Path

import httpx

from ._common import ProviderError, decision_request, post, read_decision, require_env

PROMPT = Path(__file__).with_name("prompt_decisions.yaml")


def connect(transport=None) -> httpx.Client:
    token = require_env("CLOUDFLARE_API_TOKEN", "clef")
    account = require_env("CLOUDFLARE_ACCOUNT_ID", "clef")
    return httpx.Client(base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/",
                        headers={"Authorization": f"Bearer {token}"}, timeout=90, transport=transport)


def score(client: httpx.Client, model: str, query, docs, style, p) -> list[float]:
    state, questions = decision_request(query, docs, style, p)
    raw = post(client, f"@cf/cloudflare/{model}", {"model": model, "state": state, "questions": questions}, "Cloudflare")
    if not raw.get("success"):
        raise ProviderError(f"Cloudflare success=false: {raw.get('errors')}")
    return read_decision(raw["result"].get("answers", {}), len(docs), style)
