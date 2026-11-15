"""OpenAI Decisions (gpt-6-luna, public beta): POST https://api.openai.com/v1/decisions.
Port of rerank-bench openai_decisions_rerank.py. The state is one text (query, then each document under its
id); Nouls are `predicate` questions, the Choice is a `choice` question with `{value}` options. The API has no
`criteria` field, so the criteria are appended to each predicate's instructions. A refused question carries
no ranking signal: predicate -> 0.0, choice -> uniform."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import httpx

from ._common import ProviderError, doc_id, post, require_env

PROMPT = Path(__file__).with_name("prompt_decisions.yaml")


def connect(transport=None) -> httpx.Client:
    key = require_env("OPENAI_API_KEY", "openai")
    return httpx.Client(base_url="https://api.openai.com", headers={"Authorization": f"Bearer {key}"},
                        timeout=120, transport=transport)


def _text(doc: Any) -> str:
    if isinstance(doc, Mapping):
        return "\n".join(str(v) for v in doc.values() if v)   # {"title", "text"} -> "title\ntext", as the bench
    return str(doc)


def build(model: str, query, docs, style, p) -> dict:
    crit = " True: " + p["criteria"]["true"] + " False: " + p["criteria"]["false"]
    if style == "pair":
        text = f"Query: {query}\n\nDocument:\n{_text(docs[0])}"
        qs = [{"type": "predicate", "name": "D00", "instructions": p["pair_question"] + crit}]
    else:
        ids = [doc_id(i) for i in range(len(docs))]
        text = f"Query: {query}\n\n" + "\n\n".join(f"Document `documents.{n}`:\n{_text(d)}" for n, d in zip(ids, docs))
        if style == "batch":
            qs = [{"type": "predicate", "name": n, "instructions": p["question"].format(id=n) + crit} for n in ids]
        else:
            qs = [{"type": "choice", "name": "best", "instructions": p["choice_question"], "choices": [{"value": n} for n in ids]}]
    return {"model": model, "input": text, "questions": qs}


def score(client: httpx.Client, model: str, query, docs, style, p) -> list[float]:
    raw = post(client, "/v1/decisions", build(model, query, docs, style, p), "OpenAI")
    try:
        answers = {a["name"]: a for a in raw["answers"]}
        if style == "choice":
            best = answers["best"]
            if best["type"] == "refusal":
                return [1 / len(docs)] * len(docs)
            probs = {o["value"]: o["probability"] for o in best["probabilities"]}
            return [float(probs.get(doc_id(i), 0.0)) for i in range(len(docs))]
        out = []
        for i in range(len(docs)):
            a = answers[doc_id(i)]
            out.append(0.0 if a["type"] == "refusal" else float(a["probability"]))
        return out
    except (KeyError, TypeError, ValueError) as e:
        raise ProviderError(f"OpenAI: unexpected answer shape: {e!r}") from None
