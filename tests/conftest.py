"""A fake decision-model server for every provider, at the HTTP layer (httpx / httpx2 MockTransport).

A document's relevance is 0.9 if its text contains "relevant", else 0.1; choice probabilities are those
weights normalized. Every request is recorded so tests can check URLs, auth and request bodies.
"""
import json
import re

import httpx
import httpx2
import pytest

from hev_rerank import clef, jev, openai, perplexity

ENV = {
    "TYPESAFE_API_KEY": "ts-test",
    "PERPLEXITY_API_KEY": "pplx-test",
    "CLOUDFLARE_API_TOKEN": "cf-test",
    "CLOUDFLARE_ACCOUNT_ID": "acct123",
    "OPENAI_API_KEY": "oa-test",
}


def rel(text) -> float:
    return 0.9 if "relevant" in json.dumps(text) else 0.1


def choice_probs(texts: dict) -> dict:
    w = {k: rel(t) for k, t in texts.items()}
    return {k: v / sum(w.values()) for k, v in w.items()}


def decision_answers(body):
    """Answers for the System One wire shape (Jev, Perplexity, Clef)."""
    state, out = body["state"], {}
    for name, q in body["questions"].items():
        if q["type"] == "noul":
            doc = state["document"] if "document" in state else state["documents"][name]
            out[name] = {"type": "noul", "noul": rel(doc)}
        else:
            probs = choice_probs({k: state["documents"][k] for k in q["criteria"]})
            best = max(probs, key=probs.get)
            out[name] = {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}
    return out


def openai_answers(body):
    text = body["input"]
    docs = dict(re.findall(r"Document `documents\.(D\d+)`:\n(.*?)(?=\n\nDocument `|\Z)", text, re.S))
    if not docs:
        docs = {"D00": text.split("Document:\n", 1)[1]}
    out = []
    for q in body["questions"]:
        if q["type"] == "predicate":
            out.append({"type": "predicate", "name": q["name"], "probability": rel(docs[q["name"]])})
        else:
            probs = choice_probs({c["value"]: docs[c["value"]] for c in q["choices"]})
            out.append({"type": "choice", "name": q["name"], "probabilities": [{"value": k, "probability": v} for k, v in probs.items()]})
    return out


class FakeServer:
    def __init__(self):
        self.requests = []      # (url, headers, body)
        self.override = None    # optional callable(url, body) -> (status, json) to inject errors

    def respond(self, url, headers, body):
        self.requests.append((url, headers, body))
        if self.override:
            r = self.override(url, body)
            if r is not None:
                return r
        usage = {"input_tokens": 10, "output_tokens": 1}
        if "typesafe" in url:
            return 200, {"model": body["model"], "answers": decision_answers(body), "usage": usage}
        if "perplexity" in url:
            return 200, {"model": body["model"], "answers": decision_answers(body), "usage": usage}
        if "cloudflare" in url:
            return 200, {"success": True, "result": {"model": body["model"], "answers": decision_answers(body), "usage": usage}}
        if "openai" in url:
            return 200, {"model": body["model"], "answers": openai_answers(body), "usage": usage}
        return 404, {"error": "no route"}

    def bodies(self):
        return [b for _, _, b in self.requests]


@pytest.fixture
def server(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    srv = FakeServer()

    def handler(lib):
        def h(req):
            status, payload = srv.respond(str(req.url), dict(req.headers), json.loads(req.content))
            return lib.Response(status, json=payload)
        return h

    monkeypatch.setattr("hev_rerank._common.time.sleep", lambda s: None)
    orig = {m: m.connect for m in (jev, perplexity, clef, openai)}
    monkeypatch.setattr(jev, "connect", lambda: orig[jev](transport=httpx2.MockTransport(handler(httpx2))))
    for m in (perplexity, clef, openai):
        monkeypatch.setattr(m, "connect", lambda m=m: orig[m](transport=httpx.MockTransport(handler(httpx))))
    return srv
