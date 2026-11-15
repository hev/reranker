import pytest

from hev_rerank import MODELS, MissingAPIKeyError, ProviderError, RerankResult, load_prompt, rerank
from hev_rerank.rerank import resolve_model

Q = "does vitamin D supplementation improve bone density in older adults"
DOCS = [
    "noise: sunscreen and skin",
    "relevant: vitamin D3 with calcium raised hip bone density",
    {"title": "Astronauts", "text": "bone loss in microgravity"},
    {"title": "Trial", "text": "relevant: meta-analysis of 23 trials", "url": "kept-but-not-sent"},
]
ALL_MODELS = ["jev", "perplexity", "clef", "clef-flash", "openai"]
HOST = {"jev": "api.typesafe.ai/v1/systemone", "perplexity": "api.perplexity.ai/v1/decisions",
        "clef": "api.cloudflare.com/client/v4/accounts/acct123/ai/run/@cf/cloudflare/clef",
        "clef-flash": "api.cloudflare.com/client/v4/accounts/acct123/ai/run/@cf/cloudflare/clef-flash",
        "openai": "api.openai.com/v1/decisions"}
AUTH = {"jev": "ts-test", "perplexity": "pplx-test", "clef": "cf-test", "clef-flash": "cf-test", "openai": "oa-test"}
AUTO = {"jev": "batch", "perplexity": "choice", "clef": "pair", "clef-flash": "pair", "openai": "choice"}


def shape_of(model, body):
    """Which request shape a recorded body is."""
    if model == "openai":
        qs = body["questions"]
        if qs[0]["type"] == "choice":
            return "choice"
        return "pair" if "Document:\n" in body["input"] else "batch"
    qs = list(body["questions"].values())
    if qs[0]["type"] == "choice":
        return "choice"
    return "pair" if "document" in body["state"] else "batch"


@pytest.mark.parametrize("style", ["pair", "batch", "choice", "auto"])
@pytest.mark.parametrize("model", ALL_MODELS)
def test_every_model_and_style(server, model, style):
    hits = rerank(Q, DOCS, model=model, style=style)
    want = AUTO[model] if style == "auto" else style

    assert all(isinstance(h, RerankResult) for h in hits)
    assert sorted(h.index for h in hits) == [0, 1, 2, 3]
    assert {h.index for h in hits[:2]} == {1, 3}                  # the two "relevant" docs first
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(h.document is DOCS[h.index] for h in hits)
    if want == "choice":
        assert sum(h.score for h in hits) == pytest.approx(1.0)
        assert hits[0].score == pytest.approx(0.9 / 2.0)
    else:
        assert hits[0].score == pytest.approx(0.9) and hits[-1].score == pytest.approx(0.1)

    assert len(server.requests) == (4 if want == "pair" else 1)
    for url, headers, body in server.requests:
        assert HOST[model] in url
        assert headers["authorization"] == f"Bearer {AUTH[model]}"
        assert body["model"] == MODELS[model][1]
        assert shape_of(model, body) == want
        assert "kept-but-not-sent" not in str(body)


def test_auto_table():
    assert {m: MODELS[m][2] for m in MODELS} == AUTO
    assert resolve_model("perplexity:pplx-decider-v1-27b") == ("perplexity", "pplx-decider-v1-27b", "choice")
    assert resolve_model("clef:some-new-clef") == ("clef", "some-new-clef", "pair")
    assert resolve_model("jev-latest") == ("jev", "jev-latest", "batch")     # 0.1.0's model= value
    with pytest.raises(ValueError, match="unknown model"):
        resolve_model("cohere")
    with pytest.raises(ValueError, match="unknown style"):
        rerank(Q, DOCS, style="listwise")


def test_explicit_vendor_model_id(server):
    rerank(Q, DOCS, model="perplexity:pplx-decider-v2", style="batch")
    assert server.bodies()[0]["model"] == "pplx-decider-v2"


def test_prompts_match_bench(server):
    """Jev gets prompt.yaml (bench jev generic); the others prompt_decisions.yaml (bench clef generic)."""
    rerank(Q, DOCS[:2], model="jev", style="batch")
    rerank(Q, DOCS[:2], model="jev", style="pair")
    rerank(Q, DOCS[:2], model="jev", style="choice")
    rerank(Q, DOCS[:2], model="clef", style="choice")
    rerank(Q, DOCS[:2], model="openai", style="batch")
    b = server.bodies()
    assert b[0]["questions"]["D01"]["instructions"] == "Document `documents.D01` is relevant to `query`: it contains information that answers or directly addresses it."
    assert len(b) == 6                                                  # pair: one call per document
    assert b[1]["questions"]["rel"]["instructions"].startswith("The document is relevant to the query")
    assert b[1]["questions"]["rel"]["criteria"]["true"].startswith("The document contains information")
    assert b[3]["questions"]["best"]["instructions"] == "Which document in `documents` best answers or most directly addresses `query`?"
    assert b[3]["questions"]["best"]["criteria"] == {"D00": "documents.D00", "D01": "documents.D01"}
    assert b[4]["questions"]["best"]["instructions"].startswith("Which document most directly answers or addresses the query")
    oa = b[5]["questions"][0]["instructions"]
    assert oa.startswith("Does document `documents.D00` directly answer") and " True: The document directly answers" in oa


# ---- 0.1.0 compatibility ----------------------------------------------------------------------------

def test_v010_default_call_is_jev_batch(server):
    hits = rerank(Q, DOCS)
    assert len(server.requests) == 1
    url, _, body = server.requests[0]
    assert "api.typesafe.ai/v1/systemone" in url and body["model"] == "jev-latest"
    assert set(body["state"]["documents"]) == {"D00", "D01", "D02", "D03"}
    assert body["state"]["documents"]["D03"] == {"title": "Trial", "text": "relevant: meta-analysis of 23 trials"}
    p = load_prompt()
    assert body["questions"]["D02"] == {"type": "noul", "instructions": p["question"].format(id="D02"), "criteria": p["criteria"]}
    assert [h.index for h in hits][:2] == [1, 3]


def test_v010_top_n_threshold(server):
    hits = rerank(Q, DOCS, top_n=1, threshold=0.5)
    assert [(h.index, h.score) for h in hits] == [(1, 0.9)]
    assert [h.index for h in rerank(Q, DOCS, threshold=0.5)] == [1, 3]
    assert shape_of("jev", server.bodies()[0]) == "batch"


def test_v010_keywords(server, monkeypatch):
    """client=, model="jev-latest", a 0.1.0 prompt dict, max_docs_per_call, max_state_chars, concurrency."""
    import httpx2
    from typesafe_sdk import TypeSafeClient
    from hev_rerank import jev
    client = jev.connect()                  # the fixture's mocked TypeSafeClient
    assert isinstance(client, TypeSafeClient)
    old_prompt = {"question": "Is `documents.{id}` on topic for `query`?", "criteria": {"true": "yes", "false": "no"}}
    hits = rerank(Q, DOCS, client=client, model="jev-latest", prompt=old_prompt,
                  max_docs_per_call=3, max_state_chars=10_000, concurrency=2)
    assert len(hits) == 4 and {h.index for h in hits[:2]} == {1, 3}
    assert len(server.requests) == 2
    assert server.bodies()[0]["questions"]["D00"] == {"type": "noul", "instructions": "Is `documents.D00` on topic for `query`?",
                                                      "criteria": {"true": "yes", "false": "no"}}


def test_empty_documents():
    assert rerank(Q, []) == []


# ---- choice + threshold, missing keys --------------------------------------------------------------

@pytest.mark.parametrize("model,style", [("jev", "choice"), ("perplexity", "auto"), ("openai", "auto"), ("clef", "choice")])
def test_choice_threshold_raises(server, model, style):
    with pytest.raises(ValueError, match="sum to one"):
        rerank(Q, DOCS, model=model, style=style, threshold=0.1)
    assert server.requests == []


@pytest.mark.parametrize("model,var", [
    ("jev", "TYPESAFE_API_KEY"), ("perplexity", "PERPLEXITY_API_KEY"), ("clef", "CLOUDFLARE_API_TOKEN"),
    ("clef-flash", "CLOUDFLARE_ACCOUNT_ID"), ("openai", "OPENAI_API_KEY")])
def test_missing_key_names_the_variable(server, monkeypatch, model, var):
    monkeypatch.delenv(var)
    with pytest.raises(MissingAPIKeyError, match=var):
        rerank(Q, DOCS, model=model)
    assert server.requests == []


# ---- long lists ------------------------------------------------------------------------------------

@pytest.mark.parametrize("model", ALL_MODELS)
def test_batch_splits_long_lists(server, model):
    docs = [f"doc {i}" + (" relevant" if i % 7 == 0 else "") for i in range(70)]
    hits = rerank(Q, docs, model=model, style="batch")
    assert len(server.requests) == 3
    sizes = sorted(len(b["questions"]) for b in server.bodies())
    assert sizes == [10, 30, 30]
    assert sorted(h.index for h in hits) == list(range(70))
    assert {h.index for h in hits[:10]} == {i for i in range(70) if i % 7 == 0}


def test_batch_splits_on_state_chars(server):
    docs = ["x" * 400] * 5
    rerank(Q, docs, max_state_chars=1000)
    # chunks go out concurrently, so arrival order isn't fixed
    assert sorted(len(b["questions"]) for b in server.bodies()) == [1, 2, 2]


@pytest.mark.parametrize("model", ALL_MODELS)
def test_choice_is_one_call_or_error(server, model):
    docs = [f"doc {i}" for i in range(31)]
    with pytest.raises(ValueError, match="one call"):
        rerank(Q, docs, model=model, style="choice")
    assert server.requests == []
    rerank(Q, docs, model=model, style="choice", max_docs_per_call=31)
    assert len(server.requests) == 1


def test_choice_single_document_needs_no_call(server):
    assert rerank(Q, ["only"], model="clef", style="choice") == [RerankResult(0, 1.0, "only")]
    assert server.requests == []


@pytest.mark.parametrize("model", ALL_MODELS)
def test_pair_one_call_per_document(server, model):
    docs = [f"doc {i}" for i in range(40)]
    rerank(Q, docs, model=model, style="pair", concurrency=4)
    assert len(server.requests) == 40


# ---- provider errors -------------------------------------------------------------------------------

def test_openai_refusals(server):
    def refuse(url, body):
        answers = [{"type": "refusal", "name": q["name"], "refusal": "no"} for q in body["questions"]]
        return 200, {"model": "gpt-6-luna", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}
    server.override = refuse
    assert [h.score for h in rerank(Q, DOCS, model="openai", style="batch")] == [0.0] * 4
    assert [h.score for h in rerank(Q, DOCS, model="openai", style="choice")] == [0.25] * 4


def test_retries_429_then_succeeds(server):
    seen = []
    def flaky(url, body):
        seen.append(1)
        return (429, {"error": "slow down"}) if len(seen) < 3 else None
    server.override = flaky
    hits = rerank(Q, DOCS, model="perplexity", style="batch")
    assert len(server.requests) == 3 and hits[0].score == 0.9


@pytest.mark.parametrize("model", ["perplexity", "clef", "openai"])
def test_http_error_raises(server, model):
    server.override = lambda url, body: (403, {"error": {"message": "beta access required"}})
    with pytest.raises(ProviderError, match="HTTP 403.*beta access required"):
        rerank(Q, DOCS, model=model, style="batch")


def test_clef_success_false(server):
    server.override = lambda url, body: (200, {"success": False, "errors": [{"code": 5007}]})
    with pytest.raises(ProviderError, match="success=false"):
        rerank(Q, DOCS, model="clef", style="batch")
