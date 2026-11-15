# Changelog

## 0.2.0 (2026-11-15)

- `rerank(query, docs, model=..., style=...)` covers every decision model in the
  [rerank benchmark](https://hevmind.com/research/rerankers): `"jev"`, `"perplexity"`
  (pplx-decider-v1-27b), `"clef"`, `"clef-flash"` (Cloudflare Workers AI), `"openai"` (OpenAI
  Decisions, gpt-6-luna, public beta), or any `"vendor:model-id"`.
- `style=`: `"pair"`, `"batch"`, `"choice"`, or `"auto"` (default), which picks each model's best
  measured shape: Jev batch, Perplexity choice, Clef pair, Clef-flash pair, OpenAI choice.
- Request shapes and prompts are ported from hev/rerank-bench's adapters (generic variant), one module
  per vendor. Non-Jev providers use plain HTTPS through `httpx`, keys from `PERPLEXITY_API_KEY`,
  `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`, `OPENAI_API_KEY`.
- `threshold=` with an effective style of `choice` raises `ValueError` (choice probabilities sum to
  one). Choice is always a single call; lists over `max_docs_per_call` / `max_state_chars` raise.
- A missing key raises `MissingAPIKeyError` naming the variable; provider HTTP errors raise
  `ProviderError`. 429s and 5xx are retried with backoff.
- `prompt=` accepts `pair_question` and `choice_question` alongside `question` and `criteria`.
- Default `concurrency` is 8 (was 4), so pair-style calls fan out further.
- Compatible with 0.1.0: `rerank(query, docs)` and every 0.1.0 keyword still give Jev batch, and
  `model="jev-latest"` still works.
- New: `examples/compare.py` (live run of every model and shape), mocked test suite, gated live smoke
  test (`pytest tests/test_live.py`).

## 0.1.0

- Jev as a reranker: one state, one Noul per document, batch splitting, the prompt and results.
