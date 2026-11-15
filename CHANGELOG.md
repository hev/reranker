# Changelog

## Unreleased

- Notebooks only, no package change (they still install `@v0.3.0`): both notebooks now follow the talk *Decision
  Models as Rerankers*. Part A builds the three request shapes by hand on the slides' vitamin-D example (the slides'
  exact `typesafe_sdk` code on Jev in the hosted notebook; the same typed questions sent straight to
  `pplx-decider-flash-v1.3`'s reference code in the free-tier one). Part B repeats them with `rerank()` and shows
  the same order, then SciFact, the choice-threshold error and a custom prompt. `notebooks/build.py` refuses to
  overwrite an executed notebook without `--force`.

## 0.3.0 (2026-11-15)

- Self-hosted decision models: `rerank(query, docs, model="local:<hf-id>", style=...)` runs open weights in-process
  with `transformers`, GPU or CPU, in pair, batch, choice or auto, with the same prompts and request shape as the
  hosted Perplexity and Clef providers. Supported: Perplexity deciders (`perplexity-ai/pplx-decider-flash-v1.3`,
  the default for `"local:"`, and `pplx-decider-v1.3`) and Cloudflare Clef (`Cloudflare/clef-flash`,
  `Cloudflare/clef`), each through the inference code its repository ships.
- New optional extra: `pip install "hev-rerank[local]"` (torch, transformers, huggingface-hub). The plain install
  is unchanged and does not pull torch or transformers; they are imported only on the first local call.
- `style="auto"` for local models is the best shape rerank-bench measured for those weights (`LOCAL_AUTO`:
  choice for the Perplexity deciders, pair for Clef), else batch. The hosted `MODELS` table is unchanged.
- On GPUs without bfloat16 (T4, free Colab) weights load in float16. `hev_rerank.local.load()` takes `device`,
  `dtype`, `mode` and loader options (e.g. a 4-bit `quantization_config` for Clef); pass the result as `client=`.
- Local batch requests that run out of GPU memory are split in half and retried; local pair sends all of a
  query's pair requests to the model as one padded list.
- Notebooks: `notebooks/hosted_walkthrough.ipynb` (every hosted model and shape on SciFact, calibration, the
  choice-threshold error, a custom prompt) and `notebooks/colab_free_tier.ipynb` (no keys, free Colab T4),
  committed with executed outputs; `scripts/t4/` launches and terminates the g4dn.xlarge used to execute it.

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
