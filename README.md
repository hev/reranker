# hev reranker

Decision models as rerankers: one call, any model, any request shape.

## The talk

*Decision Models as Rerankers*, a guest talk in [AI-Powered Search](https://maven.com/search-school/ai-powered-search),
Doug Turnbull and Trey Grainger's Maven course (November 23, 2026).

- **Slides:** [hevmind.com/decks/decision-models-as-rerankers](https://hevmind.com/decks/decision-models-as-rerankers/)
- **Notebooks**, companions to the slides: each request shape by hand first, as on the slides, then with `rerank()`.
  - [Free Colab GPU, no API keys](notebooks/colab_free_tier.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hev/reranker/blob/main/notebooks/colab_free_tier.ipynb): runs an open decision model on a T4 in about 15 minutes.
  - [Hosted walkthrough](notebooks/hosted_walkthrough.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hev/reranker/blob/main/notebooks/hosted_walkthrough.ipynb): Jev, Perplexity, Clef and OpenAI with your own keys.
- **Benchmark results:** [hevmind.com/research/rerankers](https://hevmind.com/research/rerankers/) compares every decision model
  and purpose-built reranker on the same BEIR shortlists: quality, price and latency. Code and raw runs are in
  [hev/rerank-bench](https://github.com/hev/rerank-bench).

## Quick start

```python
from hev_rerank import rerank

hits = rerank(query, docs, model="jev", style="auto", top_n=10)
```

Feed in the query and the candidate list (strings, or `{"title", "text"}` mappings); get back
`RerankResult(index, score, document)` best-first. Change `model=` to try another decision model,
`style=` to try another request shape. The deck's advice, *benchmark the shape on your model*, is a
one-argument change.

## Models and styles

Every model answers *is this document relevant to the query?* in one of three shapes:

- **pair**: one call per (query, document), one yes/no probability each.
- **batch**: the query and up to 30 documents in one state, one yes/no probability per document,
  evaluated in parallel.
- **choice**: one call that picks the best document; scores are the options' probabilities and sum to one.

`style="auto"` (the default) picks each model's best shape in the benchmark
([results](https://hevmind.com/research/rerankers/)).

| `model=` | Vendor model | Env vars | `auto` picks | Why |
|---|---|---|---|---|
| `"jev"` (default) | TypeSafe `jev-latest` | `TYPESAFE_API_KEY` | batch | As good as pair with far fewer calls; choice slips when many documents are relevant |
| `"perplexity"` | `pplx-decider-v1-27b` | `PERPLEXITY_API_KEY` | choice | Its best shape on quality, and the fastest |
| `"clef"` | Cloudflare `clef` | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` | pair | Pair is the shape that works for Clef |
| `"clef-flash"` | Cloudflare `clef-flash` | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` | pair | Batch and choice fall below the BM25 order |
| `"openai"` | OpenAI Decisions `gpt-6-luna` | `OPENAI_API_KEY` | choice | Clearly its best shape |

Any vendor model id works as `"vendor:model-id"`, e.g.
`model="perplexity:pplx-decider-v1.3"` or `model="clef:clef-flash"`; `auto` then uses that vendor's row.
Jev goes through `typesafe-sdk`; the others are plain HTTPS (`httpx`). A missing key raises
`MissingAPIKeyError` naming the variable.

**Choice can't be thresholded.** Pair and batch scores are absolute probabilities: use them to sort
*and* to prune (`threshold=0.1`). Choice probabilities sum to one across the list, so they only say
which documents beat the others; `threshold=` with an effective style of `choice` raises `ValueError`.
That includes `style="auto"` on Perplexity and OpenAI: pass `style="batch"` (or `"pair"`) to prune.

**Long lists.** Batch splits lists longer than `max_docs_per_call` (30) or `max_state_chars` into
concurrent calls; the scores stay comparable because each is an absolute probability. Pair sends one
call per document, `concurrency` (8) at a time. Choice is always one call, because probabilities from
separate calls aren't comparable: a list over the limits raises `ValueError` (raise the limits if your
model takes more, or use batch). A one-document choice returns probability 1 without a call.

This repo is the whole recipe: the prompts ([Jev](hev_rerank/prompt.yaml),
[the others](hev_rerank/prompt_decisions.yaml)), the request [state shape](schema/state.schema.json),
and one small module per vendor behind [`rerank()`](hev_rerank/rerank.py).

## Self-hosted: `model="local:<hf-id>"`

```sh
pip install "hev-rerank[local]"     # adds torch, transformers; the plain install never pulls them
```

```python
hits = rerank(query, docs, model="local:perplexity-ai/pplx-decider-flash-v1.3", style="batch")
```

Downloads open weights from Hugging Face and runs them in-process with `transformers`, on the GPU if there is
one, with the same prompts and request shape as the hosted Perplexity and Clef providers. No key. The first call
loads the model; later calls reuse it. `"local:"` alone is the default model below. Supported checkpoints run
through the inference code their repos ship:

| `model=` | Weights | Licence | Fits | `auto` picks |
|---|---|---|---|---|
| `"local:perplexity-ai/pplx-decider-flash-v1.3"` (default) | 4B, 8.5 GB | Apache-2.0 | free Colab T4 (float16) | choice |
| `"local:perplexity-ai/pplx-decider-v1.3"` | 27B | Apache-2.0 | 80 GB GPU | choice |
| `"local:Cloudflare/clef-flash"` | 9B | Apache-2.0 | L4 / A100 (bf16) | pair |
| `"local:Cloudflare/clef"` | 27B | Apache-2.0 | L4 in 4-bit, or 80 GB | pair |

Unmeasured checkpoints get `auto` = batch. For control over device, dtype or 4-bit loading, load the model yourself
and pass it as the client:

```python
from hev_rerank import local
m = local.load("Cloudflare/clef", quantization_config=BitsAndBytesConfig(load_in_4bit=True))
rerank(query, docs, model="local:Cloudflare/clef", client=m)
```

On GPUs without bfloat16 (T4) weights load in float16. Batch requests that run out of GPU memory are split in half
and retried (batch scores are absolute); choice can't be split and raises `ProviderError`. Pair sends every
(query, document) request to the model in one padded list. Self-hosted latency depends on your GPU and is not
comparable with the hosted APIs': [colab_free_tier.ipynb](notebooks/colab_free_tier.ipynb) measures it on a T4.

## Benchmark results

The full comparison lives at **[hevmind.com/research/rerankers](https://hevmind.com/research/rerankers/)**: every decision
model in every shape against Cohere, Voyage and Mixedbread, all reranking the same BM25 top 30 on SciFact, NFCorpus
and FiQA, with nDCG@10, price per 1,000 queries and latency. The short version: the best decision models match the
best purpose-built rerankers, and return a calibrated probability per document that rerankers don't. The write-ups:

- [Hev meets Jev](https://hevmind.com/writing/jev-as-a-reranker/)
- [Jev has company](https://hevmind.com/writing/jev-has-company/)
- [OpenAI makes four](https://hevmind.com/writing/openai-makes-four/)
- [Reranking just got cheaper](https://hevmind.com/writing/perplexity-decider-v1-2/)

Code and raw runs: [hev/rerank-bench](https://github.com/hev/rerank-bench). Jev's original runs, with confidence
intervals, determinism, position-bias and calibration checks: [RESULTS.md](RESULTS.md).

## Install

```sh
pip install hev-rerank        # or: uv add hev-rerank
export TYPESAFE_API_KEY=...   # https://console.typesafe.ai/settings/keys (and the keys of any other model you use)
```

## Use

```python
from hev_rerank import rerank

hits = rerank(
    "does vitamin D supplementation improve bone density in older adults",
    [{"title": t, "text": x} for t, x in first_stage_results],   # strings work too
    model="jev",       # or "perplexity", "clef", "clef-flash", "openai"
    style="auto",      # or "pair", "batch", "choice"
    top_n=10,          # optional
    threshold=0.1,     # optional, pair/batch only: drop candidates the model is confident are irrelevant
)
for h in hits:
    print(h.score, h.index, h.document["title"])
```

`prompt=` overrides any of `question` (batch), `pair_question`, `choice_question` and `criteria`;
keys you leave out keep the model's default. `client=` takes a ready client (a `TypeSafeClient` for
Jev, an authenticated `httpx.Client` with `base_url` for the others).
[`examples/compare.py`](examples/compare.py) runs every model on one query; pass
`pair batch choice` to try every shape. On the six-passage vitamin D example (2026-11-15, one live
call per model in its `auto` style):

| model | style | latency | D00 | D01 | D02 | D03 | D04 | D05 | order |
|---|---|---|---|---|---|---|---|---|---|
| jev | batch | 275 ms | 0.98 | 0.06 | 0.04 | 0.24 | 0.94 | 0.03 | D00 > D04 > D03 > D01 > D02 > D05 |
| perplexity | choice | 292 ms | 0.81 | 0.00 | 0.00 | 0.00 | 0.18 | 0.00 | D00 > D04 > D03 > D01 > D05 > D02 |
| clef | pair | 701 ms | 0.97 | 0.06 | 0.03 | 0.37 | 0.97 | 0.03 | D04 > D00 > D03 > D01 > D05 > D02 |
| clef-flash | pair | 319 ms | 0.87 | 0.09 | 0.08 | 0.19 | 0.88 | 0.05 | D04 > D00 > D03 > D01 > D02 > D05 |
| openai | choice | 1266 ms | 0.94 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 | D00 > D04 > D01 > D02 > D03 > D05 |

Wall time of the whole `rerank()` call from one client, including connection setup; pair is six
concurrent calls. One query, not a benchmark.

## Jev's shape, and why

- **State**: `{"query": ..., "documents": {"D00": {...}, "D01": {...}, ...}}`
  ([schema](schema/state.schema.json)).
- **Questions**: one `Noul` per document id, from [`prompt.yaml`](hev_rerank/prompt.yaml).
  All questions in a request are evaluated in parallel, so 30 documents cost about the same
  wall time as 5.
- **Score**: the Noul probability. Calibration on the SciFact run: documents scored ≥ 0.9
  were judged relevant 76% of the time; documents scored < 0.1, 0.5% of the time.

Why not one `Choice` over the documents? It is slightly better when exactly one document is
relevant (SciFact: +0.01 nDCG@10, +3 pts P@1) and much worse when many are (NFCorpus,
median 16 relevant per query: 0.316 vs 0.358). For Jev, a reranker that does not know the
corpus should ask per-document Nouls. Perplexity and OpenAI measured the other way round (table above).

## Limits

- The five named models are hosted APIs: the data leaves your environment (use `local:` to keep it in).
  OpenAI Decisions is a public beta; its numbers may shift.

Jev specifically:

- ~32k tokens per request (state + questions): ~30–50 typical passages per call.
- Rate limits are undocumented; sustained 429s appeared at ~24 requests in flight.
- Tail latency on 30-document states (p95 0.8–1.8 s) is higher than specialized hosted
  rerankers'. Fanning out per-pair calls trades 30× the calls for a p95 near 0.25 s per call.

## License

Apache-2.0.
