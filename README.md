# hev reranker

Decision models as rerankers: one call, any model, any request shape.

Notebooks: [hosted walkthrough](notebooks/hosted_walkthrough.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hev/reranker/blob/main/notebooks/hosted_walkthrough.ipynb)
· [free Colab GPU, no keys](notebooks/colab_free_tier.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hev/reranker/blob/main/notebooks/colab_free_tier.ipynb)

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

`style="auto"` (the default) picks each model's best measured shape: nDCG@10 on BM25 top-30 shortlists
of SciFact / NFCorpus / FiQA test, generic prompts
([hevmind.com/research/rerankers](https://hevmind.com/research/rerankers), code in
[hev/rerank-bench](https://github.com/hev/rerank-bench)).

| `model=` | Vendor model | Env vars | pair | batch | choice | `auto` picks | Why |
|---|---|---|---|---|---|---|---|
| `"jev"` (default) | TypeSafe `jev-latest` | `TYPESAFE_API_KEY` | 0.772 / 0.358 / 0.376 | **0.768 / 0.358 / 0.376** | 0.760 / 0.316 / 0.358 | batch | ties pair at 1/30 the calls; choice falls behind when many documents are relevant |
| `"perplexity"` | `pplx-decider-v1-27b` | `PERPLEXITY_API_KEY` | 0.752 / 0.343 / 0.374 | 0.770 / 0.352 / 0.377 | **0.776 / 0.347 / 0.382** | choice | best on SciFact and FiQA, 3–4× faster than batch (p50 0.5 s vs 1.4–2 s) |
| `"clef"` | Cloudflare `clef` | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` | **0.759 / 0.345 / 0.379** | | | pair | pair is the shape that works on Clef-flash; the full model was run in pair |
| `"clef-flash"` | Cloudflare `clef-flash` | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` | **0.769 / 0.355 / 0.372** | 0.443 / 0.228 / 0.177 | 0.511 / 0.260 / 0.216 | pair | batch and choice lose 0.10–0.33 nDCG |
| `"openai"` | OpenAI Decisions `gpt-6-luna` (public beta) | `OPENAI_API_KEY` | 0.670 / 0.306 / 0.346 | 0.681 / 0.323 / 0.346 | **0.751 / 0.311 / 0.355** | choice | +0.07 on SciFact, +0.01 on FiQA, −0.01 on NFCorpus |

nDCG@10 per cell: SciFact / NFCorpus / FiQA. Any vendor model id works as `"vendor:model-id"`, e.g.
`model="perplexity:pplx-decider-v1-27b"` or `model="clef:clef-flash"`; `auto` then uses that vendor's row.
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
one small module per vendor behind [`rerank()`](hev_rerank/rerank.py), and the [results](RESULTS.md)
of running Jev against hosted and open-weight rerankers and LLM judges on BEIR shortlists.

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
| `"local:perplexity-ai/pplx-decider-flash-v1.3"` (default) | 4B, 8.5 GB | Apache-2.0 | free Colab T4 (float16) | choice (the only shape benchmarked on it: 0.502 three-corpus mean, hosted) |
| `"local:perplexity-ai/pplx-decider-v1.3"` | 27B | Apache-2.0 | 80 GB GPU | choice (0.511 vs batch 0.507, pair 0.495) |
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

## Jev against purpose-built rerankers

nDCG@10, every system permuting the same BM25 top-30 shortlist per query. Jev uses the
domain-neutral phrasing in `prompt.yaml`, untuned per corpus.

| Reranker | SciFact (300 q) | NFCorpus (323 q) | FiQA (300 q) | p50 / p95 per query | Cost per 300 q |
|---|---|---|---|---|---|
| BM25 order, no rerank | 0.667 | 0.310 | 0.234 | | |
| MiniLM-L6 cross-encoder (local) | 0.682 | 0.336 | 0.323 | | |
| Cohere rerank-v3.5 | 0.745 | 0.340 | 0.374 | 129–277 ms / 267–698 ms | $0.60 |
| Mixedbread mxbai-rerank-large-v2 (hosted) | 0.749 | 0.324 | 0.354 | 322–380 ms / 374–460 ms | $1.05 |
| Voyage rerank-2.5 | 0.755 | 0.357 | 0.395 | 177–187 ms / 255–300 ms | ~$0.15 |
| Voyage rerank-3 | 0.755 | 0.357 | 0.402 | 173–192 ms / 285–288 ms | ~$0.15 |
| gpt-5.6-luna (reasoning off), listwise | 0.747 | 0.355 | 0.363 | 3.1–3.2 s / 3.8–5.8 s | ~$0.71 |
| Claude Haiku 4.5, listwise | 0.723 | | | 2.9 s / 11.6 s | $4.18 |
| Claude Opus 5 (low effort), listwise | 0.756 | | | 5.0 s / 7.5 s | $28.40 |
| **Jev, 30 Nouls in one state** | **0.768** | **0.358** | 0.376 | 208–238 ms / 0.8–1.8 s | $0.13–0.19 |
| Jev, one Noul per (query, doc) pair | 0.772 | 0.358 | 0.376 | 122–144 ms per call | $0.23–0.30 |

Paired-bootstrap per query, Jev's one-state shape against each reranker: at or above Cohere on
all three corpora (two wins, one tie); above Mixedbread (two wins, one interval touching zero);
trading with Voyage, the strongest purpose-built reranker here (edge on SciFact, tie on NFCorpus,
loss on FiQA by ~0.02); above every LLM used as a reranker on cost and latency, and on quality
except Opus 5, which it ties at ~1/100 the cost.

The honest one-liner: a general decision model with no reranker training lands in the same
quality and price bracket as the best purpose-built rerankers, a third of Cohere's price, and
returns a calibrated probability per document that none of them do. Voyage rerank-3 is as
good and slightly cheaper; what Jev adds over it is the probability and the fact that the same
call shape covers routing, classification and extraction gating.

Costs are the runs' actual token usage at list price (Voyage measured from its billed
`usage.total_tokens`). Full tables, confidence intervals, determinism, position-bias and
calibration checks: [RESULTS.md](RESULTS.md).

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
