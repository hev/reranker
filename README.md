# hev reranker

A reranker built on Jev, by hev.

Use [Jev](https://docs.typesafe.ai), TypeSafe's System One decision model, as a reranker.
One call carries the query and up to 30 candidate documents; the model answers one
question per document, in parallel: *is this document relevant to the query?* The answer is
a calibrated probability, so the score is both a sort key (rerank) and an absolute
threshold (prune).

This repo is the whole recipe: the [prompt](hev_rerank/prompt.yaml), the request
[state shape](schema/state.schema.json), a 90-line Python wrapper, and the
[results](RESULTS.md) of running it against hosted and open-weight rerankers and LLM
judges on BEIR shortlists.

## Results

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
export TYPESAFE_API_KEY=...   # https://console.typesafe.ai/settings/keys
```

## Use

```python
from hev_rerank import rerank

hits = rerank(
    "does vitamin D supplementation improve bone density in older adults",
    [{"title": t, "text": x} for t, x in first_stage_results],   # strings work too
    top_n=10,          # optional
    threshold=0.1,     # optional: drop candidates the model is confident are irrelevant
)
for h in hits:
    print(h.score, h.index, h.document["title"])
```

`rerank` returns `RerankResult(index, score, document)` best-first. Lists longer than
`max_docs_per_call` (30) are split into concurrent calls under Jev's ~32k-token request
budget. Scores across calls are comparable because each is an absolute probability.

## The shape, and why

- **State**: `{"query": ..., "documents": {"D00": {...}, "D01": {...}, ...}}`
  ([schema](schema/state.schema.json)).
- **Questions**: one `Noul` per document id, from [`prompt.yaml`](hev_rerank/prompt.yaml).
  All questions in a request are evaluated in parallel, so 30 documents cost about the same
  wall time as 5.
- **Score**: the Noul probability. Calibration on the SciFact run: documents scored ≥ 0.9
  were judged relevant 76% of the time; documents scored < 0.1, 0.5% of the time.

Why not one `Choice` over the documents? It is slightly better when exactly one document is
relevant (SciFact: +0.01 nDCG@10, +3 pts P@1) and much worse when many are (NFCorpus,
median 16 relevant per query: 0.316 vs 0.358). A reranker that does not know the corpus
should ask per-document Nouls.

## Limits

- ~32k tokens per request (state + questions): ~30–50 typical passages per call.
- Rate limits are undocumented; sustained 429s appeared at ~24 requests in flight.
- Tail latency on 30-document states (p95 0.8–1.8 s) is higher than specialized hosted
  rerankers'. Fanning out per-pair calls trades 30× the calls for a p95 near 0.25 s per call.
- Hosted only; the data leaves your environment.

## License

Apache-2.0.
