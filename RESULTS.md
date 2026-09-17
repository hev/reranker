# Jev as a reranker — benchmark results

Generated 2026-09-17. Jev model jev-1.13.0.

BM25 top-K shortlist per query (rank_bm25, stopworded), every reranker permutes the same shortlist. nDCG@10 via ir_measures. Latency is per query (one API call for the single-state shapes; per pair call for the pair shape). Cost is the run's actual token usage at list price ($0.042/M input for Jev; Cohere $2/1k searches; Claude list prices).

## scifact / test — 300 queries

| system | nDCG@10 | P@1 | MRR@10 | p50 | p95 | cost/run |
|---|---|---|---|---|---|---|
| jev_pair_generic | 0.772 | 0.710 | 0.756 | 122 ms | 236 ms | $0.28 |
| jev_batch_generic | 0.768 | 0.697 | 0.749 | 224 ms | 1758 ms | $0.18 |
| jev_choice | 0.762 | 0.710 | 0.754 |  |  |  |
| jev_choice_generic | 0.760 | 0.693 | 0.746 | 192 ms | 1621 ms | $0.15 |
| llm_opus5_low | 0.756 | 0.683 | 0.740 | 4950 ms | 7473 ms | $28.40 |
| jev_batch | 0.756 | 0.683 | 0.739 |  |  |  |
| jev_pair | 0.755 | 0.680 | 0.740 |  |  |  |
| rerank_voyage | 0.755 | 0.657 | 0.734 | 187 ms | 286 ms | $0.21 |
| rerank_voyage3 | 0.755 | 0.660 | 0.734 | 192 ms | 288 ms | $0.21 |
| rerank_mixedbread | 0.749 | 0.670 | 0.731 | 379 ms | 460 ms | $1.05 |
| llm_gpt56_luna_none | 0.747 | 0.653 | 0.729 | 3223 ms | 5824 ms | $0.71 |
| rerank_cohere | 0.745 | 0.660 | 0.728 | 277 ms | 698 ms | $0.60 |
| llm_haiku45 | 0.723 | 0.633 | 0.701 | 2899 ms | 11562 ms | $4.18 |
| local_mxbai-rerank-base-v1 | 0.716 | 0.633 | 0.693 | 8148 ms | 9933 ms |  |
| local_ms-marco-MiniLM-L-6-v2 | 0.682 | 0.570 | 0.653 | 211 ms | 242 ms |  |
| bm25 | 0.667 | 0.540 | 0.634 |  |  |  |

## scifact / train — 300 queries

| system | nDCG@10 | P@1 | MRR@10 | p50 | p95 | cost/run |
|---|---|---|---|---|---|---|
| jev_choice_generic | 0.776 | 0.713 | 0.766 | 253 ms | 1157 ms | $0.26 |
| jev_batch_generic | 0.774 | 0.683 | 0.751 | 298 ms | 1721 ms | $0.30 |
| jev_pair_generic | 0.773 | 0.673 | 0.754 | 117 ms | 221 ms | $0.28 |
| jev_pair | 0.767 | 0.693 | 0.740 | 126 ms | 239 ms | $0.28 |
| jev_choice | 0.756 | 0.690 | 0.747 | 202 ms | 1657 ms | $0.16 |
| jev_batch | 0.752 | 0.667 | 0.732 | 242 ms | 1738 ms | $0.18 |
| ce_minilm | 0.708 | 0.573 | 0.671 |  |  |  |
| bm25 | 0.670 | 0.567 | 0.640 |  |  |  |

## nfcorpus / test — 323 queries

| system | nDCG@10 | P@1 | MRR@10 | p50 | p95 | cost/run |
|---|---|---|---|---|---|---|
| jev_batch_generic | 0.358 | 0.511 | 0.581 | 238 ms | 1716 ms | $0.19 |
| jev_pair_generic | 0.358 | 0.523 | 0.590 | 144 ms | 295 ms | $0.30 |
| rerank_voyage3 | 0.357 | 0.514 | 0.587 | 191 ms | 285 ms | $0.23 |
| rerank_voyage | 0.357 | 0.517 | 0.584 | 183 ms | 300 ms | $0.23 |
| llm_gpt56_luna_none | 0.355 | 0.508 | 0.589 | 3062 ms | 4521 ms | $0.76 |
| rerank_cohere | 0.339 | 0.483 | 0.561 | 170 ms | 404 ms | $0.65 |
| ce_minilm | 0.336 | 0.492 | 0.567 |  |  |  |
| rerank_mixedbread | 0.324 | 0.424 | 0.528 | 380 ms | 452 ms | $1.13 |
| jev_choice_generic | 0.315 | 0.514 | 0.580 | 212 ms | 1724 ms | $0.17 |
| bm25 | 0.310 | 0.440 | 0.521 |  |  |  |

## fiqa / test — 300 queries

| system | nDCG@10 | P@1 | MRR@10 | p50 | p95 | cost/run |
|---|---|---|---|---|---|---|
| rerank_voyage3 | 0.402 | 0.490 | 0.522 | 173 ms | 288 ms | $0.15 |
| rerank_voyage | 0.395 | 0.483 | 0.517 | 177 ms | 255 ms | $0.15 |
| jev_pair_generic | 0.376 | 0.440 | 0.486 | 128 ms | 238 ms | $0.23 |
| jev_batch_generic | 0.376 | 0.443 | 0.483 | 208 ms | 785 ms | $0.13 |
| rerank_cohere | 0.374 | 0.430 | 0.482 | 129 ms | 267 ms | $0.60 |
| llm_gpt56_luna_none | 0.363 | 0.393 | 0.462 | 3131 ms | 3774 ms | $0.49 |
| jev_choice_generic | 0.358 | 0.420 | 0.480 | 191 ms | 772 ms | $0.10 |
| rerank_mixedbread | 0.354 | 0.380 | 0.447 | 322 ms | 374 ms | $1.05 |
| ce_minilm | 0.323 | 0.330 | 0.407 |  |  |  |
| bm25 | 0.234 | 0.200 | 0.279 |  |  |  |

## Validation (scifact/test)

- Determinism: 30 batch requests re-issued, 900 Noul pairs: mean |Δ| 0.004, max 0.14, zero top-1 changes.
- Position bias: reversed candidate order, per-doc Spearman 0.83 (30-Noul state) / 0.92 (Choice); nDCG@10 within 0.005.
- Calibration (pair Nouls): p≥0.9 → 76% judged relevant; p<0.1 → 0.5%.
- Random permutation of the shortlist: nDCG@10 0.12 (floor).
- Claude Opus 5 refused 12/300 claims (safety classifier); scored as zero in the table, 0.778 with BM25 order substituted on those queries.
- Jev rate limit: two concurrent jobs (~24 in flight) drew sustained 429s; runs were serialized at 6-wide.
