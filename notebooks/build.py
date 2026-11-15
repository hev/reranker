"""Writes the two notebooks in this directory (unexecuted). Edit here, rerun, then execute them:

    python notebooks/build.py [colab_free_tier.ipynb | hosted_walkthrough.ipynb] [--force]   # default: both;
        # refuses to overwrite a notebook that has executed outputs unless --force (back it up first)
    papermill notebooks/colab_free_tier.ipynb notebooks/colab_free_tier.ipynb -p HEV_RERANK_REF <sha>   # on a T4
    papermill notebooks/hosted_walkthrough.ipynb notebooks/hosted_walkthrough.ipynb -p HEV_RERANK_REF <sha>
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = "hev/reranker"


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name})")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src, tags=None):
    meta = {"tags": tags} if tags else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": src.strip("\n")}


def write(name, cells, gpu=False):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if gpu:
        meta.update(accelerator="GPU", colab={"provenance": [], "gpuType": "T4"})
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"c{i:02d}"
    (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


TALK = "the talk *Decision Models as Rerankers* (Adam Hevenor, [hevmind.com](https://hevmind.com))"

PART1 = md("""
## Part 1 · History: retrieve, then rerank

BM25 (or any first-stage retriever) is cheap enough to score a whole corpus, so it *retrieves*: it cuts millions of
documents down to a shortlist of 20–100. A reranker is too expensive for the whole corpus, so it only *reorders the
cut*. Two metrics tell those jobs apart:

- **Recall@k**: how many of the relevant documents made it into the top k. For the shortlist as a whole this is fixed
  by the retriever; a reranker can't bring back a document BM25 missed.
- **nDCG@k**: whether the relevant documents are *at the top*, discounted by rank. Reordering the cut is exactly what
  moves it (and recall@5 too, by pulling relevant documents up from places 6–30).

Part 3 measures them on SciFact: BM25's recall over the shortlist (the ceiling), and nDCG@10 before and after
reranking.
""")

PART_A = md("""
# Part A · By hand

The three request shapes from Part 2 of the deck, written out by hand on the deck's six-passage example. No
`hev-rerank` yet: it's imported in Part B.
""")

EXAMPLE = '''
query = "does vitamin D supplementation improve bone density in older adults"
docs = {"D00": "Meta-analysis of 23 trials: vitamin D alone did not meaningfully increase bone density in older adults.",
        "D01": "Dietary sources of vitamin D include oily fish, egg yolk and fortified milk. Many older adults fall short.",
        "D02": "Sunscreen reduces vitamin D synthesis in the skin only marginally in practice.",
        "D03": "Vitamin D increases intestinal calcium absorption. Deficiency impairs bone mineralization.",
        "D04": "Randomised trial: daily vitamin D3 with calcium raised hip bone density in adults over 65 versus placebo.",
        "D05": "Astronauts lose 1 to 2 percent of bone density per month in microgravity."}   # query results

part_a = {}   # each shape's scores, kept to compare with hev-rerank in Part B

def show(scores, ranked):
    for d in ranked:
        print(f"  {d}  {scores[d]:5.2f}  {docs[d][:85]}")
'''

PAIR_MD = md("""
## Part 2 · Pairwise: one call per document

Slide *"Typed questions in, probabilities out"*. One request per (query, document) with one **Noul**, a true-or-false
question answered as a probability. The answer is an absolute `p_relevant`: it means the same thing whatever else is
in the list.
""")

BATCH_MD = md("""
## Part 2 · Listwise batch: one call, every document

Slide *"One call, every document"*. All six passages go into one state, keyed `D00`…`D05`, with one Noul per
document that points at its key (`documents.D00`). Still an absolute `p_relevant` per document, from one call.
""")

CHOICE_MD = md("""
## Part 2 · Listwise choice: which is best?

Slide *"One call, one question: which is best?"*. One **Choice** whose options are the documents. The answer is
`p_best`: one unit of probability spread across the list, so it sums to one.
""")

WHEN_MD = md("""
## Part 2 · When to use which

| shape | calls for N docs | score | threshold? | use it when |
|---|---|---|---|---|
| **pairwise** | N | `p_relevant`, absolute | yes | you want the most robust per-document judgement and calls are cheap or parallel |
| **listwise batch** | 1 (or a few) | `p_relevant`, absolute | yes | you want pairwise's absolute scores at a fraction of the calls; the model also sees the other candidates |
| **listwise choice** | 1, always | `p_best`, sums to one | **no** | you need the single best answer, or few documents are relevant |

Pairwise and batch say *how likely is this one relevant*, so several documents can be near 1 (D00 and D04 here) and a
threshold prunes the rest. Choice says *which one wins*: D00 and D04 split one unit between them, and a document's
`p_best` depends on what else is in the list, so a fixed cut-off means nothing. Which shape ranks best depends on the
model: benchmark the shape on yours (Part 3).
""")

PART_B = md("""
# Part B · The same three shapes with hev-rerank

[`hev-rerank`](https://github.com/hev/reranker) wraps everything in Part A behind one call:
`rerank(query, docs, model=..., style=...)`, returning `RerankResult(index, score, document)` best-first. `style=` is
the shape: `"pair"`, `"batch"` or `"choice"`.

`prompt=` sets the wording. `slides` below is Part A's wording, so Part B asks exactly Part A's questions. (The
library's own default wording differs slightly per model; Part 3 uses the default.)
""")

SLIDES_PROMPT = '''
slides = {
    "pair_question": "`document` is relevant to `query`: it contains information that answers or directly addresses it.",
    "question": "Document `documents.{id}` is relevant to `query`: it contains information that answers or directly addresses it.",
    "choice_question": "Which document in `documents` best answers or most directly addresses `query`?",
    "criteria": {"true": "The document contains information that answers the query or directly addresses what it asks about.",
                 "false": "The document is only loosely related, on a similar topic, or does not address what the query asks."},
}
ids, texts = list(docs), list(docs.values())    # rerank takes a list; its result's .index is the position in it

def compare(style, hits):
    """Part B's scores and order next to Part A's for the same shape."""
    b = {ids[h.index]: h.score for h in hits}
    a = part_a[style]
    order_a, order_b = sorted(ids, key=a.get, reverse=True), [ids[h.index] for h in hits]
    print(f"{style}:  doc   Part A  Part B")
    for d in order_b:
        print(f"         {d}  {a[d]:6.3f}  {b[d]:6.3f}")
    print(f"  Part A order {order_a}\\n  Part B order {order_b}   same order: {order_a == order_b}, "
          f"max |score difference| {max(abs(a[d] - b[d]) for d in ids):.3f}\\n")
'''

PART4_MD = md("""
## Part 4 · Customize: your own definition of relevant

The question and its true/false criteria are plain text: change them to say what *relevant* means for your corpus.
`prompt=` overrides any of `question` (batch, with `{id}` for the document id), `pair_question`, `choice_question` and
`criteria`; anything you leave out keeps the default. Here, a stricter notion: only **clinical trial evidence**
counts, so the mechanism passage (D03) drops while the trial (D04) and the meta-analysis (D00) stay.

The deck also covers **intent**: what the person searching is trying to get done, which can change what counts as
relevant for the same query (a clinician choosing a treatment, a student writing a review). It's a concept, not a
`hev-rerank` parameter; the place to express it is the same as here, the wording of the question and criteria.
""")

STRICT = '''
strict = {
    "question": "Does document `documents.{id}` report trial or meta-analysis evidence on the question in `query`?",
    "pair_question": "Does the document report trial or meta-analysis evidence on the question in the query?",
    "criteria": {"true": "The document reports results of a trial or a meta-analysis on the query's question, whatever the result.",
                 "false": "Background, mechanisms, dietary advice or other populations: not trial evidence on the question."},
}
'''

PART5_MD = md("""
## Part 5 · Conclusions

- A decision model is a reranker once you ask it *"is this document relevant to the query?"* as a typed question:
  state in, calibrated probabilities out.
- Three request shapes: **pairwise** (one call per document), **listwise batch** (one call, one Noul per document) and
  **listwise choice** (one call, one Choice). Pairwise and batch give absolute probabilities you can threshold; choice
  gives relative ones you can only sort.
- The best shape depends on the model, so benchmark it on yours: with `hev-rerank` that's one argument.
- *Relevant* is a prompt. Write your own definition.

More: the benchmark at [hevmind.com/research/rerankers](https://hevmind.com/research/rerankers), code at
[hev/rerank-bench](https://github.com/hev/rerank-bench), and this library at
[hev/reranker](https://github.com/hev/reranker).
""")

SCIFACT = '''
import csv, io, json, math, random, re, urllib.request, zipfile
from rank_bm25 import BM25Okapi

# SciFact from BEIR (5,183 abstracts, 300 test queries with human relevance labels), 2.8 MB.
raw = urllib.request.urlopen("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip").read()
z = zipfile.ZipFile(io.BytesIO(raw))
corpus = {d["_id"]: {"title": d["title"], "text": d["text"]} for d in map(json.loads, z.open("scifact/corpus.jsonl"))}
queries = {q["_id"]: q["text"] for q in map(json.loads, z.open("scifact/queries.jsonl"))}
qrels = {}
for row in csv.DictReader(io.TextIOWrapper(z.open("scifact/qrels/test.tsv")), delimiter="\\t"):
    qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

tok = lambda s: re.findall(r"\\w+", s.lower())
corpus_ids = list(corpus)
bm25 = BM25Okapi([tok(corpus[i]["title"] + " " + corpus[i]["text"]) for i in corpus_ids])

def bm25_top(q, k=K):
    scores = bm25.get_scores(tok(q))
    return [corpus_ids[i] for i in sorted(range(len(corpus_ids)), key=lambda i: -scores[i])[:k]]

def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rel.values(), reverse=True)[:10]))
    return dcg / ideal if ideal else 0.0

qids = sorted(qrels)[:N_QUERIES]          # the first N test queries, sorted by id: the same subset every run
shortlists = {q: bm25_top(queries[q]) for q in qids}
bm25_ndcg = sum(ndcg10(shortlists[q], qrels[q]) for q in qids) / len(qids)
recall = sum(len(set(shortlists[q]) & set(qrels[q])) / len(qrels[q]) for q in qids) / len(qids)
print(f"{len(qids)} SciFact test queries, BM25 top-{K}: nDCG@10 {bm25_ndcg:.3f}, recall@{K} {recall:.3f} "
      "(a reranker can only reorder what BM25 found)")
'''

# ---------------------------------------------------------------------------------------------- free tier

FREE = [
    md(f"""
# Decision models as rerankers, on a free Colab GPU

{badge("colab_free_tier.ipynb")}

The companion notebook to {TALK}, run on an open-weight decision model on the **free Colab T4, with no API keys**.
It follows the deck: **Part A** builds the three request shapes from Part 2 of the deck by hand, on the slides'
six-passage example, and sends them straight to the model's own reference code. **Part B** does the same with
[`hev-rerank`](https://github.com/{REPO}) in one line per shape, then benchmarks the shapes on SciFact (Part 3) and
writes a custom definition of relevant (Part 4). Headings name the deck's parts, so you can find your place from a
slide.

**Model:** [`perplexity-ai/pplx-decider-flash-v1.3`](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3),
a 4B decision model distilled from Perplexity's 27B one (Apache-2.0). The slides use Jev, TypeSafe's hosted model;
this model answers the same typed questions, so its numbers are close but not identical.

**What you'll need:** a Colab runtime with a T4: *Runtime → Change runtime type → T4 GPU*. Then *Runtime → Run all*.
No keys, no account. **How long:** 11 minutes in our cold-start run on a T4 (an AWS g4dn.xlarge, the free-Colab card):
2 installing, 2 downloading and loading the 8.5 GB of weights, 6 for Part 3. Allow up to 15 when the install is
slow.

With keys, [hosted_walkthrough.ipynb](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/hosted_walkthrough.ipynb)
runs the slides' exact code on Jev and compares five hosted models.
"""),
    code('''
# Parameters. HEV_RERANK_REF is the git tag or commit of hev-rerank to install.
HEV_RERANK_REF = "v0.3.0"
MODEL = "perplexity-ai/pplx-decider-flash-v1.3"
REVISION = "d06fb3f2d97e0a52325a0d463c450cf0338f44fe"   # the pinned weights this notebook ran with
N_QUERIES = 12      # SciFact test queries per shape in Part 3: sized so the notebook finishes in ~15 minutes on a T4
K = 20              # BM25 candidates per query to rerank
BATCH_DOCS = 5      # documents per batch request on the T4 (see the note under the Part 3 results)
''', tags=["parameters"]),
    code('''
# Installs torch, transformers and hev-rerank. hev-rerank isn't imported until Part B.
%pip install -q "hev-rerank[local] @ git+https://github.com/hev/reranker@{HEV_RERANK_REF}" rank_bm25
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # fewer out-of-memory errors from fragmentation
'''),
    code('''
import torch, transformers
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then run all again."
gpu = torch.cuda.get_device_name(0)
print(f"torch {torch.__version__}, transformers {transformers.__version__}")
print(f"GPU: {gpu}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB, "
      f"bf16: {torch.cuda.is_bf16_supported(including_emulation=False)}")
'''),
    PART1,
    PART_A,
    md("""
## Part 2 · The fit: load a decision model

A decision model takes **state** (any text or JSON) plus **typed questions** about it, and returns calibrated
probabilities instead of generated text. The checkpoint ships its own inference code, `pplx_decider`; its
`decide()` takes the same request the hosted APIs do: `{"state": ..., "questions": {...}}`.

This cell downloads the weights (8.5 GB, once) and loads them on the GPU. Part B reuses this loaded model.

**Two T4 details.** The reference loader always uses bfloat16 on a GPU, which the T4 doesn't have, so we load the
backbone in float16 instead. And on the first request `transformers` warns that `causal_conv1d` and
`flash-linear-attention` aren't installed and falls back to plain PyTorch: that's expected, on a T4
`flash-linear-attention` measured slower than the fallback.
"""),
    code('''
import json, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

t0 = time.perf_counter()
checkpoint = Path(snapshot_download(MODEL, revision=REVISION,       # decision weights + the reference code
                                    allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.py", "LICENSE", "NOTICE", "source/**"],
                                    ignore_patterns=["locate-span/*"]))  # skip the span-extraction adapter
sys.path.insert(0, str(checkpoint / "source" / "src"))
import pplx_decider.model
from pplx_decider import Decider

# T4: no bfloat16. Load the backbone in float16 (bf16 where the GPU has it) and cast the readout head to match.
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
_backbone = pplx_decider.model._backbone
pplx_decider.model._backbone = lambda path, config, _bf16: _backbone(path, config, dtype)
decider = Decider.from_pretrained(checkpoint, device="cuda", mode="isolated", locate=False)
pplx_decider.model._backbone = _backbone
decider.readout.to(dtype)

load_seconds = time.perf_counter() - t0
print(f"{MODEL} in {dtype}: download + load {load_seconds:.0f} s, GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
'''),
    md("""
`mode="isolated"` reads a request's state once and answers every question from it (the layout the model was trained
on); it's what makes one batch call cheaper than six pairwise calls.

The deck's example: a query and six search results, keyed `D00`…`D05`. Two passages answer the question, one
for (D04, a trial) and one against (D00, a meta-analysis); relevance is *addresses the query*, not *agrees with it*.
D03 is related background.

The slides use TypeSafe's SDK, where `Noul(...)` and `Choice(...)` build typed questions. The decider's `decide()`
takes the same questions as plain dicts, so here `Noul` and `Choice` are two small functions that build them: the
code below reads like the slides.
"""),
    code(EXAMPLE + '''
def Noul(instructions, criteria=None):        # a true-or-false question, answered as a probability
    return {"type": "noul", "instructions": instructions, "criteria": criteria}

def Choice(instructions, criteria):           # one question, one option per key of criteria
    return {"type": "choice", "instructions": instructions, "criteria": criteria}

def show_request(request):
    """The request as the model gets it: the state, then each typed question."""
    print("state =", json.dumps(request["state"], indent=2))
    print("questions =")
    for key, q in request["questions"].items():
        print(f"  {key}: {q['type']}, {q['instructions']!r}")
        for option, text in (q["criteria"] or {}).items():
            print(f"      {option}: {text}")
'''),
    PAIR_MD,
    code('''
def pair_request(doc: str) -> dict:
    return {"state": {"query": query, "document": doc},
            "questions": {"relevant": Noul(
                instructions="`document` is relevant to `query`: "
                             "it contains information that answers or directly addresses it.",
                criteria={"true":  "The document contains information that answers the query "
                                   "or directly addresses what it asks about.",
                          "false": "The document is only loosely related, on a similar topic, "
                                   "or does not address what the query asks."})}}

show_request(pair_request(docs["D00"]))       # the request for D00; the other five differ only in `document`
'''),
    code('''
def relevance(doc: str) -> float:             # one call per document
    r = decider.decide(pair_request(doc))
    return r["answers"]["relevant"]["noul"]

t = time.perf_counter()
p_relevant = {d: relevance(text) for d, text in docs.items()}
ranked = sorted(docs, key=p_relevant.get, reverse=True)
print(f"pairwise, 6 calls, {time.perf_counter() - t:.1f} s:")
show(p_relevant, ranked)
part_a["pair"] = p_relevant
'''),
    BATCH_MD,
    code('''
questions = {
    d: Noul(  # a true-or-false question, answered as a probability
        instructions=f"Document `documents.{d}` is relevant to `query`: "
                     "it contains information that answers or directly addresses it.",
        criteria={"true":  "The document contains information that answers the query "
                           "or directly addresses what it asks about.",
                  "false": "The document is only loosely related, on a similar topic, "
                           "or does not address what the query asks."})
    for d in docs}

request = {"state": {"query": query, "documents": docs}, "questions": questions}
show_request(request)
'''),
    code('''
t = time.perf_counter()
r = decider.decide(request)                   # one call

p_relevant = {d: r["answers"][d]["noul"] for d in docs}
ranked = sorted(docs, key=p_relevant.get, reverse=True)
print(f"listwise batch, 1 call, {time.perf_counter() - t:.1f} s:")
show(p_relevant, ranked)
part_a["batch"] = p_relevant
'''),
    CHOICE_MD,
    code('''
request = {"state": {"query": query, "documents": docs},
           "questions": {"best": Choice(  # one question, one option per document
               instructions="Which document in `documents` best answers or most directly addresses `query`?",
               criteria={d: f"documents.{d}" for d in docs})}}
show_request(request)
'''),
    code('''
t = time.perf_counter()
r = decider.decide(request)                   # one call

p_best = r["answers"]["best"]["probabilities"]
ranked = sorted(docs, key=p_best.get, reverse=True)
print(f"listwise choice, 1 call, {time.perf_counter() - t:.1f} s (p_best sums to {sum(p_best.values()):.3f}):")
show(p_best, ranked)
part_a["choice"] = p_best
'''),
    WHEN_MD,
    PART_B,
    code('''
import hev_rerank
from hev_rerank import local, rerank
print(f"hev-rerank {hev_rerank.__version__}")

# rerank(model="local:...") loads the weights itself on first use; client= hands it the model Part A already loaded.
model = local.LocalModel(MODEL, decider, "cuda", dtype, "isolated")
''' + SLIDES_PROMPT),
    code('''
for style in ["pair", "batch", "choice"]:
    hits = rerank(query, texts, model=f"local:{MODEL}", style=style, prompt=slides, client=model)
    compare(style, hits)
'''),
    md("""
Each `rerank(...)` call is one shape from Part A: same model, same questions, same order. The scores can differ
in the third decimal: `hev-rerank` sends the six pairwise requests to the GPU as one padded batch rather than six
calls, and float16 arithmetic on padded batches isn't bit-identical.

## Part 3 · Benchmark: SciFact, BM25 then rerank

SciFact (from BEIR) has 5,183 paper abstracts and 300 test claims with human relevance labels. BM25 retrieves the top
K for each claim; each shape reranks them; nDCG@10 is measured before and after. From here on `rerank` uses the
library's default wording for this model (the one it was benchmarked with).
"""),
    code(SCIFACT),
    code('''
def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]

results = {}
for style in ["pair", "batch", "choice"]:
    latencies, ndcgs, t0 = [], [], time.perf_counter()
    for q in qids:
        cands = [corpus[d] for d in shortlists[q]]
        torch.cuda.synchronize(); t = time.perf_counter()
        hits = rerank(queries[q], cands, model=f"local:{MODEL}", style=style, client=model,
                      max_docs_per_call=BATCH_DOCS if style == "batch" else K)
        torch.cuda.synchronize(); latencies.append(time.perf_counter() - t)
        ndcgs.append(ndcg10([shortlists[q][h.index] for h in hits], qrels[q]))
    results[style] = {"ndcg": sum(ndcgs) / len(ndcgs), "p50": pct(latencies, 50), "p95": pct(latencies, 95),
                      "wall": time.perf_counter() - t0}
    r = results[style]
    print(f"{style:6s}  nDCG@10 {bm25_ndcg:.3f} -> {r['ndcg']:.3f}   per query p50 {r['p50']:.2f} s, "
          f"p95 {r['p95']:.2f} s   wall {r['wall']:.0f} s")
'''),
    code('''
from IPython.display import Markdown
rows = [f"| {s} | {bm25_ndcg:.3f} | {r['ndcg']:.3f} | {r['ndcg'] - bm25_ndcg:+.3f} | {r['p50']:.2f} | {r['p95']:.2f} | {r['wall']:.0f} |"
        for s, r in results.items()]
Markdown(f"**{MODEL}** on {gpu}, {len(qids)} SciFact test queries, BM25 top-{K}, {str(dtype).split('.')[-1]}\\n\\n"
         "| shape | nDCG@10 BM25 | nDCG@10 reranked | Δ | p50 s/query | p95 s/query | wall s |\\n|---|---|---|---|---|---|---|\\n"
         + "\\n".join(rows))
'''),
    md("""
**About these latencies.** They are per query (all K candidates) on a free T4, with the checkpoint's reference
PyTorch code: the T4 has no bfloat16 and no FlashAttention, and the decider's optimized path (packed sequences,
CUDA graphs) needs an Ampere or newer GPU. They are **not comparable with hosted API latencies**, which run the same
kind of model on datacenter GPUs with batching across users, and include the network. For hosted numbers see
the [benchmark](https://hevmind.com/research/rerankers). Use this notebook to compare the *shapes* with one
another on one machine, and to see the quality.

**Why batch is split into requests of five here.** This checkpoint's reference code answers each extra question
in a request by re-running its linear-attention layers over the whole shared state, so on this path a batch
request's cost grows with the square of its length, and twenty documents in one request run out of the T4's
memory. Five per request is about as fast as it gets on a T4. Batch scores are absolute probabilities, so scores
from separate requests stay comparable. (Hosted, and on Ampere GPUs with the checkpoint's fast path, the state is
read once and batch costs about the same as choice.) Choice stays one request over all candidates: its
probabilities are only comparable within one list.

With a dozen queries the nDCG@10 differences between shapes are noise-sized; the
[rerank-bench](https://github.com/hev/rerank-bench) numbers use all 300 SciFact queries plus NFCorpus and FiQA.

## Choice can't be thresholded

Pair and batch scores are calibrated probabilities, so a threshold prunes candidates the model is confident are
irrelevant. Choice scores sum to one across the list, so `hev-rerank` refuses a threshold on them.
"""),
    code('''
q = qids[0]
cands = [corpus[d] for d in shortlists[q]]
kept = rerank(queries[q], cands, model=f"local:{MODEL}", style="batch", client=model, max_docs_per_call=BATCH_DOCS, threshold=0.1)
print(f"{queries[q]!r}: batch keeps {len(kept)} of {len(cands)} candidates at threshold 0.1")
try:
    rerank(queries[q], cands, model=f"local:{MODEL}", style="choice", client=model, threshold=0.1)
except ValueError as e:
    print("ValueError:", e)
'''),
    PART4_MD,
    code(STRICT + '''
default = {ids[h.index]: h.score for h in rerank(query, texts, model=f"local:{MODEL}", style="batch", prompt=slides, client=model)}
custom = {ids[h.index]: h.score for h in rerank(query, texts, model=f"local:{MODEL}", style="batch", prompt=strict, client=model)}
print("batch, slides' wording -> trial-evidence wording")
for d in sorted(custom, key=custom.get, reverse=True):
    print(f"  {d}  {default[d]:6.3f} -> {custom[d]:6.3f}  {docs[d][:70]}")
'''),
    PART5_MD,
    md("""
## Optional: bigger models (not free tier)

The cell below is **off by default and will not fit a free T4**. It shows how to run larger open-weight decision
models with the same call, on a paid Colab GPU:

- [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) (9B, Apache-2.0) in bfloat16 needs about
  20 GB: an **L4** (24 GB) or A100. Its best measured shape is pair.
- 27B models ([`Cloudflare/clef`](https://huggingface.co/Cloudflare/clef), Apache-2.0) in **4-bit** need about
  18 GB with `bitsandbytes`: an L4 or A100.
- `perplexity-ai/pplx-decider-v1.3` (27B) runs in bfloat16 on an 80 GB A100/H100.

This cell was not executed for this notebook's saved outputs.
"""),
    code('''
RUN_BIGGER_MODELS = False   # set True on an L4 / A100 runtime (Colab Pro); not free tier

if RUN_BIGGER_MODELS:
    del model, decider; local._LOADED.clear(); torch.cuda.empty_cache()

    # Clef-flash 9B, bfloat16 on an L4 or A100.
    big = local.load("Cloudflare/clef-flash")
    for h in rerank(query, texts, model="local:Cloudflare/clef-flash", style="pair", client=big):
        print(f"  {h.score:6.3f}  {ids[h.index]}  {texts[h.index][:90]}")
    del big; torch.cuda.empty_cache()

    # A 27B decision model in 4-bit (pip install bitsandbytes).
    from transformers import BitsAndBytesConfig
    q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    big = local.load("Cloudflare/clef", quantization_config=q4)
    for h in rerank(query, texts, model="local:Cloudflare/clef", style="pair", client=big):
        print(f"  {h.score:6.3f}  {ids[h.index]}  {texts[h.index][:90]}")
'''),
]
# ---------------------------------------------------------------------------------------------- hosted

def jev(src):
    """A Part A cell: the slide code, run only when TYPESAFE_API_KEY is set."""
    return code("%%needs_jev\n" + src.strip("\n"))


SLIDE_CRITERIA = '''criteria={"true":  "The document contains information that answers the query "
                           "or directly addresses what it asks about.",
                  "false": "The document is only loosely related, on a similar topic, "
                           "or does not address what the query asks."}'''

HOSTED = [
    md(f"""
# Decision models as rerankers: the hosted walkthrough

{badge("hosted_walkthrough.ipynb")}

The companion notebook to {TALK}. It follows the deck: **Part A** runs the **exact code from the slides** on Jev,
TypeSafe's decision model, for the three request shapes of Part 2 of the deck, on the slides' six-passage example.
**Part B** does the same with [`hev-rerank`](https://github.com/{REPO}) in one line per shape, then runs five hosted
decision models in every shape on SciFact (Part 3) and writes a custom definition of relevant (Part 4). Headings name
the deck's parts, so you can find your place from a slide.

```python
rerank(query, docs, model="jev" | "perplexity" | "clef" | "clef-flash" | "openai", style="pair" | "batch" | "choice")
```

**What you'll need:** API keys for whichever models you want to run. Set them in Colab's 🔑 *Secrets* panel (with
notebook access on) or as environment variables. Part A needs `TYPESAFE_API_KEY` and is skipped without it; in
Part B, models without a key are skipped.

| model | variables |
|---|---|
| `jev` | `TYPESAFE_API_KEY` |
| `perplexity` | `PERPLEXITY_API_KEY` |
| `clef`, `clef-flash` | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` |
| `openai` | `OPENAI_API_KEY` |

**How long:** about 6 minutes with all five models (most of it Part 3), under a minute with Jev alone. No GPU
needed. No keys at all? [colab_free_tier.ipynb](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/colab_free_tier.ipynb)
does the same on an open-weight model on the free Colab GPU.
"""),
    code('''
# Parameters. HEV_RERANK_REF is the git tag or commit of hev-rerank to install.
HEV_RERANK_REF = "v0.3.0"
N_QUERIES = 30      # SciFact test queries per model and shape (pair makes K calls per query)
K = 30              # BM25 candidates per query
''', tags=["parameters"]),
    code('''
# Installs typesafe-sdk (the slides' SDK) and hev-rerank. hev-rerank isn't imported until Part B.
%pip install -q "hev-rerank @ git+https://github.com/hev/reranker@{HEV_RERANK_REF}" rank_bm25
'''),
    code('''
import os
NEEDS = {"jev": ["TYPESAFE_API_KEY"], "perplexity": ["PERPLEXITY_API_KEY"],
         "clef": ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"], "clef-flash": ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"],
         "openai": ["OPENAI_API_KEY"]}
try:                                   # Colab secrets -> environment (values are never printed)
    from google.colab import userdata
    for var in {v for vs in NEEDS.values() for v in vs}:
        try:
            os.environ.setdefault(var, userdata.get(var))
        except Exception:
            pass
except ImportError:
    pass
MODELS = [m for m, vs in NEEDS.items() if all(os.environ.get(v) for v in vs)]
for m, vs in NEEDS.items():
    print(f"{m:11s} {'ready' if m in MODELS else 'skipped: set ' + ' and '.join(vs)}")

from IPython.core.magic import register_cell_magic

@register_cell_magic
def needs_jev(line, cell):
    """%%needs_jev on the first line of a cell: run it only if TYPESAFE_API_KEY is set (Part A calls Jev)."""
    if "jev" in MODELS:
        get_ipython().ex(cell)
    else:
        print("skipped: this cell calls Jev; set TYPESAFE_API_KEY to run it")
'''),
    PART1,
    PART_A,
    md("""
## Part 2 · The fit: typed questions in, probabilities out

A decision model takes **state** (any text or JSON) plus **typed questions** about it, and returns calibrated
probabilities instead of generated text. The code in Part A is the Python tab of each slide, with the slides' `...`
filled in: all six passages and the full criteria text. Cells that call Jev start with `%%needs_jev`, which skips
them when `TYPESAFE_API_KEY` isn't set.

The deck's example: a query and six search results, keyed `D00`…`D05`. Two passages answer the question, one
for (D04, a trial) and one against (D00, a meta-analysis); relevance is *addresses the query*, not *agrees with it*.
D03 is related background.
"""),
    code(EXAMPLE),
    PAIR_MD,
    jev(f'''
from typesafe_sdk import TypeSafeClient, Noul

client = TypeSafeClient()
# query, docs: the cell above

def relevance(doc: str) -> float:              # one call per document
    r = client.system_one(
        state={{"query": query, "document": doc}},
        questions={{"relevant": Noul(  # a true-or-false question, answered as a probability
            instructions="`document` is relevant to `query`: "
                         "it contains information that answers or directly addresses it.",
            {SLIDE_CRITERIA.replace(chr(10), chr(10) + "    ")})}},
        model="jev-latest")
    return r.answers["relevant"].noul

p_relevant = {{d: relevance(text) for d, text in docs.items()}}   # {{"D00": 0.98, ...}}
ranked = sorted(docs, key=p_relevant.get, reverse=True)

print("pairwise, 6 calls:"); show(p_relevant, ranked); part_a["pair"] = p_relevant
'''),
    BATCH_MD,
    jev(f'''
from typesafe_sdk import TypeSafeClient, Noul

# query, docs: above

questions = {{
    d: Noul(  # a true-or-false question, answered as a probability
        instructions=f"Document `documents.{{d}}` is relevant to `query`: "
                     "it contains information that answers or directly addresses it.",
        {SLIDE_CRITERIA})
    for d in docs}}

r = TypeSafeClient().system_one(state={{"query": query, "documents": docs}},
                                questions=questions, model="jev-latest")

p_relevant = {{d: r.answers[d].noul for d in docs}}        # {{"D00": 0.97, "D01": 0.06, ...}}
ranked = sorted(docs, key=p_relevant.get, reverse=True)

print("listwise batch, 1 call:"); show(p_relevant, ranked); part_a["batch"] = p_relevant
'''),
    CHOICE_MD,
    jev('''
from typesafe_sdk import TypeSafeClient, Choice

r = TypeSafeClient().system_one(
    state={"query": query, "documents": docs},
    questions={"best": Choice(  # one question, one option per document
        instructions="Which document in `documents` best answers or most directly addresses `query`?",
        criteria={d: f"documents.{d}" for d in docs})},
    model="jev-latest")

p_best = r.answers["best"].probabilities   # {"D00": 0.85, "D04": 0.15, "D01": 0.0, ...}
ranked = sorted(docs, key=p_best.get, reverse=True)

print(f"listwise choice, 1 call (p_best sums to {sum(p_best.values()):.2f}):"); show(p_best, ranked); part_a["choice"] = p_best
'''),
    WHEN_MD,
    PART_B,
    code('''
import hev_rerank
from hev_rerank import rerank, MODELS as TABLE
print(f"hev-rerank {hev_rerank.__version__}")
''' + SLIDES_PROMPT),
    jev('''
for style in ["pair", "batch", "choice"]:
    hits = rerank(query, texts, model="jev", style=style, prompt=slides)
    compare(style, hits)
'''),
    md("""
Each `rerank(...)` call is one shape from Part A, with the same questions to the same model. Scores can move by a
point or two between calls: that's the hosted model's own run-to-run variation, not the library.

### Every model, every shape

`model=` swaps the decision model; each one gets its own default wording from here on (the one it was benchmarked
with). Starred: the shape `style="auto"` picks for that model, its best measured on the benchmark.
"""),
    code('''
for m in MODELS:
    for style in ["pair", "batch", "choice"]:
        hits = rerank(query, texts, model=m, style=style)
        star = "*" if TABLE[m][2] == style else " "
        print(f"{m:11s} {style:6s}{star} " + "  ".join(f"{ids[h.index]} {h.score:.2f}" for h in hits))
if not MODELS:
    print("No keys set: nothing to call. Set a key above, or use colab_free_tier.ipynb.")
'''),
    md("""
## Part 3 · Benchmark: SciFact, BM25 then every model in every shape

SciFact (from BEIR) has 5,183 paper abstracts and 300 test claims with human relevance labels. BM25 retrieves the top
K for each claim; each model and shape reranks them; nDCG@10 and latency are measured.
"""),
    code(SCIFACT),
    code('''
import time
from concurrent.futures import ThreadPoolExecutor
from hev_rerank import ProviderError

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]

def run(model, style):
    def one(q):
        t = time.perf_counter()
        hits = rerank(queries[q], [corpus[d] for d in shortlists[q]], model=model, style=style)
        return q, hits, time.perf_counter() - t
    # Four queries in flight for batch and choice; pair already sends 8 calls at a time per query, so one query
    # at a time there keeps under the providers' rate limits.
    with ThreadPoolExecutor(1 if style == "pair" else 4) as ex:
        return list(ex.map(one, qids))

results, scores = {}, {}
for m in MODELS:
    for style in ["pair", "batch", "choice"]:
        try:
            out = run(m, style)
        except ProviderError as e:                 # e.g. no beta access on this key: skip, keep going
            print(f"{m:11s} {style:6s} skipped: {str(e)[:120]}")
            continue
        nd = sum(ndcg10([shortlists[q][h.index] for h in hits], qrels[q]) for q, hits, _ in out) / len(out)
        lat = [s for _, _, s in out]
        results[m, style] = {"ndcg": nd, "p50": pct(lat, 50), "p95": pct(lat, 95)}
        scores[m, style] = {q: {shortlists[q][h.index]: h.score for h in hits} for q, hits, _ in out}
        print(f"{m:11s} {style:6s} nDCG@10 {bm25_ndcg:.3f} -> {nd:.3f}   p50 {pct(lat, 50):.2f} s  p95 {pct(lat, 95):.2f} s per query")
'''),
    code('''
from IPython.display import Markdown
rows = [f"| {m} | {s} | {r['ndcg']:.3f} | {r['ndcg'] - bm25_ndcg:+.3f} | {r['p50']:.2f} | {r['p95']:.2f} |"
        for (m, s), r in results.items()]
Markdown(f"{len(qids)} SciFact test queries, BM25 top-{K} (nDCG@10 {bm25_ndcg:.3f}). Latency per query, from wherever this ran.\\n\\n"
         "| model | shape | nDCG@10 | Δ vs BM25 | p50 s | p95 s |\\n|---|---|---|---|---|---|\\n" + "\\n".join(rows))
'''),
    md("""
With thirty queries, differences of a few points are within noise; the
[benchmark](https://hevmind.com/research/rerankers) uses 923 queries over three corpora and is where
`style="auto"` comes from.

### Calibration: do batch probabilities mean what they say?

A batch (or pair) score is a probability that the document is relevant. If it is calibrated, documents scored
around 0.8 should be relevant about 80% of the time. Bucket every (query, candidate) score and compare with the
human labels. (SciFact labels are sparse, about one relevant abstract per query, so the top buckets read low:
some "irrelevant" documents with high scores are unlabeled near-duplicates.)
"""),
    code('''
for m in MODELS:
    if (m, "batch") not in scores:
        continue
    pairs = [(s, d in qrels[q]) for q, ds in scores[m, "batch"].items() for d, s in ds.items()]
    print(f"\\n{m}, batch: {len(pairs)} scored candidates")
    print("  score bucket   n     relevant")
    for lo, hi in [(0, .01), (.01, .1), (.1, .5), (.5, .9), (.9, 1.01)]:
        b = [r for s, r in pairs if lo <= s < hi]
        if b:
            print(f"  [{lo:.2f}, {min(hi, 1):.2f})  {len(b):5d}   {sum(b) / len(b):6.1%}")
'''),
    md("""
## Choice can't be thresholded

Pair and batch scores are absolute, so they sort *and* prune: `threshold=0.1` drops candidates the model is
confident are irrelevant. Choice probabilities sum to one across the list; a document's choice score depends on
what else is in the list, so a fixed cut-off means nothing. `hev-rerank` raises instead of guessing, including
for `style="auto"` on models whose best shape is choice (Perplexity, OpenAI). This needs no key: it fails before
any call.
"""),
    code('''
try:
    rerank(query, texts, model="perplexity", style="choice", threshold=0.1)
except ValueError as e:
    print("ValueError:", e)
for m in MODELS:
    style = "batch" if TABLE[m][2] == "choice" else "auto"
    kept = rerank(query, texts, model=m, style=style, threshold=0.1)
    print(f"{m} {style}: keeps {[ids[h.index] for h in kept]} of {len(docs)} at threshold 0.1")
'''),
    PART4_MD,
    code(STRICT + '''
for m in MODELS:
    style = "batch" if TABLE[m][2] == "choice" else TABLE[m][2]
    default = {ids[h.index]: h.score for h in rerank(query, texts, model=m, style=style)}
    custom = {ids[h.index]: h.score for h in rerank(query, texts, model=m, style=style, prompt=strict)}
    print(f"\\n{m} ({style})   default -> trial-evidence prompt")
    for d in sorted(custom, key=custom.get, reverse=True):
        print(f"  {d}  {default[d]:6.3f} -> {custom[d]:6.3f}  {docs[d][:70]}")
'''),
    PART5_MD,
]


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--force"]
    which = args or ["colab_free_tier.ipynb", "hosted_walkthrough.ipynb"]
    for name, cells, gpu in [("colab_free_tier.ipynb", FREE, True), ("hosted_walkthrough.ipynb", HOSTED, False)]:
        if name not in which:
            continue
        path = HERE / name
        if path.exists() and "--force" not in sys.argv and any(
                c.get("outputs") for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"):
            sys.exit(f"{name} has executed outputs: back it up, then rerun with --force to overwrite it")
        write(name, cells, gpu=gpu)
        print("wrote", name)
