"""Writes the two notebooks in this directory (unexecuted). Edit here, rerun, then execute them:

    python notebooks/build.py [colab_free_tier.ipynb | hosted_walkthrough.ipynb]   # default: both
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


VITAMIN_D = '''
query = "does vitamin D supplementation improve bone density in older adults"
docs = [
 "Meta-analysis of 23 trials: vitamin D alone did not meaningfully increase bone density in older adults.",
 "Dietary sources of vitamin D include oily fish, egg yolk and fortified milk. Many older adults fall short.",
 "Sunscreen reduces vitamin D synthesis in the skin only marginally in practice.",
 "Vitamin D increases intestinal calcium absorption. Deficiency impairs bone mineralization.",
 "Randomised trial: daily vitamin D3 with calcium raised hip bone density in adults over 65 versus placebo.",
 "Astronauts lose 1 to 2 percent of bone density per month in microgravity.",
]
'''

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
ids = list(corpus)
bm25 = BM25Okapi([tok(corpus[i]["title"] + " " + corpus[i]["text"]) for i in ids])

def bm25_top(q, k=K):
    scores = bm25.get_scores(tok(q))
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])[:k]]

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

A decision model takes typed questions about a piece of text and returns calibrated probabilities instead of
generated text. Asked *"is this document relevant to the query?"*, it is a reranker. This notebook runs one
**on the free Colab T4, with no API keys**: [`hev-rerank`](https://github.com/{REPO}) with
`model="local:..."` downloads open weights from Hugging Face and runs them with `transformers` on the GPU.

It reranks BM25 results for SciFact queries in all three request shapes:

- **pair**: one request per (query, document), one yes/no probability each;
- **batch**: the query and all candidates in one request, one yes/no question per document;
- **choice**: one request, one question *"which document is best?"*, probabilities that sum to one.

**Model:** [`perplexity-ai/pplx-decider-flash-v1.3`](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3),
a 4B decider distilled from Perplexity's 27B one (Apache-2.0). Its 8.5 GB of weights fit the T4's 16 GB in
float16 (the T4 has no bfloat16; `hev-rerank` loads float16 there automatically).

**Before you start:** *Runtime → Change runtime type → T4 GPU*. Then *Runtime → Run all*. The whole notebook
took 15 and 17 minutes in our two runs on a T4 (an AWS g4dn.xlarge,
the free-Colab card), 6 to 9 of them installing, downloading and loading the model.

The hosted version (Jev, Perplexity, Cloudflare Clef, OpenAI; needs keys) is
[hosted_walkthrough.ipynb](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/hosted_walkthrough.ipynb).
"""),
    code('''
# Parameters. HEV_RERANK_REF is the git tag or commit of hev-rerank to install.
HEV_RERANK_REF = "v0.3.0"
MODEL = "perplexity-ai/pplx-decider-flash-v1.3"
N_QUERIES = 15      # SciFact test queries per shape: sized so the notebook finishes in ~15 minutes on a T4
K = 20              # BM25 candidates per query to rerank
BATCH_DOCS = 5      # documents per batch request on the T4 (see the note under the results)
''', tags=["parameters"]),
    code('''
%pip install -q "hev-rerank[local] @ git+https://github.com/hev/reranker@{HEV_RERANK_REF}" rank_bm25
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # fewer out-of-memory errors from fragmentation
'''),
    code('''
import torch, transformers, hev_rerank
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then run all again."
gpu = torch.cuda.get_device_name(0)
print(f"hev-rerank {hev_rerank.__version__}, torch {torch.__version__}, transformers {transformers.__version__}")
print(f"GPU: {gpu}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB, "
      f"bf16: {torch.cuda.is_bf16_supported(including_emulation=False)}")
'''),
    md("""
## Load the model

The first call downloads the weights into the Hugging Face cache (8.5 GB) and loads them on the GPU. After that
every `rerank(..., model="local:...")` call reuses the loaded model. No token is needed: the weights are public.
On the first request `transformers` warns that `causal_conv1d` and `flash-linear-attention` are not installed and
it falls back to plain PyTorch. That's expected here: on a T4, `flash-linear-attention` measured slower than the
fallback.
"""),
    code('''
import time
from hev_rerank import local, rerank

t0 = time.perf_counter()
model = local.connect(MODEL)
load_seconds = time.perf_counter() - t0
print(model)
print(f"download + load: {load_seconds:.0f} s, GPU memory in use: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
'''),
    md("## Warm-up: six passages, three shapes"),
    code(VITAMIN_D + '''
for style in ["pair", "batch", "choice"]:
    hits = rerank(query, docs, model=f"local:{MODEL}", style=style)
    print(f"\\n{style}:")
    for h in hits:
        print(f"  {h.score:6.3f}  [{h.index}] {docs[h.index][:90]}")
'''),
    md("""
Pair and batch scores are independent probabilities ("how likely is *this* document relevant?"), so several
documents can score near 1. Choice spreads one unit of probability across the list: it ranks, but a choice score
says nothing about a document on its own.

## SciFact: BM25, then rerank
"""),
    code(SCIFACT),
    code('''
import statistics

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]

results = {}
for style in ["pair", "batch", "choice"]:
    latencies, ndcgs, t0 = [], [], time.perf_counter()
    for q in qids:
        cands = [corpus[d] for d in shortlists[q]]
        torch.cuda.synchronize(); t = time.perf_counter()
        hits = rerank(queries[q], cands, model=f"local:{MODEL}", style=style, max_docs_per_call=BATCH_DOCS if style == "batch" else K)
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
Markdown(f"**{MODEL}** on {gpu}, {len(qids)} SciFact test queries, BM25 top-{K}, float16\\n\\n"
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

With fifteen queries the nDCG@10 differences between shapes are noise-sized; the
[rerank-bench](https://github.com/hev/rerank-bench) numbers use all 300 SciFact queries plus NFCorpus and FiQA.

## Thresholds: only pair and batch

Pair and batch scores are calibrated probabilities, so a threshold prunes candidates the model is confident are
irrelevant. Choice scores sum to one across the list, so `hev-rerank` refuses a threshold on them.
"""),
    code('''
q = qids[0]
cands = [corpus[d] for d in shortlists[q]]
kept = rerank(queries[q], cands, model=f"local:{MODEL}", style="batch", max_docs_per_call=BATCH_DOCS, threshold=0.1)
print(f"{queries[q]!r}: batch keeps {len(kept)} of {len(cands)} candidates at threshold 0.1")
try:
    rerank(queries[q], cands, model=f"local:{MODEL}", style="choice", threshold=0.1)
except ValueError as e:
    print("ValueError:", e)
'''),
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
    del model; local._LOADED.clear(); torch.cuda.empty_cache()

    # Clef-flash 9B, bfloat16 on an L4 or A100.
    big = local.load("Cloudflare/clef-flash")
    for h in rerank(query, docs, model="local:Cloudflare/clef-flash", style="pair", client=big):
        print(f"  {h.score:6.3f}  [{h.index}] {docs[h.index][:90]}")
    del big; torch.cuda.empty_cache()

    # A 27B decision model in 4-bit (pip install bitsandbytes).
    from transformers import BitsAndBytesConfig
    q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    big = local.load("Cloudflare/clef", quantization_config=q4)
    for h in rerank(query, docs, model="local:Cloudflare/clef", style="pair", client=big):
        print(f"  {h.score:6.3f}  [{h.index}] {docs[h.index][:90]}")
'''),
]

# ---------------------------------------------------------------------------------------------- hosted

HOSTED = [
    md(f"""
# Decision models as rerankers: the hosted walkthrough

{badge("hosted_walkthrough.ipynb")}

A decision model takes typed questions about a piece of text and returns calibrated probabilities. Asked *"is this
document relevant to the query?"*, it is a reranker. [`hev-rerank`](https://github.com/{REPO}) calls five hosted
decision models in three request shapes with one function:

```python
rerank(query, docs, model="jev" | "perplexity" | "clef" | "clef-flash" | "openai", style="pair" | "batch" | "choice")
```

- **pair**: one call per (query, document), one yes/no probability each;
- **batch**: one call, the query and every candidate in one state, one yes/no question per document;
- **choice**: one call, one question *"which document is best?"*, probabilities that sum to one.

**Keys.** Each model needs its provider's key. Set whichever you have (Colab: the 🔑 *Secrets* panel, with
notebook access on; elsewhere: environment variables). Models without a key are skipped.

| model | variables |
|---|---|
| `jev` | `TYPESAFE_API_KEY` |
| `perplexity` | `PERPLEXITY_API_KEY` |
| `clef`, `clef-flash` | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` |
| `openai` | `OPENAI_API_KEY` |

No keys at all? [colab_free_tier.ipynb](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/colab_free_tier.ipynb)
runs an open-weight decision model on the free Colab GPU instead.
"""),
    code('''
# Parameters. HEV_RERANK_REF is the git tag or commit of hev-rerank to install.
HEV_RERANK_REF = "v0.3.0"
N_QUERIES = 30      # SciFact test queries per model and shape (pair makes K calls per query)
K = 30              # BM25 candidates per query
''', tags=["parameters"]),
    code('''
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
'''),
    md("## Warm-up: six passages"),
    code(VITAMIN_D + '''
import hev_rerank
from hev_rerank import rerank, MODELS as TABLE
print(f"hev-rerank {hev_rerank.__version__}")
for m in MODELS:
    style = TABLE[m][2]                      # what style="auto" picks for this model
    print(f"\\n{m} ({style}, its best measured shape):")
    for h in rerank(query, docs, model=m):
        print(f"  {h.score:6.3f}  [{h.index}] {docs[h.index][:90]}")
if not MODELS:
    print("No keys set: nothing to call. Set a key above, or use colab_free_tier.ipynb.")
'''),
    md("""
The trial that answers the question and the meta-analysis that contradicts it should both come first: relevance
is *addresses the query*, not *agrees with it*. The background passage on calcium absorption is related but
doesn't answer it.

## SciFact: BM25, then every model in every shape
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

## Calibration: do batch probabilities mean what they say?

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
    rerank(query, docs, model="perplexity", style="choice", threshold=0.1)
except ValueError as e:
    print("ValueError:", e)
for m in MODELS:
    style = "batch" if TABLE[m][2] == "choice" else "auto"
    kept = rerank(query, docs, model=m, style=style, threshold=0.1)
    print(f"{m} {style}: keeps {[h.index for h in kept]} of {len(docs)} at threshold 0.1")
'''),
    md("""
## Your own prompt

The question and its true/false criteria are plain text: change them to say what *relevant* means for your
corpus. `prompt=` overrides any of `question` (batch, with `{id}` for the document id), `pair_question`,
`choice_question` and `criteria`; anything you leave out keeps the default. Here, a stricter notion: only
**clinical trial evidence** counts.
"""),
    code('''
from hev_rerank import load_prompt
print("default batch question (Perplexity, Clef, OpenAI):", load_prompt(hev_rerank.local.PROMPT)["question"])
strict = {
    "question": "Does document `documents.{id}` report trial or meta-analysis evidence on the question in `query`?",
    "pair_question": "Does the document report trial or meta-analysis evidence on the question in the query?",
    "criteria": {"true": "The document reports results of a trial or a meta-analysis on the query's question, whatever the result.",
                 "false": "Background, mechanisms, dietary advice or other populations: not trial evidence on the question."},
}
for m in MODELS:
    style = "batch" if TABLE[m][2] == "choice" else TABLE[m][2]
    default = {h.index: h.score for h in rerank(query, docs, model=m, style=style)}
    custom = {h.index: h.score for h in rerank(query, docs, model=m, style=style, prompt=strict)}
    print(f"\\n{m} ({style})   default -> trial-evidence prompt")
    for i in sorted(custom, key=lambda i: -custom[i]):
        print(f"  [{i}] {default[i]:6.3f} -> {custom[i]:6.3f}  {docs[i][:70]}")
'''),
]


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["colab_free_tier.ipynb", "hosted_walkthrough.ipynb"]
    if "colab_free_tier.ipynb" in which:
        write("colab_free_tier.ipynb", FREE, gpu=True)
    if "hosted_walkthrough.ipynb" in which:
        write("hosted_walkthrough.ipynb", HOSTED)
    print("wrote", *sorted(p.name for p in HERE.glob("*.ipynb")))
