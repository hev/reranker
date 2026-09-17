"""First-stage BM25 over a handful of passages, then Jev reranks the shortlist."""
from rank_bm25 import BM25Okapi
from jev_rerank import rerank

passages = [
    {"title": "Vitamin D and bone density", "text": "Randomised trial: supplementation raised bone mineral density in adults over 65."},
    {"title": "Vitamin D sources", "text": "Dietary sources include oily fish, egg yolk and fortified milk."},
    {"title": "Calcium absorption", "text": "Vitamin D increases intestinal calcium absorption; deficiency impairs it."},
    {"title": "Sunscreen use", "text": "Sunscreen reduces cutaneous vitamin D synthesis only marginally in practice."},
]
query = "does vitamin D supplementation improve bone density in older adults"
bm25 = BM25Okapi([(p["title"] + " " + p["text"]).lower().split() for p in passages])
shortlist = [passages[i] for i in sorted(range(len(passages)), key=lambda i: -bm25.get_scores(query.lower().split())[i])]
for r in rerank(query, shortlist, top_n=3, threshold=0.1):
    print(f"{r.score:.2f}  {r.document['title']}")
