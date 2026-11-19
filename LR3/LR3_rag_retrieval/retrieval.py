import math
import re
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def ranked(scores, depth):
    order = sorted(scores, key=lambda doc: (-scores[doc], int(doc)))[:depth]
    return [dict(doc_id=doc, rank=i, score=float(scores[doc]))
            for i, doc in enumerate(order, 1)]


class BM25:
    def __init__(self, texts, k1, b):
        posting = defaultdict(list)
        lengths = {}
        for doc, text in texts.items():
            counts = Counter(tokens(text))
            lengths[doc] = sum(counts.values())
            for term, frequency in counts.items():
                posting[term].append((doc, frequency))
        total = len(texts)
        average = sum(lengths.values()) / total
        norm = {doc: k1 * (1 - b + b * length / average) for doc, length in lengths.items()}
        self.weights = {}
        for term, entries in posting.items():
            df = len(entries)
            idf = math.log1p((total - df + 0.5) / (df + 0.5))
            self.weights[term] = [(doc, idf * frequency * (k1 + 1) / (frequency + norm[doc]))
                                  for doc, frequency in entries]

    def search(self, claim, depth):
        scores = defaultdict(float)
        for term in sorted(set(tokens(claim))):
            for doc, weight in self.weights.get(term, ()):
                scores[doc] += weight
        return ranked(scores, depth)


def fuse(dense, bm25, constant, depth):
    parts = defaultdict(dict)
    for technique, records in (("dense", dense), ("bm25", bm25)):
        for record in records[:depth]:
            parts[record["doc_id"]][technique] = Fraction(1, constant + record["rank"])
    scores = {doc: sum(weights.values()) for doc, weights in parts.items()}
    rows = ranked(scores, depth)
    for row in rows:
        row["contributions"] = {key: float(parts[row["doc_id"]].get(key, 0))
                                for key in ("dense", "bm25")}
    return rows


def checks():
    toy = BM25({"1": "a a b", "2": "a c", "3": "b"}, 1.2, 0.75)
    result = toy.search("a A", 20)
    assert [r["doc_id"] for r in result] == ["1", "2"]
    idf = math.log(1.6)
    np.testing.assert_allclose([r["score"] for r in result], [idf * 4.4 / 3.65, idf])
    assert toy.search("zero", 20) == []
    a = ranked({"1": 3, "2": 2}, 20)
    b = ranked({"3": 3, "2": 2}, 20)
    assert [r["doc_id"] for r in fuse(a, b, 60, 20)] == ["2", "1", "3"]
    assert tokens("ALDH1 IL-6") == ["aldh1", "il", "6"]
