"""Hybrid lexical + optional dense retrieval over evidence-bearing facts."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Callable, Sequence
from .core import Fact


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


@dataclass(frozen=True)
class Hit:
    fact: Fact
    score: float
    lexical_score: float
    semantic_score: float


class HybridRetriever:
    """BM25 and optional embedding similarity; no model download on import."""
    def __init__(self, facts: Sequence[Fact], embed: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
                 lexical_weight: float = 0.6, semantic_weight: float = 0.4):
        if lexical_weight < 0 or semantic_weight < 0 or lexical_weight + semantic_weight <= 0:
            raise ValueError("Invalid retrieval weights")
        self.facts = list(facts)
        self.docs = [tokenize(f"{f.subject} {f.predicate} {f.object}") for f in facts]
        self.embed = embed
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self._vectors = None
        self.avgdl = sum(map(len, self.docs)) / len(self.docs) if self.docs else 0
        self.df = Counter(t for doc in self.docs for t in set(doc))

    def _bm25(self, tokens: list[str], doc: list[str]) -> float:
        if not doc:
            return 0.0
        tf = Counter(doc)
        n = len(self.docs)
        result = 0.0
        for term in set(tokens):
            count = tf.get(term, 0)
            if not count:
                continue
            idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
            result += idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * len(doc) / max(self.avgdl, 1)))
        return result

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b) or not a:
            raise ValueError("Invalid embedding dimensions")
        numerator = sum(x * y for x, y in zip(a, b))
        norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return numerator / norms if norms else 0.0

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not query.strip() or not self.facts:
            return []
        lexical = [self._bm25(tokenize(query), doc) for doc in self.docs]
        semantic = [0.0] * len(self.facts)
        if self.embed:
            if self._vectors is None:
                self._vectors = list(self.embed([" ".join([f.subject, f.predicate, f.object]) for f in self.facts]))
                if len(self._vectors) != len(self.facts):
                    raise ValueError("Embedding count mismatch")
            question_vectors = list(self.embed([query]))
            if len(question_vectors) != 1:
                raise ValueError("Embedder must return one query vector")
            semantic = [max(0, self._cosine(question_vectors[0], v)) for v in self._vectors]
        max_lex = max(lexical, default=0.0)
        hits = [Hit(f, self.lexical_weight * (l / max_lex if max_lex else 0.0)
                    + self.semantic_weight * s if self.embed else (l / max_lex if max_lex else 0.0),
                    l, s)
                for f, l, s in zip(self.facts, lexical, semantic)]
        hits = [h for h in hits if h.lexical_score > 0 or h.semantic_score > 0]
        return sorted(hits, key=lambda h: (-h.score, h.fact.subject, h.fact.predicate))[:top_k]
