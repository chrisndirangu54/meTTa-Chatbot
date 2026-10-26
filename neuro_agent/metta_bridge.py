"""Optional MeTTa integration with safe escaping and provenance retained in Python."""
from __future__ import annotations
from .core import Fact, KnowledgeBase


def _metta_string(value: str) -> str:
    import json
    return json.dumps(value, ensure_ascii=False)


class MeTTaBridge:
    def __init__(self, kb: KnowledgeBase):
        try:
            from hyperon import MeTTa
        except ImportError as exc:
            raise RuntimeError("Install optional dependency 'hyperon' to use MeTTaBridge") from exc
        self.runner = MeTTa()
        self.kb = kb

    def ingest(self, fact: Fact) -> None:
        # String atoms are quoted so untrusted text cannot inject MeTTa expressions.
        self.runner.run(f"(add-atom &self (fact {_metta_string(fact.subject)} {_metta_string(fact.predicate)} {_metta_string(fact.object)}))")
        self.kb.add(fact)

    def query(self, subject: str, predicate: str) -> list[Fact]:
        # Provenance and confidence are resolved through the canonical fact store.
        return self.kb.query(subject=subject, predicate=predicate)
