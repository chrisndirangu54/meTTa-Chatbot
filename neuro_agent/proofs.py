"""Deterministic provenance-preserving symbolic derivation with proof steps."""
from dataclasses import dataclass
from .core import Fact, KnowledgeBase


@dataclass(frozen=True)
class Proof:
    conclusion: Fact
    premises: tuple[Fact, ...]
    rule: str | None


def explain(kb: KnowledgeBase, subject: str, predicate: str) -> list[Proof]:
    """Explain canonical facts and single-hop rules; no invented intermediate proofs."""
    originals = kb.query(infer=False)
    conclusions = kb.query(subject=subject, predicate=predicate)
    proofs = []
    for conclusion in conclusions:
        if conclusion in originals:
            proofs.append(Proof(conclusion, (conclusion,), None))
            continue
        matches = []
        for premise in originals:
            for rule in kb._rules:
                if (premise.subject == conclusion.subject and premise.object == conclusion.object
                    and premise.predicate == rule.antecedent
                    and conclusion.predicate == rule.consequent
                    and conclusion.source == "rule:" + rule.antecedent + "=>" + rule.consequent + "|" + premise.source):
                    matches.append((premise, rule))
        for premise, rule in matches:
            proofs.append(Proof(conclusion, (premise,), rule.antecedent + " => " + rule.consequent))
    return proofs
