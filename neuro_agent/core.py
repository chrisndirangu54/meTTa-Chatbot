"""Auditable neuro-symbolic agent with explicit policy and evidence handling.

This is a deterministic, dependency-light foundation. A MeTTa adapter and
LLM planners can be added without giving the model unrestricted tool access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import re
import threading


class PolicyError(PermissionError):
    pass


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    object: str
    source: str
    confidence: float = 1.0
    observed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not all([self.subject.strip(), self.predicate.strip(), self.object.strip(), self.source.strip()]):
            raise ValueError("Facts must include subject, predicate, object, and provenance")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be in [0, 1]")


@dataclass(frozen=True)
class Rule:
    """Single-hop symbolic implication: (s, antecedent, o) -> (s, consequent, o)."""
    antecedent: str
    consequent: str
    confidence: float = 1.0


class KnowledgeBase:
    def __init__(self, facts: Iterable[Fact] = (), rules: Iterable[Rule] = ()) -> None:
        self._facts = list(facts)
        self._rules = list(rules)
        self._lock = threading.RLock()

    def add(self, fact: Fact) -> None:
        with self._lock:
            if fact not in self._facts:
                self._facts.append(fact)

    def add_rule(self, rule: Rule) -> None:
        with self._lock:
            self._rules.append(rule)

    def query(self, subject: str | None = None, predicate: str | None = None, object: str | None = None, infer: bool = True) -> list[Fact]:
        with self._lock:
            candidates = list(self._facts)
            if infer:
                # Bounded, single-hop inference avoids circular rule explosion.
                for fact in self._facts:
                    for rule in self._rules:
                        if fact.predicate == rule.antecedent:
                            candidates.append(Fact(fact.subject, rule.consequent, fact.object, "rule:" + rule.antecedent + "=>" + rule.consequent + "|" + fact.source, fact.confidence * rule.confidence))
        return [f for f in candidates if (subject is None or f.subject.casefold() == subject.casefold())
                and (predicate is None or f.predicate.casefold() == predicate.casefold())
                and (object is None or f.object.casefold() == object.casefold())]

    def save(self, path: str | Path) -> None:
        from dataclasses import asdict
        data = {"schema_version": 1, "facts": [asdict(f) for f in self.query(infer=False)],
                "rules": [asdict(r) for r in self._rules]}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported knowledge-base schema")
        return cls([Fact(**f) for f in data["facts"]], [Rule(**r) for r in data["rules"]])


@dataclass(frozen=True)
class Tool:
    name: str
    function: Callable[..., Any]
    description: str
    mutates_state: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", tool.name):
            raise ValueError("Tool names must be snake_case")
        if tool.name in self._tools:
            raise ValueError("Duplicate tool")
        self._tools[tool.name] = tool

    def execute(self, name: str, *, approved: bool = False, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise PolicyError(f"Tool not allowlisted: {name}")
        tool = self._tools[name]
        if tool.mutates_state and not approved:
            raise PolicyError(f"Explicit approval required: {name}")
        return tool.function(**kwargs)


@dataclass
class AgentResult:
    answer: str
    evidence: list[Fact] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    abstained: bool = False


class Agent:
    """Bounded plan/act/observe agent. Planner is injected, never trusted as authority.

    Planner returns a dict with either {'type':'query', ...} or
    {'type':'tool', 'name':'...', 'args': {...}}. Tool invocations remain policy-gated.
    """

    def __init__(self, kb: KnowledgeBase, tools: ToolRegistry | None = None,
                 planner: Callable[[str, list[dict[str, Any]]], dict[str, Any]] | None = None,
                 max_steps: int = 4, max_observation_chars: int = 4000) -> None:
        if not 1 <= max_steps <= 20:
            raise ValueError("max_steps must be 1..20")
        self.kb = kb
        self.tools = tools or ToolRegistry()
        self.planner = planner
        self.max_steps = max_steps
        self.max_observation_chars = max_observation_chars

    def ask(self, question: str, *, approved_tools: Iterable[str] = ()) -> AgentResult:
        if not question.strip():
            raise ValueError("Question cannot be blank")
        if self.planner is None:
            matches = self.kb.query(subject=question.strip())
            if not matches:
                return AgentResult("I do not have verified evidence for that question.", abstained=True)
            return AgentResult("; ".join(f"{f.subject} {f.predicate} {f.object} [source: {f.source}]" for f in matches), matches)

        trace: list[dict[str, Any]] = []
        evidence: list[Fact] = []
        for _ in range(self.max_steps):
            step = self.planner(question, list(trace))
            if not isinstance(step, dict):
                raise ValueError("Planner must return an action dictionary")
            kind = step.get("type")
            if kind == "query":
                found = self.kb.query(subject=step.get("subject"), predicate=step.get("predicate"),
                                      object=step.get("object"))
                evidence.extend(f for f in found if f not in evidence)
                trace.append({"action": "query", "count": len(found),
                              "observations": [f"{f.subject} {f.predicate} {f.object} ({f.source})" for f in found[:20]]})
            elif kind == "tool":
                name = step.get("name")
                args = step.get("args", {})
                if not isinstance(name, str) or not isinstance(args, dict):
                    raise ValueError("Invalid tool action")
                observation = self.tools.execute(name, approved=name in approved_tools, **args)
                trace.append({"action": "tool", "name": name,
                              "observation": str(observation)[:self.max_observation_chars]})
            elif kind == "finish":
                # Free-form planner statements are not presented as verified facts.
                if not evidence:
                    return AgentResult("No independently verified evidence found.", actions=trace, abstained=True)
                return AgentResult("; ".join(f"{f.subject} {f.predicate} {f.object} [source: {f.source}]" for f in evidence),
                                   evidence, trace)
            else:
                raise ValueError(f"Unsupported planner action: {kind}")
        if evidence:
            return AgentResult("; ".join(f"{f.subject} {f.predicate} {f.object} [source: {f.source}]" for f in evidence),
                               evidence, trace)
        return AgentResult("Step limit reached without verified evidence.", actions=trace, abstained=True)
