# Neuro-symbolic Agent V2 — initial implementation

This incremental upgrade leaves the existing `metta.py` entry point unchanged.

## Delivered
- Fact store with mandatory provenance, bounded confidence, JSON persistence, and single-hop symbolic inference.
- Optional `hyperon` MeTTa adapter with escaped string atoms.
- Bounded plan/act/observe runner with injectable planner.
- Allowlisted tools; mutating tools require explicit per-request authorization.
- Evidence-first answers, abstention when no verified facts are found.
- Standard-library unit tests and CI workflow.

## Example

```python
from neuro_agent import Agent, Fact, KnowledgeBase, Rule, Tool, ToolRegistry

kb = KnowledgeBase(
    [Fact("Nairobi", "capital_of", "Kenya", "reference:verified-dataset", 0.99)],
    [Rule("capital_of", "administrative_capital_of")]
)
result = Agent(kb).ask("Nairobi")
print(result.answer)
```

Run locally with `python -m unittest discover -s tests -p 'test_neuro_agent.py' -v`.
Optional MeTTa adapter: `pip install hyperon`.

## Architectural boundaries
The existing large model stack stays opt-in and is *not* initialized by importing
`neuro_agent`. No arbitrary shell, web requests, file writes, or unapproved external
mutations are exposed as tools. Tool outputs are observations, **not** verified
facts. Add reviewed facts with traceable sources before presenting them as evidence.

The provided MeTTa adapter writes data to the MeTTa atomspace but currently
serves queries through the provenance-aware Python fact store. It is a bridge,
not a complete symbolic proof engine.

## Next engineering milestones
1. MeTTa rule compilation and proof-graph extraction with provenance lineage.
2. Hybrid retrieval (BM25 + embedding reranking), document-level citations, and
   adversarial prompt-injection tests.
3. Structured LLM planner adapter with strict schema and per-step budget.
4. Permissioned connectors, persistent episodic memory, and audit/event logs.
5. Benchmarks covering answer correctness, citation faithfulness, reasoning,
   security, latency, cost, and calibration; compare to a plain RAG baseline.
6. Integrate the new runtime into the existing Gradio UI after integration tests.

## Important
This is a functioning **foundation**, not a claim of frontier-model performance.
It does not train a GNN, validate web evidence, or autonomously execute workflows.
No deployment or performance benchmark has been run as part of this change.
