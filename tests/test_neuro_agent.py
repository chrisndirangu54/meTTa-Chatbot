import tempfile
import unittest
from pathlib import Path

from neuro_agent import Agent, Fact, KnowledgeBase, PolicyError, Rule, Tool, ToolRegistry


class KnowledgeTests(unittest.TestCase):
    def test_grounded_query_and_abstention(self):
        kb = KnowledgeBase([Fact("Nairobi", "capital_of", "Kenya", "source:test")])
        self.assertIn("source:test", Agent(kb).ask("Nairobi").answer)
        self.assertTrue(Agent(kb).ask("Unknown").abstained)

    def test_inference_with_provenance(self):
        kb = KnowledgeBase([Fact("A", "parent_of", "B", "source:example", 0.8)],
                           [Rule("parent_of", "ancestor_of", 0.9)])
        result = kb.query(subject="A", predicate="ancestor_of")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.72)
        self.assertIn("source:example", result[0].source)

    def test_persistence_roundtrip(self):
        kb = KnowledgeBase([Fact("A", "relation", "B", "test")])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "kb.json"
            kb.save(path)
            self.assertEqual(KnowledgeBase.load(path).query(subject="A")[0].object, "B")

    def test_invalid_confidence(self):
        with self.assertRaises(ValueError):
            Fact("a", "b", "c", "src", 1.5)


class AgentPolicyTests(unittest.TestCase):
    def test_tool_requires_approval(self):
        registry = ToolRegistry()
        touched = []
        registry.register(Tool("write_note", lambda text: touched.append(text), "writes", mutates_state=True))
        with self.assertRaises(PolicyError):
            registry.execute("write_note", text="x")
        self.assertEqual(touched, [])
        registry.execute("write_note", approved=True, text="x")
        self.assertEqual(touched, ["x"])

    def test_unknown_tool_rejected(self):
        with self.assertRaises(PolicyError):
            ToolRegistry().execute("arbitrary_function")

    def test_planner_cannot_fabricate_grounded_response(self):
        agent = Agent(KnowledgeBase(), planner=lambda q, t: {"type": "finish", "answer": "fabricated"})
        self.assertTrue(agent.ask("anything").abstained)

    def test_step_budget(self):
        agent = Agent(KnowledgeBase(), planner=lambda q, t: {"type": "query", "subject": "missing"}, max_steps=2)
        result = agent.ask("anything")
        self.assertEqual(len(result.actions), 2)
        self.assertTrue(result.abstained)

    def test_plan_query_and_finish(self):
        kb = KnowledgeBase([Fact("Nairobi", "capital_of", "Kenya", "source:test")])
        def planner(q, trace):
            return {"type": "query", "subject": "Nairobi"} if not trace else {"type": "finish"}
        result = Agent(kb, planner=planner).ask("What's Kenya's capital?")
        self.assertFalse(result.abstained)
        self.assertEqual(len(result.evidence), 1)


if __name__ == "__main__":
    unittest.main()
