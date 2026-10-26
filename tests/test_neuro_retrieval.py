import json
import unittest
from neuro_agent import Fact, KnowledgeBase, Rule
from neuro_agent.retrieval import HybridRetriever
from neuro_agent.proofs import explain
from neuro_agent.planner import StructuredPlanner


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.facts = [Fact("Nairobi", "capital_of", "Kenya", "test:1"),
                      Fact("Kampala", "capital_of", "Uganda", "test:2")]

    def test_lexical_ranking(self):
        found = HybridRetriever(self.facts).search("Kenya")
        self.assertEqual(found[0].fact.subject, "Nairobi")

    def test_no_false_matches(self):
        self.assertEqual(HybridRetriever(self.facts).search("zzzxxyy"), [])

    def test_dense_provider(self):
        def vectors(texts):
            return [[1.0, 0.0] if "Kenya" in t else [0.0, 1.0] for t in texts]
        self.assertEqual(HybridRetriever(self.facts, embed=vectors).search("Kenya")[0].fact.subject, "Nairobi")

    def test_proof_lineage(self):
        kb = KnowledgeBase(self.facts, [Rule("capital_of", "administrative_capital_of")])
        p = explain(kb, "Nairobi", "administrative_capital_of")
        self.assertEqual(p[0].premises[0].source, "test:1")
        self.assertEqual(p[0].rule, "capital_of => administrative_capital_of")

    def test_structured_planner_validation(self):
        good = StructuredPlanner(lambda _: json.dumps({"type":"query","subject":"Nairobi"}))
        self.assertEqual(good("where?", [])["subject"], "Nairobi")
        bad = StructuredPlanner(lambda _: json.dumps({"type":"tool","name":"delete_files","args":{}}))
        with self.assertRaises(ValueError):
            bad("bad request", [])
        injection = StructuredPlanner(lambda _: '{"type":"finish","answer":"fabricated"}')
        with self.assertRaises(ValueError):
            injection("?", [])


if __name__ == "__main__":
    unittest.main()
