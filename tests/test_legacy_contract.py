"""Static integration-contract checks on the legacy entry point without model downloads."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "metta.py"


class LegacyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(LEGACY.read_text(encoding="utf-8"), filename="metta.py")
        cls.functions = {node.name: node for node in ast.walk(cls.tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        cls.classes = {node.name for node in ast.walk(cls.tree) if isinstance(node, ast.ClassDef)}

    def test_expected_runtime_components_present(self):
        for name in ("KnowledgeGraph", "NeuralExtractor", "DASClient", "PatternGNN", "AutonomousGoalAgent"):
            with self.subTest(component=name):
                self.assertIn(name, self.classes)
        for name in ("run_chatbot", "query_metta_dynamic", "autonomous_goal_setting"):
            self.assertIn(name, self.functions)

    def test_legacy_does_not_embed_literal_secrets(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.lower() in {"api_key", "google_api_key", "gemini_api_key"}:
                        self.assertFalse(isinstance(node.value, ast.Constant) and bool(node.value.value),
                                         "Potential hard-coded credential")

    def test_public_gradio_share_disabled(self):
        func = self.functions["run_chatbot"]
        launches = [n for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "launch"]
        self.assertTrue(launches)
        for call in launches:
            share = next((kw.value for kw in call.keywords if kw.arg == "share"), None)
            self.assertIsNotNone(share)
            self.assertIsInstance(share, ast.Constant)
            self.assertIs(share.value, False)


if __name__ == "__main__":
    unittest.main()
