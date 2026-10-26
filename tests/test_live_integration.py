"""Opt-in integration tests. Run with RUN_LIVE_INTEGRATION=1 after installing requirements.

Uses actual Hyperon, FAISS, sentence-transformers, Redis and optionally Gemini.
Never invokes the legacy autonomous loop or shares a public Gradio URL.
"""
import importlib.util
import os
import unittest
from uuid import uuid4


def enabled(flag):
    return os.getenv("RUN_LIVE_INTEGRATION") == "1" and (flag is None or os.getenv(flag) == "1")


@unittest.skipUnless(enabled(None), "Set RUN_LIVE_INTEGRATION=1 to test real model dependencies")
class NeuralSymbolicIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for package in ("hyperon", "faiss", "sentence_transformers", "numpy"):
            if importlib.util.find_spec(package) is None:
                raise unittest.SkipTest(f"Missing optional package: {package}")

    def test_meTTa_atomspace_roundtrip(self):
        from neuro_agent import Fact, KnowledgeBase
        from neuro_agent.metta_bridge import MeTTaBridge
        kb = KnowledgeBase()
        bridge = MeTTaBridge(kb)
        subject = "integration-" + uuid4().hex
        bridge.ingest(Fact(subject, "type", "probe", "integration:test"))
        self.assertEqual(bridge.query(subject, "type")[0].object, "probe")
        # Inspect actual Hyperon atomspace rather than relying only on Python storage.
        matches = bridge.runner.run('!(match &self (fact $s "type" "probe") $s)')
        self.assertTrue(matches)

    def test_real_embeddings_and_faiss(self):
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        queries = ["Nairobi is the capital of Kenya", "A telescope studies distant stars"]
        vectors = np.asarray(model.encode(queries), dtype=np.float32)
        self.assertEqual(vectors.ndim, 2)
        self.assertTrue(np.isfinite(vectors).all())
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        dists, ids = index.search(vectors[:1], 1)
        self.assertEqual(int(ids[0][0]), 0)
        self.assertLess(float(dists[0][0]), 1e-4)

    def test_real_bert_forward(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        model_name = os.getenv("BERT_MODEL", "bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).eval()
        inputs = tokenizer("A test of neural extraction.", return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        self.assertTrue(torch.isfinite(out.last_hidden_state).all())
        self.assertEqual(out.last_hidden_state.shape[0], 1)


@unittest.skipUnless(enabled("RUN_REDIS_INTEGRATION"), "Requires RUN_LIVE_INTEGRATION=1 and RUN_REDIS_INTEGRATION=1")
class RedisIntegrationTests(unittest.TestCase):
    def test_persistence_across_clients(self):
        import redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/15")
        key = "neuro-agent-integration:" + uuid4().hex
        client1 = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2, decode_responses=True)
        client2 = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2, decode_responses=True)
        try:
            self.assertTrue(client1.ping())
            client1.set(key, "persisted", ex=60)
            self.assertEqual(client2.get(key), "persisted")
        finally:
            client1.delete(key)
            client1.close()
            client2.close()


@unittest.skipUnless(enabled("RUN_GEMINI_INTEGRATION"), "Requires live flag and RUN_GEMINI_INTEGRATION=1")
class GeminiIntegrationTests(unittest.TestCase):
    def test_real_model_response(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            self.skipTest("GEMINI_API_KEY is missing")
        llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                                      google_api_key=key, temperature=0, max_retries=1,
                                      timeout=30)
        answer = llm.invoke("Respond only with the word READY.")
        self.assertIn("READY", str(answer.content).upper())
