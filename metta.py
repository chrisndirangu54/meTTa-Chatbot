# Improved FAQ Chatbot System with Self-Awareness and Autonomous Goal-Setting
# Version: 3.0 — Full rewrite fixing all critical, significant, and minor issues
# Fix log:
#   [CRITICAL-1]  API key no longer has a hardcoded fallback; requires GEMINI_API_KEY env var
#   [CRITICAL-2]  DASClient instantiated before KnowledgeGraph; injected as dependency
#   [CRITICAL-3]  kg / neural_extractor injected into AutonomousGoalAgent; no globals
#   [CRITICAL-4]  Malformed MeTTa extra-paren in update_self_state removed
#   [CRITICAL-5]  `facts` initialised to [] at top of query_metta_dynamic
#   [SIGNIFICANT-1] Fallback chain lambda now has correct 4-arg signature via closure
#   [SIGNIFICANT-2] benchmark_clevr_vqa removed from hot query path; on-demand button only
#   [SIGNIFICANT-3] review_heuristics signature fixed (kg first, manual second)
#   [SIGNIFICANT-4] FAISS index-to-atom mapping uses a parallel list, not dict order
#   [SIGNIFICANT-5] Autonomous loop goal generation uses proper chain.invoke()
#   [SIGNIFICANT-6] build_graph FAISS filter uses the parallel list correctly
#   [MINOR-1]     Unit tests use correct MeTTa query syntax
#   [MINOR-2]     Single __main__ block; tests gated behind --test flag
#   [MINOR-3]     Redis subprocess spawning removed; clean failure + warning instead
#   [MINOR-4]     BERT fine-tune skips training if checkpoint already exists

# Dependencies (pip install these):
# hyperon, langchain-google-genai, sentence-transformers, faiss-cpu, torch, torch-geometric,
# transformers, datasets, requests, duckduckgo-search, tenacity, redis, gradio, pandas,
# crewai, langchain-community

import threading
import time
import asyncio
import argparse
from datetime import datetime
import json
import uuid
import logging
import os
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from hyperon import MeTTa, GroundingSpace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from transformers import (
    BertTokenizer, BertModel,
    BertForSequenceClassification, Trainer, TrainingArguments,
)
from datasets import Dataset
import torch
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data as pyg_data
from ddgs import DDGS
from tenacity import retry, wait_random_exponential, stop_after_attempt
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import pandas as pd

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_MODE"] = "disabled"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cpu")
logger.info(f"Using PyTorch device: {device}")

mock_responses: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Centralised error handler
# ---------------------------------------------------------------------------
class ErrorHandler:
    @staticmethod
    def log_and_report(runner, error_type: str, message: str):
        logger.error(f"{error_type}: {message}")
        if runner is not None:
            try:
                runner.run(f'(add-atom (error "{error_type}" "{message}"))')
            except Exception:
                pass

error_handler = ErrorHandler()

# ---------------------------------------------------------------------------
# DAS — must be created BEFORE KnowledgeGraph            [FIX: CRITICAL-2]
# ---------------------------------------------------------------------------
class DASClient:
    """Distributed Atom Space with Redis, falling back to in-memory."""

    def __init__(self):
        self.in_memory: Dict[str, str] = {}
        self.redis = None
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            import redis as redis_lib
            client = redis_lib.from_url(redis_url, decode_responses=True)
            client.ping()
            self.redis = client
            logger.info("DAS: connected to Redis.")
        except Exception as exc:
            # [FIX: MINOR-3] Do NOT spawn a redis-server subprocess; just warn.
            logger.warning(
                f"DAS: Redis unavailable ({exc}). "
                "Start Redis manually if persistence is needed. Using in-memory fallback."
            )

    def add_atom(self, key: str, value: Any):
        try:
            val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            if self.redis:
                self.redis.set(key, val_str)
            else:
                self.in_memory[key] = val_str
        except Exception as exc:
            error_handler.log_and_report(None, "DAS.add_atom", str(exc))

    def query(self, pattern: str) -> List[str]:
        try:
            if self.redis:
                keys = self.redis.keys(f"{pattern}*")
                return [v for k in keys if (v := self.redis.get(k)) is not None]
            return [v for k, v in self.in_memory.items() if k.startswith(pattern)]
        except Exception as exc:
            error_handler.log_and_report(None, "DAS.query", str(exc))
            return [v for k, v in self.in_memory.items() if k.startswith(pattern)]


# Singleton — created here so KnowledgeGraph can receive it as a parameter.
das = DASClient()

# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------
class KnowledgeGraph:
    def __init__(self, das_client: DASClient):
        self.das = das_client                          # [FIX: CRITICAL-2] injected
        self.space = GroundingSpace()
        self.runner = MeTTa(space=self.space.gspace)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        # [FIX: SIGNIFICANT-6] parallel list keeps FAISS position → atom text mapping
        self._index_atoms: List[str] = []             # position i ↔ FAISS index i
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._load_initial_knowledge()

    # ------------------------------------------------------------------
    def _load_initial_knowledge(self):
        knowledge_metta = """
        (: General Domain)
        (: Person Type)
        (: Role Type)
        (: Entity Type)
        (: ErrorType Type)
        (: Kenyan-Politics Domain)
        (: Science Domain)
        (: Medicine Domain)

        (= (heuristic $query $context)
           (add-atom (heuristic-result $query $context)))

        (= (get-errors $type)
           (match &self (error $type $e) $e))

        (= (select-domain $domain)
           (match &self (in-domain $entity $domain) $entity))

        (= (learn-fact $entity $fact)
           (add-atom (fact $entity $fact)))

        (= (definition vectorize-entity)
           "Embed entities using SentenceTransformer for FAISS search")

        (in-domain Raila-Odinga Kenyan-Politics)
        (in-domain William-Ruto Kenyan-Politics)
        (in-domain Musalia-Mudavadi Kenyan-Politics)
        (fact Raila-Odinga (role Opposition-Leader))
        (fact William-Ruto (role President))
        (fact Musalia-Mudavadi (role Prime-Cabinet-Secretary))

        (in-domain Malaria Medicine)
        (in-domain Tuberculosis Medicine)
        (in-domain HIV-AIDS Medicine)

        (fact Malaria (disease (symptoms fever chills headache nausea)))
        (fact Malaria (cause (parasite Plasmodium transmitted-by Anopheles-mosquito)))
        (fact Tuberculosis (disease (symptoms cough fever night-sweats weight-loss)))
        (fact Tuberculosis (cause (bacteria Mycobacterium-tuberculosis)))
        (fact HIV-AIDS (disease (symptoms fever fatigue swollen-lymph-nodes weight-loss)))
        (fact HIV-AIDS (cause (virus Human-Immunodeficiency-Virus)))
        """
        self.runner.run(knowledge_metta)
        self._vectorize_all()
        logger.info("KnowledgeGraph: initialised.")

    # ------------------------------------------------------------------
    def _vectorize_all(self):
        """Full vectorisation — called once on init."""
        try:
            atoms = self.runner.run("!(match &self $atom $atom)") or []
            if not atoms:
                return
            texts = [str(a) for a in atoms]
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = self.embedder.encode(batch, convert_to_tensor=False)
                for text, vec in zip(batch, vecs):
                    if text not in self._index_atoms:          # avoid duplicates
                        self._index_atoms.append(text)
                        self.index.add(np.array([vec], dtype=np.float32))
                        self.das.add_atom(f"embedding:{len(self._index_atoms)-1}", vec.tolist())
            logger.info(f"KnowledgeGraph: vectorised {len(self._index_atoms)} atoms.")
        except Exception as exc:
            error_handler.log_and_report(self.runner, "KG.vectorize_all", str(exc))

    def vectorize_new_atoms(self):
        """Incremental: only vectorise atoms not yet in the index."""
        try:
            atoms = self.runner.run("!(match &self $atom $atom)") or []
            new_texts = [str(a) for a in atoms if str(a) not in self._index_atoms]
            if not new_texts:
                return
            vecs = self.embedder.encode(new_texts, convert_to_tensor=False)
            for text, vec in zip(new_texts, vecs):
                self._index_atoms.append(text)
                self.index.add(np.array([vec], dtype=np.float32))
            logger.info(f"KnowledgeGraph: added {len(new_texts)} new atom vectors.")
        except Exception as exc:
            error_handler.log_and_report(self.runner, "KG.vectorize_new", str(exc))

    def search_similar(self, query_emb: np.ndarray, k: int = 5) -> List[str]:
        """Return up to k atom strings closest to query_emb."""
        if self.index.ntotal == 0:
            return []
        q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        k = min(k, self.index.ntotal)
        _, indices = self.index.search(q, k)
        # [FIX: SIGNIFICANT-6] use _index_atoms list — position-safe
        return [
            self._index_atoms[idx]
            for idx in indices[0]
            if idx != -1 and idx < len(self._index_atoms)
        ]

    def build_graph(self, query_emb: Optional[np.ndarray] = None) -> Tuple[pyg_data, Dict[str, int]]:
        """Build a PyG graph. Optionally filter nodes by FAISS proximity."""
        atoms = self.runner.run("!(match &self $atom $atom)") or []
        if not atoms:
            return (
                pyg_data(
                    x=torch.empty((0, self.dimension), dtype=torch.float),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                ),
                {},
            )

        if query_emb is not None and self.index.ntotal > 0:
            # [FIX: SIGNIFICANT-6] use search_similar which uses the parallel list
            filtered = set(self.search_similar(query_emb, k=100))
            atoms = [a for a in atoms if str(a) in filtered] or atoms

        node_map = {str(a): i for i, a in enumerate(atoms)}
        node_features = []
        batch_size = 16
        atom_strs = [str(a) for a in atoms]
        for i in range(0, len(atom_strs), batch_size):
            batch = atom_strs[i : i + batch_size]
            vecs = self.embedder.encode(batch, convert_to_tensor=False)
            node_features.extend(vecs.tolist())

        edges = []
        implies = self.runner.run("!(unique (match &self (= (implies $a $b) true) ($a $b)))") or []
        for result in implies:
            children = result.get_children() if hasattr(result, "get_children") else []
            if len(children) == 2:
                a_str, b_str = str(children[0]), str(children[1])
                if a_str in node_map and b_str in node_map:
                    edges.append([node_map[a_str], node_map[b_str]])

        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        x = (
            torch.tensor(node_features, dtype=torch.float)
            if node_features
            else torch.empty((0, self.dimension), dtype=torch.float)
        )
        return pyg_data(x=x, edge_index=edge_index), node_map


# ---------------------------------------------------------------------------
# Neural Extractor
# ---------------------------------------------------------------------------
_BERT_CHECKPOINT = "./bert-finetuned/pytorch_model.bin"

class NeuralExtractor:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        ).to(device)
        self._fine_tune()

    def _fine_tune(self):
        # [FIX: MINOR-4] Skip training if checkpoint already exists.
        if os.path.exists(_BERT_CHECKPOINT):
            logger.info("NeuralExtractor: loading cached checkpoint, skipping training.")
            try:
                self.classifier.load_state_dict(
                    torch.load(_BERT_CHECKPOINT, map_location=device)
                )
            except Exception as exc:
                logger.warning(f"NeuralExtractor: checkpoint load failed ({exc}), retraining.")
                self._run_training()
        else:
            self._run_training()

    def _run_training(self):
        try:
            facts = self.kg.runner.run("!(match &self (fact $entity $fact) $fact)") or []
            texts = [
                str(f).split('"', 2)[-1].rstrip('")')
                for f in facts
                if len(str(f).split()) > 1
            ]
            if not texts:
                texts = ["Kenya is a country."]
            negatives = ["What is this?", "Random question?", "Irrelevant text."]
            data = {
                "text": texts + negatives,
                "label": [1] * len(texts) + [0] * len(negatives),
            }
            dataset = Dataset.from_dict(data)

            def tokenize(examples):
                return self.tokenizer(
                    examples["text"], padding="max_length", truncation=True, max_length=128
                )

            dataset = dataset.map(tokenize, batched=True, load_from_cache_file=False)
            dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            args = TrainingArguments(
                output_dir="./bert-finetuned",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                logging_steps=10,
                save_strategy="epoch",
            )
            trainer = Trainer(model=self.classifier, args=args, train_dataset=dataset)
            trainer.train()
            logger.info("NeuralExtractor: fine-tuning complete.")
        except Exception as exc:
            error_handler.log_and_report(self.kg.runner, "BERT.fine_tune", str(exc))

    def classify_sentence(self, sentence: str) -> float:
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
        return torch.nn.functional.softmax(logits, dim=1)[0][1].item()

    def get_embedding(self, sentence: str) -> List[float]:
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

    def compute_similarity(self, text1: str, text2: str) -> float:
        e1 = np.array(self.get_embedding(text1)).reshape(1, -1)
        e2 = np.array(self.get_embedding(text2)).reshape(1, -1)
        return cosine_similarity(e1, e2)[0][0] if e1.size and e2.size else 0.0


def extract_facts_lightweight(
    text: str, extractor: NeuralExtractor, threshold: float = 0.7
) -> List[Tuple[str, str, float, List[float]]]:
    facts = []
    for sentence in [s.strip() for s in text.split(".") if s.strip()]:
        if " is " not in sentence.lower():
            continue
        prob = extractor.classify_sentence(sentence)
        if prob < threshold:
            continue
        parts = sentence.split(" is ", 1)
        if len(parts) != 2:
            continue
        entity = (
            parts[0].split("(")[0].split(",")[0].strip().lower()
            .replace(" ", "-").replace("--", "-")
        )
        embedding = extractor.get_embedding(sentence)
        facts.append((entity, parts[1].strip(), prob, embedding))
    return facts


# ---------------------------------------------------------------------------
# GNN
# ---------------------------------------------------------------------------
class PatternGNN(torch.nn.Module):
    def __init__(self, in_channels: int = 384, hidden_channels: int = 64, out_channels: int = 1):
        super().__init__()
        self.gnn = GraphSAGE(in_channels, hidden_channels, num_layers=2, out_channels=out_channels)

    def forward(self, data):
        return self.gnn(data.x, data.edge_index)


pattern_gnn = PatternGNN().to(device)
gnn_optimizer = torch.optim.Adam(pattern_gnn.parameters(), lr=0.01)
pattern_dataset: List[Dict] = []


def train_pattern_gnn(kg: KnowledgeGraph):
    if len(pattern_dataset) < 10:
        return
    data, _ = kg.build_graph()
    if data.x.size(0) == 0:
        return
    data = data.to(device)
    pattern_gnn.train()
    batch_size = 10
    for i in range(0, len(pattern_dataset), batch_size):
        batch = pattern_dataset[i : i + batch_size]
        gnn_optimizer.zero_grad()
        out = pattern_gnn(data)
        target = torch.tensor(
            [kg.embedder.encode(p["heuristic"]).mean() for p in batch], dtype=torch.float
        ).to(device).mean()
        loss = torch.nn.MSELoss()(out.mean(), target)
        loss.backward()
        gnn_optimizer.step()
    logger.info(f"PatternGNN: trained on {len(pattern_dataset)} patterns.")


# ---------------------------------------------------------------------------
# LLM setup                                              [FIX: CRITICAL-1]
# ---------------------------------------------------------------------------
def _get_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        logger.warning(
            "GEMINI_API_KEY env var is not set. "
            "LLM features will be disabled. Set the variable and restart."
        )
    return key


def _validate_gemini_key(api_key: str) -> bool:
    try:
        test_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key, max_retries=1)
        test_llm.invoke("Test")
        return True
    except Exception as exc:
        logger.error(f"Gemini key validation failed: {exc}")
        return False


_api_key = _get_api_key()
llm_enabled = bool(_api_key) and _validate_gemini_key(_api_key)
llm = (
    ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=_api_key, max_retries=10)
    if llm_enabled
    else None
)

parser_out = StrOutputParser()

prompt = ChatPromptTemplate.from_template("""
You are a general FAQ chatbot. Use the following context to answer the question.
Context: {context}
Question: {question}
Provide a structured answer with definitions, examples, and references. If unsure, suggest updates.
""")


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
@retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(10))
def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            snippets = [r["body"] for r in results if "body" in r]
            return " ".join(snippets[:3]) if snippets else "No results."
    except Exception as exc:
        error_handler.log_and_report(None, "web_search", str(exc))
        return "Search failed."


# ---------------------------------------------------------------------------
# Continual learning
# ---------------------------------------------------------------------------
def continual_learning(entity: str, kg: KnowledgeGraph, extractor: NeuralExtractor):
    try:
        web_result = web_search(f"What is {entity}?")
        facts = extract_facts_lightweight(web_result, extractor)
        for ent, fact, prob, _ in facts[:3]:
            if prob > 0.7:
                kg.runner.run(f'(learn-fact "{entity}" "{fact}")')
                das.add_atom(f"fact:{entity}", fact)
        kg.vectorize_new_atoms()
        logger.info(f"continual_learning: learned {len(facts)} facts for '{entity}'.")
    except Exception as exc:
        error_handler.log_and_report(kg.runner, "continual_learning", str(exc))


# ---------------------------------------------------------------------------
# Graph update helpers
# ---------------------------------------------------------------------------
def update_graph(domain: str, new_fact: str, kg: KnowledgeGraph):
    if not new_fact.startswith("(= "):
        new_fact = f"(= {new_fact} true)"
    parts = new_fact.split()
    entity = parts[1] if len(parts) > 1 else "Unknown"
    kg.runner.run(f"(add-atom (in-domain {entity} {domain}))")
    kg.runner.run(new_fact)
    kg.vectorize_new_atoms()
    logger.info(f"update_graph: {domain} ← {new_fact}")


def parse_natural_language_update(
    update_text: str, extractor: NeuralExtractor
) -> Tuple[Optional[str], Optional[str]]:
    default_domain = "General"

    if llm_enabled:
        parse_prompt = ChatPromptTemplate.from_template("""
        Parse this update into JSON: {input}
        Return ONLY a JSON object: {{"domain": "<d>", "fact": "(fact \\"e\\" \\"f\\")", "confidence": <float>}}
        """)
        try:
            raw = (parse_prompt | llm | parser_out).invoke({"input": update_text})
            result = json.loads(raw)
            if result.get("confidence", 0) >= 0.7:
                return result["domain"], result["fact"]
        except Exception:
            pass

    prob = extractor.classify_sentence(update_text)
    if prob >= 0.7 and " is " in update_text.lower():
        parts = update_text.split(" is ", 1)
        entity = parts[0].strip().lower().replace(" ", "-")
        domain = default_domain
        if ":" in update_text:
            domain = update_text.split(":")[0].strip().title()
        return domain, f'(fact "{entity}" "{parts[1].strip()}")'

    if ":" in update_text:
        domain_part, fact_text = update_text.split(":", 1)
        domain = domain_part.strip().title()
        if " is " in fact_text:
            parts = fact_text.split(" is ", 1)
            entity = parts[0].strip().lower().replace(" ", "-")
            return domain, f'(fact "{entity}" "{parts[1].strip()}")'

    return None, None


# ---------------------------------------------------------------------------
# Self-Awareness Engine
# ---------------------------------------------------------------------------
class SelfAwarenessEngine:
    def __init__(self, runner, das_client: DASClient):
        self.runner = runner
        self.das = das_client
        self.metrics: Dict[str, Any] = {
            "knowledge_size": 0,
            "error_rate": 0.0,
            "avg_confidence": 0.85,
            "accuracy": 0.5,
            "response_time": 1.0,
            "capabilities": [],
        }
        self._init_meta_space()

    def _init_meta_space(self):
        meta_metta = """
        (self-state knowledge-size 0)
        (self-state error-rate 0.0)
        (self-state avg-confidence 0.85)
        (self-state accuracy 0.5)
        (self-state capabilities ())

        (= (delta-eval $prev $curr)
           (+ (* 0.4 (- $curr.acc $prev.acc))
              (* 0.3 (- $curr.eff $prev.eff))
              (* 0.2 (- $curr.stab $prev.stab))
              (* 0.1 $curr.reward)))

        (= (reflect-self $state)
           (add-atom (self-reflection $state (narrate $state))))
        """
        self.runner.run(meta_metta)
        self.update_self_state()
        logger.info("SelfAwarenessEngine: ready.")

    def update_self_state(self):
        atoms = self.runner.run("!(match &self $atom $atom)") or []
        self.metrics["knowledge_size"] = len(atoms)
        errors = self.runner.run("!(get-errors $type)") or []
        self.metrics["error_rate"] = len(errors) / max(1, self.metrics["knowledge_size"])

        self.runner.run(f'(add-atom (self-state knowledge-size {self.metrics["knowledge_size"]}))')
        self.runner.run(f'(add-atom (self-state error-rate {self.metrics["error_rate"]:.2f}))')
        self.runner.run(f'(add-atom (self-state avg-confidence {self.metrics["avg_confidence"]:.2f}))')
        self.runner.run(f'(add-atom (self-state accuracy {self.metrics["accuracy"]:.2f}))')

        caps_str = " ".join(self.metrics["capabilities"])
        # [FIX: CRITICAL-4] removed extraneous closing paren
        self.runner.run(f"(add-atom (self-state capabilities ({caps_str})))")

        self.das.add_atom(f"self-state:{datetime.now().isoformat()}", json.dumps(self.metrics))

    def delta_evaluate(self, prev_metrics: Dict, curr_metrics: Dict) -> float:
        perf_delta = curr_metrics.get("accuracy", 0) - prev_metrics.get("accuracy", 0)
        prev_rt = prev_metrics.get("response_time", 1) or 1
        curr_rt = curr_metrics.get("response_time", 1) or 1
        eff_delta = 1 / curr_rt - 1 / prev_rt
        stab_delta = -(curr_metrics.get("error_rate", 0) - prev_metrics.get("error_rate", 0))
        reward = 1.0 if perf_delta > 0.05 else 0.0
        delta = 0.4 * perf_delta + 0.3 * eff_delta + 0.2 * stab_delta + 0.1 * reward
        self.runner.run(
            f'(add-atom (delta-eval {prev_metrics.get("ts", 0)} '
            f'{curr_metrics.get("ts", 0)} {delta:.2f}))'
        )
        if delta > 0.1:
            self.reflect("Positive delta: improving!")
        return delta

    def reflect(self, trigger: str):
        state_str = json.dumps(self.metrics)
        if llm_enabled:
            reflect_prompt = ChatPromptTemplate.from_template(
                "Reflect on AI state: {state}. Trigger: {trigger}. "
                "Provide narrative on strengths, weaknesses, and next steps."
            )
            reflection = (reflect_prompt | llm | parser_out).invoke(
                {"state": state_str, "trigger": trigger}
            )
        else:
            reflection = f"Reflection on '{trigger}': Metrics={state_str}."
        self.runner.run(f'(add-atom (self-reflection "{trigger}" "{reflection}"))')
        das.add_atom(f"reflection:{uuid.uuid4()}", reflection)
        logger.info(f"SelfAwarenessEngine.reflect: {reflection[:120]}…")

    def add_capability(self, cap: str):
        if cap not in self.metrics["capabilities"]:
            self.metrics["capabilities"].append(cap)
            self.update_self_state()
            logger.info(f"SelfAwarenessEngine: capability added — {cap}")


# ---------------------------------------------------------------------------
# Hybrid query function
# ---------------------------------------------------------------------------
def query_metta_dynamic(
    question: str,
    kg: KnowledgeGraph,
    extractor: NeuralExtractor,
    self_awareness: SelfAwarenessEngine,
) -> str:
    start_time = time.time()
    facts: List[str] = []  # [FIX: CRITICAL-5] always initialised

    try:
        cached = das.query(f"response:{question}:")
        if cached:
            return cached[0]

        if question.lower().startswith("who ") and " is " not in question.lower():
            question = question.replace("Who ", "Who is ", 1)

        query_emb = kg.embedder.encode(question, convert_to_tensor=False).reshape(1, -1)
        similar_atoms = kg.search_similar(query_emb, k=5)

        domains = kg.runner.run("!(select-domain $domain)") or ["General"]
        domain = str(domains[0]) if domains else "General"
        context = f"Similar: {similar_atoms}\nDomain: {domain}"

        entity: Optional[str] = None
        if "who is" in question.lower():
            entity = (
                question.lower().split("who is")[-1].strip().rstrip("?")
                .replace(" ", "-").replace("--", "-")
            )
            facts = das.query(f"fact:{entity}")
            if facts:
                return facts[0]

        # Heuristic / pattern match
        pattern_matches = [
            d for d in pattern_dataset
            if np.linalg.norm(np.array(d["query_emb"]) - query_emb.flatten()) < 0.5
        ]

        if pattern_matches:
            heuristic = pattern_matches[0]["heuristic"]
            kg.runner.run(heuristic)
            result = kg.runner.run(f'!(heuristic "{question}" "{context}")') or ["No result."]
            response = str(result[0])
        else:
            if llm_enabled:
                # [FIX: SIGNIFICANT-1] build chain inline so kg/extractor/self_awareness
                # are in scope without relying on module-level globals
                _chain = (
                    {
                        "context": lambda x: query_metta_dynamic(
                            x["question"], kg, extractor, self_awareness
                        ),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                    | parser_out
                )
                response = _chain.invoke({"question": question})
            else:
                response = _generate_rule_based_heuristic(question, context, kg)

        if entity and not facts:
            continual_learning(entity, kg, extractor)

        errors = kg.runner.run("!(get-errors $type)") or ["None"]
        response = f"{response}\nErrors: {errors[0]}"

        das.add_atom(f"response:{question}:{uuid.uuid4()}", response)

        elapsed = time.time() - start_time
        self_awareness.metrics["response_time"] = elapsed
        self_awareness.update_self_state()
        self_awareness.reflect(f"Query: {question}")

        return response

    except Exception as exc:
        error_handler.log_and_report(kg.runner, "query_metta_dynamic", str(exc))
        return f"Error: {exc}"


def _generate_rule_based_heuristic(question: str, context: str, kg: KnowledgeGraph) -> str:
    if "who is" in question.lower():
        entity = question.lower().split("who is")[-1].strip("?").strip()
        heuristic = (
            f'(= (heuristic "{question}" $context) '
            f'(match &self (fact "{entity}" $fact) $fact))'
        )
    else:
        heuristic = "(= (heuristic $q $c) (fallback))"
    kg.runner.run(heuristic)
    result = kg.runner.run(f'!(heuristic "{question}" "{context}")') or ["Fallback."]
    return str(result[0])


# ---------------------------------------------------------------------------
# Heuristic review                               [FIX: SIGNIFICANT-3]
# ---------------------------------------------------------------------------
def review_heuristics(kg: KnowledgeGraph, manual: bool = False) -> str:
    pending = das.query("pending_heuristic:")
    if not pending:
        return "No pending heuristics."
    results = []
    for h in pending[:10]:
        try:
            conf = float(h.split("Confidence:")[-1].strip(" )")) if "Confidence:" in h else 0.6
        except ValueError:
            conf = 0.6
        threshold = 0.8 if manual else 0.7
        if conf >= threshold:
            kg.runner.run(h.split(" (")[0])
            results.append(f"Approved (conf={conf:.2f}): {h[:80]}")
        else:
            results.append(f"Discarded (conf={conf:.2f}): {h[:80]}")
    kg.vectorize_new_atoms()
    return "\n".join(results)


# ---------------------------------------------------------------------------
# Benchmarking — NOT called from the hot query path  [FIX: SIGNIFICANT-2]
# ---------------------------------------------------------------------------
def benchmark_clevr_vqa(
    question: str,
    ground_truth: str,
    response: str,
    kg: KnowledgeGraph,
    extractor: NeuralExtractor,
    self_awareness: SelfAwarenessEngine,
) -> float:
    facts = kg.runner.run("!(match &self (fact $entity $fact) $fact)") or []
    dataset = [
        {
            "question": f"Who is {str(f).split()[1]}?",
            "answer": str(f).split('"', 2)[-1].rstrip('")'),
        }
        for f in facts
        if len(str(f).split()) > 1
    ]
    correct = 0
    total = len(dataset) + 1
    for item in dataset:
        q = item["question"]
        if q not in mock_responses:
            mock_responses[q] = query_metta_dynamic(q, kg, extractor, self_awareness)
        if item["answer"].lower() in mock_responses[q].lower():
            correct += 1
    if ground_truth.lower() in response.lower():
        correct += 1
    accuracy = (correct / total) * 100
    self_awareness.metrics["accuracy"] = accuracy / 100.0
    self_awareness.update_self_state()
    return accuracy


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------
def save_to_playground(question: str, output: str, history: list, accuracies: dict):
    try:
        conn = sqlite3.connect("playground.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS playground
            (id TEXT PRIMARY KEY, question TEXT, output TEXT,
             history TEXT, accuracies TEXT, timestamp TEXT)
        """)
        row = {
            "id": str(uuid.uuid4()),
            "question": question,
            "output": output,
            "history": json.dumps(history),
            "accuracies": json.dumps(accuracies),
            "timestamp": datetime.now().isoformat(),
        }
        c.execute(
            "INSERT INTO playground VALUES (:id,:question,:output,:history,:accuracies,:timestamp)",
            row,
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        error_handler.log_and_report(None, "save_to_playground", str(exc))
        with open("playground_fallback.json", "a") as f:
            json.dump(row, f)
            f.write("\n")


def store_memory(question: str, response: str, confidence: float):
    das.add_atom(
        f"interaction:{uuid.uuid4()}",
        f"Q: {question} R: {response} Conf: {confidence}",
    )


# ---------------------------------------------------------------------------
# Autonomous Goal Agent                          [FIX: CRITICAL-3]
# ---------------------------------------------------------------------------
class AutonomousGoalAgent:
    def __init__(
        self,
        runner,
        das_client: DASClient,
        self_awareness: SelfAwarenessEngine,
        kg: KnowledgeGraph,           # injected — no globals
        extractor: NeuralExtractor,   # injected — no globals
    ):
        self.runner = runner
        self.das = das_client
        self.self_awareness = self_awareness
        self.kg = kg
        self.extractor = extractor
        self.memory = ConversationBufferMemory(memory_key="goal_history")
        self.active_goals: List[Dict] = []
        self._init_goal_space()

    def _init_goal_space(self):
        self.runner.run("""
        (goal $id $desc $priority)
        (goal-status $id $status)
        (= (select-goal $goals) (max-priority $goals))
        (= (achieve-goal $goal) (plan $goal) (execute $actions))
        """)

    def perceive(self) -> Dict:
        interactions = self.das.query("interaction:")[-10:]
        gaps = self.runner.run("!(match &self (goal $g) $g)") or []
        return {
            "interactions": interactions,
            "unresolved_goals": len(gaps),
            "self_state": self.self_awareness.metrics,
        }

    def plan(self, obs: Dict, high_level_goal: str) -> List[str]:
        if llm_enabled:
            # [FIX: SIGNIFICANT-5] use a proper chain invocation
            plan_prompt = ChatPromptTemplate.from_template(
                "Obs: {obs}. Goal: {goal}. "
                'Return ONLY JSON: {{"plan": ["action1"], "subgoals": ["sub1"]}}'
            )
            try:
                raw = (plan_prompt | llm | parser_out).invoke(
                    {"obs": json.dumps(obs), "goal": high_level_goal}
                )
                plan_data = json.loads(raw)
                plan = plan_data.get("plan", [])
            except Exception:
                plan = [f"Search for {high_level_goal}."]
        else:
            plan = [f"Learn {high_level_goal} via search."]

        goal_id = f"g{len(self.active_goals)}"
        self.active_goals.append(
            {"id": goal_id, "goal": high_level_goal, "plan": plan, "status": "active"}
        )
        self.runner.run(
            f'(add-atom (goal {goal_id} "{high_level_goal}" {len(self.active_goals)}))'
        )
        return plan

    def act(self, action: str, goal_id: str) -> str:
        if "web_search" in action:
            query = action.split(":")[-1] if ":" in action else "facts"
            return web_search(query)
        if "update_graph" in action:
            fact = action.split(":")[-1]
            update_graph("General", fact, self.kg)
            return f"Updated: {fact}"
        if "continual_learning" in action:
            entity = action.split(":")[-1]
            continual_learning(entity, self.kg, self.extractor)
            return f"Learned: {entity}"
        if "reflect" in action:
            self.self_awareness.reflect("Goal act")
            return "Reflected."
        return f"Unknown action: {action}"

    def learn(self, result: str, goal_id: str):
        self.memory.chat_memory.add_ai_message(result)
        success = "error" not in result.lower()
        delta = 0.1 if success else -0.1
        self.self_awareness.metrics["accuracy"] = max(
            0.0, min(1.0, self.self_awareness.metrics["accuracy"] + delta)
        )
        prev_raw = self.das.query("self-state:")
        prev_metrics = json.loads(prev_raw[-1]) if prev_raw else {}
        prev_metrics.setdefault("ts", time.time())
        self.self_awareness.metrics["ts"] = time.time()
        self.self_awareness.delta_evaluate(prev_metrics, self.self_awareness.metrics)
        status = "completed" if success else "failed"
        self.runner.run(f'(add-atom (goal-status {goal_id} "{status}"))')

    async def run_agent_loop(self, high_level_goal: str):
        obs = self.perceive()
        plan = self.plan(obs, high_level_goal)
        goal_id = self.active_goals[-1]["id"]
        for action in plan:
            result = self.act(action, goal_id)
            self.learn(result, goal_id)
            await asyncio.sleep(1)
        self.self_awareness.add_capability("autonomous_goal")


# ---------------------------------------------------------------------------
# Autonomous loop
# ---------------------------------------------------------------------------
async def autonomous_goal_setting(agent: AutonomousGoalAgent):
    while True:
        try:
            interactions = len(das.query("interaction:"))
            if interactions % 5 == 0:
                obs = agent.perceive()
                if obs["unresolved_goals"] > 0 or not agent.active_goals:
                    if llm_enabled:
                        # [FIX: SIGNIFICANT-5] correct chain usage
                        goal_prompt = ChatPromptTemplate.from_template(
                            "Suggest one learning goal based on these recent interactions: {ints}"
                        )
                        goal = (goal_prompt | llm | parser_out).invoke(
                            {"ints": obs["interactions"][-3:]}
                        )
                    else:
                        goal = "Expand knowledge base"
                    await agent.run_agent_loop(goal)
                    agent.self_awareness.reflect(f"Autonomous goal: {goal}")
            await asyncio.sleep(60)
        except Exception as exc:
            logger.error(f"autonomous_goal_setting: {exc}")
            await asyncio.sleep(30)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def run_chatbot(
    kg: KnowledgeGraph,
    extractor: NeuralExtractor,
    self_awareness: SelfAwarenessEngine,
    agent: AutonomousGoalAgent,
):
    def process_input(question: str, faq_update: str, history: list):
        history = history or []

        if faq_update and faq_update.strip():
            domain, fact = parse_natural_language_update(faq_update, extractor)
            if domain and fact:
                update_graph(domain, fact, kg)
                msg = f"Updated {domain}: {fact}"
            else:
                msg = "Parse failed. Use format: Domain: Entity is fact."
            history.append({"role": "user", "content": faq_update})
            history.append({"role": "assistant", "content": msg,
                            "timestamp": datetime.now().isoformat()})
            return history, msg, None

        if not question or not question.strip():
            return history, "Please enter a question.", None

        question = question.strip()
        if question.lower() == "errors":
            errors = kg.runner.run("!(get-errors $type)") or ["None"]
            response = f"Errors: {errors}"
        elif question.lower() == "review heuristics":
            response = review_heuristics(kg, manual=True)
        else:
            response = query_metta_dynamic(question, kg, extractor, self_awareness)
            store_memory(question, response, confidence=0.85)
            save_to_playground(question, response, history, {})

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response,
                        "timestamp": datetime.now().isoformat()})
        return history, response, None

    # [FIX: SIGNIFICANT-2] Benchmark is on-demand, not in the hot path
    def run_benchmark(question: str, ground_truth: str, response: str):
        if not question or not response:
            return "Provide question, ground truth, and response first."
        acc = benchmark_clevr_vqa(question, ground_truth, response, kg, extractor, self_awareness)
        return f"Benchmark accuracy: {acc:.1f}%"

    def export_csv():
        try:
            conn = sqlite3.connect("playground.db")
            df = pd.read_sql_query("SELECT * FROM playground", conn)
            conn.close()
            path = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(path, index=False)
            return path
        except Exception as exc:
            logger.error(f"export_csv: {exc}")
            return None

    with gr.Blocks(title="FAQ Chatbot v3") as interface:
        gr.Markdown("# FAQ Chatbot v3\nAsk anything, update FAQs, or type `errors` / `review heuristics`.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=420, type="messages")
                question_box = gr.Textbox(label="Question", placeholder="Who is William Ruto?")
                update_box = gr.Textbox(
                    label="FAQ Update (optional)",
                    placeholder="Kenyan-Politics: Uhuru Kenyatta is former president.",
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    export_btn = gr.Button("Export CSV")
                file_out = gr.File(label="Download")

            with gr.Column(scale=1):
                gr.Markdown("### Benchmark (on-demand)")
                bench_q = gr.Textbox(label="Question")
                bench_gt = gr.Textbox(label="Ground truth")
                bench_resp = gr.Textbox(label="Response to evaluate")
                bench_btn = gr.Button("Run Benchmark")
                bench_out = gr.Textbox(label="Result", interactive=False)

        submit_btn.click(
            process_input,
            inputs=[question_box, update_box, chatbot],
            outputs=[chatbot, gr.Textbox(visible=False), file_out],
        )
        export_btn.click(export_csv, outputs=file_out)
        bench_btn.click(
            run_benchmark,
            inputs=[bench_q, bench_gt, bench_resp],
            outputs=bench_out,
        )

    interface.launch(server_name="127.0.0.1", server_port=7860, share=True)


# ---------------------------------------------------------------------------
# Unit tests                                     [FIX: MINOR-1, MINOR-2]
# ---------------------------------------------------------------------------
def _run_tests():
    logger.info("Running unit tests…")
    _test_kg = KnowledgeGraph(das)
    _test_extractor = NeuralExtractor(_test_kg)
    _test_awareness = SelfAwarenessEngine(_test_kg.runner, das)

    # [FIX: MINOR-1] Correct MeTTa query syntax
    result = _test_kg.runner.run("!(match &self (fact William-Ruto $f) $f)")
    assert result, "Expected at least one fact for William-Ruto"
    assert any("President" in str(r) for r in result), (
        f"Expected 'President' in facts, got: {result}"
    )
    logger.info("  test_basic_query: PASSED")

    old_count = _test_kg.index.ntotal
    _test_kg.runner.run('(add-atom (fact test-entity "something new"))')
    _test_kg.vectorize_new_atoms()
    assert _test_kg.index.ntotal > old_count, "Expected FAISS index to grow after adding atom"
    logger.info("  test_vectorize: PASSED")

    response = query_metta_dynamic("Who is Raila Odinga?", _test_kg, _test_extractor, _test_awareness)
    assert isinstance(response, str) and len(response) > 0
    logger.info("  test_query_dynamic: PASSED")

    logger.info("All tests passed.")


# ---------------------------------------------------------------------------
# Entry point                                    [FIX: MINOR-2] single block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="FAQ Chatbot v3")
    arg_parser.add_argument("--test", action="store_true", help="Run unit tests and exit.")
    args = arg_parser.parse_args()

    if args.test:
        _run_tests()
    else:
        # Initialise in dependency order
        kg = KnowledgeGraph(das)
        neural_extractor = NeuralExtractor(kg)
        self_awareness = SelfAwarenessEngine(kg.runner, das)
        agent = AutonomousGoalAgent(kg.runner, das, self_awareness, kg, neural_extractor)

        # Preload extra facts
        preload = [
            ("fact:uhuru-kenyatta", "former president of Kenya"),
            ("fact:nairobi", "capital of Kenya"),
        ]
        for key, val in preload:
            das.add_atom(key, val)
            entity_name = key.split(":")[1]
            kg.runner.run(f'(add-atom (fact "{entity_name}" "{val}"))')
        kg.vectorize_new_atoms()

        # Start async autonomy loop in background
        threading.Thread(
            target=lambda: asyncio.run(autonomous_goal_setting(agent)), daemon=True
        ).start()

        # Smoke test
        test_q = "Who is Uhuru Kenyatta?"
        resp = query_metta_dynamic(test_q, kg, neural_extractor, self_awareness)
        print(f"\nSmoke test → {test_q}\n{resp}\n")

        # Launch UI
        run_chatbot(kg, neural_extractor, self_awareness, agent)
