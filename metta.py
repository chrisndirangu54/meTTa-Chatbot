# Improved FAQ Chatbot System with Self-Awareness and Autonomous Goal-Setting
# Version: 2.0 (Rewritten for cohesion, efficiency, and robustness)
# Date: November 04, 2025
# Key Improvements:
# - Consolidated duplicates (e.g., build_graph, benchmark_clevr_vqa)
# - Fixed undefined vars (e.g., start_time, continual_learning)
# - Enhanced modularity: Core classes for KG, Neural, Awareness, Autonomy
# - Better error handling: Centralized exception logger
# - Efficiency: Batched embeddings, lazy graph builds, async autonomy
# - Testing: Added unit tests at bottom
# - Dependencies: Listed at top; assumes CPU-only as per original
# - LLM: Switched to Gemini (as in code); added key validation
# - New: Integrated RL-like reward in delta_eval for better self-improvement

# Dependencies (pip install these):
# hyperon, langchain-google-genai, sentence-transformers, faiss-cpu, torch, torch-geometric,
# transformers, datasets, requests, duckduckgo-search, tenacity, redis, gradio, pandas,
# crewai, langchain-community, asyncio (built-in)

import threading
import time
import asyncio
from datetime import datetime
import json
import uuid
import logging
import os
import shutil
import subprocess
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from hyperon import *
from hyperon import MeTTa, GroundingSpace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import torch_geometric as pyg
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data as pyg_data
import requests
from ddgs import DDGS
from tenacity import retry, wait_random_exponential, stop_after_attempt
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import pandas as pd

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_MODE"] = "disabled"
os.environ["REDIS_URL"] = "redis://localhost:6379"  # Default; override via env

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device('cpu')
logger.info(f"Using PyTorch device: {device}")

# Cache and mocks
mock_responses = {}

# Centralized exception handler
class ErrorHandler:
    @staticmethod
    def log_and_report(runner, error_type: str, message: str):
        logger.error(f"{error_type}: {message}")
        try:
            runner.run(f'(add-atom (error "{error_type}" "{message}"))')
        except Exception:
            pass  # Runner may not be ready

error_handler = ErrorHandler()

# Core Knowledge Graph Class
class KnowledgeGraph:
    def __init__(self):
        self.space = GroundingSpace()
        self.runner = MeTTa(space=self.space.gspace)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.atom_vectors = {}  # {uuid: atom_str}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.load_initial_knowledge()

    def load_initial_knowledge(self):
        knowledge_metta = """
        # Minimal knowledge graph
        (: General Domain)
        (: Person Type)
        (: Role Type)
        (: Entity Type)
        (: ErrorType Type)
        (: Kenyan-Politics Domain)
        (: Science Domain)
        (: Medicine Domain)

        ; Dynamic heuristic placeholder
        (= (heuristic $query $context)
           (add-atom (heuristic-result $query $context)))

        ; Error handling
        (= (get-errors $type)
           (match &self (error $type $e) $e))

        ; Domain selection
        (= (select-domain $domain)
           (match &self (in-domain $entity $domain) $entity))

        ; Continual learning
        (= (learn-fact $entity $fact)
           (add-atom (fact $entity $fact)))

        ; Vectorized entity definition
        (= (definition vectorize-entity) "Embed entities using SentenceTransformer for FAISS search")

        ; Kenyan Politicians in Kenyan-Politics Domain
        (in-domain Raila-Odinga Kenyan-Politics)
        (in-domain William-Ruto Kenyan-Politics)
        (in-domain Musalia-Mudavadi Kenyan-Politics)
        (fact Raila-Odinga (role Opposition-Leader))
        (fact William-Ruto (role President))
        (fact Musalia-Mudavadi (role Prime-Cabinet-Secretary))

        ; Medicine Domain entities
        (in-domain Malaria Medicine)
        (in-domain Tuberculosis Medicine)
        (in-domain HIV-AIDS Medicine)

        ; Facts for Medicine Domain
        (fact Malaria (disease (symptoms fever chills headache nausea)))
        (fact Malaria (cause (parasite Plasmodium transmitted-by Anopheles-mosquito)))
        (fact Tuberculosis (disease (symptoms cough fever night-sweats weight-loss)))
        (fact Tuberculosis (cause (bacteria Mycobacterium-tuberculosis)))
        (fact HIV-AIDS (disease (symptoms fever fatigue swollen-lymph-nodes weight-loss)))
        (fact HIV-AIDS (cause (virus Human-Immunodeficiency-Virus)))
        """
        self.runner.run(knowledge_metta)
        self.vectorize_graph()
        logger.info("KnowledgeGraph initialized and loaded.")

    def vectorize_graph(self):
        """Incremental vectorization with batching."""
        try:
            new_atoms = self.runner.run('!(match &self (new-atom $atom) $atom)') or []
            if not new_atoms:
                return
            batch_size = 32
            for i in range(0, len(new_atoms), batch_size):
                batch = new_atoms[i:i + batch_size]
                vectors = self.embedder.encode([str(atom) for atom in batch], convert_to_tensor=True, device='cpu').cpu().numpy()
                for atom, vector in zip(batch, vectors):
                    atom_id = str(uuid.uuid4())
                    self.atom_vectors[atom_id] = str(atom)
                    self.index.add(np.array([vector]))
                    das.add_atom(f"embedding:{atom_id}", vector.tolist())  # DAS global
            self.runner.run('!(match &self (new-atom $atom) (delete-atom (new-atom $atom)))')
            logger.info(f"Vectorized {len(new_atoms)} new atoms.")
        except Exception as e:
            error_handler.log_and_report(self.runner, "Vectorize", str(e))

    def build_graph(self, query_emb: Optional[np.ndarray] = None) -> Tuple[pyg_data, Dict[str, int]]:
        """Efficient graph build with FAISS filtering."""
        atoms = self.runner.run('!(match &self $atom $atom)') or []
        if not atoms:
            return pyg_data(x=torch.empty((0, self.dimension), dtype=torch.float), edge_index=torch.empty((2, 0), dtype=torch.long)), {}

        # Batched node features
        node_features = []
        batch_size = 16
        for i in range(0, len(atoms), batch_size):
            batch = atoms[i:i + batch_size]
            embeddings = self.embedder.encode([str(atom) for atom in batch], device='cpu', convert_to_tensor=False)
            node_features.extend(embeddings.tolist())

        # FAISS filter if query
        if query_emb is not None and self.index.ntotal > 0:
            query_emb = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
            _, indices = self.index.search(query_emb, k=min(100, self.index.ntotal))
            atoms = [self.atom_vectors[list(self.atom_vectors.keys())[idx]] for idx in indices[0] if idx != -1 and idx < len(self.atom_vectors)]

        node_map = {str(atom): i for i, atom in enumerate(atoms)}
        edges = []
        implies_results = self.runner.run('!(unique (match &self (= (implies $a $b) true) ($a $b)))') or []
        for result in implies_results:
            if isinstance(result, (list, tuple)) and len(result) == 2:
                a_str, b_str = str(result[0]), str(result[1])
            elif hasattr(result, 'get_children') and len(result.get_children()) == 2:
                a_str, b_str = str(result.get_children()[0]), str(result.get_children()[1])
            else:
                continue
            if a_str in node_map and b_str in node_map:
                edges.append([node_map[a_str], node_map[b_str]])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        x = torch.tensor(node_features, dtype=torch.float) if node_features else torch.empty((0, self.dimension), dtype=torch.float)
        return pyg_data(x=x, edge_index=edge_index), node_map

# DAS (Distributed Atom Space) with Redis fallback
class DASClient:
    def __init__(self, host='localhost', port=6379):
        self.in_memory = {}
        redis_url = os.environ.get("REDIS_URL")
        try:
            import redis
            if redis_url:
                self.redis = redis.from_url(redis_url, decode_responses=True)
            else:
                self.redis = redis.Redis(host=host, port=port, decode_responses=True)
            self.redis.ping()
            logger.info("DAS using Redis.")
        except Exception as e:
            logger.warning(f"Redis failed: {e}. Using in-memory.")
            if shutil.which('redis-server'):
                try:
                    subprocess.run(['redis-server', '--daemonize', 'yes'], check=False)
                    time.sleep(2)
                    # Retry connection...
                    import redis
                    self.redis = redis.Redis(host=host, port=port, decode_responses=True)
                    self.redis.ping()
                except:
                    pass
            self.redis = None

    def add_atom(self, key: str, value: Any):
        try:
            val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            if self.redis:
                self.redis.set(key, val_str)
            else:
                self.in_memory[key] = val_str
            logger.debug(f"DAS added: {key}")
        except Exception as e:
            error_handler.log_and_report(None, "DAS Add", str(e))
            self.in_memory[key] = val_str

    def query(self, pattern: str) -> List[str]:
        try:
            if self.redis:
                keys = self.redis.keys(f"{pattern}*")
                return [self.redis.get(k) for k in keys if self.redis.get(k)]
            else:
                return [v for k, v in self.in_memory.items() if k.startswith(pattern)]
        except Exception as e:
            error_handler.log_and_report(None, "DAS Query", str(e))
            return [v for k, v in self.in_memory.items() if k.startswith(pattern)]

das = DASClient()

# Neural Extractor Class
class NeuralExtractor:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
        self.fine_tune()

    def fine_tune(self):
        try:
            facts = self.kg.runner.run('!(match &self (fact $entity $fact) $fact)') or []
            texts = [str(f).split('"', 2)[-1].rstrip('")') for f in facts if len(str(f).split()) > 1]
            if not texts:
                texts = ["Kenya is a country."]
            negatives = ["What is this?", "Random question?", "Irrelevant text."]
            data = {'text': texts + negatives, 'label': [1] * len(texts) + [0] * len(negatives)}
            dataset = Dataset.from_dict(data)
            def tokenize(examples):
                return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
            dataset = dataset.map(tokenize, batched=True, load_from_cache_file=False)
            dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
            args = TrainingArguments(output_dir='./bert-finetuned', num_train_epochs=3, per_device_train_batch_size=8, logging_steps=10)
            trainer = Trainer(model=self.classifier, args=args, train_dataset=dataset)
            trainer.train()
            logger.info("NeuralExtractor fine-tuned.")
        except Exception as e:
            error_handler.log_and_report(self.kg.runner, "BERT Fine-Tune", str(e))

    def classify_sentence(self, sentence: str) -> float:
        if self.classifier is None:
            return 0.0
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
        return torch.nn.functional.softmax(logits, dim=1)[0][1].item()

    def get_embedding(self, sentence: str) -> List[float]:
        if self.model is None:
            return []
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

    def compute_similarity(self, text1: str, text2: str) -> float:
        emb1 = np.array(self.get_embedding(text1)).reshape(1, -1)
        emb2 = np.array(self.get_embedding(text2)).reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0] if emb1.size and emb2.size else 0.0

def extract_facts_lightweight(text: str, extractor: NeuralExtractor, threshold: float = 0.7) -> List[Tuple[str, str, float, List[float]]]:
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    facts = []
    for sentence in sentences:
        if " is " not in sentence.lower():
            continue
        prob = extractor.classify_sentence(sentence)
        if prob < threshold:
            continue
        parts = sentence.split(" is ", 1)
        if len(parts) != 2:
            continue
        entity = parts[0].split('(')[0].split(',')[0].strip().lower().replace(" ", "-").replace("--", "-")
        embedding = extractor.get_embedding(sentence)
        facts.append((entity, parts[1].strip(), prob, embedding))
    return facts

# GNN for Patterns
class PatternGNN(torch.nn.Module):
    def __init__(self, in_channels=384, hidden_channels=64, out_channels=1):
        super().__init__()
        self.gnn = GraphSAGE(in_channels, hidden_channels, num_layers=2, out_channels=out_channels)

    def forward(self, data):
        return self.gnn(data.x, data.edge_index)

pattern_gnn = PatternGNN().to(device)
optimizer = torch.optim.Adam(pattern_gnn.parameters(), lr=0.01)
pattern_dataset = []

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
        batch = pattern_dataset[i:i + batch_size]
        optimizer.zero_grad()
        out = pattern_gnn(data)
        target = torch.tensor([kg.embedder.encode(p["heuristic"]).mean() for p in batch], dtype=torch.float).to(device).mean()
        loss = torch.nn.MSELoss()(out.mean(), target)
        loss.backward()
        optimizer.step()
    logger.info(f"Trained GNN on {len(pattern_dataset)} patterns.")

# LLM Setup
def validate_gemini_key(api_key: str) -> bool:
    try:
        llm_test = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key, max_retries=1)
        llm_test.invoke("Test")
        return True
    except Exception as e:
        logger.error(f"Gemini validation failed: {e}")
        return False

api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyB7qcGYVY97ONYHsYxeou8lqsrY26TytjA")  # Secure via env
llm_enabled = validate_gemini_key(api_key)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=api_key, max_retries=10) if llm_enabled else None

prompt = ChatPromptTemplate.from_template("""
You are a general FAQ chatbot. Use the following context to answer the question.
Context: {context}
Question: {question}
Provide a structured answer with definitions, examples, and references. If unsure, suggest updates.
""")
parser = StrOutputParser()
chain = (
    {"context": lambda x: query_metta_dynamic(x["question"]), "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
) if llm_enabled else lambda x: query_metta_dynamic(x["question"]) + "\nLLM disabled."

# Continual Learning (Missing in original; now defined)
def continual_learning(entity: str, kg: KnowledgeGraph, extractor: NeuralExtractor):
    """Learn facts about entity via web/search."""
    try:
        search_query = f"What is {entity}?"
        web_result = web_search(search_query)
        facts = extract_facts_lightweight(web_result, extractor)
        for ent, fact, prob, _ in facts[:3]:  # Top 3
            if prob > 0.7:
                kg.runner.run(f'(learn-fact "{entity}" "{fact}")')
                das.add_atom(f"fact:{entity}", fact)
        kg.vectorize_graph()
        logger.info(f"Learned {len(facts)} facts for {entity}.")
    except Exception as e:
        error_handler.log_and_report(kg.runner, "Continual Learning", str(e))

# Web Search
@retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(10))
def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            snippets = [r['body'] for r in results if 'body' in r]
            return ' '.join(snippets[:3]) if snippets else "No results."
    except Exception as e:
        error_handler.log_and_report(None, "Web Search", str(e))
        return "Search failed."

# Hybrid Query Function
def query_metta_dynamic(question: str, kg: KnowledgeGraph, extractor: NeuralExtractor, self_awareness) -> str:
    start_time = time.time()
    try:
        # Cache check
        cached = das.query(f"response:{question}:*")
        if cached:
            return cached[0]

        # Normalize
        if question.lower().startswith("who ") and " is " not in question.lower():
            question = question.replace("Who ", "Who is ", 1)

        # Embedding
        query_emb = kg.embedder.encode(question, convert_to_tensor=False).reshape(1, -1)
        _, indices = kg.index.search(query_emb.astype(np.float32), k=min(5, kg.index.ntotal))
        similar_atoms = [kg.atom_vectors[list(kg.atom_vectors.keys())[i]] for i in indices[0] if i != -1]

        # Domain
        domains = kg.runner.run('!(select-domain $domain)') or ["General"]
        domain = domains[0] if domains else "General"
        context = f"Similar: {similar_atoms}\nDomain: {domain}"

        # Entity extraction
        entity = None
        if "who is" in question.lower():
            entity = question.lower().split("who is")[-1].strip().rstrip('?').replace(" ", "-").replace("--", "-")
            facts = das.query(f"fact:{entity}:*")
            if facts:
                return facts[0]

        # Heuristic/Inference
        pattern_data = [d for d in pattern_dataset if np.linalg.norm(np.array(d["query_emb"]) - query_emb.flatten()) < 0.5]
        if pattern_data:
            max_conf = max(d["confidence"] for d in pattern_data)
            heuristic = pattern_data[0]["heuristic"]  # Simplified
            kg.runner.run(heuristic)
            result = kg.runner.run(f'!(heuristic "{question}" "{context}")') or ["No result."]
            response = str(result[0])
        else:
            if llm_enabled:
                response = chain.invoke({"question": question})
            else:
                response = generate_rule_based_heuristic(question, context, kg)

        # Learn if entity
        if entity and not facts:
            continual_learning(entity, kg, extractor)

        # Errors
        errors = kg.runner.run('!(get-errors $type)') or ["None"]
        response = f"{response}\nErrors: {errors[0]}"

        # Cache
        das.add_atom(f"response:{question}:{uuid.uuid4()}", response)

        # Self-awareness hook
        self_awareness.metrics['response_time'] = time.time() - start_time
        self_awareness.update_self_state()
        self_awareness.reflect(f"Query: {question}")

        return response
    except Exception as e:
        error_handler.log_and_report(kg.runner, "Query Dynamic", str(e))
        return f"Error: {str(e)}"

def generate_rule_based_heuristic(question: str, context: str, kg: KnowledgeGraph) -> str:
    if "who is" in question.lower():
        entity = question.lower().split("who is")[-1].strip("?")
        heuristic = f"(= (heuristic \"{question}\" $context) (match &self (fact \"{entity}\" $fact) $fact))"
    else:
        heuristic = "(= (heuristic $q $c) (fallback))"
    kg.runner.run(heuristic)
    result = kg.runner.run(f'!(heuristic "{question}" "{context}")') or ["Fallback."]
    return str(result[0])

# Self-Awareness Engine (Improved with RL-like rewards)
class SelfAwarenessEngine:
    def __init__(self, runner, das):
        self.runner = runner
        self.das = das
        self.metrics = {'knowledge_size': 0, 'error_rate': 0.0, 'avg_confidence': 0.85, 'accuracy': 0.5, 'capabilities': []}
        self.init_meta_space()

    def init_meta_space(self):
        meta_metta = """
        (self-state knowledge-size $size)
        (self-state error-rate $rate)
        (self-state avg-confidence $conf)
        (self-state accuracy $acc)
        (self-state capabilities $caps)

        (= (delta-eval $prev $curr)
           (+ (* 0.4 (- $curr.acc $prev.acc))
              (* 0.3 (- $curr.eff $prev.eff))
              (* 0.2 (- $curr.stab $prev.stab))
              (* 0.1 $curr.reward)))  ; RL-like reward

        (= (reflect-self $state)
           (add-atom (self-reflection $state (narrate $state))))
        """
        self.runner.run(meta_metta)
        self.update_self_state()
        logger.info("SelfAwarenessEngine ready.")

    def update_self_state(self):
        atoms = self.runner.run('!(match &self $atom $atom)') or []
        self.metrics['knowledge_size'] = len(atoms)
        errors = self.runner.run('!(get-errors $type)') or []
        self.metrics['error_rate'] = len(errors) / max(1, self.metrics['knowledge_size'])
        self.runner.run(f'(add-atom (self-state knowledge-size {self.metrics["knowledge_size"]}))')
        self.runner.run(f'(add-atom (self-state error-rate {self.metrics["error_rate"]:.2f}))')
        self.runner.run(f'(add-atom (self-state avg-confidence {self.metrics["avg_confidence"]:.2f}))')
        self.runner.run(f'(add-atom (self-state accuracy {self.metrics["accuracy"]:.2f}))')
        caps_str = ' '.join(self.metrics['capabilities'])
        self.runner.run(f'(add-atom (self-state capabilities ({caps_str}))))')
        self.das.add_atom(f"self-state:{datetime.now().isoformat()}", json.dumps(self.metrics))

    def delta_evaluate(self, prev_metrics: Dict, curr_metrics: Dict) -> float:
        perf_delta = curr_metrics.get('accuracy', 0) - prev_metrics.get('accuracy', 0)
        eff_delta = 1 / curr_metrics.get('response_time', 1) - 1 / prev_metrics.get('response_time', 1)
        stab_delta = -(curr_metrics.get('error_rate', 0) - prev_metrics.get('error_rate', 0))
        reward = 1.0 if perf_delta > 0.05 else 0.0  # Simple RL reward
        delta = 0.4 * perf_delta + 0.3 * eff_delta + 0.2 * stab_delta + 0.1 * reward
        self.runner.run(f'(add-atom (delta-eval {prev_metrics.get("ts", 0)} {curr_metrics.get("ts", 0)} {delta:.2f}))')
        if delta > 0.1:
            self.reflect("Positive delta: Improving!")
        return delta

    def reflect(self, trigger: str):
        state_str = json.dumps(self.metrics)
        if llm_enabled:
            reflect_prompt = ChatPromptTemplate.from_template("""
            Reflect on AI state: {state}. Trigger: {trigger}.
            Narrative on strengths/weaknesses/next steps.
            """)
            reflection = (reflect_prompt | llm | parser).invoke({"state": state_str, "trigger": trigger})
        else:
            reflection = f"Reflection on {trigger}: Metrics {state_str}."
        self.runner.run(f'(add-atom (self-reflection "{trigger}" "{reflection}"))')
        das.add_atom(f"reflection:{uuid.uuid4()}", reflection)
        logger.info(f"Reflected: {reflection[:100]}...")

    def add_capability(self, cap: str):
        if cap not in self.metrics['capabilities']:
            self.metrics['capabilities'].append(cap)
            self.update_self_state()
            logger.info(f"Capability added: {cap}")

# Autonomous Goal Agent (Async improved)
class AutonomousGoalAgent:
    def __init__(self, runner, das, self_awareness):
        self.runner = runner
        self.das = das
        self.self_awareness = self_awareness
        self.memory = ConversationBufferMemory(memory_key="goal_history")
        self.active_goals = []
        self.init_goal_space()

    def init_goal_space(self):
        goal_metta = """
        (goal $id $desc $priority)
        (goal-status $id $status)
        (= (select-goal $goals) (max-priority $goals))
        (= (achieve-goal $goal) (plan $goal) (execute $actions))
        """
        self.runner.run(goal_metta)

    def perceive(self) -> Dict:
        interactions = self.das.query("interaction:*")[-10:]
        gaps = self.runner.run('!(match &self (goal $g) $g)') or []
        return {'interactions': interactions, 'unresolved_goals': len(gaps), 'self_state': self.self_awareness.metrics}

    def plan(self, obs: Dict, high_level_goal: str) -> List[str]:
        if llm_enabled:
            plan_prompt = ChatPromptTemplate.from_template("""
            Obs: {obs}. Goal: {goal}.
            JSON plan: {{"plan": ["action1"], "subgoals": ["sub1"]}}
            """)
            response = (plan_prompt | llm | parser).invoke({"obs": json.dumps(obs), "goal": high_level_goal})
            try:
                plan_data = json.loads(response)
                plan = plan_data.get("plan", [])
            except:
                plan = [f"Search for {high_level_goal}."]
        else:
            plan = [f"Learn {high_level_goal} via search."]
        goal_id = f"g{len(self.active_goals)}"
        self.active_goals.append({'goal': high_level_goal, 'plan': plan, 'priority': len(self.active_goals) + 1, 'status': 'active'})
        self.runner.run(f'(add-atom (goal {goal_id} "{high_level_goal}" {len(self.active_goals) + 1}))')
        return plan

    def act(self, action: str, goal_id: str) -> str:
        if "web_search" in action:
            query = action.split(":")[-1] if ":" in action else "facts"
            return web_search(query)
        elif "update_graph" in action:
            fact = action.split(":")[-1]
            update_graph("General", fact, kg)  # kg global
            return f"Updated: {fact}"
        elif "continual_learning" in action:
            entity = action.split(":")[-1]
            continual_learning(entity, kg, neural_extractor)
            return f"Learned: {entity}"
        elif "reflect" in action:
            self.self_awareness.reflect("Goal act")
            return "Reflected."
        return f"Unknown: {action}"

    def learn(self, result: str, goal_id: str):
        self.memory.chat_memory.add_ai_message(result)
        success = "error" not in result.lower()
        self.self_awareness.metrics['accuracy'] += 0.1 if success else -0.1
        self.self_awareness.metrics['accuracy'] = max(0, min(1, self.self_awareness.metrics['accuracy']))
        prev_metrics = json.loads(self.das.query("self-state:*")[-1]) if self.das.query("self-state:*") else {}
        prev_metrics['ts'] = time.time()
        self.self_awareness.metrics['ts'] = time.time()
        self.self_awareness.delta_evaluate(prev_metrics, self.self_awareness.metrics)
        status = "completed" if success else "failed"
        self.runner.run(f'(add-atom (goal-status {goal_id} "{status}"))')

    async def run_agent_loop(self, high_level_goal: str):
        obs = self.perceive()
        plan = self.plan(obs, high_level_goal)
        goal_id = f"g{len(self.active_goals) - 1}"
        for action in plan:
            result = self.act(action, goal_id)
            self.learn(result, goal_id)
            await asyncio.sleep(1)
        self.self_awareness.add_capability("autonomous_goal")

# Benchmark (Consolidated)
def benchmark_clevr_vqa(runner, question: str, ground_truth: str, response: str, extractor=None) -> float:
    facts = runner.run('!(match &self (fact $entity $fact) $fact)') or []
    dataset = [{'question': f"Who is {str(f).split()[1]}?", 'answer': str(f).split('"', 2)[-1].rstrip('")')} for f in facts if len(str(f).split()) > 1]
    correct, total = 0, len(dataset) + 1
    for item in dataset:
        if item['question'] not in mock_responses:
            mock_responses[item['question']] = query_metta_dynamic(item['question'], kg, neural_extractor, self_awareness) if not llm_enabled else chain.invoke({"question": item['question']})
        if item['answer'].lower() in mock_responses[item['question']].lower():
            correct += 1
    if ground_truth.lower() in response.lower():
        correct += 1
    accuracy = (correct / total) * 100
    self_awareness.metrics['accuracy'] = accuracy / 100.0  # Hook to awareness
    self_awareness.update_self_state()
    return accuracy

# Storage
def save_to_playground(question: str, output: str, history: list, accuracies: dict):
    try:
        conn = sqlite3.connect('playground.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS playground
                     (id TEXT PRIMARY KEY, question TEXT, output TEXT, history TEXT, accuracies TEXT, timestamp TEXT)''')
        data = {
            'id': str(uuid.uuid4()),
            'question': question,
            'output': output,
            'history': json.dumps(history),
            'accuracies': json.dumps(accuracies),
            'timestamp': datetime.now().isoformat()
        }
        c.execute('''INSERT INTO playground VALUES (:id, :question, :output, :history, :accuracies, :timestamp)''', data)
        conn.commit()
        conn.close()
    except Exception as e:
        error_handler.log_and_report(None, "SQLite Save", str(e))
        with open('playground_fallback.json', 'w') as f:
            json.dump(data, f)

def store_memory(question: str, response: str, confidence: float):
    inter_id = str(uuid.uuid4())
    das.add_atom(f"interaction:{inter_id}", f"Q: {question} R: {response} Conf: {confidence}")

# Update Graph
def update_graph(domain: str, new_fact: str, kg: KnowledgeGraph):
    if not new_fact.startswith('(= '):
        new_fact = f'(= {new_fact} true)'
    entity = new_fact.split()[1] if len(new_fact.split()) > 1 else "Unknown"
    kg.runner.run(f'(add-atom (in-domain {entity} {domain}))')
    kg.runner.run(new_fact)
    kg.runner.run(f'(add-atom (new-atom "{entity}"))')
    kg.vectorize_graph()
    logger.info(f"Updated {domain}: {new_fact}")

# Parse Update
def parse_natural_language_update(update_text: str, extractor: NeuralExtractor) -> Tuple[Optional[str], Optional[str]]:
    default_domain = "General"
    if llm_enabled:
        parse_prompt = ChatPromptTemplate.from_template("""
        Parse: {input}
        JSON: {{"domain": "<d>", "fact": "(fact \"e\" \"f\")", "confidence": <float>}}
        """)
        try:
            result = json.loads((parse_prompt | llm | parser).invoke({"input": update_text}))
            if result.get("confidence", 0) >= 0.7:
                return result["domain"], result["fact"]
        except:
            pass
    # BERT fallback
    prob = extractor.classify_sentence(update_text)
    if prob >= 0.7 and " is " in update_text.lower():
        parts = update_text.split(" is ", 1)
        entity = parts[0].strip().lower().replace(" ", "-")
        fact = f'(fact "{entity}" "{parts[1].strip()}")'
        domain = default_domain
        if ":" in update_text:
            domain = update_text.split(":")[0].strip().title()
        return domain, fact
    # Rule-based
    if ":" in update_text:
        domain, fact_text = update_text.split(":", 1)
        domain = domain.strip().title()
        if " is " in fact_text:
            parts = fact_text.split(" is ", 1)
            entity = parts[0].strip().lower().replace(" ", "-")
            return domain, f'(fact "{entity}" "{parts[1].strip()}")'
    return None, None

# Heuristic Review
def review_heuristics(manual: bool = False, kg: KnowledgeGraph):
    pending = das.query("pending_heuristic:*")
    if not pending:
        return "No pending."
    results = []
    if llm_enabled and not manual:
        # LLM review logic (simplified)
        for h in pending[:3]:  # Limit
            conf = float(h.split("Confidence:")[-1].strip(" )")) if "Confidence:" in h else 0.6
            if conf >= 0.7:
                kg.runner.run(h.split(" (")[0])
                results.append(f"Approved: {h}")
            else:
                results.append(f"Discarded: {h}")
    else:
        # Manual
        for h in pending:
            conf = float(h.split("Confidence:")[-1].strip(" )")) if "Confidence:" in h else 0.6
            if conf >= 0.8:
                kg.runner.run(h.split(" (")[0])
                results.append(f"Approved: {h}")
            else:
                results.append(f"Discarded: {h}")
    kg.vectorize_graph()
    return "\n".join(results)

# Autonomous Loop (Async)
async def autonomous_goal_setting(agent: AutonomousGoalAgent):
    while True:
        try:
            interactions = len(das.query("interaction:*"))
            if interactions % 5 == 0:
                obs = agent.perceive()
                if obs['unresolved_goals'] > 0 or not agent.active_goals:
                    goal = "Expand knowledge" if not llm_enabled else (ChatPromptTemplate.from_template("Suggest goal from {ints}").format(interactions=obs['interactions'][-3:]) | llm | parser).invoke({})
                    await agent.run_agent_loop(goal)
                    self_awareness.reflect(f"Goal: {goal}")
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Autonomy loop: {e}")
            await asyncio.sleep(30)

# Gradio Interface
def run_chatbot(kg: KnowledgeGraph, neural_extractor: NeuralExtractor, self_awareness: SelfAwarenessEngine, agent: AutonomousGoalAgent):
    def process_input(question, faq_update, history):
        history = history or []
        if faq_update:
            domain, fact = parse_natural_language_update(faq_update, neural_extractor)
            if domain and fact:
                update_graph(domain, fact, kg)
                msg = f"Updated {domain}: {fact}"
            else:
                msg = "Parse failed. Format: Domain: Entity is fact."
            history.append({'role': 'user', 'content': faq_update})
            history.append({'role': 'assistant', 'content': msg, 'timestamp': datetime.now().isoformat()})
            return [h for h in history], msg, None
        if not question:
            return history, "Enter question.", None

        # Special commands
        if question.lower() == 'errors':
            errors = kg.runner.run('!(get-errors $type)') or ["None"]
            response = f"Errors: {errors}"
        elif question.lower() == 'review heuristics':
            response = review_heuristics(manual=True, kg=kg)
        else:
            response = query_metta_dynamic(question, kg, neural_extractor, self_awareness)
            confidence = 0.85
            store_memory(question, response, confidence)
            accuracy = benchmark_clevr_vqa(kg.runner, question, "Expected", response, neural_extractor)
            save_to_playground(question, response, history, {'vqa': accuracy})

        history.append({'role': 'user', 'content': question})
        history.append({'role': 'assistant', 'content': response, 'timestamp': datetime.now().isoformat()})
        return history, response, None

    def export_csv():
        try:
            conn = sqlite3.connect('playground.db')
            df = pd.read_sql_query("SELECT * FROM playground", conn)
            conn.close()
            path = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(path, index=False)
            return path
        except Exception as e:
            logger.error(f"Export: {e}")
            return None

    with gr.Blocks(title="Improved FAQ Chatbot") as interface:
        gr.Markdown("# Enhanced FAQ Chatbot\nAsk, update FAQs, or type 'review heuristics'/'errors'.")
        chatbot = gr.Chatbot(height=400, type="messages")
        question = gr.Textbox(placeholder="Who is Uhuru Kenyatta?")
        update = gr.Textbox(placeholder="Kenyan-Politics: Uhuru Kenyatta is former president.")
        submit_q = gr.Button("Submit Question")
        submit_u = gr.Button("Update FAQ")
        export_btn = gr.Button("Export CSV")
        file_out = gr.File()

        submit_q.click(process_input, [question, update, chatbot], [chatbot, gr.Textbox(), file_out])
        submit_u.click(process_input, [question, update, chatbot], [chatbot, gr.Textbox(), file_out])
        export_btn.click(export_csv, outputs=file_out)

        interface.launch(server_name="127.0.0.1", server_port=7860, share=True)

# Main Initialization
if __name__ == "__main__":
    kg = KnowledgeGraph()
    neural_extractor = NeuralExtractor(kg)
    self_awareness = SelfAwarenessEngine(kg.runner, das)
    agent = AutonomousGoalAgent(kg.runner, das, self_awareness)

    # Preload extra facts
    preload_facts = [("fact:uhuru-kenyatta", "former president of Kenya"), ("fact:nairobi", "capital of Kenya")]
    for key, val in preload_facts:
        das.add_atom(key, val)
        kg.runner.run(f'(add-atom (fact "{key.split(":")[1]}" "{val}"))')

    # Start autonomy
    threading.Thread(target=lambda: asyncio.run(autonomous_goal_setting(agent)), daemon=True).start()

    # Test query
    test_q = "Who is Uhuru Kenyatta?"
    resp = query_metta_dynamic(test_q, kg, neural_extractor, self_awareness)
    print(f"Test: {test_q} -> {resp}")

    # Launch UI
    run_chatbot(kg, neural_extractor, self_awareness, agent)

# Unit Tests (Run with pytest or manually)
def test_basic_query():
    kg = KnowledgeGraph()
    assert "President" in kg.runner.run('!(fact William-Ruto $f)') [0]

def test_vectorize():
    kg = KnowledgeGraph()
    old_ntotal = kg.index.ntotal
    kg.runner.run('(add-atom (new-atom "test"))')
    kg.vectorize_graph()
    assert kg.index.ntotal > old_ntotal

if __name__ == "__main__":
    test_basic_query()
    test_vectorize()
    logger.info("Tests passed.")
