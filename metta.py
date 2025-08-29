
# Install compatible versions

import threading
import time
from hyperon import *
from hyperon import MeTTa
from hyperon import GroundingSpace
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import logging
import uuid
import json
import redis
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from typing import Dict, List, Any
import torch
import torch_geometric as pyg
from torch_geometric.nn import GraphSAGE
import requests
from ddgs import DDGS
from tenacity import retry, wait_random_exponential, stop_after_attempt
import sqlite3
import subprocess
import os
import shutil
from sklearn.metrics.pairwise import cosine_similarity
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_MODE"] = "disabled"
os.environ["REDIS_URL"] = "redis://your-redis-host:6379"


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device('cpu')

logger.info("Use pytorch device_name: cpu")
# Cache for responses
mock_responses = {}

# Initialize MeTTa and Atomspace
space = GroundingSpace()
runner = MeTTa(space=space.gspace)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS for vector search
dimension = 384
index = faiss.IndexFlatL2(dimension)
atom_vectors = {}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Pattern dataset for GNN training
pattern_dataset = []

# DAS Setup with Redis
class DASClient:
    def __init__(self, host='127.0.0.1', port=6379):
        self.redis = None
        self.in_memory = {}
        # Allow overriding Redis connection via environment variables
        redis_host = os.environ.get("REDIS_HOST", host)
        redis_port = int(os.environ.get("REDIS_PORT", port))
        redis_url = os.environ.get("REDIS_URL")

        # Try connecting first; if connection fails, try to start a local redis-server if available.
        try:
            if redis_url:
                self.redis = redis.from_url(redis_url, decode_responses=True)
            else:
                self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis.ping()
            logger.info("DAS client initialized with Redis backend")
        except Exception as e:
            logger.warning(f"Could not connect to Redis at {redis_url or f'{redis_host}:{redis_port}'}: {e}")
            # Attempt to start local redis-server if available on PATH
            if shutil.which('redis-server'):
                try:
                    logger.info("Attempting to start local redis-server...")
                    subprocess.run(['redis-server', '--daemonize', 'yes'], check=False)
                    time.sleep(1)
                    if redis_url:
                        self.redis = redis.from_url(redis_url, decode_responses=True)
                    else:
                        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                    self.redis.ping()
                    logger.info("DAS client initialized with Redis backend after starting redis-server")
                except Exception as e2:
                    logger.error(f"Failed to start/connect to local redis-server: {e2}. Falling back to in-memory storage.")
                    self.redis = None
                    try:
                        runner.run(f'(add-atom (error "DAS Redis Start Failed" "{str(e2)}"))')
                    except Exception:
                        logger.debug("Runner unavailable while reporting DAS Redis start failure")
            else:
                logger.info("Redis not available; using in-memory DAS. To enable Redis set REDIS_URL or install/start redis-server.")
                logger.info("To install Redis:")
                logger.info("  - Ubuntu/Debian: `sudo apt-get install redis-server`")
                logger.info("  - macOS: `brew install redis`")
                self.redis = None
                try:
                    runner.run(f'(add-atom (error "DAS Redis Not Installed" "{str(e)}"))')
                except Exception:
                    logger.debug("Runner unavailable while reporting DAS Redis not installed")

    def add_atom(self, atom: str, value: Any):
        try:
            if self.redis:
                self.redis.set(atom, str(value))
            else:
                self.in_memory[atom] = str(value)
            logger.info(f"Added atom {atom} to DAS")
            logger.debug(f"In-memory storage: {self.in_memory}")
        except redis.ConnectionError as e:
            logger.error(f"DAS add_atom failed: {e}. Using in-memory storage.")
            self.in_memory[atom] = str(value)
            logger.debug(f"In-memory storage: {self.in_memory}")
            runner.run(f'(add-atom (error "DAS Connection" "{str(e)}"))')
        except Exception as e:
            logger.error(f"DAS add_atom failed: {e}")
            runner.run(f'(add-atom (error "DAS General" "{str(e)}"))')
    def query(self, pattern: str) -> List[str]:
        try:
            if self.redis:
                keys = self.redis.keys(f"{pattern}*")
                results = [self.redis.get(key) for key in keys]
            else:
                results = [value for key, value in self.in_memory.items() if key.startswith(pattern)]
            logger.info(f"DAS query returned {len(results)} results")
            return results
        except redis.ConnectionError as e:
            logger.error(f"DAS query failed: {e}. Using in-memory storage.")
            runner.run(f'(add-atom (error "DAS Query" "{str(e)}"))')
            return [value for key, value in self.in_memory.items() if key.startswith(pattern)]
        except Exception as e:
            logger.error(f"DAS query failed: {e}")
            runner.run(f'(add-atom (error "DAS General" "{str(e)}"))')
            return []

das = DASClient()
knowledge_metta = """


# Minimal knowledge graph
(: General Domain)
(: Person Type)
(: Role Type)
(: Entity Type)
(: ErrorType Type)
(: Kenyan-Politics Domain)
(: Science Domain)
(: Medicine Domain)  ; Added Medicine domain

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


runner.run(knowledge_metta)
logger.info("Minimal knowledge graph loaded.")

# GNN Setup
class PatternGNN(torch.nn.Module):
    def __init__(self, in_channels=384, hidden_channels=64, out_channels=1):
        super().__init__()
        self.gnn = GraphSAGE(in_channels, hidden_channels, num_layers=2, out_channels=out_channels)
    
    def forward(self, data):
        return self.gnn(data.x, data.edge_index)

pattern_gnn = PatternGNN()
pattern_gnn.to(torch.device('cpu'))  # Force cpu

optimizer = torch.optim.Adam(pattern_gnn.parameters(), lr=0.01)

# Build graph for GNN


from hyperon import Atom, ExpressionAtom
import numpy as np
import torch
import torch_geometric.data as pyg_data  # Use explicit import for clarity
from typing import Optional, Tuple, Dict
def build_graph(query_emb: Optional[np.ndarray] = None) -> Tuple[pyg_data.Data, Dict[str, int]]:
    device = torch.device('cpu')
    logger.debug("Building graph with query_emb: %s", query_emb is not None)
    atoms = runner.run('!(match &self $atom $atom)') or []
    if not atoms:
        logger.warning("No atoms found in GroundingSpace")
        return pyg_data.Data(x=torch.empty((0, 384), dtype=torch.float).to(device), edge_index=torch.empty((2, 0), dtype=torch.long).to(device)), {}

    node_features = []
    batch_size = 16
    try:
        for i in range(0, len(atoms), batch_size):
            batch = atoms[i:i + batch_size]
            embeddings = embedder.encode([str(atom) for atom in batch], device='cpu', convert_to_tensor=False)
            node_features.extend(embeddings.tolist())
    except Exception as e:
        logger.error(f"Node feature encoding failed: {e}")
        runner.run(f'(add-atom (error "Node Encoding" "{str(e)}"))')
        return pyg_data.Data(x=torch.empty((0, 384), dtype=torch.float).to(device), edge_index=torch.empty((2, 0), dtype=torch.long).to(device)), {}

    if query_emb is not None:
        if not isinstance(query_emb, np.ndarray) or query_emb.shape != (1, 384):
            logger.error(f"Invalid query embedding: shape={query_emb.shape if isinstance(query_emb, np.ndarray) else 'not numpy array'}")
            query_emb = None
        else:
            try:
                if index.ntotal == 0:
                    logger.warning("FAISS index is empty, skipping search")
                    query_emb = None
                else:
                    query_emb = np.asarray(query_emb, dtype=np.float32)
                    if query_emb.ndim == 1:
                        query_emb = query_emb.reshape(1, -1)
                    distances, indices = index.search(query_emb, k=min(100, index.ntotal))
                    atoms = [atom_vectors[list(atom_vectors.keys())[i]] for i in indices[0] if i != -1 and i < len(atom_vectors)]
                    logger.debug(f"Filtered %d atoms using FAISS", len(atoms))
            except Exception as e:
                logger.error(f"FAISS search failed: {e}")
                runner.run(f'(add-atom (error "FAISS Search" "{str(e)}"))')
                query_emb = None

    node_map = {str(atom): i for i, atom in enumerate(atoms)}
    edges = []
    implies_results = runner.run('!(unique (match &self (= (implies $a $b) true) ($a $b)))') or []
    logger.debug(f"MeTTa implies query returned {len(implies_results)} results: {implies_results}")
    if not implies_results:
        logger.info("No implication rules found in GroundingSpace, proceeding with empty edges")
    for result in implies_results:
        try:
            if isinstance(result, list) and len(result) == 2:
                a, b = result
            elif isinstance(result, Atom):
                children = result.get_children()
                if len(children) != 2:
                    logger.warning("Implies result has %d children, expected 2: %s", len(children), children)
                    continue
                a, b = children
            else:
                logger.warning("Unexpected implies result type: %s, value: %s", type(result), result)
                continue
            a_str, b_str = str(a), str(b)
            if a_str in node_map and b_str in node_map:
                edges.append((node_map[a_str], node_map[b_str]))
            else:
                logger.debug(f"Skipping edge: {a_str} or {b_str} not in node_map")
        except Exception as e:
            logger.error(f"Error processing implies result: {e}, result: {result}")
            runner.run(f'(add-atom (error "Implies Processing" "{str(e)}"))')

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device) if edges else torch.empty((2, 0), dtype=torch.long).to(device)
    x = torch.tensor(node_features, dtype=torch.float).to(device) if node_features else torch.empty((0, 384), dtype=torch.float).to(device)
    logger.debug("Graph built: nodes=%d, edges=%d", x.size(0), edge_index.size(1))
    return pyg_data.Data(x=x, edge_index=edge_index), node_map
        # Train GNN
def train_pattern_gnn():
    global pattern_dataset
    try:
        if len(pattern_dataset) < 10:
            return
        data, _ = build_graph()
        if data.x.size(0) == 0 or data.edge_index.size(1) == 0:
            logger.info("Empty graph, skipping GNN training")
            return
        device = torch.device('cpu')  # Force cpu
        data = data.to(device)
        pattern_gnn.to(device)  # Move model to cpu
        pattern_gnn.train()
        batch_size = 10
        for i in range(0, len(pattern_dataset), batch_size):
            batch = pattern_dataset[i:i+batch_size]
            optimizer.zero_grad()
            out = pattern_gnn(data)
            target = torch.tensor([embedder.encode(p["heuristic"]).mean() for p in batch], dtype=torch.float).to(device).mean()
            loss = torch.nn.MSELoss()(out.mean(), target)
            loss.backward()
            optimizer.step()
        logger.info(f"Trained GNN with {len(pattern_dataset)} patterns")
    except Exception as e:
        logger.error(f"GNN training failed: {e}")
        runner.run(f'(add-atom (error "GNN Training" "{str(e)}"))')
# Vectorize graph
def vectorize_graph():
    """
    Vectorize new atoms in the knowledge graph using CPU-only operations.
    """
    global index, atom_vectors, embedder
    try:
        # Initialize globals if not defined
        if 'embedder' not in globals():
            embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        if 'index' not in globals():
            index = faiss.IndexFlatL2(384)  # Dimension for all-MiniLM-L6-v2
        if 'atom_vectors' not in globals():
            atom_vectors = {}

        new_atoms = runner.run('!(match &self (new-atom $atom) $atom)') or []
        logger.debug(f"New atoms: {new_atoms}")
        if new_atoms:
            # Process in batches to reduce memory usage
            batch_size = 32
            for i in range(0, len(new_atoms), batch_size):
                batch = new_atoms[i:i + batch_size]
                # Encode with CPU explicitly
                vectors = embedder.encode([str(atom) for atom in batch], convert_to_tensor=True, device='cpu')
                vectors = vectors.cpu().numpy()  # Convert to NumPy for FAISS
                for atom, vector in zip(batch, vectors):
                    atom_id = str(uuid.uuid4())
                    atom_vectors[atom_id] = str(atom)
                    index.add(np.array([vector]))
                    logger.debug(f"Adding atom embedding:{atom_id} to DAS")
                    das.add_atom(f"embedding:{atom_id}", vector.tolist())
            # Clean up new atoms
            runner.run('!(match &self (new-atom $atom) (delete-atom (new-atom $atom)))')
        logger.info("Knowledge graph incrementally vectorized.")
    except Exception as e:
        logger.error(f"Vectorize graph failed: {e}")
        runner.run(f'(add-atom (error "Vectorize Graph" "{str(e)}"))')

vectorize_graph()

# Neural Extractor
class NeuralExtractor:
    def __init__(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased').to('cpu')  # For embeddings
            self.classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            self.fine_tune()
            logger.info("BERT models initialized and fine-tuned")
        except Exception as e:
            logger.error(f"BERT initialization failed: {e}")
            runner.run(f'(add-atom (error "BERT Init" "{str(e)}"))')
            self.classifier = None
            self.model = None

    def fine_tune(self):
        try:
            facts = runner.run('!(match &self (fact $entity $fact) $fact)') or []
            texts = [str(f).split('"', 2)[-1].rstrip('")') for f in facts if len(str(f).split()) > 1]
            if not texts:
                texts = ["Kenya is a country."]
            # Add negative examples for better binary classification
            negatives = [
                "What is this?", "This sentence is not a fact.", "Random question?",
                "How does it work?", "Irrelevant text without entities."
            ]
            data = {'text': texts + negatives, 'label': [1] * len(texts) + [0] * len(negatives)}
            dataset = Dataset.from_dict(data)
            def tokenize_function(examples):
                return tokenizer(examples['text'], padding='max_length', truncation=True)
            # Disable cache hashing
            dataset = dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
            dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
            training_args = TrainingArguments(
                output_dir='./bert-finetuned',
                num_train_epochs=3,
                per_device_train_batch_size=8,
                logging_dir='./logs',
                logging_steps=10,
            )
            trainer = Trainer(model=self.classifier, args=training_args, train_dataset=dataset)
            trainer.train()
            logger.info("BERT fine-tuned for fact extraction with positives and negatives")
        except Exception as e:
            logger.error(f"BERT fine-tuning failed: {e}")
            runner.run(f'(add-atom (error "BERT Fine-Tune" "{str(e)}"))')
            raise

    # New method: Classify if a sentence is a fact
    def classify_sentence(self, sentence: str) -> float:
        if self.classifier is None:
            return 0.0  # Fallback if init failed
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs[0][1].item()  # Probability for label 1 (fact)

    # New method: Get BERT embedding for a sentence
    def get_embedding(self, sentence: str) -> list:
        if self.model is None:
            return []
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling
        # New method: Compute cosine similarity between two texts using embeddings
    def compute_similarity(self, text1: str, text2: str) -> float:
        if self.model is None:
            return 0.0
        emb1 = np.array(self.get_embedding(text1)).reshape(1, -1)
        emb2 = np.array(self.get_embedding(text2)).reshape(1, -1)
        if emb1.size == 0 or emb2.size == 0:
            return 0.0
        return cosine_similarity(emb1, emb2)[0][0]

def extract_facts_lightweight(text: str, threshold: float = 0.7) -> List[tuple]:
    sentences = text.split('.')
    facts = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or " is " not in sentence.lower():
            logger.debug(f"Skipping sentence without 'is' or empty: '{sentence}'")
            continue
        try:
            prob = neural_extractor.classify_sentence(sentence)
            if prob < threshold:
                logger.debug(f"Skipping low-confidence fact (prob={prob:.2f}): '{sentence}'")
                continue
            parts = sentence.split(" is ", 1)
            if len(parts) != 2 or not all(parts):
                logger.debug(f"Skipping malformed sentence: '{sentence}'")
                continue
            entity, fact = parts
            # Clean entity: remove parentheticals, commas, etc.
            entity = entity.split('(')[0].split(',')[0].strip().lower().replace(" ", "-").replace("--", "-")
            embedding = neural_extractor.get_embedding(sentence)
            facts.append((entity, fact.strip(), prob, embedding))
            logger.debug(f"Extracted fact: entity='{entity}', fact='{fact.strip()}', prob={prob:.2f}")
        except ValueError as e:
            logger.warning(f"Failed to extract fact from sentence '{sentence}': {e}")
            continue
    if not facts:
        logger.info(f"No valid facts extracted from text: {text[:100]}...")
    else:
        logger.info(f"Extracted {len(facts)} facts from text")
    return facts
neural_extractor = NeuralExtractor()



# Configure logging
logger = logging.getLogger(__name__)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging

# Configure logging
logger = logging.getLogger(__name__)

def validate_openai_key(api_key: str) -> bool:
    try:
        test_client = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key, max_retries=1)
        test_client.invoke("Test prompt")
        return True
    except Exception as e:
        logger.error(f"Gemini API key validation failed: {e}")
        runner.run(f'(add-atom (error "Gemini" "{str(e)}"))')
        return False

api_key = "AIzaSyB7qcGYVY97ONYHsYxeou8lqsrY26TytjA"  # Replace with your actual Gemini API key
llm_enabled = validate_openai_key(api_key)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=api_key, max_retries=10) if llm_enabled else None

prompt = ChatPromptTemplate.from_template("""
You are a general FAQ chatbot. Use the following context to answer the question.
Context: {context}
Question: {question}
Provide a structured answer with definitions, examples, and references. If unsure, suggest updates.
""")
parser = StrOutputParser()
chain = {"context": lambda x: query_metta_dynamic(x["question"]), "question": RunnablePassthrough()} | prompt | llm | parser if llm_enabled else lambda x: query_metta_dynamic(x["question"]) + "\nLLM disabled."

# Hybrid Reasoning with GNN
def infer_reasoning_pattern(question: str, context: str) -> dict:
    try:
        query_emb = embedder.encode(question).tolist()
        context_emb = embedder.encode(context).tolist()
        pattern_data = [d for d in pattern_dataset if np.linalg.norm(np.array(d["query_emb"]) - np.array(query_emb)) < 0.5]
        if not pattern_data:
            logger.info(f"No matching patterns for query: {question}")
            return {"heuristic": f"(= (heuristic \"{question}\" $context) (fallback-answer))", "confidence": 0.75}
        
        confidences = [d["confidence"] for d in pattern_data]
        max_confidence = max(confidences) if confidences else 0.75
        best_pattern = pattern_data[confidences.index(max_confidence)]["heuristic"] if confidences else f"(= (heuristic \"{question}\" $context) (fallback-answer))"
        return {"heuristic": best_pattern, "confidence": max_confidence}
    except Exception as e:
        logger.error(f"infer_reasoning_pattern failed for query '{question}': {e}")
        return {"heuristic": f"(= (heuristic \"{question}\" $context) (fallback-answer))", "confidence": 0.75}

def query_metta_dynamic(question: str) -> str:
    try:
        normalized_question = question
        if question.lower().startswith("who ") and " is " not in question.lower():
            normalized_question = question.replace("Who ", "Who is ", 1)
            logger.info(f"Normalized query: '{question}' to '{normalized_question}'")
        question = normalized_question

        cached_response = das.query(f"response:{question}:*")
        if cached_response:
            logger.info(f"Returning cached response for query: {question}")
            return f"{cached_response[0]}\nRecent Errors: None"

        query_emb = embedder.encode(question, convert_to_tensor=False)
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        logger.debug(f"Encoded query '{question}' to embedding")
        _, indices = index.search(query_emb, k=min(5, index.ntotal))
        similar_atoms = [atom_vectors[list(atom_vectors.keys())[i]] for i in indices[0] if i != -1 and i < len(atom_vectors)]
        domains = runner.run(f'!(select-domain $domain)') or ["General"]
        logger.debug(f"MeTTa domains query returned: {domains}")
        domain = domains[0] if isinstance(domains, list) and domains else "General"
        context = f"Similar facts: {similar_atoms}\nDomain: {domain}"
        logger.debug(f"Context for query '{question}': {context}")

        entity = None
        if "who is" in question.lower():
            entity = question.lower().split("who is")[-1].strip().rstrip('?').replace(" ", "-").replace("--", "-")
            logger.info(f"Extracted entity: {entity}")
            facts = das.query(f"fact:{entity}:*") or das.query(f"fact:{entity.title()}:*")
            logger.debug(f"Facts for {entity}: {facts}")
            if facts:
                return f"{facts[0]}\nRecent Errors: None"

        if entity and not facts:
            logger.info(f"Triggering continual learning for entity: {entity}")
            try:
                continual_learning(entity)
                facts = das.query(f"fact:{entity}:*") or das.query(f"fact:{entity.title()}:*")
                if facts:
                    return f"{facts[0]}\nRecent Errors: None"
            except Exception as e:
                logger.error(f"Continual learning failed: {e}")
                runner.run(f'(add-atom (error "Continual Learning" "{str(e)}"))')

        heuristic_response = infer_reasoning_pattern(question, context)
        logger.debug(f"Heuristic response: {heuristic_response}")
        if heuristic_response["confidence"] >= 0.7:
            runner.run(heuristic_response["heuristic"])
            result = runner.run(f'!(heuristic "{question}" "{context}")')
            logger.debug(f"MeTTa heuristic query returned: {result}")
            response = result[0] if isinstance(result, list) and result else "No heuristic result."
            das.add_atom(f"heuristic:{uuid.uuid4()}", f"{heuristic_response['heuristic']} (Confidence: {heuristic_response['confidence']})")
        else:
            if llm_enabled:
                heuristic_prompt = ChatPromptTemplate.from_template("""
                Generate a MeTTa heuristic rule to process the query dynamically...
                Output in JSON format:
                {
                  "heuristic": "(= (heuristic \\"{question}\\" $context) $action)",
                  "confidence": <float>
                }
                """)
                try:
                    heuristic_response = json.loads((heuristic_prompt | llm | parser).invoke({"question": question, "context": context}))
                    logger.debug(f"LLM-generated heuristic: {heuristic_response}")
                    if heuristic_response["confidence"] >= 0.7:
                        runner.run(heuristic_response["heuristic"])
                        result = runner.run(f'!(heuristic "{question}" "{context}")')
                        logger.debug(f"MeTTa heuristic query returned: {result}")
                        response = result[0] if isinstance(result, list) and result else "No heuristic result."
                        das.add_atom(f"heuristic:{uuid.uuid4()}", f"{heuristic_response['heuristic']} (Confidence: {heuristic_response['confidence']})")
                        pattern_dataset.append({
                            "query_emb": query_emb.tolist(),
                            "context_emb": embedder.encode(context, convert_to_tensor=False).tolist(),
                            "heuristic": heuristic_response["heuristic"],
                            "confidence": heuristic_response["confidence"]
                        })
                        train_pattern_gnn()
                    else:
                        das.add_atom(f"pending_heuristic:{uuid.uuid4()}", f"{heuristic_response['heuristic']} (Confidence: {heuristic_response['confidence']})")
                        response = f"Low-confidence heuristic (score: {heuristic_response['confidence']}). Stored for review."
                except Exception as e:
                    logger.error(f"LLM heuristic generation failed: {e}")
                    runner.run(f'(add-atom (error "LLM Heuristic" "{str(e)}"))')
                    response = generate_rule_based_heuristic(question, context)
            else:
                logger.info(f"LLM disabled, using rule-based heuristic for query: {question}")
                response = generate_rule_based_heuristic(question, context)

        errors = runner.run(f'!(get-errors $type)') or ["None"]
        logger.debug(f"MeTTa errors query returned: {errors}")
        response = f"{response}\nRecent Errors: {errors[0] if isinstance(errors, list) and errors else errors}"
        das.add_atom(f"response:{question}:{uuid.uuid4()}", response)
        logger.info(f"Dynamic MeTTa query result for '{question}': {response}")
        return response
    except Exception as e:
        logger.error(f"Dynamic MeTTa query failed for '{question}': {e}")
        runner.run(f'(add-atom (error "Query" "{str(e)}"))')
        return f"Sorry, I couldn't process that request due to an internal error: {str(e)}"     
def generate_rule_based_heuristic(question: str, context: str) -> str:
    """Generate a simple MeTTa heuristic when LLM is unavailable."""
    if "who is" in question.lower():
        entity = question.lower().split("who is")[-1].strip("?")
        heuristic = f"(= (heuristic \"{question}\" $context) (match &self (fact \"{entity}\" $fact) $fact))"
        confidence = 0.75
    elif "what is" in question.lower():
        entity = question.lower().split("what is")[-1].strip("?")
        heuristic = f"(= (heuristic \"{question}\" $context) (match &self (fact $entity \"{entity}\") $entity))"
        confidence = 0.75
    else:
        heuristic = f"(= (heuristic \"{question}\" $context) (match &self (fact $entity $fact) $entity))"
        confidence = 0.6
    das.add_atom(f"heuristic:{uuid.uuid4()}", f"{heuristic} (Confidence: {confidence})")
    runner.run(heuristic)
    result = runner.run(f'!(heuristic "{question}" "{context}")') or ["No rule-based result"]
    return result[0] if result else f"No rule-based result. Confidence: {confidence}"

# Heuristic Review
def review_heuristics(manual=False):
    try:
        pending_heuristics = das.query("pending_heuristic:*")
        if not pending_heuristics:
            logger.info("No pending heuristics to review.")
            return "No pending heuristics."
        
        results = []
        if llm_enabled and not manual:
            review_prompt = ChatPromptTemplate.from_template("""
            Review the following pending heuristics and decide whether to approve, refine, or discard each one...
            Output in JSON format:
            [
              {
                "original": "<original heuristic>",
                "decision": "approve|refine|discard",
                "refined": "<refined heuristic or null>",
                "confidence": <float>
              }
            ]
            """)
            try:
                review_results = json.loads((review_prompt | llm | parser).invoke({
                    "heuristics": pending_heuristics,
                    "interactions": das.query("interaction:*")[-5:] or ["No recent interactions"]
                }))
                for result in review_results:
                    original = result["original"]
                    decision = result["decision"]
                    refined = result["refined"]
                    confidence = result["confidence"]
                    if decision == "approve" and confidence >= 0.7:
                        runner.run(original)
                        das.add_atom(f"heuristic:{uuid.uuid4()}", f"{original} (Confidence: {confidence})")
                        results.append(f"Approved heuristic: {original}")
                    elif decision == "refine" and refined and confidence >= 0.7:
                        runner.run(refined)
                        das.add_atom(f"heuristic:{uuid.uuid4()}", f"{refined} (Confidence: {confidence})")
                        results.append(f"Refined heuristic: {refined}")
                    else:
                        results.append(f"Discarded heuristic: {original}")
                    das.add_atom(f"reviewed_heuristic:{uuid.uuid4()}", f"{original} -> {decision}")
            except Exception as e:
                logger.error(f"LLM heuristic review failed: {e}")
                runner.run(f'(add-atom (error "LLM Heuristic Review" "{str(e)}"))')
                results = manual_heuristic_review(pending_heuristics)
        else:
            logger.info("LLM disabled or manual review requested, using manual heuristic review")
            results = manual_heuristic_review(pending_heuristics)
        
        vectorize_graph()
        logger.info(f"Reviewed {len(pending_heuristics)} heuristics: {results}")
        return "\n".join(results)
    except Exception as e:
        logger.error(f"Heuristic review failed: {e}")
        runner.run(f'(add-atom (error "Heuristic Review" "{str(e)}"))')
        return f"Failed to review heuristics: {str(e)}"
    
def manual_heuristic_review(pending_heuristics: List[str]) -> List[str]:
    """Manually review heuristics when LLM is unavailable."""
    results = []
    for heuristic in pending_heuristics:
        try:
            # Extract confidence from heuristic string (e.g., "(Confidence: 0.6)")
            confidence = float(heuristic.split("Confidence:")[-1].strip(" )")) if "Confidence:" in heuristic else 0.6
            original = heuristic.split(" (Confidence:")[0]
            if confidence >= 0.8:
                runner.run(original)
                das.add_atom(f"heuristic:{uuid.uuid4()}", f"{original} (Confidence: {confidence})")
                results.append(f"Approved heuristic: {original}")
            else:
                results.append(f"Discarded heuristic: {original} (low confidence: {confidence})")
            das.add_atom(f"reviewed_heuristic:{uuid.uuid4()}", f"{original} -> {'approve' if confidence >= 0.8 else 'discard'}")
        except Exception as e:
            logger.error(f"Manual heuristic review failed for {heuristic}: {e}")
            results.append(f"Failed to review heuristic: {heuristic}")
    return results


    
def generate_fallback_rule(entity: str, facts: List[tuple]) -> str:
    """Generate a fallback MeTTa rule when LLM is unavailable."""
    if not facts:
        return ""
    fact = facts[0][1]  # Use first fact
    return f"(= (implies (fact \"{entity}\" \"{fact}\") (role \"{entity}\" entity)) true)"

# Web Search
def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            if not results:
                logger.info(f"No web results found for query: {query}")
                return "No web results found."
            snippets = [result['body'] for result in results if 'body' in result]
            combined = ' '.join(snippets[:3])
            logger.info(f"Web search for '{query}' returned {len(snippets)} snippets")
            return combined if combined else "No web results found."
    except Exception as e:
        logger.error(f"Web search failed for '{query}': {e}")
        runner.run(f'(add-atom (error "Web Search" "{str(e)}"))')
        return "No web results found."

# Benchmarking

logger = logging.getLogger(__name__)

def benchmark_clevr_vqa(runner: MeTTa, question: str, ground_truth: str, current_response: str, neural_extractor=None) -> float:
    """
    Benchmark VQA performance using MeTTa facts and a specific question.
    
    Args:
        runner: MeTTa instance for querying facts.
        question: The specific question to evaluate (e.g., "Who is Uhuru Kenyatta").
        ground_truth: The expected answer for the question.
        current_response: The model's response to the question.
        neural_extractor: Optional NeuralExtractor instance for tokenizer access.
    
    Returns:
        float: Accuracy percentage.
    """
    try:
        # Query MeTTa for facts
        facts = runner.run('!(match &self (fact $entity $fact) $fact)') or []
        # Construct dataset from MeTTa facts
        dataset: List[Dict[str, str]] = [
            {'question': f"Who is {entity}?", 'answer': fact}
            for entity, fact in [
                (str(f).split()[1], str(f).split('"', 2)[-1].rstrip('")'))
                for f in facts if len(str(f).split()) > 1
            ]
        ]
        correct = 0
        total = len(dataset) + 1  # Include the input question
        
        # Evaluate dataset questions
        for item in dataset:
            if item['question'] not in mock_responses:
                try:
                    if llm_enabled:
                        # Assuming chain is a langchain pipeline
                        mock_responses[item['question']] = safe_invoke(chain, {"question": item['question']})
                    else:
                        # Fallback to MeTTa reasoning
                        mock_responses[item['question']] = query_metta_dynamic(item['question'], neural_extractor=neural_extractor) + "\nLLM disabled."
                except Exception as e:
                    logger.error(f"Benchmark query failed for {item['question']}: {e}")
                    mock_responses[item['question']] = "Error in response"
                    runner.run(f'(add-atom (error "Benchmark Query" "{str(e)}"))')
            response = mock_responses[item['question']]
            # Case-insensitive comparison
            if item['answer'].lower() in response.lower():
                correct += 1
        
        # Evaluate the input question using current_response
        if ground_truth.lower() in current_response.lower():
            correct += 1
        else:
            logger.debug(f"Input question failed: question={question}, ground_truth={ground_truth}, current_response={current_response}")
        
        accuracy = correct / total * 100
        logger.info(f"CLEVR/VQA Accuracy: {accuracy:.2f}%")
        return accuracy
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        runner.run(f'(add-atom (error "Benchmark" "{str(e)}"))')
        return 0.0
# SQLite Storage
def save_to_playground(question: str, output: str, history: list, accuracies: dict):
    try:
        conn = sqlite3.connect('playground.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS playground (
            id TEXT PRIMARY KEY,
            question TEXT,
            output TEXT,
            history TEXT,
            accuracies TEXT,
            timestamp TEXT
        )''')
        data = {
            'id': str(uuid.uuid4()),
            'question': question,
            'output': output,
            'history': json.dumps(history),
            'accuracies': json.dumps(accuracies),
            'timestamp': logging.Formatter().formatTime(logging.makeLogRecord({}))
        }
        cursor.execute('''INSERT INTO playground (id, question, output, history, accuracies, timestamp)
                         VALUES (:id, :question, :output, :history, :accuracies, :timestamp)''', data)
        conn.commit()
        conn.close()
        logger.info("Saved to SQLite playground.db")
    except sqlite3.Error as e:
        logger.error(f"SQLite save failed: {e}")
        runner.run(f'(add-atom (error "SQLite Save" "{str(e)}"))')
        with open('playground_output.json', 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Fallback: Saved to playground_output.json")

# Memory Storage
def store_memory(question: str, response: str, confidence: float = 0.85):
    inter_id = str(uuid.uuid4())
    das.add_atom(f"interaction:{inter_id}", f"Query: {question} Response: {response} Confidence: {confidence}")
    
def build_graph(query_emb: Optional[np.ndarray] = None) -> Tuple[pyg_data.Data, Dict[str, int]]:
    device = torch.device('cpu')
    logger.debug("Building graph with query_emb: %s", query_emb is not None)
    atoms = runner.run('!(match &self $atom $atom)') or []
    if not atoms:
        logger.warning("No atoms found in GroundingSpace")
        return pyg_data.Data(x=torch.empty((0, 384), dtype=torch.float).to(device), edge_index=torch.empty((2, 0), dtype=torch.long).to(device)), {}

    node_features = []
    batch_size = 16
    try:
        for i in range(0, len(atoms), batch_size):
            batch = atoms[i:i + batch_size]
            embeddings = embedder.encode([str(atom) for atom in batch], device='cpu', convert_to_tensor=False)
            node_features.extend(embeddings.tolist())
    except Exception as e:
        logger.error(f"Node feature encoding failed: {e}")
        runner.run(f'(add-atom (error "Node Encoding" "{str(e)}"))')
        return pyg_data.Data(x=torch.empty((0, 384), dtype=torch.float).to(device), edge_index=torch.empty((2, 0), dtype=torch.long).to(device)), {}

    if query_emb is not None:
        if not isinstance(query_emb, np.ndarray) or query_emb.shape != (1, 384):
            logger.error(f"Invalid query embedding: shape={query_emb.shape if isinstance(query_emb, np.ndarray) else 'not numpy array'}")
            query_emb = None
        else:
            try:
                if index.ntotal == 0:
                    logger.warning("FAISS index is empty, skipping search")
                    query_emb = None
                else:
                    query_emb = np.asarray(query_emb, dtype=np.float32)
                    if query_emb.ndim == 1:
                        query_emb = query_emb.reshape(1, -1)
                    distances, indices = index.search(query_emb, k=min(100, index.ntotal))
                    atoms = [atom_vectors[list(atom_vectors.keys())[i]] for i in indices[0] if i != -1 and i < len(atom_vectors)]
                    logger.debug(f"Filtered %d atoms using FAISS", len(atoms))
            except Exception as e:
                logger.error(f"FAISS search failed: {e}")
                runner.run(f'(add-atom (error "FAISS Search" "{str(e)}"))')
                query_emb = None

    node_map = {str(atom): i for i, atom in enumerate(atoms)}
    edges = []
    implies_results = runner.run('!(unique (match &self (= (implies $a $b) true) ($a $b)))') or []
    logger.debug(f"MeTTa implies query returned {len(implies_results)} results: {implies_results}")
    if not implies_results:
        logger.info("No implication rules found in GroundingSpace, proceeding with empty edges")
    for result in implies_results:
        try:
            if isinstance(result, list) and len(result) == 2:
                a, b = result
            elif isinstance(result, Atom):
                children = result.get_children()
                if len(children) != 2:
                    logger.warning("Implies result has %d children, expected 2: %s", len(children), children)
                    continue
                a, b = children
            else:
                logger.warning("Unexpected implies result type: %s, value: %s", type(result), result)
                continue
            a_str, b_str = str(a), str(b)
            if a_str in node_map and b_str in node_map:
                edges.append((node_map[a_str], node_map[b_str]))
            else:
                logger.debug(f"Skipping edge: {a_str} or {b_str} not in node_map")
        except Exception as e:
            logger.error(f"Error processing implies result: {e}, result: {result}")
            runner.run(f'(add-atom (error "Implies Processing" "{str(e)}"))')

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device) if edges else torch.empty((2, 0), dtype=torch.long).to(device)
    x = torch.tensor(node_features, dtype=torch.float).to(device) if node_features else torch.empty((0, 384), dtype=torch.float).to(device)
    logger.debug("Graph built: nodes=%d, edges=%d", x.size(0), edge_index.size(1))
    return pyg_data.Data(x=x, edge_index=edge_index), node_map
            # New helper function for BERT fallback rule
def bert_fallback_rule(entity: str, facts: List[tuple], existing_facts: list) -> str:
    context_text = f"Entity: {entity} Facts: {facts} Existing: {existing_facts}"
    
    # Pre-defined rule templates
    rule_templates = [
        "(= (implies (fact \"{entity}\" \"{fact}\") (role \"{entity}\" leader)) true)",
        "(= (implies (fact \"{entity}\" \"{fact}\") (role \"{entity}\" politician)) true)",
        "(= (implies (fact \"{entity}\" \"{fact}\") (role \"{entity}\" entity)) true)",  # Default
    ]
    
    # Format with first fact if available
    fact = facts[0][1] if facts else "unknown"
    formatted_templates = [t.format(entity=entity, fact=fact) for t in rule_templates]
    
    # Compute similarities
    similarities = [neural_extractor.compute_similarity(context_text, templ) for templ in formatted_templates]
    max_sim_idx = np.argmax(similarities)
    best_rule = formatted_templates[max_sim_idx]
    confidence = similarities[max_sim_idx]
    
    if confidence < 0.7:
        best_rule = rule_templates[-1].format(entity=entity, fact=fact)
    
    return best_rule

def preload_facts():
    initial_facts = [
        ("fact:uhuru-kenyatta", "former president of Kenya, served 2013-2022, son of Jomo Kenyatta, AU-Kenya Peace Envoy"),
        ("fact:nairobi", "capital of Kenya")
    ]
    for key, value in initial_facts:
        das.add_atom(key, value)
        runner.run(f'(add-atom (fact "{key.split(":")[1]}" "{value}"))')
        # Add implication rule
        entity = key.split(":")[1]
        implies_rule = f"(= (implies (fact \"{entity}\" \"{value}\") (entity \"{entity}\" known)) true)"
        runner.run(implies_rule)
        das.add_atom(f"implies:{entity}:{uuid.uuid4()}", implies_rule)
        logger.info(f"Preloaded fact: {key}, value='{value}'")
        logger.info(f"Preloaded implies rule: {implies_rule}")
# Autonomous Goal Setting
def autonomous_goal_setting():
    while True:
        time.sleep(300)
        goals = runner.run('!(match &self (goal $g) $g)')
        interactions = das.query("interaction:*")[-5:] or ["No recent interactions"]
        if not goals and llm_enabled:
            goal_prompt = ChatPromptTemplate.from_template("""
            Suggest a learning goal based on recent interactions...
            Return: "Learn about <entity>"
            """)
            new_goal = (goal_prompt | llm | parser).invoke({"interactions": interactions})
            runner.run(f'(set-goal "{new_goal}")')
            logger.info(f"Dynamic goal set: {new_goal}")
        if goals:
            goal = goals[0]
            if "Learn about" in goal:
                entity = goal.split("Learn about")[-1].strip()
                continual_learning(entity)
        train_pattern_gnn()
        review_heuristics(manual=False)

# Retry Decorator
@retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(10))
def safe_invoke(chain, input_dict):
    try:
        logger.info(f"Attempting LLM invocation for input: {input_dict}")
        return chain.invoke(input_dict)
    except Exception as e:
        logger.error(f"LLM invocation failed after retries: {e}")
        runner.run(f'(add-atom (error "LLM Invoke" "{str(e)}"))')
        return f"LLM unavailable: {str(e)}. Falling back to MeTTa reasoning."

import gradio as gr
import pandas as pd
from datetime import datetime

# Function to format chat history for Gradio
def format_chat_history(history):
    chat_display = []
    for entry in history:
        if isinstance(entry, str) and entry.startswith("Updated"):
            chat_display.append([None, f"System: {entry}"])
        else:
            try:
                entry_data = json.loads(entry) if isinstance(entry, str) else entry
                question = entry_data.get("question", "Unknown")
                response = entry_data.get("output", "No response")
                # Ensure format is [question, response] for Gradio Chatbot
                chat_display.append([question, f"{response}\nConfidence: {entry_data.get('confidence', 0.85):.2f}\nTimestamp: {entry_data.get('timestamp', 'Unknown')}"])
            except (json.JSONDecodeError, AttributeError):
                chat_display.append([None, f"System: Invalid history entry: {entry}"])
    return chat_display

# Function to export chat history to CSV
def export_chat_to_csv():
    try:
        conn = sqlite3.connect('playground.db')
        cursor = conn.cursor()
        cursor.execute("SELECT question, output, confidence, timestamp FROM playground")
        rows = cursor.fetchall()
        conn.close()
        df = pd.DataFrame(rows, columns=["Question", "Response", "Confidence", "Timestamp"])
        csv_path = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported chat history to {csv_path}")
        return csv_path
    except sqlite3.Error as e:
        logger.error(f"CSV export failed: {e}")
        runner.run(f'(add-atom (error "CSV Export" "{str(e)}"))')
        return None
def parse_natural_language_update(update_text: str) -> tuple:
    """
    Parse natural language update to extract domain and fact.
    Tries LLM first, then BERT, then rule-based parsing.
    Returns: (domain, fact) or (None, None) if parsing fails.
    """
    try:
        # Default domain if not specified
        default_domain = "General"
        fact = update_text.strip()
        
        # Try LLM first if enabled
        if llm_enabled:
            try:
                parse_prompt = ChatPromptTemplate.from_template("""
                Parse the following input to extract a domain and fact for a knowledge graph update.
                Input: {input}
                Return in JSON format:
                {
                  "domain": "<domain>",
                  "fact": "<MeTTa fact like (fact \"entity\" \"fact\")>",
                  "confidence": <float>
                }
                If the input is invalid, return {"domain": null, "fact": null, "confidence": 0.0}.
                Valid domains: General, Kenyan-Politics, Science.
                Example input: "Kenyan-Politics: Uhuru Kenyatta is a former president."
                Example output: {"domain": "Kenyan-Politics", "fact": "(fact \"uhuru-kenyatta\" \"a former president\")", "confidence": 0.9}
                """)
                result = json.loads((parse_prompt | llm | parser).invoke({"input": update_text}))
                if result["domain"] and result["fact"] and result["confidence"] >= 0.7:
                    logger.info(f"LLM-parsed update: domain='{result['domain']}', fact='{result['fact']}', confidence={result['confidence']:.2f}")
                    return result["domain"], result["fact"]
                else:
                    logger.debug(f"LLM rejected update (low confidence or invalid): {result}")
                    # Fall through to BERT
            except Exception as e:
                logger.error(f"LLM parsing failed for update '{update_text}': {e}")
                runner.run(f'(add-atom (error "LLM Update Parse" "{str(e)}"))')
        
        # BERT-based fallback
        if neural_extractor.classifier is not None and neural_extractor.model is not None:
            prob = neural_extractor.classify_sentence(update_text)
            if prob < 0.7:
                logger.debug(f"BERT rejected update as non-fact (prob={prob:.2f}): '{update_text}'")
                runner.run(f'(add-atom (error "BERT Update Parse" "Input not a valid fact: {update_text}"))')
                # Fall through to rule-based
            else:
                # Extract entity and fact
                if " is " in update_text.lower():
                    parts = update_text.split(" is ", 1)
                    if len(parts) != 2 or not all(parts):
                        logger.debug(f"Invalid fact structure in BERT parsing: '{update_text}'")
                        runner.run(f'(add-atom (error "BERT Update Parse" "Invalid fact structure: {update_text}"))')
                        return None, None
                    entity, fact = parts
                    entity = entity.split('(')[0].split(',')[0].strip().lower().replace(" ", "-").replace("--", "-")
                    fact = fact.strip()
                    
                    # Detect domain
                    domain = default_domain
                    if ":" in update_text:
                        potential_domain = update_text.split(":")[0].strip().title()
                        if potential_domain in ["General", "Kenyan-Politics", "Science"]:
                            domain = potential_domain
                    elif " in " in update_text.lower():
                        domain = update_text.lower().split(" in ")[-1].split(" ")[0].title()
                    elif "about" in update_text.lower():
                        domain = update_text.lower().split("about")[-1].split(" ")[0].title()
                    
                    metta_fact = f'(fact "{entity}" "{fact}")'
                    logger.info(f"BERT-parsed update: domain='{domain}', fact='{metta_fact}', prob={prob:.2f}")
                    return domain, metta_fact
        
        # Rule-based fallback
        logger.info(f"BERT unavailable, using rule-based parsing for update: '{update_text}'")
        domain = default_domain
        if ":" in update_text:
            parts = update_text.split(":", 1)
            if len(parts) == 2:
                domain = parts[0].strip().title()
                fact = parts[1].strip()
        elif " in " in update_text.lower():
            parts = update_text.split(" in ", 1)
            if len(parts) == 2:
                fact = parts[0].strip()
                domain = parts[1].strip().title()
        elif "about" in update_text.lower():
            parts = update_text.split("about", 1)
            if len(parts) == 2:
                fact = parts[0].strip()
                domain = parts[1].strip().title()
        else:
            fact = update_text
        
        if not fact or " is " not in fact.lower():
            logger.debug(f"Invalid fact in rule-based parsing: '{update_text}'")
            runner.run(f'(add-atom (error "Rule-Based Update Parse" "Invalid fact format: {update_text}"))')
            return None, None
        
        parts = fact.split(" is ", 1)
        if len(parts) != 2 or not all(parts):
            logger.debug(f"Invalid fact structure in rule-based parsing: '{update_text}'")
            runner.run(f'(add-atom (error "Rule-Based Update Parse" "Invalid fact structure: {update_text}"))')
            return None, None
        entity = parts[0].split('(')[0].split(',')[0].strip().lower().replace(" ", "-").replace("--", "-")
        metta_fact = f'(fact "{entity}" "{parts[1].strip()}")'
        
        logger.info(f"Rule-based parsed update: domain='{domain}', fact='{metta_fact}'")
        return domain, metta_fact
    
    except Exception as e:
        logger.error(f"Failed to parse update '{update_text}': {e}")
        runner.run(f'(add-atom (error "Update Parse" "{str(e)}"))')
        return None, None
    
# Modified run_chatbot with Gradio interface
def run_chatbot():
    def format_chat_history(history):
        return [{'role': msg['role'], 'content': msg['content']} for msg in history]

    def process_input(question, faq_update, history):
        if not question and not faq_update:
            return history, "Please enter a question or an FAQ update.", None
        history = history or []

        # Handle FAQ update
        if faq_update:
            domain, fact = parse_natural_language_update(faq_update)
            if domain and fact:
                update_graph(domain, fact)
                update_msg = f"Updated {domain}: {fact}"
                history.append({'role': 'user', 'content': f"FAQ Update: {faq_update}"})
                history.append({'role': 'assistant', 'content': update_msg, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                logger.info(f"Processed FAQ update: {update_msg}")
                return format_chat_history(history), update_msg, None
            else:
                error_msg = f"Failed to parse FAQ update: '{faq_update}'. Use format like 'Kenyan-Politics: Uhuru Kenyatta is a former president.'"
                history.append({'role': 'user', 'content': f"FAQ Update: {faq_update}"})
                history.append({'role': 'assistant', 'content': error_msg, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                return format_chat_history(history), error_msg, None

        # Normalize query for "Who" questions
        normalized_question = question
        if question.lower().startswith("who ") and " is " not in question.lower():
            normalized_question = question.replace("Who ", "Who is ", 1)
            logger.info(f"Normalized query: '{question}' to '{normalized_question}'")
        question = normalized_question

        if question.lower() == 'errors':
            errors = runner.run('!(get-errors $type)') or ["No recent errors"]
            history.append({'role': 'user', 'content': question})
            history.append({'role': 'assistant', 'content': f"Recent Errors: {errors}", 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            return format_chat_history(history), f"Recent Errors: {errors}", None
        if question.lower() == 'review heuristics':
            review_result = review_heuristics(manual=True)
            history.append({'role': 'user', 'content': question})
            history.append({'role': 'assistant', 'content': review_result, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            return format_chat_history(history), review_result, None
        if question.lower().startswith('update'):
            parts = question[6:].strip().split(' ', 1)
            domain = parts[0] if len(parts) > 1 else "General"
            new_fact = parts[1] if len(parts) > 1 else ""
            update_graph(domain, new_fact)
            update_msg = f"Updated {domain}: {new_fact}"
            history.append({'role': 'user', 'content': question})
            history.append({'role': 'assistant', 'content': update_msg, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            return format_chat_history(history), update_msg, None
        try:
            if llm_enabled:
                response = safe_invoke(chain, {"question": question})
            else:
                logger.warning("LLM disabled, falling back to MeTTa reasoning")
                response = query_metta_dynamic(question) + "\nNote: LLM is disabled due to API quota issues."
        except Exception as e:
            logger.error(f"Query processing failed for '{question}': {e}")
            runner.run(f'(add-atom (error "Query Processing" "{str(e)}"))')
            response = f"Sorry, I couldn't process that request due to an internal error: {str(e)}"
        confidence = 0.85
        store_memory(question, response, confidence)
        history.append({'role': 'user', 'content': question})
        history.append({'role': 'assistant', 'content': response, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        accuracy = benchmark_clevr_vqa(runner, question, "Unknown", response)
        save_to_playground(question, response, history, {'clevr_vqa': accuracy})
        if "who is" in question.lower() or "what is" in question.lower():
            entity = (question.lower().split("who is")[-1].strip().rstrip('?').replace(" ", "-").replace("--", "-") if "who is" in question.lower() else
                      question.lower().split("what is")[-1].strip().rstrip('?').replace(" ", "-").replace("--", "-"))
            if entity:
                logger.info(f"Extracted entity for continual learning: {entity}")
                try:
                    continual_learning(entity)
                except Exception as e:
                    logger.error(f"Continual learning failed for {entity}: {e}")
                    runner.run(f'(add-atom (error "Continual Learning" "{str(e)}"))')
                    response += f"\nFailed to learn facts for {entity}: {str(e)}"
                    history[-1]['content'] = response
        return format_chat_history(history), response, None

    def launch_gradio():
        server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
        server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
        max_attempts = 20
        for port in range(server_port, server_port + max_attempts):
            try:
                with gr.Blocks(title="General FAQ Chatbot") as interface:
                    gr.Markdown("""
                    # General FAQ Chatbot
                    Ask questions, update FAQs with natural language (e.g., 'Kenyan-Politics: Uhuru Kenyatta is a former president.' or 'Add to Science: Gravity is a force.'), review heuristics with `review heuristics`, or check errors with `errors`.  
                    Note: LLM is disabled due to OpenAI API quota issues. Using MeTTa reasoning and BERT-based fallbacks.  
                    Note: Redis is required for persistent storage. Install with: `sudo apt-get install redis-server` (Ubuntu) or `brew install redis` (macOS).
                    """)
                    chatbot = gr.Chatbot(label="Chat History", height=400, type="messages")
                    question_input = gr.Textbox(label="Ask a question", placeholder="Type your question here (e.g., 'Who is Uhuru Kenyatta?')")
                    faq_update_input = gr.Textbox(label="Update FAQ", placeholder="Enter FAQ update (e.g., 'Kenyan-Politics: Uhuru Kenyatta is a former president.')")
                    submit_button = gr.Button("Submit Question")
                    faq_submit_button = gr.Button("Submit FAQ Update")
                    csv_button = gr.Button("Export Chat History as CSV")
                    csv_output = gr.File(label="Download CSV")

                    submit_button.click(
                        fn=process_input,
                        inputs=[question_input, faq_update_input, chatbot],
                        outputs=[chatbot, gr.Textbox(label="Response"), csv_output]
                    )
                    faq_submit_button.click(
                        fn=process_input,
                        inputs=[question_input, faq_update_input, chatbot],
                        outputs=[chatbot, gr.Textbox(label="Response"), csv_output]
                    )
                    csv_button.click(
                        fn=export_chat_to_csv,
                        inputs=None,
                        outputs=csv_output
                    )
                    logger.info(f"Attempting to launch Gradio on port {port}")
                    interface.launch(server_name=server_name, server_port=port, share=True)
                    logger.info(f"Gradio launched successfully on http://{server_name}:{port}")
                    return
            except Exception as e:
                logger.error(f"Failed to launch Gradio on port {port}: {e}")
                runner.run(f'(add-atom (error "Gradio Launch" "Port {port}: {str(e)}"))')
                if "Cannot find empty port" in str(e):
                    continue
                else:
                    print(f"Gradio launch failed: {e}. Check logs or try a different port range.")
                    return
        print(f"Failed to find an available port in range {server_port}-{server_port + max_attempts - 1}. Set GRADIO_SERVER_PORT to a free port or free up port {server_port}.")
        runner.run(f'(add-atom (error "Gradio Launch" "No available ports in range {server_port}-{server_port + max_attempts - 1}"))')

    threading.Thread(target=autonomous_goal_setting, daemon=True).start()
    test_question = "Who is Uhuru Kenyatta"
    history = []
    history, response, _ = process_input(test_question, "", history)
    print(f"Question: {test_question}\nResponse: {response}")
    launch_gradio()
    
# Update Graph
def update_graph(domain: str, new_fact: str):
    try:
        if not new_fact.startswith('(= '):
            new_fact = f'(= {new_fact} true)'
        entity = new_fact.split()[1] if len(new_fact.split()) > 1 else "Unknown"
        runner.run(f'(add-atom (in-domain {entity} {domain}) true)')
        runner.run(new_fact)
        runner.run(f'(add-atom (new-atom "{entity}"))')
        vectorize_graph()
        logger.info(f"Graph updated in domain {domain} with: {new_fact}")
    except Exception as e:
        logger.error(f"Graph update failed: {e}")
        runner.run(f'(add-atom (error "Update" "{str(e)}"))')

if __name__ == "__main__":
    run_chatbot()
