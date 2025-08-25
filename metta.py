


import threading
import time
from hyperon import *
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
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Setup logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MeTTa and Atomspace
space = AtomSpace()
runner = MeTTa(space=space)

# Vectorization: SentenceTransformer for embedding atoms
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS for vector search (modular, scalable)
dimension = 384  # MiniLM embedding size
index = faiss.IndexFlatL2(dimension)
atom_vectors = {}  # {atom_id: atom_text}

# DAS Setup 
class DASClient:
    def __init__(self, host='localhost', port=6379):
        try:
            self.redis = redis.Redis(host=host, port=port, decode_responses=True)
            logger.info("DAS client initialized with Redis backend")
        except Exception as e:
            logger.error(f"DAS initialization failed: {e}")
            sys.exit(1)

    def add_atom(self, atom: str, value: Any):
        try:
            self.redis.set(atom, str(value))
            logger.info(f"Added atom {atom} to DAS")
        except Exception as e:
            logger.error(f"DAS add_atom failed: {e}")

    def query(self, pattern: str) -> List[str]:
        try:
            keys = self.redis.keys(f"{pattern}*")
            results = [self.redis.get(key) for key in keys]
            logger.info(f"DAS query returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"DAS query failed: {e}")
            return []

das = DASClient()

# Load initial general knowledge graph in MeTTa
knowledge_metta = """
(: General Domain)
(: Person Type)
(: Role Type)
(: Entity Type)

; Modular domain example
(: Kenyan-Politics Domain)
(add-atom (in-domain Uhuru-Kenyatta Kenyan-Politics) true)
(: Science Domain)

; General inference rules
(= (definition Person) "A human individual with roles and relationships.")
(= (example Person) "Example: Individuals like Uhuru Kenyatta.")
(= (definition Role) "A position or function, e.g., President.")
(= (example Role) "Example: President of a country.")

; Dynamic relationship extraction
(= (get-roles $entity)
   (match &self (role $entity $r) $r))

(= (get-relations $entity $type)
   (match &self ($type $entity $other) ($type $other)))

; Reverse inference
(= (son-of $x $y) (father-of $y $x))

; Domain selection
(= (select-domain $domain)
   (match &self (in-domain $entity $domain) $entity))

; Self-reflection rule
(= (reflect $interaction $confidence)
   (if (< $confidence 0.7)
      (refine-rule $interaction)
      "No refinement needed"))

; Refine strategy
(= (refine-rule $interaction)
   (add-atom (improved-rule $interaction "Refined")))

; Cross-domain transfer
(= (transfer-pattern $domain1 $domain2 $pattern)
   (match &self (pattern $domain1 $pattern)
      (add-atom (pattern $domain2 $pattern))))

; Goal setting
(= (set-goal $goal)
   (add-atom (goal $goal)))

; Continual learning rule
(= (learn-fact $entity $fact)
   (add-atom (fact $entity $fact)))
"""
runner.run(knowledge_metta)
logger.info("Initial knowledge graph loaded.")

# Vectorize atoms and store in FAISS/DAS
def vectorize_graph():
    global index, atom_vectors
    atoms = runner.run('!(match &self $atom $atom)') or []
    index = faiss.IndexFlatL2(dimension)
    atom_vectors = {}
    vectors = []
    for atom in atoms:
        atom_text = str(atom)
        embedding = embedder.encode(atom_text)
        atom_id = str(uuid.uuid4())
        atom_vectors[atom_id] = atom_text
        vectors.append(embedding)
        das.add_atom(f"embedding:{atom_id}", embedding.tolist())
    if vectors:
        index.add(np.array(vectors))
    logger.info("Knowledge graph vectorized.")

vectorize_graph()

# Neural Extractor for continual learning 
class NeuralExtractor:
    def __init__(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            self.fine_tune()
            logger.info("BERT models initialized and fine-tuned")
        except Exception as e:
            logger.error(f"BERT initialization failed: {e}")
            sys.exit(1)

    def fine_tune(self):
        data = {
            'text': [
                "Uhuru Kenyatta is president of Kenya.",
                "Jomo Kenyatta is father of Uhuru Kenyatta.",
                "Kenya is a country."
            ],
            'label': [1, 1, 1]
        }
        dataset = Dataset.from_dict(data)
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
            output_dir='./bert-finetuned',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir='./logs',
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.classifier,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()
        logger.info("BERT fine-tuned for fact extraction")

    def extract_facts(self, text: str) -> List[tuple]:
        # Generate candidate facts (simplified)
        templates = [
            f"{text.split(' ')[-1]} is a [MASK].",
            f"{text.split(' ')[-1]} is [MASK] of Kenya."
        ]
        candidate_facts = []
        for template in templates:
            inputs = self.tokenizer(template, return_tensors='pt')
            outputs = self.classifier(**inputs)
            logits = outputs.logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            mask_token = template.replace("[MASK]", self.tokenizer.decode(torch.argmax(logits, dim=-1)))
            candidate_facts.append((text, mask_token, prob))
        return candidate_facts

neural_extractor = NeuralExtractor()

# LLM Integration (LangChain with GPT-4o-mini)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a general FAQ chatbot. Use the following context from the knowledge graph to answer the question.
Context: {context}

Question: {question}

Provide a structured answer with definitions, examples, and references. If unsure, suggest updates.
""")

parser = StrOutputParser()

chain = {
    "context": lambda x: query_metta_dynamic(x["question"]),
    "question": RunnablePassthrough()
} | prompt | llm | parser

# Dynamic MeTTa query with vector search
def query_metta_dynamic(question: str) -> str:
    try:
        query_emb = embedder.encode(question)
        _, indices = index.search(np.array([query_emb]), k=5)
        similar_atoms = [atom_vectors[list(atom_vectors.keys())[i]] for i in indices[0] if i < len(atom_vectors)]
        entity = question.split("Who is")[-1].strip().rstrip('?') if "Who is" in question else question.split()[-1].rstrip('?')
        domains = runner.run(f'!(match &self (in-domain {entity} $d) $d)') or ["General"]
        domain = domains[0] if domains else "General"
        faqs = runner.run(f'!(get-faq "{question}" {domain})') or ["No FAQ"]
        roles = runner.run(f'!(get-roles {entity})') or ["Unknown"]
        relations = runner.run(f'!(get-relations {entity} father-of)') or ["None"]
        definition = runner.run(f'!(definition {entity})') or runner.run('!(definition Person)') or ["No definition"]
        example = runner.run(f'!(example {entity})') or runner.run('!(example Person)') or ["No example"]
        context = (
            f"Domain: {domain}\n"
            f"FAQ: {question} → {faqs[0]}\n"
            f"Similar facts: {similar_atoms}\n"
            f"Roles: {roles}\n"
            f"Relations: {relations}\n"
            f"Definition: {definition}\n"
            f"Example: {example}"
        )
        logger.info(f"Dynamic MeTTa query result: {context}")
        return context
    except Exception as e:
        logger.error(f"Dynamic MeTTa query failed: {e}")
        return "Error querying knowledge graph."

# Update knowledge graph (user-fed data, modular)
def update_graph(domain: str, new_fact: str):
    try:
        if not new_fact.startswith('(= '):
            new_fact = f'(= {new_fact} true)'
        entity = new_fact.split()[1] if len(new_fact.split()) > 1 else "Unknown"
        runner.run(f'(add-atom (in-domain {entity} {domain}) true)')
        runner.run(new_fact)
        vectorize_graph()
        logger.info(f"Graph updated in domain {domain} with: {new_fact}")
    except Exception as e:
        logger.error(f"Graph update failed: {e}")

# Autonomous goal setting and execution (narcissist/realist)
def autonomous_goal_setting():
    while True:
        time.sleep(30)  # Check every 30 seconds
        goals = runner.run('!(match &self (goal $g) $g)')
        if goals:
            goal = goals[0]
            if "Learn about" in goal:
                entity = goal.split("Learn about")[-1].strip()
                facts = neural_extractor.extract_facts(entity)
                for entity, fact, conf in facts:
                    runner.run(f'!(learn-fact "{entity}" "{fact}")')
                vectorize_graph()
                logger.info(f"Autonomous learning goal executed for {entity}")
            elif "Reflect on" in goal:
                interactions = das.query("interaction:*")
                for inter in interactions[-5:]:  # Last 5
                    runner.run(f'!(reflect "{inter}" 0.6)')  # Mock confidence
                logger.info("Self-reflection goal executed")
        else:
            # Set default goal
            runner.run('(set-goal "Learn about Uhuru Kenyatta")')
            logger.info("Default goal set")

# Memory storage 
def store_memory(question: str, response: str, confidence: float = 0.85):
    inter_id = str(uuid.uuid4())
    das.add_atom(f"interaction:{inter_id}", f"Query: {question} Response: {response} Confidence: {confidence}")

# Self-reflection in MeTTa (already in knowledge_metta)

# Cross-domain transfer 
def cross_domain_transfer(domain1: str, domain2: str, pattern: str):
    runner.run(f'(transfer-pattern {domain1} {domain2} "{pattern}")')
    logger.info(f"Transferred pattern from {domain1} to {domain2}")

# Continual learning function (narcissist/optimist)
def continual_learning(entity: str):
    web_result = web_search(f"Who is {entity}?")
    facts = neural_extractor.extract_facts(web_result)
    for entity, fact, conf in facts:
        runner.run(f'!(learn-fact "{entity}" "{fact}")')
    vectorize_graph()
    logger.info(f"Continual learning: Added facts for {entity}")

# Web Search Mock (enhanced for continual learning)
def web_search(query: str) -> str:
    mock_results = {
        "Who is Uhuru Kenyatta?": "Uhuru Kenyatta is a Kenyan politician, fourth president of Kenya (2013–2022), son of Jomo Kenyatta.",
        "What is E=mc2?": "Einstein’s mass-energy equivalence equation."
    }
    return mock_results.get(query, "No web results found.")

# Benchmarking 
def benchmark_clevr_vqa(runner: MeTTa, question: str, ground_truth: str) -> float:
    try:
        mock_dataset = [
            {'question': 'Who is Uhuru Kenyatta?', 'answer': 'President of Kenya'},
            {'question': 'What is E=mc2?', 'answer': 'Einstein’s equation'}
        ]
        correct = 0
        total = len(mock_dataset) + 1
        for item in mock_dataset:
            response = chain.invoke({"question": item['question']})
            if item['answer'].lower() in response.lower():
                correct += 1
        response = chain.invoke({"question": question})
        if ground_truth.lower() in response.lower():
            correct += 1
        accuracy = correct / total * 100
        logger.info(f"CLEVR/VQA Accuracy: {accuracy:.2f}%")
        return accuracy
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        return 0.0

# Save to playground 
def save_to_playground(question: str, output: str, history: list, accuracies: dict):
    try:
        data = {
            'question': question,
            'final_output': output,
            'history': history,
            'accuracies': accuracies,
            'timestamp': logging.Formatter().formatTime(logging.makeLogRecord({}))
        }
        with open('playground_output.json', 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Saved to playground_output.json for metta-lang.dev")
    except Exception as e:
        logger.error(f"Playground save failed: {e}")

# Chatbot loop
def run_chatbot():
    print("General FAQ Chatbot. Type 'exit' to quit, 'update <domain> <fact>' to add knowledge (e.g., update Science (= (E=mc2 Equation) true)).")
    history = []
    threading.Thread(target=autonomous_goal_setting, daemon=True).start()  # Autonomous loop
    while True:
        question = input("Ask a question: ")
        if question.lower() == 'exit':
            break
        if question.lower().startswith('update'):
            parts = question[6:].strip().split(' ', 1)
            domain = parts[0] if len(parts) > 1 else "General"
            new_fact = parts[1] if len(parts) > 1 else ""
            update_graph(domain, new_fact)
            print("Knowledge graph updated.")
            history.append(f"Updated {domain}: {new_fact}")
            continue
        response = chain.invoke({"question": question})
        confidence = 0.85  # Mock
        store_memory(question, response, confidence)  # Memory
        history.append(response)
        accuracy = benchmark_clevr_vqa(runner, question, "President of Kenya" if "Uhuru Kenyatta" in question else "Unknown")
        save_to_playground(question, response, history, {'clevr_vqa': accuracy})
        # Self-reflection
        runner.run(f'!(reflect "Query: {question}" {confidence})')
        # Continual learning example
        entity = question.split("Who is")[-1].strip().rstrip('?') if "Who is" in question else ""
        if entity:
            continual_learning(entity)
        # Cross-domain transfer example
        cross_domain_transfer("Kenyan-Politics", "Science", "(implies (role X president) (role X leader))")
        print("Response:", response)

if __name__ == "__main__":
    run_chatbot()
import threading
import time
from hyperon import *
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
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Setup logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MeTTa and Atomspace
space = AtomSpace()
runner = MeTTa(space=space)

# Vectorization: SentenceTransformer for embedding atoms
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS for vector search (modular, scalable)
dimension = 384  # MiniLM embedding size
index = faiss.IndexFlatL2(dimension)
atom_vectors = {}  # {atom_id: atom_text}

# DAS Setup 
class DASClient:
    def __init__(self, host='localhost', port=6379):
        try:
            self.redis = redis.Redis(host=host, port=port, decode_responses=True)
            logger.info("DAS client initialized with Redis backend")
        except Exception as e:
            logger.error(f"DAS initialization failed: {e}")
            sys.exit(1)

    def add_atom(self, atom: str, value: Any):
        try:
            self.redis.set(atom, str(value))
            logger.info(f"Added atom {atom} to DAS")
        except Exception as e:
            logger.error(f"DAS add_atom failed: {e}")

    def query(self, pattern: str) -> List[str]:
        try:
            keys = self.redis.keys(f"{pattern}*")
            results = [self.redis.get(key) for key in keys]
            logger.info(f"DAS query returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"DAS query failed: {e}")
            return []

das = DASClient()

# Load initial general knowledge graph in MeTTa
knowledge_metta = """
(: General Domain)
(: Person Type)
(: Role Type)
(: Entity Type)

; Modular domain example
(: Kenyan-Politics Domain)
(add-atom (in-domain Uhuru-Kenyatta Kenyan-Politics) true)
(: Science Domain)

; General inference rules
(= (definition Person) "A human individual with roles and relationships.")
(= (example Person) "Example: Individuals like Uhuru Kenyatta.")
(= (definition Role) "A position or function, e.g., President.")
(= (example Role) "Example: President of a country.")

; Dynamic relationship extraction
(= (get-roles $entity)
   (match &self (role $entity $r) $r))

(= (get-relations $entity $type)
   (match &self ($type $entity $other) ($type $other)))

; Reverse inference
(= (son-of $x $y) (father-of $y $x))

; Domain selection
(= (select-domain $domain)
   (match &self (in-domain $entity $domain) $entity))

; Self-reflection rule
(= (reflect $interaction $confidence)
   (if (< $confidence 0.7)
      (refine-rule $interaction)
      "No refinement needed"))

; Refine strategy
(= (refine-rule $interaction)
   (add-atom (improved-rule $interaction "Refined")))

; Cross-domain transfer
(= (transfer-pattern $domain1 $domain2 $pattern)
   (match &self (pattern $domain1 $pattern)
      (add-atom (pattern $domain2 $pattern))))

; Goal setting
(= (set-goal $goal)
   (add-atom (goal $goal)))

; Continual learning rule
(= (learn-fact $entity $fact)
   (add-atom (fact $entity $fact)))
"""
runner.run(knowledge_metta)
logger.info("Initial knowledge graph loaded.")

# Vectorize atoms and store in FAISS/DAS
def vectorize_graph():
    global index, atom_vectors
    atoms = runner.run('!(match &self $atom $atom)') or []
    index = faiss.IndexFlatL2(dimension)
    atom_vectors = {}
    vectors = []
    for atom in atoms:
        atom_text = str(atom)
        embedding = embedder.encode(atom_text)
        atom_id = str(uuid.uuid4())
        atom_vectors[atom_id] = atom_text
        vectors.append(embedding)
        das.add_atom(f"embedding:{atom_id}", embedding.tolist())
    if vectors:
        index.add(np.array(vectors))
    logger.info("Knowledge graph vectorized.")

vectorize_graph()

# Neural Extractor for continual learning 
class NeuralExtractor:
    def __init__(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            self.fine_tune()
            logger.info("BERT models initialized and fine-tuned")
        except Exception as e:
            logger.error(f"BERT initialization failed: {e}")
            sys.exit(1)

    def fine_tune(self):
        data = {
            'text': [
                "Uhuru Kenyatta is president of Kenya.",
                "Jomo Kenyatta is father of Uhuru Kenyatta.",
                "Kenya is a country."
            ],
            'label': [1, 1, 1]
        }
        dataset = Dataset.from_dict(data)
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
            output_dir='./bert-finetuned',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir='./logs',
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.classifier,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()
        logger.info("BERT fine-tuned for fact extraction")

    def extract_facts(self, text: str) -> List[tuple]:
        # Generate candidate facts (simplified)
        templates = [
            f"{text.split(' ')[-1]} is a [MASK].",
            f"{text.split(' ')[-1]} is [MASK] of Kenya."
        ]
        candidate_facts = []
        for template in templates:
            inputs = self.tokenizer(template, return_tensors='pt')
            outputs = self.classifier(**inputs)
            logits = outputs.logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            mask_token = template.replace("[MASK]", self.tokenizer.decode(torch.argmax(logits, dim=-1)))
            candidate_facts.append((text, mask_token, prob))
        return candidate_facts

neural_extractor = NeuralExtractor()

# LLM Integration (LangChain with GPT-4o-mini)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a general FAQ chatbot. Use the following context from the knowledge graph to answer the question.
Context: {context}

Question: {question}

Provide a structured answer with definitions, examples, and references. If unsure, suggest updates.
""")

parser = StrOutputParser()

chain = {
    "context": lambda x: query_metta_dynamic(x["question"]),
    "question": RunnablePassthrough()
} | prompt | llm | parser

# interactions = das.query("interaction:*")
                for inter in interactions[-5:]:  # Last 5
                    runner.run(f'!(reflect "{inter}" 0.6)')  # Mock confidence
                logger.info("Self-reflection goal executed")
        else:
            # Set default goal
            runner.run('(set-goal "Learn about Uhuru Kenyatta")')
            logger.info("Default goal set")

# Memory storage 
def store_memory(question: str, response: str, confidence: float = 0.85):
    inter_id = str(uuid.uuid4())
    das.add_atom(f"interaction:{inter_id}", f"Query: {question} Response: {response} Confidence: {confidence}")

# Self-reflection in MeTTa (already in knowledge_metta)

# Cross-domain transfer 
def cross_domain_transfer(domain1: str, domain2: str, pattern: str):
    runner.run(f'(transfer-pattern {domain1} {domain2} "{pattern}")')
    logger.info(f"Transferred pattern from {domain1} to {domain2}")

# Continual learning function (narcissist/optimist)
.append(response)
        accuracy = benchmark_clevr_vqa(runner, question, "President of Kenya" if "Uhuru Kenyatta" in question else "Unknown")
        save_to_playground(question, response, history, {'clevr_vqa': accuracy})
        # Self-reflection
        runner.run(f'!(reflect "Query: {question}" {confidence})')
        # Continual learning example
        entity = question.split("Who is")[-1].strip().rstrip('?') if "Who is" in question else ""
        if entity:
            continual_learning(entity)
        # Cross-domain transfer example
        cross_domain_transfer("Kenyan-Politics", "Science", "(implies (role X president) (role X leader))")
        print("Response:", response)

if __name__ == "__main__":
    run_chatbot()
