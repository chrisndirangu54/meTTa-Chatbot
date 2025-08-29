General FAQ Chatbot

A modular FAQ chatbot built with MeTTa for symbolic reasoning, Graph Neural Networks (GNNs) for pattern inference, BERT for natural language understanding, and Gradio for an interactive web interface. The system supports continual learning, web search integration, and distributed atom storage (DAS) with Redis or in-memory fallback. It is designed to handle general knowledge queries, perform autonomous goal setting, and allow dynamic updates to the knowledge base.

Features
- Symbolic Reasoning: Uses MeTTa for logic-based query processing and knowledge graph management.
- Neural Integration: Employs BERT for fact extraction and GraphSAGE for reasoning pattern inference.
- Continual Learning: Automatically learns new facts for unrecognized entities via web search.
- Interactive Interface: Gradio-based UI for asking questions, updating FAQs, and reviewing system performance.
- Distributed Storage: Supports Redis for persistent atom storage, with in-memory fallback if Redis is unavailable.
- Error Handling: Robust logging and error tracking with MeTTa-based error reporting.
- Benchmarking: Evaluates performance using a CLEVR/VQA-inspired accuracy metric.
- FAQ Updates: Supports natural language updates to the knowledge graph (e.g., "Kenyan-Politics: Uhuru Kenyatta is a former president.").

Requirements
- Python: 3.8+
- Dependencies:
  - hyperon
  - langchain-openai
  - langchain-google-genai
  - sentence-transformers
  - faiss-cpu
  - redis
  - transformers
  - datasets
  - torch
  - torch-geometric
  - requests
  - duckduckgo-search (aliased as ddgs)
  - tenacity
  - sqlite3
  - gradio
  - pandas
- Optional:
  - Redis server for persistent storage (install via `sudo apt-get install redis-server` on Ubuntu or `brew install redis` on macOS).
  - Gemini API key for LLM functionality (set via GOOGLE_API_KEY environment variable).

InstallationClone the Repository:
git clone <repository-url>
cd general-faq-chatbot
Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
pip install hyperon langchain-openai langchain-google-genai sentence-transformers faiss-cpu redis transformers datasets torch torch-geometric requests duckduckgo-search tenacity gradio pandas
Install Redis (optional, for persistent storage):Ubuntu/Debian:
sudo apt-get install redis-server
macOS:
brew install redis

Set Environment Variables:For Gemini API:
export GOOGLE_API_KEY="your-gemini-api-key"
For Redis (if not using default 127.0.0.1:6379):
export REDIS_HOST="your-redis-host"
export REDIS_PORT="your-redis-port"
export REDIS_URL="your-redis-url"  # Optional
For Gradio (optional, to specify server details):
export GRADIO_SERVER_NAME="your-server-name"  # Default: 127.0.0.1
export GRADIO_SERVER_PORT="your-server-port"  # Default: 7860

Disable GPU (if needed):
The code is configured to use CPU-only operations. Ensure CUDA_VISIBLE_DEVICES="" is set in the environment:
export CUDA_VISIBLE_DEVICES=""




Usage

1. Run the Chatbot:
   python chatbot.py
   This starts the Gradio interface (default: http://127.0.0.1:7860) and initiates autonomous goal setting in the background.

2. Interact via Gradio:
   - Ask Questions: Enter queries like "Who is Uhuru Kenyatta?" or "What is gravity?" in the question input box.
   - Update FAQs: Add new facts using natural language, e.g., "Kenyan-Politics: Uhuru Kenyatta is a former president." or "Add to Science: Gravity is a force."
   - Review Heuristics: Type `review heuristics` to evaluate and refine reasoning patterns.
   - Check Errors: Type `errors` to view recent system errors.
   - Export History: Click "Export Chat History as CSV" to download interaction logs.

3. Example Queries:
   - Who is Uhuru Kenyatta?
   - What is the capital of Kenya?
   - Science: The speed of light is 299,792,458 meters per second.
   - review heuristics
   - errors

Architecture

- MeTTa: Core symbolic reasoning engine using a GroundingSpace for knowledge representation.
- Graph Neural Network (GNN): GraphSAGE model for learning reasoning patterns from MeTTa atoms and implications.
- BERT: Fine-tuned for fact extraction and sentence classification, with embeddings for similarity comparisons.
- FAISS: Vector search for efficient retrieval of similar atoms based on SentenceTransformer embeddings.
- Distributed Atomspace (DAS): Redis-backed storage for atoms, with in-memory fallback if Redis is unavailable.
- LangChain with Gemini: Optional LLM integration for heuristic generation and query augmentation (disabled if API key is invalid or quota is exceeded).
- Gradio: Web-based UI for user interaction and system management.
- SQLite: Stores chat history and benchmarking results for persistence and export.
- Web Search: Integrates DuckDuckGo for real-time fact retrieval during continual learning.

Continual Learning
The system automatically learns new facts when it encounters unknown entities:
1. Queries the web using DuckDuckGo for relevant information.
2. Extracts facts using a lightweight BERT-based extractor.
3. Updates the MeTTa knowledge graph and FAISS index.
4. Stores facts in DAS (Redis or in-memory).
5. Trains the GNN on new reasoning patterns.

Error Handling
- Errors are logged to the console and stored in MeTTa as (error <type> <message>) atoms.
- Common issues (e.g., Redis connection failures, LLM quota limits) trigger fallbacks to in-memory storage or rule-based reasoning.
- Check errors via the `errors` command in the Gradio interface.

Benchmarking
The system evaluates performance using a CLEVR/VQA-inspired metric:
- Compares responses against known MeTTa facts and a provided ground truth.
- Accuracy is logged and stored in SQLite for analysis.
- Access benchmarking results via exported CSV files.

Limitations
- LLM Dependency: Gemini API key is required for full functionality; without it, the system falls back to BERT and rule-based methods.
- Redis Requirement: Persistent storage requires a running Redis server; otherwise, in-memory storage is used, which is not persistent.
- CPU-Only: The system is configured for CPU to ensure compatibility, which may limit performance on large datasets.
- Port Conflicts: Gradio may fail to launch if the default port (7860) is in use. Set GRADIO_SERVER_PORT to a free port if needed.

Troubleshooting
- Redis Connection Issues:
  - Ensure Redis is installed and running (redis-server).
  - Verify REDIS_HOST, REDIS_PORT, or REDIS_URL environment variables.
  - If Redis is unavailable, the system uses in-memory storage (non-persistent).
- LLM Errors:
  - Check GOOGLE_API_KEY is valid and quota is not exceeded.
  - Without a valid key, the system falls back to BERT and rule-based reasoning.
- Gradio Launch Failure:
  - Free up port 7860 or set GRADIO_SERVER_PORT to an available port.
  - Check logs for specific errors (e.g., Cannot find empty port).
- Dependency Issues:
  - Ensure all dependencies are installed correctly.
  - Use a virtual environment to avoid conflicts.

Contributing
Contributions are welcome! Please submit pull requests or open issues for bugs, feature requests, or improvements.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or support, contact the maintainers via GitHub issues or email.



<img width="3840" height="2314" alt="Untitled diagram _ Mermaid Chart-2025-08-29-055923" src="https://github.com/user-attachments/assets/d5db358f-5431-4cbc-9d39-f02cc1f3da68" />

