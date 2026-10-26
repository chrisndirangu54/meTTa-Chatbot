# Integration validation matrix

## CI (safe, no secrets or model downloads)
```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q neuro_agent metta.py
```
The legacy contract suite uses AST checks; this **does not** import or launch `metta.py`. Live integration suites skip by default.

## Neural + symbolic integration (downloads pretrained model weights)
Install optional packages: `pip install hyperon numpy faiss-cpu sentence-transformers transformers torch`.
Then:
```bash
RUN_LIVE_INTEGRATION=1 python -m unittest discover -s tests -p 'test_live_integration.py' -v
```
Tests instantiate a real MeTTa space, perform actual BERT inference, compute real embeddings, and exercise a real FAISS index. These verify underlying component interoperability, **not** the full legacy Gradio workflow or fine-tuning pipeline.

## Redis persistence
Start an isolated Redis instance; use a dedicated test DB and avoid production credentials.
```bash
pip install redis
RUN_LIVE_INTEGRATION=1 RUN_REDIS_INTEGRATION=1 REDIS_URL=redis://127.0.0.1:6379/15 python -m unittest discover -s tests -p 'test_live_integration.py' -v
```
The test uses separate clients and deletes its temporary key. It does not stop or wipe Redis.

## Gemini API (may incur provider charges)
```bash
pip install langchain-google-genai
RUN_LIVE_INTEGRATION=1 RUN_GEMINI_INTEGRATION=1 GEMINI_API_KEY=<SECRET> GEMINI_MODEL=gemini-2.5-flash python -m unittest discover -s tests -p 'test_live_integration.py' -v
```
Use a secret manager for hosted runs; never commit API keys. A valid model accessible to your account is required. This makes one actual API request.

## Legacy application full-system validation: still pending
The original entry point imports a large dependency stack, instantiates networks and model weights, can fine-tune BERT, launches Gradio and starts a background autonomous loop. Testing it safely requires a dedicated test container with model cache and resource budgets; use its `--test` flag first in an isolated environment and monitor its resource usage. **Do not run its default production main path as an automated CI test.**

## Known gaps
- Legacy `metta.py` uses `gemini-1.5-flash`; provider availability and quotas need review.
- Legacy training and external web search need deterministic fixtures and limits.
- No end-to-end Gradio user flow, concurrency, cost, security or throughput test has been performed.
- Live tests must report actual PASS / FAIL / SKIP; skipped checks are **not** evidence of successful integration.
