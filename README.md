# ResumeForge 🔨

> AI-powered resume analysis, job matching, optimization, and review — in a single Python file.

## What it does

Upload your resume (PDF, DOCX, or TXT) along with a job description and target role. ResumeForge runs a four-agent [LangGraph](https://github.com/langchain-ai/langgraph) workflow that:

1. **Analyzes** your resume — extracts skills, experience, projects, achievements, weak sections, and missing information.
2. **Matches** it against the job description — produces a 0–100 fit score with gap analysis.
3. **Optimizes** the resume — rewrites bullet points, improves keyword coverage, and strengthens action verbs without hallucinating new facts.
4. **Reviews** the result — scores ATS relevance, clarity, grammar, and consistency, then flags any hallucinations. If quality is insufficient, it re-optimizes automatically.

## Stack

| Layer | Technology |
|---|---|
| API server | FastAPI + LangServe |
| Workflow | LangGraph |
| LLM | Any model via OpenRouter |
| LLM client | langchain-openai |
| Document parsing | PyMuPDF · python-docx |

## Quick start

### 1. Clone

```
git clone https://github.com/vidwansai66/ResumeForge.git
cd ResumeForge
```

### 2. Create a virtual environment

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Configure

```
cp .env.example .env
```

Edit .env with your OpenRouter API key from https://openrouter.ai/

### 5. Run

```
uvicorn server:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs
LangServe Playground: http://127.0.0.1:8000/api/playground/

## API

POST /resume/analyze — full workflow (file, target_role, job_description)
POST /resume/extract — extract raw text from a file
GET  /health         — liveness check

## License

MIT
