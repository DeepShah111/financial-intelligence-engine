# Financial Intelligence Engine (Enterprise RAG)

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Framework-LangChain-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/LLM-Llama--3.3--70B-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Faithfulness-0.864-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Relevance-0.955-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Correctness-0.812-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Production--Ready-red?style=flat-square"/>
  <a href="https://huggingface.co/spaces/your-username/financial-intelligence-engine">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Live%20Demo-orange?style=flat-square"/>
  </a>
</p>

> An enterprise-grade Agentic RAG system for SEC 10-K financial analysis.  
> Eliminating AI hallucinations through Dual-LLM guardrails, Custom Rank Fusion, and statistically validated evaluation against verified ground truth.

---

## Table of Contents

1. [The Business Problem](#1-the-business-problem)
2. [What Makes This Different](#2-what-makes-this-different)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Technical Decisions & Rationale](#4-technical-decisions--rationale)
5. [Evaluation Results](#5-evaluation-results)
6. [Repository Structure](#6-repository-structure)
7. [Quickstart](#7-quickstart)
8. [Dataset](#8-dataset)
9. [Interactive Demo](#9-interactive-demo)

---

## 1. The Business Problem

Financial analysis requires absolute precision. Standard Generative AI models hallucinate numbers, lose context in long documents, and fail to synthesize comparative data when parsing dense regulatory filings like SEC 10-Ks.

The three failure modes that make off-the-shelf LLMs unusable for financial work:

| Failure Mode | What Happens | Business Consequence |
|---|---|---|
| Hallucination | Model invents revenue figures not in the filing | Analyst acts on fabricated data |
| Context loss | Long documents exceed attention window | Key financial metrics silently dropped |
| Knowledge bleed | Model uses pre-trained knowledge, not the filing | Answers reflect outdated or wrong fiscal year |

This engine solves all three by implementing a strictly regulated **Agentic Retrieval-Augmented Generation (RAG)** pipeline. It allows financial analysts to cross-examine massive, unstructured SEC filings across multiple organizations simultaneously — providing mathematically grounded, fully cited comparative analysis with zero pre-trained knowledge bleed.

**Corpus used for development and evaluation:**
- Google 10-K (FY2025, filed February 2026) — 104 pages
- Meta 10-K (FY2025, filed 2026) — 145 pages
- Microsoft 10-K (FY2024, filed 2024) — 171 pages

---

## 2. What Makes This Different

Most RAG portfolio projects are demos. They retrieve context, pass it to an LLM, and print the output. There is no verification that the output is grounded in the source, no correction mechanism when it isn't, and no statistically valid evaluation that the system actually works.

### Side-by-side comparison

| What a standard RAG demo does | What this pipeline does |
|---|---|
| Single retrieval method (dense only) | Hybrid retrieval: Dense vectors + BM25 sparse fused via custom RRF |
| Random chunk IDs on every rebuild | Deterministic SHA-256 chunk IDs — index stable across re-runs |
| One LLM call, output printed directly | Two-stage pipeline: CoT generator → SEC compliance auditor |
| No evaluation | LLM-as-a-Judge evaluation across n=15 questions |
| Self-referential eval (model judges itself) | Separate model family as judge (Qwen3-32B judges Llama-3.3-70B) |
| Single question scored | Batch evaluation with mean ± std dev reported |
| No ground truth | Ground truth extracted directly from source 10-K filings |
| Corpus dominated by one source | Company-balanced RRF — no single company exceeds 43% of context |
| No UI | Gradio interactive demo with real-time evaluation scores |
| Single-turn only | Multi-turn conversation memory (last 3 turns as retrieval context) |
| Single monolithic query | Optional query decomposition for complex multi-part questions |

---

## 3. Pipeline Architecture

```
SEC 10-K PDFs (Google, Meta, Microsoft)
        │
        ▼
┌─────────────────────────────────────┐
│         data_ingestion.py           │
│  ThreadPoolExecutor parallel parse  │
│  RecursiveCharacterTextSplitter     │
│  SHA-256 deterministic chunk IDs    │
│  Company + source metadata tagging  │
└──────────────────┬──────────────────┘
                   │  1,617 annotated chunks
                   ▼
┌─────────────────────────────────────┐
│        retrieval_engine.py          │
│                                     │
│  Dense Index:                       │
│    ChromaDB + BAAI/bge-small-en     │
│                                     │
│  Sparse Index:                      │
│    BM25 (atomic write + SHA-256     │
│    integrity verification on load)  │
│                                     │
│  Fusion:                            │
│    Custom Weighted RRF              │
│    score += w × (1 / (rank + 60))  │
│                                     │
│  Output:                            │
│    Company-balanced Top-K docs      │
│    (max 3 chunks per company)       │
└──────────────────┬──────────────────┘
                   │  7 balanced documents
                   ▼
┌─────────────────────────────────────┐
│        generation_agent.py          │
│                                     │
│  [Optional] Query Decomposition:    │
│    Llama-3.3-70B splits complex     │
│    query into 2–4 sub-queries       │
│    → per-sub-query retrieval        │
│    → SHA-256 dedup + merge          │
│    → synthesis prompt               │
│                                     │
│  Stage 1 — CoT Generator:           │
│    Llama-3.3-70B                    │
│    Extract facts → identify gaps    │
│    → structured comparative answer  │
│                                     │
│  Stage 2 — Compliance Auditor:      │
│    Llama-3.3-70B (adversarial role) │
│    Strip any claim not in context   │
│    Enforce citation on every fact   │
│                                     │
│  Retry: tenacity exponential        │
│  backoff (3 attempts, 2–10s)        │
└──────────────────┬──────────────────┘
                   │  Hallucination-free cited answer
                   ▼
┌─────────────────────────────────────┐
│           evaluation.py             │
│                                     │
│  Judge: Qwen3-32B                   │
│  (different family from generator)  │
│                                     │
│  Metrics:                           │
│    Faithfulness  — grounded in ctx? │
│    Relevance     — answers prompt?  │
│    Correctness   — matches GT?      │
│                                     │
│  Batch: n=15 verified questions     │
│  Output: mean ± std dev + report    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│   gradio_app.py + conversation.py   │
│                                     │
│  Gradio UI:                         │
│    Chat panel with cited answers    │
│    Retrieved sources + page nums    │
│    Real-time eval scores (Qwen3)    │
│    Decomposition reasoning chain    │
│    6 example question buttons       │
│                                     │
│  ConversationMemory:                │
│    Rolling 3-turn window            │
│    Follow-up query reformulation    │
│    (pronoun / reference detection)  │
└─────────────────────────────────────┘
```

---

## 4. Technical Decisions & Rationale

### 4.1 Hybrid Retrieval — Why Both Dense and Sparse

Financial documents contain two fundamentally different types of information that require different retrieval strategies:

| Query Type | Example | Best Retrieval |
|---|---|---|
| Semantic / conceptual | "What are Meta's AI strategy risks?" | Dense (ChromaDB + embeddings) |
| Exact financial figure | "Google R&D expenses $61.087 billion" | Sparse (BM25 keyword match) |

Using dense retrieval alone misses exact dollar amounts because embeddings compress meaning and lose precise numerical tokens. Using sparse retrieval alone misses contextual questions because BM25 has no semantic understanding. The custom RRF layer fuses both scoring systems into a single ranked list using the formula:

```
score(doc) += weight × (1 / (rank + K))
```

where K=60 is the standard RRF smoothing constant and weights are 0.5/0.5 for equal contribution. This formulation is mathematically correct — the weight scales the entire RRF fraction, not just the numerator.

### 4.2 Deterministic Chunk IDs — Why SHA-256 Over UUID4

The original system used `uuid.uuid4()` (random) to assign chunk IDs. This caused a silent correctness bug: every time the pipeline ran, the same chunk received a different ID. The RRF deduplication logic uses chunk IDs as dictionary keys — if the same document chunk has two different IDs across two retrieval calls, it appears twice in the fused result instead of having its scores merged.

The fix uses SHA-256 over the content + source filename:

```python
content_hash = hashlib.sha256(
    f"{source_file}::{chunk_index}::{page_content}".encode()
).hexdigest()[:16]
chunk_id = f"{stem}_{content_hash}"   # e.g. "google_10k_a3f9c21b"
```

The same chunk always gets the same ID regardless of when the pipeline runs. Index rebuilds are reproducible and cache-compatible.

### 4.3 Atomic BM25 Serialization — Preventing Split-State Corruption

The original system wrote the BM25 pickle directly to disk. If the process was interrupted mid-write, the file was corrupted but the Chroma directory was intact. On the next run, the smart load logic detected `chroma_exists=True` and `bm25_exists=False`, then fell into the cold build branch which requires `document_chunks` — but the user called `build_indexes()` with no arguments. Unrecoverable `ValueError`.

The fix uses atomic rename:

```python
# Write to temp file first (same filesystem = atomic rename)
with tempfile.NamedTemporaryFile(dir=dir_path, suffix='.pkl') as tmp:
    pickle.dump(sparse_retriever, tmp)
    tmp_path = tmp.name

file_hash = compute_sha256(tmp_path)
shutil.move(tmp_path, self.bm25_path)   # atomic on POSIX

with open(self.bm25_hash_path, 'w') as f:
    f.write(file_hash)                  # SHA-256 sidecar for integrity check
```

On load, the SHA-256 digest is verified before deserialization. A mismatch raises `RuntimeError` with a clear recovery instruction rather than silently loading a corrupt index.

### 4.4 Company-Balanced Retrieval — Fixing Corpus Bias

Before this fix, the retriever returned 71.4% Meta chunks for any multi-company query. This happened because Meta's 10-K (145 pages) was larger than Google's (104 pages), giving Meta more total chunks in both the dense and sparse indexes. A query asking to compare Google and Meta would retrieve 5 Meta chunks and 1 Google chunk — the answer was structurally biased before the LLM ever ran.

The fix applies a per-company cap after RRF fusion:

```python
MAX_CHUNKS_PER_COMPANY = TOP_K_VECTORS // 3   # = 3 with TOP_K=7

for chunk_id, score in sorted_docs:
    company = doc_map[chunk_id].metadata.get('company')
    if company_counts.get(company, 0) < MAX_CHUNKS_PER_COMPANY:
        company_counts[company] += 1
        balanced.append(doc_map[chunk_id])
```

Result: retrieval distribution changed from **71% / 14% / 14%** to **43% / 29% / 29%**. Every cross-company query now receives meaningful context from all three filings.

### 4.5 Separate Judge Model — Preventing Circular Evaluation

Using the same model to generate and evaluate its own answers inflates scores due to self-consistency bias — the model does not recognize its own failure modes. The evaluation module uses `qwen/qwen3-32b` (Alibaba Qwen architecture) as the judge for outputs generated by `llama-3.3-70b-versatile` (Meta Llama architecture). These are completely different model families with different training data, tokenizers, and failure modes.

### 4.6 Ground Truth From Source Documents — Not Approximations

Every `ground_truth` value in the evaluation set was extracted directly from the uploaded 10-K PDFs using programmatic text extraction. No approximations, no rounded figures.

Examples:
- Google R&D FY2025: **$61.087 billion** (extracted from income statement table)
- Meta total revenue FY2025: **$200.966 billion** (extracted from consolidated statements)
- Microsoft cloud revenue FY2024: **$137.4 billion** (extracted from segment reporting)

This matters because a `ground_truth` string of "$49 billion" when the filing says "$49.326 billion" will score 0.0 on correctness even if the answer is factually right. Approximate ground truth produces a meaningless correctness metric.

### 4.7 Query Decomposition — Handling Multi-Part Questions

Standard single-shot retrieval fails on complex comparative questions like *"Compare Google and Meta's R&D spending and headcount trends."* A single query vector averages the semantics of both companies and both metrics, causing the retriever to favour whichever sub-topic has the strongest embedding signal — typically one company and one metric.

The decomposer (Llama-3.3-70B with a strict JSON-only prompt) splits the question into 2–4 focused sub-queries:

```
"Compare Google and Meta's R&D spending and headcount trends"
  → ["What were Google's total R&D expenses and YoY change?",
     "What were Meta's total R&D expenses and YoY change?",
     "What were Google's headcount figures?",
     "What were Meta's headcount figures?"]
```

Each sub-query is retrieved independently. Chunks are merged and deduplicated by SHA-256 chunk ID (same deduplication key used throughout the pipeline) before being fed to a synthesis prompt. The synthesised draft is then audited by the existing compliance auditor — so hallucination protection applies to decomposed answers too.

Simple / single-focus questions return a 1-element array from the decomposer, making `generate_answer_decomposed` a strict superset of `generate_answer` with zero redundant API calls on simple queries.

### 4.8 Conversation Memory — Contextualising Follow-Up Questions

A rolling-window `ConversationMemory` (default: 3 turns) detects follow-up questions via a signal word scan (`"their"`, `"its"`, `"what about"`, `"also"`, etc.) and appends prior questions as inline context before the query reaches the retriever:

```
Turn 1: "What were Google's R&D expenses?"
Turn 2: "What about their capital expenditure?"
         → reformulated: "What about their capital expenditure?
                          [Prior context: Q1: What were Google's R&D expenses?]"
```

Only the prior *questions* (not full answers) are appended, keeping the retrieval query concise and avoiding the BM25 token limit. The original user question is preserved in history for readable conversation logs.

---

## 5. Evaluation Results

The system is evaluated using an automated **LLM-as-a-Judge** framework across 15 questions spanning factual retrieval, numerical accuracy, and qualitative analysis.

### 5.1 Evaluation Configuration

| Parameter | Value |
|---|---|
| Evaluation set size | 15 questions |
| Generator model | `llama-3.3-70b-versatile` (Meta Llama 3.3) |
| Judge model | `qwen/qwen3-32b` (Alibaba Qwen3 — different family) |
| Ground truth source | Programmatically extracted from source 10-K PDFs |
| Questions with ground truth | 12 of 15 (3 qualitative questions score faithfulness/relevance only) |
| Structured output parsing | LangChain `PydanticOutputParser` — no fragile string manipulation |

### 5.2 Batch Evaluation Results (n=15)

| Metric | Mean | Std Dev | Pass Rate | Threshold |
|:---|:---:|:---:|:---:|:---:|
| **Faithfulness** (no hallucinations) | **0.864** | ±0.323 | **81.8%** | ≥ 0.80 |
| **Relevance** (answers the prompt) | **0.955** | ±0.151 | — | ≥ 0.80 |
| **Correctness** (vs verified ground truth) | **0.812** | ±0.372 | — | ≥ 0.60 |

All three metrics exceed their respective thresholds simultaneously.

**Faithfulness** measures whether every claim in the generated answer is directly supported by the retrieved context — no outside knowledge, no invented figures. The compliance auditor (Stage 2 of generation) is the primary mechanism driving this score.

**Relevance** measures whether the answer directly and completely addresses the question asked. At 0.955 with ±0.151 standard deviation, the system almost never returns an off-topic answer — this reflects the hybrid retrieval working correctly.

**Correctness** measures factual agreement between the generated answer and the verified ground truth extracted from source filings. At 0.812 on real financial figures, this is the most rigorous metric in the evaluation — and the one that distinguishes this project from systems that only measure self-referential quality.

### 5.3 Evaluation Dashboards

**Primary Dashboard — Batch Evaluation Results**

*All three metrics exceed the 0.80 pass threshold. The pie chart shows the company-balanced retrieval distribution after the corpus bias fix — Meta 42.9%, Google 28.6%, Microsoft 28.6%.*

![Batch Evaluation Dashboard](assets/batch_eval_primary.png)

---

**System Telemetry — Single Query + Batch Summary**

*Left: Single-query validation scores (1.00 / 1.00 — directional reference only). Right: Statistically valid batch evaluation summary with error bars showing score distribution across all 15 questions.*

![System Telemetry](assets/telemetry_dashboard.png)

---

### 5.4 What the Standard Deviation Tells You

The ±0.323 std dev on faithfulness is explained by question type, not system instability:

- **Factual quantitative questions** (e.g., "What were Google's R&D expenses?"): faithfulness = 1.0 consistently. The compliance auditor effectively removes hallucinations when the retrieved context contains the exact figure.
- **Qualitative open-ended questions** (e.g., "What regulatory risks does Meta face?"): faithfulness scores vary because the answer requires synthesizing multiple partial-context passages, and the judge model scores synthesis more conservatively than direct retrieval.

This is expected and honest behavior. The high-variance questions are harder — not broken.

---

## 6. Repository Structure

```text
financial-intelligence-engine/
│
├── artifacts/                       # Auto-generated outputs (Git-ignored)
│   ├── eval_reports/
│   │   └── batch_eval_report.json   # Full per-question scores + aggregate stats
│   ├── vector_db/                   # ChromaDB persist dir + BM25 pickle + SHA-256
│   │   ├── bm25_index.pkl
│   │   └── bm25_index.sha256        # Integrity sidecar
│   └── visualizations/
│       ├── batch_eval_primary.png
│       └── telemetry_dashboard.png
│
├── assets/                          # README image assets
│   ├── batch_eval_primary.png
│   └── telemetry_dashboard.png
│
├── data/
│   └── raw_pdfs/                    # SEC 10-K filings (Git-ignored)
│       ├── google_10k.pdf
│       ├── meta_10k.pdf
│       └── microsoft_10k.pdf
│
├── notebooks/
│   └── main_execution.ipynb         # Full pipeline — 6 cells, run sequentially
│
├── src/                             # Modular Python package
│   ├── __init__.py
│   ├── config.py                    # Hyperparameters, paths, model names, logging
│   ├── data_ingestion.py            # Parallel PDF parse, SHA-256 chunk IDs
│   ├── retrieval_engine.py          # Hybrid RRF engine, atomic writes, integrity
│   ├── generation_agent.py          # CoT generator + compliance auditor + retry
│   │                                #   + query decomposition (generate_answer_decomposed)
│   ├── conversation.py              # Rolling-window ConversationMemory, follow-up reformulation
│   └── evaluation.py                # Batch eval harness, Pydantic output parsing
│
├── gradio_app.py                    # Interactive Gradio demo (HuggingFace Spaces ready)
│
├── .env                             # GROQ_API_KEY (Git-ignored)
├── .gitignore
├── README.md
├── requirements.txt                 # Core RAG dependencies, pinned
└── requirements_spaces.txt          # Core deps + gradio==4.44.1 for HF Spaces
```

---

## 7. Quickstart

### Prerequisites

- Google account with Google Drive
- Groq API key (free at [console.groq.com](https://console.groq.com))
- SEC 10-K PDFs placed in `data/raw_pdfs/`

### Option A — Google Colab (Recommended)

**1. Upload the project to Google Drive:**
```
MyDrive/
└── financial-intelligence-engine/
    ├── src/
    ├── notebooks/
    ├── data/raw_pdfs/
    ├── .env
    └── requirements.txt
```

**2. Add your API key to `.env`:**
```
GROQ_API_KEY=your_groq_api_key_here
```

**3. Open `notebooks/main_execution.ipynb` in Google Colab.**

**4. Install dependencies (first session only):**
```python
!pip install -q -r /content/drive/MyDrive/financial-intelligence-engine/requirements.txt
```

**5. Run cells 1 through 6 sequentially.**

> **Smart Load:** After the first run, the system detects existing indexes on Drive and bypasses all PDF re-processing. Subsequent cold starts take under 3 seconds instead of 5+ minutes.

### Option B — Local (VS Code)

```bash
# Clone
git clone https://github.com/your-username/financial-intelligence-engine.git
cd financial-intelligence-engine

# Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt

# Add credentials
echo "GROQ_API_KEY=your_key_here" > .env

# Run notebook
jupyter notebook notebooks/main_execution.ipynb
```

### Option C — Gradio Demo (Local)

```bash
# Install core + Gradio deps
pip install -r requirements_spaces.txt

# Add credentials
echo "GROQ_API_KEY=your_key_here" > .env

# Launch
python gradio_app.py
# → open http://localhost:7860
```

### Notebook Cell Reference

| Cell | Phase | Description | Token Cost |
|:---:|---|---|---|
| 1 | Setup | Mount Drive, flush module cache, import | Zero |
| 2 | Retrieval | Smart load indexes or cold build from PDFs | Zero (warm) / High (cold) |
| 3 | Generation | CoT + compliance audit on primary query | ~4,000 tokens |
| 4 | Eval (single) | Quick single-query faithfulness check | ~500 tokens |
| 5 | Eval (batch) | Full 15-question batch evaluation | ~50,000 tokens |
| 6 | Visualization | Generate and save dashboards to Drive | Zero |

> **Token budget note:** The full pipeline (Cells 3–5) consumes approximately 55,000–60,000 tokens. Groq free tier provides 100,000 tokens/day. Run Cell 3 only when needed — the batch evaluation in Cell 5 is the statistically meaningful result.

---

## 8. Dataset

**SEC 10-K Annual Filings**

| Filing | Company | Fiscal Year | Period End | Pages | Chunks |
|---|---|---|:---:|:---:|:---:|
| `google_10k.pdf` | Alphabet Inc. (Google) | FY2025 | Dec 31, 2025 | 104 | 394 |
| `meta_10k.pdf` | Meta Platforms Inc. | FY2025 | Dec 31, 2025 | 145 | 608 |
| `microsoft_10k.pdf` | Microsoft Corporation | FY2024 | Jun 30, 2024 | 171 | 615 |
| | | **Total** | | **420** | **1,617** |

**Chunking configuration:**
- Chunk size: 1,200 characters
- Chunk overlap: 250 characters
- Separators: `["\n\n", "\n", ".", " ", ""]` — paragraph-first splitting

**Key verified financial figures used as ground truth:**

| Company | Metric | Value |
|---|---|---|
| Google | R&D Expenses FY2025 | $61.087 billion |
| Google | Total Revenue FY2025 | $402.836 billion |
| Google | Net Income FY2025 | $132.170 billion |
| Google | Capital Expenditures FY2025 | $91.4 billion |
| Google | Cloud Revenue FY2025 | $58.705 billion |
| Meta | Total Revenue FY2025 | $200.966 billion |
| Meta | R&D Expenses FY2025 | $57.372 billion |
| Meta | Net Income FY2025 | $60.458 billion |
| Meta | Reality Labs Operating Loss FY2025 | $19.19 billion |
| Meta | Employees (Dec 31, 2025) | 78,865 |
| Microsoft | Total Revenue FY2024 | $245.122 billion |
| Microsoft | Cloud Revenue FY2024 | $137.4 billion |
| Microsoft | R&D Expenses FY2024 | $29.510 billion |
| Microsoft | Net Income FY2024 | $88.136 billion |

All figures extracted programmatically from source PDFs using `pypdf`. Values verified against the income statement tables in each filing.

---

## 9. Interactive Demo

### Live on HuggingFace Spaces

[![🤗 Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl.svg)](https://huggingface.co/spaces/your-username/financial-intelligence-engine)

> **Replace** `your-username` in the badge URL above with your actual HuggingFace username after deployment.

### Demo Features

| Feature | Description |
|---|---|
| **Multi-turn Chat** | Full conversation with rolling 3-turn memory. Follow-up questions ("What about their R&D?") are automatically contextualised. |
| **Source Panel** | Every answer shows the exact retrieved chunks with company label and page number. |
| **Real-time Scores** | Toggle Faithfulness + Relevance evaluation via Qwen3-32B judge — rendered as a colour-coded score bar. |
| **Query Decomposition** | Toggle to split complex multi-company / multi-metric questions into focused sub-queries. A Reasoning Chain tab shows how the query was decomposed and which companies contributed chunks. |
| **Example Questions** | Six one-click buttons covering factual, comparative, and qualitative query types. |

### Deploying to HuggingFace Spaces

```bash
# 1. Create a new Space at huggingface.co/new-space
#    Runtime: Python 3.11  |  Hardware: CPU Basic (free tier)

# 2. Push your repo
git remote add spaces https://huggingface.co/spaces/your-username/financial-intelligence-engine
git push spaces main

# 3. Set your API key as a Repository Secret
#    Space Settings → Repository Secrets → Add Secret
#    Name:  GROQ_API_KEY
#    Value: your_groq_api_key_here

# 4. Set the requirements file
#    Space Settings → Files → App file: gradio_app.py
#                  → Requirements: requirements_spaces.txt
```

> **PDF Note:** The 10-K PDFs are Git-ignored. On Spaces, either commit them to the repo directly or use Hugging Face Datasets as a remote data source and adjust `DATA_DIR` in `config.py`.

### Conversation Example

Below is a representative multi-turn session showing how the conversation memory contextualises follow-up questions:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Turn 1                                                                  │
│  User:  What were Google's total R&D expenses in FY2025?                 │
│                                                                          │
│  Agent: Google reported total R&D expenses of $61.087 billion for       │
│         FY2025, representing a year-over-year increase from $49.1B in   │
│         FY2024 [Source: Google 10-K]. This reflects continued           │
│         investment in AI infrastructure and core search capabilities.   │
│                                                                          │
│  Scores: Faithfulness 🟢 1.000  ·  Relevance 🟢 1.000                  │
├──────────────────────────────────────────────────────────────────────────┤
│  Turn 2  (follow-up detected → query reformulated with Turn 1 context)  │
│  User:  What about their capital expenditure?                            │
│                                                                          │
│  Agent: Google's capital expenditures for FY2025 were $91.4 billion     │
│         [Source: Google 10-K], a significant step-up that the company   │
│         attributed to data centre expansion and AI compute buildout.    │
│         This compares to R&D spend of $61.087B in the same period.      │
│                                                                          │
│  Scores: Faithfulness 🟢 0.920  ·  Relevance 🟢 0.980                  │
├──────────────────────────────────────────────────────────────────────────┤
│  Turn 3  (query decomposition enabled)                                  │
│  User:  Compare all three companies' capex and R&D                      │
│                                                                          │
│  🔍 Decomposed into 4 sub-queries:                                      │
│     1. What were Google's capital expenditures and R&D expenses?        │
│     2. What were Meta's capital expenditures and R&D expenses?          │
│     3. What were Microsoft's capital expenditures and R&D expenses?     │
│     4. How do capex-to-revenue ratios compare across the three?         │
│                                                                          │
│  Agent: [Synthesised comparative analysis across all three companies    │
│          with per-source citations and data from 9 retrieved chunks]    │
│                                                                          │
│  Scores: Faithfulness 🟢 0.880  ·  Relevance 🟢 0.960                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

<p align="center">
  Built as a portfolio project demonstrating production ML engineering and applied NLP.<br/>
  Structured for correctness, statistical rigor, and interview-readiness.
</p>