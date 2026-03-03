# Financial Intelligence Engine (Enterprise RAG)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Llama-3](https://img.shields.io/badge/LLM-Llama--3.3--70B-purple)
![Status](https://img.shields.io/badge/Status-Production--Ready-red)

> **An Enterprise-Grade Agentic RAG System for SEC 10-K Financial Analysis.**
> *Eliminating AI Hallucinations through Dual-LLM Guardrails and Custom Rank Fusion.*

---

## Executive Summary & Business Case

Financial analysis requires absolute precision. Standard Generative AI models hallucinate numbers, lose context in long documents, and fail to synthesize comparative data when parsing dense documents like SEC 10-K filings. 

This engine solves the "hallucination problem" by implementing a strictly regulated **Agentic Retrieval-Augmented Generation (RAG)** pipeline. It allows financial analysts to cross-examine massive, unstructured SEC filings across multiple organizations simultaneously, providing mathematically grounded, fully cited comparative analysis with zero pre-trained knowledge bleed.

---

## Technical Architecture & Methodology

This system bypasses standard wrapper APIs in favor of a highly optimized, custom-engineered backend:

### 1. Parallel I/O Data Ingestion
Reading hundreds of dense PDFs is an I/O-bound bottleneck. The ingestion module utilizes Python's `concurrent.futures.ThreadPoolExecutor` to asynchronously parse and chunk financial filings, completely eliminating CPU idle time during document loading.

### 2. Custom Reciprocal Rank Fusion (RRF)
To capture both semantic meaning and exact financial terminology, the engine utilizes a Hybrid Retrieval approach:
* **Dense Vectors:** ChromaDB powered by `BAAI/bge-small-en-v1.5` for contextual understanding.
* **Sparse Keywords:** BM25 Index for exact-match vocabulary.
* *The Engine:* A custom mathematical RRF algorithm normalizes and fuses these disparate scoring scales, utilizing deterministic UUIDs to prevent dictionary overwrite collisions during indexing.

### 3. Agentic Guardrails (The "Merciless Auditor")
Standard RAG pipelines pass retrieved context directly to the LLM. This pipeline implements a two-pass **Generator/Critic** workflow:
* **Chain-of-Thought Generator:** Extracts raw facts, maps strategic overlaps, and explicitly flags missing data.
* **Compliance Auditor (Critic):** A strict secondary LLM prompt tasked purely with hallucination destruction. It strips any metric, year, or dollar amount not explicitly cited in the retrieved context.

### 4. Zero-Compute Warm Starts
Vectorizing 1,500+ document chunks per run wastes expensive compute tokens. The engine implements a disk-persistence layer using `pickle` (for BM25 serialization) and Chroma directory states, dropping system boot time from 5 minutes to under 3 seconds.

---

## System Telemetry & Evaluation

The system's accuracy is actively monitored using an automated **LLM-as-a-Judge** quantitative evaluation suite, forced into strict JSON outputs via LangChain's `PydanticOutputParser`. 

### Validation Metrics
* **Faithfulness Score (0.80+):** Validates that the final output is derived *exclusively* from the source documents with zero outside hallucinations.
* **Context Relevance (0.90+):** Confirms the custom RRF engine successfully retrieves data that directly answers the user's prompt.

### Visual Evidence
*The retrieved context distribution demonstrates the hybrid search dynamically allocating weight across multiple filings to answer complex comparative queries.*

![System Telemetry](assets/telemetry_dashboard.png)

---

## Repository Structure

```text
financial-intelligence-engine/
│
├── artifacts/                  # Auto-generated outputs
│   ├── eval_reports/           # JSON LLM-as-a-Judge grading
│   └── vector_db/              # ChromaDB & BM25 Pickles (Git-ignored)
│
├── assets/                     # README visual assets
│   └── telemetry_dashboard.png 
│
├── data/
│   └── raw_pdfs/               # SEC 10-K Filings (Git-ignored)
│
├── notebooks/
│   └── main_execution.ipynb    # Interactive system pipeline execution
│
├── src/                        # Modularized Source Code
│   ├── __init__.py
│   ├── config.py               # Global hyperparameters and directory paths
│   ├── data_ingestion.py       # Parallelized PDF loading and chunking
│   ├── evaluation.py           # LLM-as-a-Judge quantitative grading
│   ├── generation_agent.py     # Dual-LLM Generator & Critic guardrails
│   └── retrieval_engine.py     # Custom RRF Hybrid Search engine
│
├── .env                        # API Key configurations (Git-ignored)
├── .gitignore                  # Version control exclusions
├── README.md                   # Project Documentation
└── requirements.txt            # Package dependencies

Gemini said
Here is the complete, raw Markdown code for your FAANG-level README.md.

You can copy this entire block and paste it directly into your empty README.md file in VS Code.

Markdown
# Financial Intelligence Engine (Enterprise RAG)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Llama-3](https://img.shields.io/badge/LLM-Llama--3.3--70B-purple)
![Status](https://img.shields.io/badge/Status-Production--Ready-red)

> **An Enterprise-Grade Agentic RAG System for SEC 10-K Financial Analysis.**
> *Eliminating AI Hallucinations through Dual-LLM Guardrails and Custom Rank Fusion.*

---

## Executive Summary & Business Case

Financial analysis requires absolute precision. Standard Generative AI models hallucinate numbers, lose context in long documents, and fail to synthesize comparative data when parsing dense documents like SEC 10-K filings. 

This engine solves the "hallucination problem" by implementing a strictly regulated **Agentic Retrieval-Augmented Generation (RAG)** pipeline. It allows financial analysts to cross-examine massive, unstructured SEC filings across multiple organizations simultaneously, providing mathematically grounded, fully cited comparative analysis with zero pre-trained knowledge bleed.

---

## Technical Architecture & Methodology

This system bypasses standard wrapper APIs in favor of a highly optimized, custom-engineered backend:

### 1. Parallel I/O Data Ingestion
Reading hundreds of dense PDFs is an I/O-bound bottleneck. The ingestion module utilizes Python's `concurrent.futures.ThreadPoolExecutor` to asynchronously parse and chunk financial filings, completely eliminating CPU idle time during document loading.

### 2. Custom Reciprocal Rank Fusion (RRF)
To capture both semantic meaning and exact financial terminology, the engine utilizes a Hybrid Retrieval approach:
* **Dense Vectors:** ChromaDB powered by `BAAI/bge-small-en-v1.5` for contextual understanding.
* **Sparse Keywords:** BM25 Index for exact-match vocabulary.
* *The Engine:* A custom mathematical RRF algorithm normalizes and fuses these disparate scoring scales, utilizing deterministic UUIDs to prevent dictionary overwrite collisions during indexing.

### 3. Agentic Guardrails (The "Merciless Auditor")
Standard RAG pipelines pass retrieved context directly to the LLM. This pipeline implements a two-pass **Generator/Critic** workflow:
* **Chain-of-Thought Generator:** Extracts raw facts, maps strategic overlaps, and explicitly flags missing data.
* **Compliance Auditor (Critic):** A strict secondary LLM prompt tasked purely with hallucination destruction. It strips any metric, year, or dollar amount not explicitly cited in the retrieved context.

### 4. Zero-Compute Warm Starts
Vectorizing 1,500+ document chunks per run wastes expensive compute tokens. The engine implements a disk-persistence layer using `pickle` (for BM25 serialization) and Chroma directory states, dropping system boot time from 5 minutes to under 3 seconds.

---

## System Telemetry & Evaluation

The system's accuracy is actively monitored using an automated **LLM-as-a-Judge** quantitative evaluation suite, forced into strict JSON outputs via LangChain's `PydanticOutputParser`. 

### Validation Metrics
* **Faithfulness Score (0.80+):** Validates that the final output is derived *exclusively* from the source documents with zero outside hallucinations.
* **Context Relevance (0.90+):** Confirms the custom RRF engine successfully retrieves data that directly answers the user's prompt.

### Visual Evidence
*The retrieved context distribution demonstrates the hybrid search dynamically allocating weight across multiple filings to answer complex comparative queries.*

![System Telemetry](assets/telemetry_dashboard.png)

---

## Repository Structure

```text
financial-intelligence-engine/
│
├── artifacts/                  # Auto-generated outputs
│   ├── eval_reports/           # JSON LLM-as-a-Judge grading
│   └── vector_db/              # ChromaDB & BM25 Pickles (Git-ignored)
│
├── assets/                     # README visual assets
│   └── telemetry_dashboard.png 
│
├── data/
│   └── raw_pdfs/               # SEC 10-K Filings (Git-ignored)
│
├── notebooks/
│   └── main_execution.ipynb    # Interactive system pipeline execution
│
├── src/                        # Modularized Source Code
│   ├── __init__.py
│   ├── config.py               # Global hyperparameters and directory paths
│   ├── data_ingestion.py       # Parallelized PDF loading and chunking
│   ├── evaluation.py           # LLM-as-a-Judge quantitative grading
│   ├── generation_agent.py     # Dual-LLM Generator & Critic guardrails
│   └── retrieval_engine.py     # Custom RRF Hybrid Search engine
│
├── .env                        # API Key configurations (Git-ignored)
├── .gitignore                  # Version control exclusions
├── README.md                   # Project Documentation
└── requirements.txt            # Package dependencies


How to Run
This pipeline is optimized for remote execution (Google Colab) with seamless Drive integration and local VS Code connections.

Clone & Setup: Clone the repository and install requirements.txt.

Secure Credentials: Create a .env file in the root directory and add GROQ_API_KEY=your_key.

Smart Load: Open notebooks/main_execution.ipynb.

Execute: Run the cells sequentially. The system will automatically detect if a vector database exists on your disk. If found, it bypasses the PDF extraction phase for a Zero-Compute Warm Start.