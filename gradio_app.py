"""
Gradio Interactive Demo — Financial Intelligence Engine.
Built for Gradio 4.44.1. Deploys on Render (Jina embeddings, Groq LLM).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from src.config import setup_environment, get_logger
from src.data_ingestion import load_and_chunk_pdfs
from src.retrieval_engine import HybridRetrievalEngine
from src.generation_agent import FinancialGenerationAgent
from src.evaluation import RAGEvaluator
from src.conversation import ConversationMemory
from langchain_core.documents import Document

logger = get_logger("gradio_app")


# ── Example Questions ─────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS: list[str] = [
    "What were Google's total R&D expenses in FY2025?",
    "Compare Meta and Microsoft's net income for their most recent fiscal year.",
    "What AI infrastructure investments did Google announce in their 10-K?",
    "How did Meta's Reality Labs perform — what was its operating loss?",
    "Compare the capital expenditures of all three companies.",
    "What regulatory and AI-related risks did Microsoft highlight in their filing?",
]


# ── Global State (module-level) ───────────────────────────────────────────────
# The agent and evaluator hold no per-user data, so one shared copy is safe.
# Conversation memory is NOT here any more: it lives in a per-session gr.State,
# so each visitor gets their own history (see chat() and the UI wiring below).
_agent:       Optional[FinancialGenerationAgent] = None
_evaluator:   Optional[RAGEvaluator]             = None
_initialized: bool                               = False


# ── Pipeline Initialisation ───────────────────────────────────────────────────
def initialize_pipeline() -> str:
    global _agent, _evaluator, _initialized

    if _initialized:
        return "Pipeline already initialized and ready."

    api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ("GROQ_API_KEY not found. Add it as an environment variable "
                "(Render) or in a .env file (local).")

    try:
        setup_environment()
        engine = HybridRetrievalEngine()
        try:
            retriever = engine.build_indexes()
            logger.info("[Gradio] Warm start: indexes loaded from disk.")
        except ValueError:
            logger.info("[Gradio] Cold start: building indexes from PDFs...")
            chunks    = load_and_chunk_pdfs()
            retriever = engine.build_indexes(document_chunks=chunks)

        _agent     = FinancialGenerationAgent(retriever=retriever, api_key=api_key)
        _evaluator = RAGEvaluator(api_key=api_key)
        _initialized = True
        return "Pipeline initialized and ready. Ask your first question below."

    except FileNotFoundError as exc:
        return f"PDF files not found: {exc}. Place 10-K PDFs in data/raw_pdfs/."
    except Exception as exc:
        logger.error("[Gradio] Initialization error: %s", exc)
        return f"Initialization failed: {exc}"


# ── Display Formatters ────────────────────────────────────────────────────────
def _format_sources(docs: list[Document]) -> str:
    if not docs:
        return "*No source chunks were retrieved for this query.*"
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        company  = doc.metadata.get("company", "Unknown")
        page     = doc.metadata.get("page", "N/A")
        # PyPDFLoader counts pages from 0, but a PDF viewer shows page 1 first.
        # Add 1 so the number shown here matches the real page in the filing.
        if isinstance(page, int):
            page = page + 1
        src_file = doc.metadata.get("source_file", "")
        content  = doc.page_content[:600].strip()
        ellipsis = "…" if len(doc.page_content) > 600 else ""
        parts.append(
            f"### Source {i} — {company} 10-K &nbsp;·&nbsp; Page {page}\n"
            f"*`{src_file}`*\n\n{content}{ellipsis}"
        )
    return "\n\n---\n\n".join(parts)


def _format_scores(scores: dict) -> str:
    if not scores:
        return "*Evaluation scores will appear here when enabled.*"

    def _emoji(val: float) -> str:
        return "🟢" if val >= 0.85 else "🟡" if val >= 0.70 else "🔴"

    def _bar(val: float) -> str:
        filled = round(val * 10)
        return "█" * filled + "░" * (10 - filled)

    lines: list[str] = [
        "### Real-Time Evaluation Scores",
        "*Judge model: a different family from the generator (anti-circular-bias)*\n",
    ]
    for metric, val in scores.items():
        if isinstance(val, float):
            lines.append(f"**{metric.capitalize()}** &nbsp; {_emoji(val)} &nbsp; "
                         f"`{val:.3f}` &nbsp; `[{_bar(val)}]`")
        else:
            lines.append(f"**{metric.capitalize()}**: `{val}`")
    lines.append("\n---\n*Thresholds — 🟢 ≥ 0.85 · 🟡 ≥ 0.70 · 🔴 < 0.70*")
    return "\n\n".join(lines)


def _format_reasoning(sub_queries: list[str], docs: list[Document]) -> str:
    if not sub_queries:
        return "*Enable 'Query Decomposition' to see the reasoning chain here.*"
    companies: dict[str, int] = {}
    for doc in docs:
        company = doc.metadata.get("company", "Unknown")
        companies[company] = companies.get(company, 0) + 1
    company_breakdown = "  ·  ".join(
        f"**{co}** {cnt} chunk{'s' if cnt > 1 else ''}"
        for co, cnt in sorted(companies.items())
    )
    lines: list[str] = [
        "### Query Decomposition Reasoning Chain",
        f"Original query split into **{len(sub_queries)} sub-queries**, "
        f"each retrieved independently, then merged and synthesised.\n",
    ]
    for i, sq in enumerate(sub_queries, 1):
        lines.append(f"**Sub-query {i}:** {sq}")
    lines.append(f"\n**Merged retrieval:** {len(docs)} unique chunks  \n"
                 f"**By company:** {company_breakdown}")
    return "\n\n".join(lines)


# ── Main Chat Handler (tuple history format for Gradio 4.x) ───────────────────
def chat(user_message, history, use_decomposition, run_evaluation, memory):
    history = history or []
    # Per-session memory: gr.State starts as None for a new visitor.
    if memory is None:
        memory = ConversationMemory(max_turns=3)

    if not user_message.strip():
        return history, "", "", "", memory

    if not _initialized:
        warning = "Pipeline not initialized. Click 'Initialize Pipeline' at the top first."
        return history + [[user_message, warning]], "", "", "", memory

    retrieval_query = memory.reformulate_query(user_message)
    sources_md = scores_md = reasoning_md = ""

    try:
        if use_decomposition:
            final_answer, docs, sub_queries = _agent.generate_answer_decomposed(retrieval_query)
            reasoning_md = _format_reasoning(sub_queries, docs)
        else:
            final_answer, docs = _agent.generate_answer(retrieval_query)

        sources_md = _format_sources(docs)

        if run_evaluation and _evaluator is not None:
            scores = _evaluator.evaluate(question=retrieval_query,
                                         answer=final_answer, context_docs=docs)
            scores_md = _format_scores(scores)

        memory.add_turn(user_message, final_answer)
        return history + [[user_message, final_answer]], sources_md, scores_md, reasoning_md, memory

    except Exception as exc:
        logger.error("[Gradio] Generation error: %s", exc)
        error_msg = (f"An error occurred during generation: `{exc}`. "
                     "This may be a rate-limit or network issue; the pipeline retries "
                     "automatically. If it persists, wait 30s and try again.")
        return history + [[user_message, error_msg]], "", "", "", memory


def clear_conversation(memory):
    if memory is not None:
        memory.clear()
    return [], "", "", "", memory


# ── Gradio UI ─────────────────────────────────────────────────────────────────
_CUSTOM_CSS = """
.source-panel .prose { font-size: 0.84em; line-height: 1.55; }
.score-panel  .prose { font-size: 0.88em; }
.app-footer { font-size: 0.78em; color: #888; text-align: center; margin-top: 1rem; }
.example-btn { font-size: 0.80em !important; padding: 4px 8px !important; }
"""

with gr.Blocks(title="Financial Intelligence Engine", css=_CUSTOM_CSS) as demo:

    gr.Markdown("""
# Financial Intelligence Engine
### Enterprise Agentic RAG — SEC 10-K Analysis · Google · Meta · Microsoft

Ask any question about the three 10-K filings. The engine retrieves context via
**Hybrid Dense + BM25** retrieval fused with **Custom RRF**, generates a cited
answer through a **Chain-of-Thought → Compliance Auditor** pipeline, and optionally
scores the response with an independent **LLM-as-a-Judge** in real time.
    """)

    with gr.Row():
        init_btn  = gr.Button("Initialize Pipeline", variant="primary",   scale=2)
        clear_btn = gr.Button("Clear Conversation",  variant="secondary", scale=1)

    init_status = gr.Textbox(
        label="Pipeline Status",
        value="Pipeline not yet initialized — click 'Initialize Pipeline' to begin.",
        interactive=False, lines=2,
    )

    gr.Markdown("---")

    # Per-visitor conversation memory (None until the visitor's first message).
    memory_state = gr.State(None)

    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=400):
            chatbot = gr.Chatbot(label="Conversation", height=480)

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about the 10-K filings…",
                    label="Your Question", lines=2, scale=5, show_label=False,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

            with gr.Row():
                use_decomp = gr.Checkbox(
                    label="Query Decomposition (recommended for multi-company / multi-metric questions)",
                    value=False,
                )
            with gr.Row():
                run_eval = gr.Checkbox(
                    label="Real-Time Evaluation Scores (adds a few seconds per query)",
                    value=True,
                )

            gr.Markdown("**Example questions — click to load:**")
            with gr.Row():
                for q in EXAMPLE_QUESTIONS[:3]:
                    gr.Button(q, size="sm", elem_classes=["example-btn"]).click(
                        fn=lambda x=q: x, outputs=msg_input,
                    )
            with gr.Row():
                for q in EXAMPLE_QUESTIONS[3:]:
                    gr.Button(q, size="sm", elem_classes=["example-btn"]).click(
                        fn=lambda x=q: x, outputs=msg_input,
                    )

        with gr.Column(scale=2, min_width=320):
            with gr.Tabs():
                with gr.Tab("Retrieved Sources"):
                    sources_display = gr.Markdown(
                        "*Sources will appear here after your first query.*",
                        elem_classes=["source-panel"])
                with gr.Tab("Evaluation Scores"):
                    scores_display = gr.Markdown(
                        "*Enable 'Real-Time Evaluation Scores' and submit a query.*",
                        elem_classes=["score-panel"])
                with gr.Tab("Reasoning Chain"):
                    reasoning_display = gr.Markdown(
                        "*Enable 'Query Decomposition' to see the sub-query reasoning chain.*")
                with gr.Tab("How It Works"):
                    gr.Markdown("""
**Retrieval** — your query goes to ChromaDB (dense) and BM25 (sparse) simultaneously,
fused via Custom Reciprocal Rank Fusion. A company-balance filter caps any single
company at 3 chunks to prevent corpus bias.

**Generation** — Stage 1 (Chain-of-Thought) extracts facts and writes a cited analysis;
Stage 2 (Compliance Auditor) removes any claim not grounded in the retrieved context.

**Query Decomposition** *(optional)* — complex questions are split into focused
sub-queries, each retrieved independently, then merged (deduplicated by SHA-256 chunk id)
and synthesised.

**Evaluation** *(optional)* — an independent LLM-as-a-Judge scores Faithfulness and
Relevance in real time.

**Conversation Memory** — the last 3 turns are stored; follow-up questions are
automatically enriched with prior context before retrieval.
                    """)

    gr.Markdown("""
---
<div class="app-footer">
Hybrid Dense (ChromaDB) + BM25 → Custom RRF → Chain-of-Thought → independent Judge.
Corpus: Google, Meta, Microsoft 10-K filings — 1,617 annotated chunks.
<br>
Data note: Microsoft's filing covers the fiscal year ended June 30, 2024 (one year earlier
than the others), so comparisons across companies may span different periods.
</div>
    """)

    # ── Event Wiring ──────────────────────────────────────────────────────────
    init_btn.click(fn=initialize_pipeline, outputs=init_status)

    _inputs  = [msg_input, chatbot, use_decomp, run_eval, memory_state]
    _outputs = [chatbot, sources_display, scores_display, reasoning_display, memory_state]

    submit_btn.click(fn=chat, inputs=_inputs, outputs=_outputs).then(
        fn=lambda: "", outputs=msg_input)
    msg_input.submit(fn=chat, inputs=_inputs, outputs=_outputs).then(
        fn=lambda: "", outputs=msg_input)
    clear_btn.click(fn=clear_conversation,
                    inputs=[memory_state],
                    outputs=[chatbot, sources_display, scores_display, reasoning_display, memory_state])


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))