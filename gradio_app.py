"""
Gradio Interactive Demo — Financial Intelligence Engine.

Deploy-ready for HuggingFace Spaces.

Features:
- Multi-turn chat with ConversationMemory (last 3 turns as retrieval context).
- Retrieved source chunks displayed with company label and page number.
- Final cited answer rendered in the chat panel.
- Real-time Faithfulness + Relevance scores via LLM-as-a-Judge (Qwen3-32B).
- Query Decomposition toggle: breaks complex questions into sub-queries.
- Reasoning chain panel: shows decomposed sub-queries and retrieval stats.
- Six example questions as one-click buttons.

Environment:
    Set GROQ_API_KEY as an environment variable or in a .env file.
    On HuggingFace Spaces, set it as a Repository Secret named GROQ_API_KEY.

Pipeline initialisation:
    The RAG pipeline (embedding model + ChromaDB + BM25) is initialised once
    on the first click of "⚙️ Initialize Pipeline". Warm start (existing
    artifacts/) skips re-embedding; cold start builds indexes from raw PDFs.
    Both paths are handled transparently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# ── Path bootstrap (supports both Spaces and local layouts) ───────────────────
# Spaces layout: repo root is the cwd; src/ is a package inside it.
# Local layout:  gradio_app.py lives at repo root alongside src/.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # no-op on Spaces (env vars come from Secrets); harmless locally

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


# ── Global Pipeline State ─────────────────────────────────────────────────────
# Initialised lazily on first "Initialize Pipeline" click to keep cold startup
# fast. Using module-level variables is safe because Gradio shares state across
# tabs within a single server process but not across separate worker processes.
_agent:       Optional[FinancialGenerationAgent] = None
_evaluator:   Optional[RAGEvaluator]             = None
_initialized: bool                               = False


# ── Pipeline Initialisation ───────────────────────────────────────────────────

def initialize_pipeline() -> str:
    """
    Build or warm-load the full RAG pipeline.

    Returns a status string displayed in the Pipeline Status textbox.
    Idempotent: calling it a second time returns immediately if already ready.
    """
    global _agent, _evaluator, _initialized

    if _initialized:
        return "✅ Pipeline already initialized and ready."

    api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "❌ GROQ_API_KEY not found.\n"
            "• Local: add GROQ_API_KEY=your_key to a .env file at project root.\n"
            "• HuggingFace Spaces: add it as a Repository Secret named GROQ_API_KEY."
        )

    try:
        setup_environment()
        engine = HybridRetrievalEngine()

        # Smart load: warm start if both indexes exist; cold build otherwise.
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

        return "✅ Pipeline initialized and ready. Ask your first question below."

    except FileNotFoundError as exc:
        return (
            f"❌ PDF files not found: {exc}\n"
            "Place your 10-K PDFs in data/raw_pdfs/ and try again."
        )
    except Exception as exc:
        logger.error("[Gradio] Initialization error: %s", exc)
        return f"❌ Initialization failed: {exc}"


# ── Display Formatters ────────────────────────────────────────────────────────

def _format_sources(docs: list[Document]) -> str:
    """
    Render retrieved source chunks as a readable Markdown string.

    Each chunk shows company label, page number, source filename, and up to
    600 characters of content so the panel stays scannable without scrolling.
    """
    if not docs:
        return "*No source chunks were retrieved for this query.*"

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        company  = doc.metadata.get("company", "Unknown")
        page     = doc.metadata.get("page", "N/A")
        src_file = doc.metadata.get("source_file", "")
        content  = doc.page_content[:600].strip()
        ellipsis = "…" if len(doc.page_content) > 600 else ""

        parts.append(
            f"### 📄 Source {i} — {company} 10-K &nbsp;·&nbsp; Page {page}\n"
            f"*`{src_file}`*\n\n"
            f"{content}{ellipsis}"
        )

    return "\n\n---\n\n".join(parts)


def _format_scores(scores: dict) -> str:
    """
    Render LLM-as-a-Judge evaluation scores as a colour-coded Markdown panel.

    Colour coding thresholds (matching evaluation.py pass criteria):
        🟢  ≥ 0.85  — strong
        🟡  ≥ 0.70  — acceptable
        🔴  < 0.70  — needs attention
    """
    if not scores:
        return "*Evaluation scores will appear here when 'Real-time Evaluation' is enabled.*"

    def _emoji(val: float) -> str:
        if val >= 0.85:
            return "🟢"
        if val >= 0.70:
            return "🟡"
        return "🔴"

    def _bar(val: float) -> str:
        filled = round(val * 10)
        return "█" * filled + "░" * (10 - filled)

    lines: list[str] = [
        "### 📊 Real-Time Evaluation Scores",
        "*Judge model: Qwen3-32B — different family from generator (anti-circular-bias)*\n",
    ]

    for metric, val in scores.items():
        if isinstance(val, float):
            lines.append(
                f"**{metric.capitalize()}** &nbsp; {_emoji(val)} &nbsp; "
                f"`{val:.3f}` &nbsp; `[{_bar(val)}]`"
            )
        else:
            lines.append(f"**{metric.capitalize()}**: `{val}`")

    lines.append(
        "\n---\n"
        "*Thresholds — 🟢 ≥ 0.85 · 🟡 ≥ 0.70 · 🔴 < 0.70*  \n"
        "*Benchmark: Faithfulness 0.864 · Relevance 0.955 · Correctness 0.812 (n=15)*"
    )
    return "\n\n".join(lines)


def _format_reasoning(sub_queries: list[str], docs: list[Document]) -> str:
    """
    Render the query decomposition reasoning chain as Markdown.

    Shows each sub-query that was retrieved independently, then a retrieval
    summary listing which companies contributed chunks to the merged context.
    """
    if not sub_queries:
        return "*Enable 'Query Decomposition' to see the reasoning chain here.*"

    companies: dict[str, int] = {}
    for doc in docs:
        company = doc.metadata.get("company", "Unknown")
        companies[company] = companies.get(company, 0) + 1

    company_breakdown: str = "  ·  ".join(
        f"**{co}** {cnt} chunk{'s' if cnt > 1 else ''}"
        for co, cnt in sorted(companies.items())
    )

    lines: list[str] = [
        "### 🔍 Query Decomposition Reasoning Chain",
        f"Original query split into **{len(sub_queries)} sub-queries** "
        f"— each retrieved independently, then merged and synthesised.\n",
    ]

    for i, sq in enumerate(sub_queries, 1):
        lines.append(f"**Sub-query {i}:** {sq}")

    lines.append(
        f"\n**Merged retrieval:** {len(docs)} unique chunks  \n"
        f"**By company:** {company_breakdown}  \n"
        f"\n*Chunks deduplicated by SHA-256 chunk ID → "
        f"fed to synthesis prompt → compliance auditor.*"
    )
    return "\n\n".join(lines)


# ── Main Chat Handler ─────────────────────────────────────────────────────────

def chat(
    user_message:     str,
    history:          list,
    memory:           ConversationMemory,
    use_decomposition: bool,
    run_evaluation:   bool,
) -> tuple[list, ConversationMemory, str, str, str]:
    """
    Process one user turn through the full RAG pipeline.

    Args:
        user_message:      Raw text from the input box.
        history:           Current Gradio chatbot history ([[user, bot], ...]).
        memory:            ConversationMemory state (passed via gr.State).
        use_decomposition: If True, run generate_answer_decomposed instead of
                           generate_answer.
        run_evaluation:    If True, call RAGEvaluator.evaluate after generation.

    Yields (via return — Gradio handles streaming at the component level):
        Updated history, updated memory state, sources_md, scores_md, reasoning_md.
    """
    if not user_message.strip():
        return history, memory, "", "", ""

    if not _initialized:
        warning = (
            "⚠️ Pipeline not initialized. "
            "Click **⚙️ Initialize Pipeline** at the top of the page first."
        )
        return (
            history + [[str(user_message), str(warning)]],
            memory,
            "",
            "",
            "",
        )

    # Reformulate the query using conversation history (handles follow-ups).
    retrieval_query: str = memory.reformulate_query(user_message)

    sources_md   = ""
    scores_md    = ""
    reasoning_md = ""

    try:
        if use_decomposition:
            logger.info("[Gradio] Running decomposed generation...")
            final_answer, docs, sub_queries = _agent.generate_answer_decomposed(
                retrieval_query
            )
            reasoning_md = _format_reasoning(sub_queries, docs)
        else:
            logger.info("[Gradio] Running standard generation...")
            final_answer, docs = _agent.generate_answer(retrieval_query)

        sources_md = _format_sources(docs)

        if run_evaluation and _evaluator is not None:
            logger.info("[Gradio] Running real-time evaluation...")
            scores    = _evaluator.evaluate(
                question=retrieval_query,
                answer=final_answer,
                context_docs=docs,
            )
            scores_md = _format_scores(scores)

        # Store completed turn in memory (original question, not reformulated).
        memory.add_turn(user_message, final_answer)

        updated_history = history + [[str(user_message), str(final_answer)]]
        return updated_history, memory, sources_md, scores_md, reasoning_md

    except Exception as exc:
        logger.error("[Gradio] Generation error: %s", exc)
        error_msg = (
            f"❌ An error occurred during generation: `{exc}`\n\n"
            "This may be a Groq rate-limit or transient network issue. "
            "The pipeline retries automatically (up to 3×). "
            "If the error persists, wait 30 s and try again."
        )
        return (
            history + [[str(user_message), str(error_msg)]],
            memory,
            "",
            "",
            "",
        )


def clear_conversation(
    memory: ConversationMemory,
) -> tuple[list, ConversationMemory, str, str, str]:
    """Reset the chatbot panel and conversation memory."""
    memory.clear()
    return [], memory, "", "", ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_CUSTOM_CSS = """
/* Tighten source / score panels */
.source-panel .prose { font-size: 0.84em; line-height: 1.55; }
.score-panel  .prose { font-size: 0.88em; }

/* Slightly de-emphasise the footer branding */
.app-footer { font-size: 0.78em; color: #888; text-align: center; margin-top: 1rem; }

/* Compact example buttons */
.example-btn { font-size: 0.80em !important; padding: 4px 8px !important; }
"""

with gr.Blocks(
    title="Financial Intelligence Engine",
) as demo:

    # ── Persistent State ──────────────────────────────────────────────────────
    # gr.State carries ConversationMemory across turns for a single browser session.
    memory_state = gr.State(ConversationMemory(max_turns=3))

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("""
# 📊 Financial Intelligence Engine
### Enterprise Agentic RAG — SEC 10-K Analysis · Google · Meta · Microsoft

Ask any question about the three 10-K filings. The engine retrieves context
via **Hybrid Dense + BM25** retrieval fused with **Custom RRF**, generates a
cited answer through a **Chain-of-Thought → Compliance Auditor** pipeline,
and optionally scores the response with an independent **LLM-as-a-Judge** in real time.
    """)

    # ── Pipeline Initialisation Row ───────────────────────────────────────────
    with gr.Row():
        init_btn   = gr.Button("⚙️ Initialize Pipeline", variant="primary",   scale=2)
        clear_btn  = gr.Button("🗑️ Clear Conversation",  variant="secondary", scale=1)

    init_status = gr.Textbox(
        label="Pipeline Status",
        value="Pipeline not yet initialized — click ⚙️ Initialize Pipeline to begin.",
        interactive=False,
        lines=2,
    )

    gr.Markdown("---")

    # ── Main Panel ────────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left: Chat ────────────────────────────────────────────────────────
        with gr.Column(scale=3, min_width=400):

            chatbot = gr.Chatbot(
                label="Conversation",
                height=480,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about the 10-K filings…",
                    label="Your Question",
                    lines=2,
                    scale=5,
                    show_label=False,
                )
                submit_btn = gr.Button("Send ▶", variant="primary", scale=1, min_width=80)

            with gr.Row():
                use_decomp = gr.Checkbox(
                    label="🔀 Query Decomposition  (recommended for multi-company / multi-metric questions)",
                    value=False,
                    info=(
                        "Decomposes the query into focused sub-queries, "
                        "retrieves each independently, then synthesises."
                    ),
                )
            with gr.Row():
                run_eval = gr.Checkbox(
                    label="📏 Real-Time Evaluation Scores  (adds ~5–10 s per query)",
                    value=True,
                    info=(
                        "Runs Qwen3-32B as an impartial judge after each answer "
                        "to score Faithfulness and Relevance."
                    ),
                )

            # ── Example Questions ─────────────────────────────────────────────
            gr.Markdown("**💡 Example questions — click to load:**")
            with gr.Row():
                for q in EXAMPLE_QUESTIONS[:3]:
                    gr.Button(q, size="sm", elem_classes=["example-btn"]).click(
                        fn=lambda x=q: x,
                        outputs=msg_input,
                    )
            with gr.Row():
                for q in EXAMPLE_QUESTIONS[3:]:
                    gr.Button(q, size="sm", elem_classes=["example-btn"]).click(
                        fn=lambda x=q: x,
                        outputs=msg_input,
                    )

        # ── Right: Sources + Scores + Reasoning ───────────────────────────────
        with gr.Column(scale=2, min_width=320):

            with gr.Tabs():

                with gr.Tab("📄 Retrieved Sources"):
                    sources_display = gr.Markdown(
                        "*Sources will appear here after your first query.*",
                        elem_classes=["source-panel"],
                    )

                with gr.Tab("📊 Evaluation Scores"):
                    scores_display = gr.Markdown(
                        "*Enable 'Real-Time Evaluation Scores' and submit a query.*",
                        elem_classes=["score-panel"],
                    )

                with gr.Tab("🔍 Reasoning Chain"):
                    reasoning_display = gr.Markdown(
                        "*Enable 'Query Decomposition' to see the sub-query reasoning chain.*",
                    )

                with gr.Tab("ℹ️ How It Works"):
                    gr.Markdown("""
**Retrieval**
1. Your query is passed to **ChromaDB** (dense vector search, BAAI/bge-small-en embeddings) and **BM25** (sparse keyword search) simultaneously.
2. Results are fused via **Custom Reciprocal Rank Fusion (RRF)**: `score += w × (1 / (rank + 60))`.
3. A **company-balance filter** ensures no single company exceeds 3 chunks, preventing corpus bias.

**Generation**
4. **Stage 1 — Chain-of-Thought (Llama-3.3-70B):** extracts raw facts, identifies gaps, writes a structured comparative analysis with citations.
5. **Stage 2 — Compliance Auditor (Llama-3.3-70B):** reviews the draft as an SEC auditor, removes any claim not grounded in the retrieved context.

**Query Decomposition** *(optional)*
6. Complex multi-part questions are decomposed into 2–4 focused sub-queries. Each is retrieved independently. Results are merged, deduplicated by SHA-256 chunk ID, and synthesised.

**Evaluation** *(optional, real-time)*
7. **LLM-as-a-Judge (Qwen3-32B):** independently scores Faithfulness (are all claims grounded in context?) and Relevance (does the answer address the question?).

**Conversation Memory**
8. The last 3 turns are stored. Follow-up questions ("What about their R&D?") are automatically enriched with prior context before retrieval.
                    """)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
---
<div class="app-footer">

**Architecture:** Hybrid Dense (ChromaDB · BAAI/bge-small-en) + BM25 → Custom RRF → Llama-3.3-70B CoT → Qwen3-32B Judge  
**Benchmark:** Faithfulness **0.864** · Relevance **0.955** · Correctness **0.812** (n=15 verified questions, ground truth from 10-K source)  
**Corpus:** Google 10-K FY2025 · Meta 10-K FY2025 · Microsoft 10-K FY2024 — 1,617 annotated chunks

</div>
        """,
    )

    # ── Event Wiring ──────────────────────────────────────────────────────────

    init_btn.click(
        fn=initialize_pipeline,
        outputs=init_status,
    )

    # Shared submit logic (button click or Enter key in textbox).
    _submit_inputs  = [msg_input, chatbot, memory_state, use_decomp, run_eval]
    _submit_outputs = [chatbot, memory_state, sources_display, scores_display, reasoning_display]

    submit_btn.click(
        fn=chat,
        inputs=_submit_inputs,
        outputs=_submit_outputs,
    ).then(
        fn=lambda: "",    # clear the input box after submission
        outputs=msg_input,
    )

    msg_input.submit(
        fn=chat,
        inputs=_submit_inputs,
        outputs=_submit_outputs,
    ).then(
        fn=lambda: "",
        outputs=msg_input,
    )

    clear_btn.click(
        fn=clear_conversation,
        inputs=[memory_state],
        outputs=[chatbot, memory_state, sources_display, scores_display, reasoning_display],
    )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.queue()       # enables concurrency and progress events
    demo.launch(
        server_name="0.0.0.0",   # required for HuggingFace Spaces / Docker
        server_port=7860,
        share=False,             # set True for a temporary public link locally
    )
