"""
Generation Agent with Chain-of-Thought (CoT) Reasoning and Self-Correction.

UPGRADES vs previous version:
- Retry logic via tenacity: all LLM calls automatically retry up to
  MAX_API_RETRIES times with exponential backoff. Prevents silent failures
  under Groq rate limits or transient network errors.
- Explicit request_timeout on ChatGroq to surface hung calls instead of
  waiting indefinitely.
- All prompts, logic, and dual-LLM (generator + critic) architecture
  are fully preserved from the original.
- Full type annotations on all public methods.

NEW — Query Decomposition (generate_answer_decomposed):
- Decomposes complex multi-part questions into 2–4 focused sub-queries.
- Retrieves context independently for each sub-query.
- Deduplicates and merges all retrieved chunks by chunk_id.
- Synthesises sub-results into a single coherent cited answer via a
  dedicated synthesis prompt, then audits with the existing compliance
  auditor for hallucination removal.
- Returns (final_answer, all_source_docs, sub_queries) so callers can
  surface the decomposition reasoning chain in a UI.
"""

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import (
    logger,
    GENERATOR_MODEL,
    MAX_API_RETRIES,
    API_RETRY_MIN_WAIT,
    API_RETRY_MAX_WAIT,
)


# ── Retry Decorator ───────────────────────────────────────────────────────────
# Applied to every LLM call. Retries on any Exception (covers rate limits,
# timeouts, and transient network errors from Groq).
# wait_exponential: 2s → 4s → 8s → cap 10s between attempts.
_llm_retry = retry(
    stop=stop_after_attempt(MAX_API_RETRIES),
    wait=wait_exponential(
        multiplier=1,
        min=API_RETRY_MIN_WAIT,
        max=API_RETRY_MAX_WAIT,
    ),
    reraise=True,  # if all retries fail, re-raise the original exception
)


# ── Query Decomposition Prompt ────────────────────────────────────────────────
# Asks the LLM to split a complex question into 2–4 atomic sub-queries, each
# independently answerable via a single targeted retrieval pass.
# Simple / single-focus questions are returned unchanged as a 1-element array.
_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial research query analyst specialising in SEC 10-K filings.

Your task: analyse the user's question and decide whether it requires decomposition.

Rules:
1. If the question asks about multiple companies, multiple metrics, or multiple time periods, decompose it into 2–4 focused sub-queries — one per distinct retrieval need.
2. If the question is already atomic and focused (single fact, single company), return it unchanged as a 1-element array.
3. Every sub-query must be fully self-contained (include the company name, metric name, and fiscal year if inferable from the original).
4. You MUST respond with ONLY a valid JSON array of strings. No explanation, no preamble, no markdown code fences.

Example — complex:
Input:  "Compare Google and Meta's R&D spending and headcount trends"
Output: ["What were Google's total R&D expenses and year-over-year change?", "What were Meta's total R&D expenses and year-over-year change?", "What were Google's total headcount and hiring trends?", "What were Meta's total headcount and hiring trends?"]

Example — simple:
Input:  "What was Google's net income in FY2025?"
Output: ["What was Google's net income in FY2025?"]"""),
    ("human", "{question}"),
])


# ── Synthesis Prompt ──────────────────────────────────────────────────────────
# Takes the merged context from all sub-query retrieval passes and writes a
# single coherent comparative answer covering every decomposed sub-aspect.
_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Lead Financial Data Scientist. Multiple sub-queries were retrieved independently to answer the user's original question. Synthesise all findings into a single, coherent, professional comparative analysis.

Rules:
- Use structured bullet points with clear per-company or per-topic section headers.
- Cite the source immediately after every claim (e.g., [Source: Meta 10-K]).
- If a sub-query returned no relevant data, explicitly state the filing does not provide that information.
- Do NOT invent or infer any numbers absent from the context.

Sub-queries that were researched:
{sub_queries_formatted}

Combined Retrieved Context (all sub-queries merged and deduplicated):
{context}"""),
    ("human", "Original question: {question}\n\nSynthesize a complete, cited answer covering every sub-aspect above."),
])


class FinancialGenerationAgent:
    """
    Two-stage LLM pipeline: Chain-of-Thought generator → Compliance auditor.

    Stage 1 (Generator): Llama-3.3-70B reasons step-by-step using retrieved
                         context, then writes a structured comparative analysis.
    Stage 2 (Auditor):   The same model reviews the draft as an SEC Compliance
                         Auditor — removing any claim not grounded in the context.

    Both stages use the same GENERATOR_MODEL intentionally: the auditor prompt
    role-plays a distinct persona (compliance reviewer vs. data scientist),
    creating adversarial tension within a single model family. The evaluator
    in evaluation.py uses a DIFFERENT model family to avoid circular bias.
    """

    def __init__(self, retriever, api_key: str) -> None:
        self.retriever = retriever
        self.llm = ChatGroq(
            model=GENERATOR_MODEL,
            temperature=0,
            api_key=api_key,
            request_timeout=60,   # surface hung calls; don't wait indefinitely
        )

        # ── Generator Prompt (CoT Analysis) ──────────────────────────────────
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Lead Financial Data Scientist. Your objective is to answer the user's query using ONLY the provided SEC 10-K context.

You MUST structure your response strictly in two parts:

<thought_process>
1. Extract the raw facts from the context for each company.
2. Identify the strategic overlaps and stark differences.
3. Note any data requested by the prompt that is explicitly missing from the context.
</thought_process>

<final_answer>
Synthesize your findings into a professional, comparative analysis. 
- Use structured bullet points.
- You MUST cite the source immediately after every claim (e.g., [Source: Meta 10-K]).
- If data is missing (e.g., specific 2025 budgets), explicitly state that the filings do not provide forward-looking numbers for that year.
</final_answer>

Context:
{context}"""),
            ("human", "{question}"),
        ])

        # ── Auditor Prompt (Compliance Review) ───────────────────────────────
        self.critic_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an SEC Compliance Auditor. Review the <final_answer> section of the Draft Answer against the Source Context.
If the draft contains ANY numbers, metrics, or claims not explicitly present in the context, rewrite it to remove them.
Ensure every claim has a citation.

Original User Question:
{question}

Source Context:
{context}

Draft Answer:
{draft}"""),
            ("human", "Audit this draft. Output ONLY the finalized, hallucination-free <final_answer> text. Do not include the <thought_process>."),
        ])

    # ── Internal: Retryable LLM Calls ────────────────────────────────────────
    @_llm_retry
    def _invoke_generator(self, context: str, question: str) -> str:
        """
        Stage 1: Generate a CoT draft answer. Retried on failure.

        Args:
            context:  Concatenated retrieved document chunks.
            question: The user's original query.

        Returns:
            Raw LLM response string including <thought_process> and <final_answer>.
        """
        chain = self.qa_prompt | self.llm
        return chain.invoke({"context": context, "question": question}).content

    @_llm_retry
    def _invoke_auditor(self, context: str, question: str, draft: str) -> str:
        """
        Stage 2: Audit the draft for hallucinations. Retried on failure.

        Args:
            context:  Same retrieved context passed to the generator.
            question: The user's original query.
            draft:    The raw generator output to be audited.

        Returns:
            Final hallucination-free <final_answer> text.
        """
        chain = self.critic_prompt | self.llm
        return chain.invoke({
            "context":  context,
            "draft":    draft,
            "question": question,
        }).content

    # ── Public: Full Answer Generation ───────────────────────────────────────
    def generate_answer(self, query: str) -> tuple[str, list[Document]]:
        """
        Execute the full two-stage RAG generation pipeline.

        Args:
            query: Natural language question from the user.

        Returns:
            Tuple of (final_answer: str, source_docs: list[Document]).
            final_answer is the audited, hallucination-checked response.
            source_docs are the exact chunks used to construct the answer.

        Raises:
            Exception: Propagates after MAX_API_RETRIES exhausted on LLM calls.
        """
        logger.info("Retrieving documents for query: '%s'", query)
        docs: list[Document] = self.retriever.invoke(query)

        if not docs:
            logger.warning("Retriever returned zero documents for query: '%s'", query)

        # Build context string with explicit company attribution per chunk.
        context: str = "\n\n".join([
            f"[Source: {d.metadata.get('company', 'Unknown')} 10-K]: {d.page_content}"
            for d in docs
        ])

        logger.info("Step 1: Executing Chain-of-Thought Analysis (%s)...", GENERATOR_MODEL)
        draft_response: str = self._invoke_generator(context, query)

        logger.info("Step 2: Running Strict Compliance Audit (%s)...", GENERATOR_MODEL)
        final_response: str = self._invoke_auditor(context, query, draft_response)

        return final_response, docs

    # ── Internal: Query Decomposition Helpers ─────────────────────────────────

    @_llm_retry
    def _invoke_decomposer(self, question: str) -> list[str]:
        """
        Decompose a complex question into focused sub-queries via LLM.

        The LLM is instructed to return ONLY a JSON array of strings.
        Falls back to the original question as a single-element list on any
        parse failure, so the caller always receives a valid list[str].

        Args:
            question: The original user question.

        Returns:
            List of 1–4 sub-query strings.
        """
        chain = _DECOMPOSE_PROMPT | self.llm
        raw: str = chain.invoke({"question": question}).content.strip()

        # Strip accidental markdown code fences if the model adds them.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            sub_queries: list[str] = json.loads(raw)
            if not isinstance(sub_queries, list) or not all(
                isinstance(q, str) for q in sub_queries
            ):
                raise ValueError("Parsed value is not a list of strings.")
            logger.info(
                "Query decomposed into %d sub-queries: %s",
                len(sub_queries), sub_queries,
            )
            return sub_queries
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Decomposer returned unparseable output (%s). "
                "Falling back to original question as single sub-query.", exc
            )
            return [question]

    @_llm_retry
    def _invoke_synthesizer(
        self, sub_queries: list[str], context: str, question: str
    ) -> str:
        """
        Synthesise answers from multiple sub-query retrievals into one response.

        Args:
            sub_queries: The list of decomposed sub-queries (for prompt context).
            context:     Merged, deduplicated context from all sub-query retrievals.
            question:    The original user question for the synthesis prompt.

        Returns:
            Raw synthesised answer string (before compliance audit).
        """
        sub_queries_formatted: str = "\n".join(
            f"  {i}. {q}" for i, q in enumerate(sub_queries, 1)
        )
        chain = _SYNTHESIS_PROMPT | self.llm
        return chain.invoke({
            "sub_queries_formatted": sub_queries_formatted,
            "context":               context,
            "question":              question,
        }).content

    # ── Public: Decomposed Answer Generation ─────────────────────────────────
    def generate_answer_decomposed(
        self, query: str
    ) -> tuple[str, list[Document], list[str]]:
        """
        Execute the full query-decomposition RAG pipeline.

        Pipeline:
          1. Decompose the query into 1–4 focused sub-queries (LLM).
          2. Retrieve documents independently for each sub-query.
          3. Merge and deduplicate all retrieved chunks by chunk_id.
          4. Synthesise the merged context into a single comparative answer (LLM).
          5. Audit the synthesised answer for hallucinations (existing auditor).

        For simple questions the decomposer returns a 1-element list, making
        this method functionally equivalent to generate_answer — no wasted calls.

        Args:
            query: Natural language question from the user.

        Returns:
            Tuple of:
              - final_answer (str):        Audited, hallucination-free response.
              - all_source_docs (list):    Deduplicated chunks used across all sub-queries.
              - sub_queries (list[str]):   The decomposed sub-queries for UI display.

        Raises:
            Exception: Propagates after MAX_API_RETRIES exhausted on any LLM call.
        """
        logger.info(
            "[Decomposition] Decomposing query: '%s'", query
        )
        sub_queries: list[str] = self._invoke_decomposer(query)

        # Retrieve context for each sub-query independently and merge.
        doc_map: dict[str, Document] = {}   # chunk_id → Document (deduplication)

        for i, sub_query in enumerate(sub_queries, 1):
            logger.info(
                "[Decomposition] Sub-query %d/%d retrieval: '%s'",
                i, len(sub_queries), sub_query,
            )
            docs: list[Document] = self.retriever.invoke(sub_query)
            for doc in docs:
                chunk_id: str = doc.metadata.get(
                    "chunk_id",
                    # Fallback hash if chunk_id absent (should not occur in production).
                    f"fallback_{i}_{hash(doc.page_content)}",
                )
                doc_map[chunk_id] = doc   # later sub-queries overwrite on collision; content is identical

        all_docs: list[Document] = list(doc_map.values())

        if not all_docs:
            logger.warning(
                "[Decomposition] All sub-query retrievals returned zero documents."
            )

        # Build merged context string with explicit company attribution.
        merged_context: str = "\n\n".join([
            f"[Source: {d.metadata.get('company', 'Unknown')} 10-K]: {d.page_content}"
            for d in all_docs
        ])

        logger.info(
            "[Decomposition] Synthesis over %d deduplicated chunks (%d sub-queries).",
            len(all_docs), len(sub_queries),
        )
        synthesised_draft: str = self._invoke_synthesizer(
            sub_queries, merged_context, query
        )

        logger.info(
            "[Decomposition] Running compliance audit on synthesised answer..."
        )
        final_response: str = self._invoke_auditor(
            merged_context, query, synthesised_draft
        )

        return final_response, all_docs, sub_queries