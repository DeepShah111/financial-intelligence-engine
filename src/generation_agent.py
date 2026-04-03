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
"""

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