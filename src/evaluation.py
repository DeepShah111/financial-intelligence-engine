"""
Enterprise LLM-as-a-Judge Evaluation Module.

UPGRADES vs previous version:
- Uses EVALUATOR_MODEL (Mixtral-8x7B) — deliberately DIFFERENT from the
  GENERATOR_MODEL (Llama-3.3-70B). Using the same model to judge its own
  outputs inflates scores due to self-consistency bias. This is the single
  most important evaluation fix.
- Batch evaluation harness (run_batch_evaluation) — scores across N≥10
  questions and reports mean ± std dev. Single-question evaluation is
  statistically meaningless and is an immediate interview red flag.
- Ground truth support — optional reference answer can be provided for
  each question. When present, enables Answer Correctness scoring
  (LLM compares generated answer vs. ground truth).
- Retry logic via tenacity on all LLM judge calls.
- Full type annotations throughout.
- Pydantic structured output parsing preserved from original.
"""

import json
import statistics
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    logger,
    EVAL_REPORTS_DIR,
    EVALUATOR_MODEL,
    MAX_API_RETRIES,
    API_RETRY_MIN_WAIT,
    API_RETRY_MAX_WAIT,
)


# ── Retry Decorator ───────────────────────────────────────────────────────────
_eval_retry = retry(
    stop=stop_after_attempt(MAX_API_RETRIES),
    wait=wait_exponential(
        multiplier=1,
        min=API_RETRY_MIN_WAIT,
        max=API_RETRY_MAX_WAIT,
    ),
    reraise=True,
)


# ── Pydantic Output Schemas ───────────────────────────────────────────────────
class EvaluationScores(BaseModel):
    """Structured output for RAG quality metrics without ground truth."""
    faithfulness: float = Field(
        description=(
            "Score from 0.0 to 1.0. Measures whether the Answer is derived "
            "ONLY from the provided Context (1.0 = no outside knowledge used; "
            "0.0 = answer is hallucinated)."
        )
    )
    relevance: float = Field(
        description=(
            "Score from 0.0 to 1.0. Measures whether the Answer directly and "
            "completely addresses the Question (1.0 = perfectly answers the prompt; "
            "0.0 = off-topic or evasive)."
        )
    )


class EvaluationScoresWithGroundTruth(EvaluationScores):
    """Extended structured output when a ground truth reference answer is available."""
    correctness: float = Field(
        description=(
            "Score from 0.0 to 1.0. Measures factual agreement between the "
            "Answer and the Ground Truth reference. Penalizes wrong numbers, "
            "missing key facts, or contradictory claims (1.0 = fully correct; "
            "0.0 = factually wrong or contradicts the reference)."
        )
    )


# ── Evaluator Class ───────────────────────────────────────────────────────────
class RAGEvaluator:
    """
    LLM-as-a-Judge evaluator for the Financial Intelligence RAG system.

    Deliberately uses EVALUATOR_MODEL (Mixtral-8x7B) — a different model
    family than the GENERATOR_MODEL (Llama-3.3-70B) — to prevent circular
    self-evaluation bias.

    Supports:
        - Single question scoring (evaluate)
        - Batch scoring over a test set (run_batch_evaluation)
        - Optional ground truth comparison (Answer Correctness metric)
    """

    def __init__(self, api_key: str) -> None:
        self.llm = ChatGroq(
            model=EVALUATOR_MODEL,
            temperature=0,
            api_key=api_key,
            request_timeout=60,
        )
        self.parser_base = PydanticOutputParser(pydantic_object=EvaluationScores)
        self.parser_gt   = PydanticOutputParser(
            pydantic_object=EvaluationScoresWithGroundTruth
        )

        # ── Evaluation Prompt (no ground truth) ──────────────────────────────
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an impartial AI quality auditor evaluating a Retrieval-Augmented Generation (RAG) system.
Analyze the provided Question, Context, and Answer. Score the following metrics strictly and objectively.

1. 'faithfulness' (0.0–1.0): Is EVERY claim in the Answer directly supported by the Context?
   - 1.0: Every fact traces back to a specific passage in the Context.
   - 0.5: Most claims are grounded; one or two minor unsupported details.
   - 0.0: Claims are made from outside knowledge or invented.

2. 'relevance' (0.0–1.0): Does the Answer completely and directly address the Question?
   - 1.0: The question is fully answered; no important aspect is omitted.
   - 0.5: Partially answers; misses one significant part of the question.
   - 0.0: The answer is off-topic or refuses to engage with the question.

{format_instructions}"""),
            ("human", "Question: {question}\n\nContext: {context}\n\nAnswer: {answer}"),
        ])

        # ── Evaluation Prompt (with ground truth) ────────────────────────────
        self.eval_prompt_gt = ChatPromptTemplate.from_messages([
            ("system", """You are an impartial AI quality auditor evaluating a Retrieval-Augmented Generation (RAG) system.
Analyze the provided Question, Context, Answer, and Ground Truth reference. Score the following metrics strictly and objectively.

1. 'faithfulness' (0.0–1.0): Is EVERY claim in the Answer directly supported by the Context?
2. 'relevance' (0.0–1.0): Does the Answer completely and directly address the Question?
3. 'correctness' (0.0–1.0): Does the Answer factually agree with the Ground Truth?
   - 1.0: All key facts, numbers, and conclusions match the Ground Truth.
   - 0.5: Mostly correct; minor factual discrepancies.
   - 0.0: Contradicts or significantly diverges from the Ground Truth.

{format_instructions}"""),
            ("human", (
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Answer: {answer}\n\n"
                "Ground Truth: {ground_truth}"
            )),
        ])

    # ── Internal: Retryable Judge Call ────────────────────────────────────────
    @_eval_retry
    def _call_judge(self, chain, inputs: dict) -> str:
        """Execute a judge LLM chain with retry logic."""
        return chain.invoke(inputs)

    # ── Public: Single Question Evaluation ───────────────────────────────────
    def evaluate(
        self,
        question: str,
        answer: str,
        context_docs: list[Document],
        ground_truth: Optional[str] = None,
    ) -> dict:
        """
        Score a single RAG response on faithfulness, relevance, and optionally correctness.

        Args:
            question:      The original user question.
            answer:        The generated answer to evaluate.
            context_docs:  The retrieved documents used to generate the answer.
            ground_truth:  Optional reference answer from the 10-K source.
                           When provided, adds an Answer Correctness score.

        Returns:
            Dict with keys: faithfulness, relevance, [correctness if ground_truth provided].
            Returns {"faithfulness": "Error", "relevance": "Error"} on parse failure.
        """
        logger.info(
            "Running LLM-as-a-Judge evaluation (Judge model: %s)...", EVALUATOR_MODEL
        )

        context_text: str = "\n".join([d.page_content for d in context_docs])

        try:
            if ground_truth:
                chain  = self.eval_prompt_gt | self.llm
                parser = self.parser_gt
                inputs = {
                    "question":       question,
                    "context":        context_text,
                    "answer":         answer,
                    "ground_truth":   ground_truth,
                    "format_instructions": parser.get_format_instructions(),
                }
            else:
                chain  = self.eval_prompt | self.llm
                parser = self.parser_base
                inputs = {
                    "question": question,
                    "context":  context_text,
                    "answer":   answer,
                    "format_instructions": parser.get_format_instructions(),
                }

            response = self._call_judge(chain, inputs)
            scores   = parser.invoke(response).model_dump()

            logger.info("Evaluation scores: %s", scores)
            return scores

        except Exception as exc:
            logger.error("Failed to parse LLM evaluation response: %s", exc)
            return {"faithfulness": "Error", "relevance": "Error"}

    # ── Public: Batch Evaluation ──────────────────────────────────────────────
    def run_batch_evaluation(
        self,
        eval_set: list[dict],
        agent,
        save_report: bool = True,
    ) -> dict:
        """
        Run evaluation across a full test set and compute aggregate statistics.

        This is the production-grade evaluation method. Single-question evaluation
        is not statistically meaningful; N≥10 questions with mean ± std dev is
        the minimum bar for a credible RAG evaluation.

        Args:
            eval_set:    List of dicts with keys:
                           - "question" (str, required)
                           - "ground_truth" (str, optional)
            agent:       FinancialGenerationAgent instance for generating answers.
            save_report: If True, save full per-question results to EVAL_REPORTS_DIR.

        Returns:
            Dict with:
              - per_question_results: list of individual score dicts
              - mean_faithfulness, std_faithfulness
              - mean_relevance, std_relevance
              - mean_correctness, std_correctness  (only if ground_truth provided)
              - n: number of questions evaluated
              - pass_rate: fraction of questions where faithfulness >= 0.8

        Example eval_set:
            [
                {
                    "question": "What were Google's total R&D expenses in 2024?",
                    "ground_truth": "Google reported R&D expenses of $49.1 billion in FY2024."
                },
                {
                    "question": "Compare Meta and Microsoft capital expenditure.",
                    "ground_truth": "Meta capex was $37.3B; Microsoft capex was $55.7B in FY2024."
                },
            ]
        """
        if not eval_set:
            raise ValueError("eval_set must contain at least one question.")

        logger.info(
            "Starting batch evaluation over %d questions (Judge: %s)...",
            len(eval_set), EVALUATOR_MODEL,
        )

        per_question_results: list[dict] = []

        for i, item in enumerate(eval_set, 1):
            question     = item.get("question", "")
            ground_truth = item.get("ground_truth")   # None if not provided

            if not question:
                logger.warning("Skipping eval_set item %d — missing 'question' key.", i)
                continue

            logger.info(
                "  Evaluating question %d/%d: '%s'", i, len(eval_set), question[:60]
            )

            try:
                answer, docs = agent.generate_answer(question)
                scores = self.evaluate(
                    question=question,
                    answer=answer,
                    context_docs=docs,
                    ground_truth=ground_truth,
                )
                per_question_results.append({
                    "question":     question,
                    "ground_truth": ground_truth,
                    "answer":       answer,
                    "scores":       scores,
                })
            except Exception as exc:
                logger.error(
                    "  Question %d failed during batch eval: %s", i, exc
                )
                per_question_results.append({
                    "question": question,
                    "scores":   {"faithfulness": "Error", "relevance": "Error"},
                })

        # ── Aggregate Statistics ──────────────────────────────────────────────
        def _extract_numeric(results: list[dict], key: str) -> list[float]:
            return [
                r["scores"][key]
                for r in results
                if isinstance(r["scores"].get(key), float)
            ]

        faith_scores = _extract_numeric(per_question_results, "faithfulness")
        relev_scores = _extract_numeric(per_question_results, "relevance")
        corr_scores  = _extract_numeric(per_question_results, "correctness")

        def _safe_stats(scores: list[float]) -> tuple[float, float]:
            if not scores:
                return 0.0, 0.0
            mean = sum(scores) / len(scores)
            std  = statistics.stdev(scores) if len(scores) > 1 else 0.0
            return round(mean, 4), round(std, 4)

        mean_faith, std_faith = _safe_stats(faith_scores)
        mean_relev, std_relev = _safe_stats(relev_scores)
        mean_corr,  std_corr  = _safe_stats(corr_scores)

        pass_rate = (
            sum(1 for s in faith_scores if s >= 0.8) / len(faith_scores)
            if faith_scores else 0.0
        )

        aggregate: dict = {
            "n":                    len(per_question_results),
            "evaluator_model":      EVALUATOR_MODEL,
            "mean_faithfulness":    mean_faith,
            "std_faithfulness":     std_faith,
            "mean_relevance":       mean_relev,
            "std_relevance":        std_relev,
            "faithfulness_pass_rate": round(pass_rate, 4),
            "per_question_results": per_question_results,
        }

        if corr_scores:
            aggregate["mean_correctness"] = mean_corr
            aggregate["std_correctness"]  = std_corr

        logger.info(
            "Batch evaluation complete. "
            "Faithfulness: %.3f ± %.3f | Relevance: %.3f ± %.3f | "
            "Pass rate: %.1f%% | n=%d",
            mean_faith, std_faith, mean_relev, std_relev,
            pass_rate * 100, len(per_question_results),
        )

        if save_report:
            report_path = f"{EVAL_REPORTS_DIR}/batch_eval_report.json"
            with open(report_path, "w") as f:
                # Exclude full per-question answers from the summary log for brevity
                summary = {k: v for k, v in aggregate.items()
                           if k != "per_question_results"}
                json.dump({"summary": summary, "details": per_question_results}, f, indent=4)
            logger.info("Batch evaluation report saved to: %s", report_path)

        return aggregate