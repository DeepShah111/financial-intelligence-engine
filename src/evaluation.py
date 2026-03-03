"""
Enterprise LLM-as-a-Judge Evaluation Module.
Computes quantitative scores for RAG Faithfulness and Answer Relevance.
"""
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
# UPGRADE: Imports for bulletproof structured outputs
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.config import logger, EVAL_REPORTS_DIR

# UPGRADE: Define strict Pydantic schema for output validation
class EvaluationScores(BaseModel):
    faithfulness: float = Field(description="Score from 0.0 to 1.0 indicating if Answer is derived ONLY from Context.")
    relevance: float = Field(description="Score from 0.0 to 1.0 indicating if Answer directly addresses the Question.")

class RAGEvaluator:
    def __init__(self, api_key):
        # We use the same high-powered model for evaluation
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)
        
        # UPGRADE: Initialize the parser
        self.parser = PydanticOutputParser(pydantic_object=EvaluationScores)
        
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an impartial AI auditor evaluating a Generative AI system. 
            Analyze the provided Question, Context, and Answer. 
            
            Provide a score from 0.0 to 1.0 for the following two metrics:
            1. 'faithfulness': Is the Answer derived ONLY from the Context? (1.0 = Perfect, no outside info. 0.0 = Hallucinated).
            2. 'relevance': Does the Answer directly address the Question? (1.0 = Perfectly answers the prompt. 0.0 = Off-topic).
            
            {format_instructions}"""),
            ("human", "Question: {question}\n\nContext: {context}\n\nAnswer: {answer}")
        ])

    def evaluate(self, question: str, answer: str, context_docs: list) -> dict:
        logger.info("📊 Running Quantitative LLM-as-a-Judge Evaluation...")
        
        # Combine all the text the retriever found
        context_text = "\n".join([d.page_content for d in context_docs])
        
        # Run the evaluation chain
        chain = self.eval_prompt | self.llm
        # UPGRADE: Inject the Pydantic format instructions directly into the prompt
        response = chain.invoke({
            "question": question, 
            "context": context_text, 
            "answer": answer,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        try:
            # UPGRADE: Use Pydantic parser instead of fragile string manipulation
            scores_obj = self.parser.invoke(response)
            scores = scores_obj.model_dump()
            
            # Save the report to our artifacts directory
            report_path = f"{EVAL_REPORTS_DIR}/latest_eval.json"
            with open(report_path, "w") as f:
                json.dump({"question": question, "scores": scores}, f, indent=4)
                
            logger.info(f"✅ Evaluation Complete. Report saved to {report_path}")
            return scores
            
        except Exception as e:
            logger.error(f"Failed to parse LLM evaluation: {e}")
            return {"faithfulness": "Error", "relevance": "Error"}