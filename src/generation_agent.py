"""
Generation Agent with Chain-of-Thought (CoT) Reasoning and Self-Correction.
Forces the LLM to map out analytical steps before generating the final answer.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.config import logger

class FinancialGenerationAgent:
    def __init__(self, retriever, api_key):
        self.retriever = retriever
        # Meta Llama 3.3 70B - highly capable of advanced reasoning
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)
        
        # 1. The Generator Prompt (Restored to Original Stable Version)
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
            ("human", "{question}")
        ])
        
        # 2. The Critic Prompt (Restored to Original Stable Version)
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
            ("human", "Audit this draft. Output ONLY the finalized, hallucination-free <final_answer> text. Do not include the <thought_process>.")
        ])

    def generate_answer(self, query: str):
        logger.info(f"🔍 Retrieving documents for query: '{query}'")
        docs = self.retriever.invoke(query)
        
        context = "\n\n".join([f"[Source: {d.metadata.get('company')} 10-K]: {d.page_content}" for d in docs])
        
        logger.info("✍️ Step 1: Executing Chain-of-Thought Analysis (Llama-3.3)...")
        draft_chain = self.qa_prompt | self.llm
        draft_response = draft_chain.invoke({"context": context, "question": query}).content
        
        logger.info("🛡️ Step 2: Running Strict Compliance Audit (Llama-3.3)...")
        audit_chain = self.critic_prompt | self.llm
        final_response = audit_chain.invoke({"context": context, "draft": draft_response, "question": query}).content
        
        return final_response, docs