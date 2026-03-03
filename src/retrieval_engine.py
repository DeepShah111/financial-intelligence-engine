"""
Enterprise Hybrid Retrieval Engine.
Combines Dense Vectors (ChromaDB) and Sparse Keywords (BM25).
Features a custom Reciprocal Rank Fusion (RRF) implementation to guarantee stability.
"""
import os
import pickle # UPGRADE: Added for BM25 disk serialization
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from src.config import logger, VECTOR_DB_DIR, TOP_K_VECTORS

class CustomHybridRetriever:
    """Mathematical Engine for Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever, sparse_retriever, dense_weight=0.5, sparse_weight=0.5):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def invoke(self, query: str):
        # 1. Fetch results independently
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)
        
        # 2. Apply Reciprocal Rank Fusion Math
        # Score = Weight * (1 / (Rank + K_constant))
        rrf_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(dense_docs):
            # UPGRADE: Use unique chunk_id instead of raw text to prevent dictionary overwrite collisions
            chunk_id = doc.metadata.get('chunk_id', hash(doc.page_content))
            doc_map[chunk_id] = doc # Store the actual document object
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (self.dense_weight / (rank + 60))
            
        for rank, doc in enumerate(sparse_docs):
            # UPGRADE: Use unique chunk_id
            chunk_id = doc.metadata.get('chunk_id', hash(doc.page_content))
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (self.sparse_weight / (rank + 60))
            
        # 3. Sort by highest fused RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 4. Return Top K unique documents
        return [doc_map[chunk_id] for chunk_id, score in sorted_docs[:TOP_K_VECTORS]]


class HybridRetrievalEngine:
    def __init__(self):
        self.vector_db_dir = VECTOR_DB_DIR
        self.bm25_path = os.path.join(VECTOR_DB_DIR, "bm25_index.pkl") # Store BM25 with Chroma
        # Using a highly-ranked, cost-free open-source embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    def build_indexes(self, document_chunks=None):
        # UPGRADE: Smart Load Logic - Check if indexes already exist on disk
        chroma_exists = os.path.exists(self.vector_db_dir) and len(os.listdir(self.vector_db_dir)) > 0
        bm25_exists = os.path.exists(self.bm25_path)

        if chroma_exists and bm25_exists:
            logger.info("[2/4] 🚀 Smart Load: Found existing indexes on disk. Bypassing embedding compute...")
            
            # 1. Load ChromaDB directly from the saved directory
            vector_store = Chroma(
                persist_directory=self.vector_db_dir,
                embedding_function=self.embedding_model
            )
            
            # 2. Deserialize the BM25 engine from the pickle file
            with open(self.bm25_path, 'rb') as f:
                sparse_retriever = pickle.load(f)
                
        else:
            if not document_chunks:
                raise ValueError("❌ No existing database found. You must provide document_chunks for the initial build.")
                
            logger.info("[2/4] Building Dense Vector Database (ChromaDB) from scratch...")
            vector_store = Chroma.from_documents(
                documents=document_chunks,
                embedding=self.embedding_model,
                persist_directory=self.vector_db_dir
            )
            
            logger.info("[2/4] Building Sparse Keyword Index (BM25)...")
            sparse_retriever = BM25Retriever.from_documents(document_chunks)
            sparse_retriever.k = TOP_K_VECTORS
            
            logger.info("[2/4] Serializing BM25 Index to disk for future warm-starts...")
            with open(self.bm25_path, 'wb') as f:
                pickle.dump(sparse_retriever, f)

        # Initialize the retrievers
        dense_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_VECTORS})
        
        logger.info("[2/4] Initializing Custom Reciprocal Rank Fusion (RRF)...")
        self.ensemble_retriever = CustomHybridRetriever(dense_retriever, sparse_retriever)
        
        logger.info("✅ Hybrid Retrieval Engine Ready.")
        return self.ensemble_retriever