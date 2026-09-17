# One-time script: build the search index locally, then exit. Run: python build_index.py
from dotenv import load_dotenv
load_dotenv()

from src.data_ingestion import load_and_chunk_pdfs
from src.retrieval_engine import HybridRetrievalEngine

print("Loading and chunking PDFs...")
chunks = load_and_chunk_pdfs()
print(f"Got {len(chunks)} chunks. Building index (this embeds via Jina — will take ~15 min)...")

engine = HybridRetrievalEngine()
engine.build_indexes(document_chunks=chunks)

print("Index built successfully. You can now commit artifacts/vector_db.")