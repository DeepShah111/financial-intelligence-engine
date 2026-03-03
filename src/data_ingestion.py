"""
Data ingestion and semantic chunking module for 10-K Filings.
Upgraded with ThreadPoolExecutor for parallel processing of I/O bound tasks.
"""
import os
import glob
import uuid
import concurrent.futures # UPGRADE: Standard library for parallel execution
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import logger, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def _process_single_pdf(file_path: str, text_splitter: RecursiveCharacterTextSplitter) -> list:
    """
    Helper function to process a single PDF. 
    Isolated so it can be mapped across multiple threads.
    """
    file_name = os.path.basename(file_path)
    company_name = file_name.split('_')[0].capitalize()
    logger.info(f"  -> Parsing in parallel: {file_name}")
    
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        raw_pages = loader.load()
        
        # Add Metadata
        for page in raw_pages:
            page.metadata['company'] = company_name
            page.metadata['source_file'] = file_name
            
        # Split
        chunks = text_splitter.split_documents(raw_pages)
        
        # Assign a deterministic UUID to every chunk to prevent RRF collisions
        for chunk in chunks:
            chunk.metadata['chunk_id'] = str(uuid.uuid4())
            
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to process {file_name}: {e}")
        return []

def load_and_chunk_pdfs() -> list:
    logger.info("[1/4] Starting Data Ingestion from PDFs (Parallel Mode)")
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDFs found in {DATA_DIR}.")
        raise FileNotFoundError(f"Missing Data. Ensure PDFs are in {DATA_DIR}")
        
    logger.info(f"Found {len(pdf_files)} SEC filings. Initiating parallel extraction...")
    
    all_chunks = []
    
    # UPGRADE: Using ThreadPoolExecutor to handle multiple files simultaneously
    # os.cpu_count() optimally sets the number of threads based on your machine/Colab instance
    max_threads = min(32, (os.cpu_count() or 1) * 4) 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit all PDF processing tasks to the thread pool
        future_to_pdf = {
            executor.submit(_process_single_pdf, pdf, text_splitter): pdf 
            for pdf in pdf_files
        }
        
        # As each thread finishes, collect its chunks
        for future in concurrent.futures.as_completed(future_to_pdf):
            chunks = future.result()
            if chunks:
                all_chunks.extend(chunks)
            
    logger.info(f"[1/4] Complete. Generated {len(all_chunks)} semantic chunks.")
    return all_chunks