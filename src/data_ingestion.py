"""
Data ingestion and semantic chunking module for SEC 10-K Filings.

UPGRADES vs previous version:
- Deterministic chunk IDs using SHA-256 hash of (content + source_file).
  Previously used uuid4() (random), which broke index integrity on re-runs
  because the same chunk got a different ID each time.
- Full type annotations on all public and private functions.
- Raises a structured ChunkingError instead of silently returning [] on failure,
  so the caller knows which file failed and why.
- max_workers capped at min(32, cpu*4) — preserved from original (correct).
- chunk_index added to metadata for positional context within a document.
"""

import os
import glob
import hashlib
import concurrent.futures
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import logger, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


# ── Custom Exception ──────────────────────────────────────────────────────────
class ChunkingError(RuntimeError):
    """Raised when a PDF cannot be parsed or chunked."""


# ── Deterministic ID Generation ───────────────────────────────────────────────
def _make_chunk_id(page_content: str, source_file: str, chunk_index: int) -> str:
    """
    Generate a deterministic, collision-resistant chunk ID.

    Uses SHA-256 over (content + source + index) so the same chunk always
    gets the same ID across pipeline re-runs. This is critical for:
      - BM25 dedup logic in the RRF retriever
      - Cache invalidation correctness
      - Reproducible experiment tracking

    Args:
        page_content:  The raw text content of the chunk.
        source_file:   The PDF filename the chunk came from.
        chunk_index:   The sequential position of this chunk within its document.

    Returns:
        A 16-character hex string prefixed with the source file stem.
        Example: "google_10k_a3f9c21b7e4d0012"
    """
    raw = f"{source_file}::{chunk_index}::{page_content}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    stem = os.path.splitext(source_file)[0]   # e.g. "google_10k"
    return f"{stem}_{digest}"


# ── Single-File Processor ─────────────────────────────────────────────────────
def _process_single_pdf(
    file_path: str,
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """
    Parse, chunk, and annotate a single PDF file.

    Isolated as a top-level function (not a lambda or inner function) so it can
    be safely pickled and mapped across threads by ThreadPoolExecutor.

    Args:
        file_path:     Absolute path to the PDF.
        text_splitter: Pre-configured RecursiveCharacterTextSplitter instance.
                       Shared across threads — LangChain splitters are stateless
                       and therefore thread-safe.

    Returns:
        List of annotated Document chunks. Returns [] only on hard failure
        (logged as ERROR). Raises ChunkingError propagated to the caller.

    Raises:
        ChunkingError: If the PDF cannot be loaded or produces zero pages.
    """
    file_name: str    = os.path.basename(file_path)
    company_name: str = file_name.split("_")[0].capitalize()
    logger.info("  -> Parsing in parallel: %s", file_name)

    try:
        loader = PyPDFLoader(file_path)
        raw_pages: list[Document] = loader.load()

        if not raw_pages:
            raise ChunkingError(f"PDF produced zero pages: {file_name}")

        # Annotate pages with document-level metadata before splitting.
        # Metadata set here propagates to every chunk derived from these pages.
        for page in raw_pages:
            page.metadata["company"]      = company_name
            page.metadata["source_file"]  = file_name

        chunks: list[Document] = text_splitter.split_documents(raw_pages)

        if not chunks:
            raise ChunkingError(f"Text splitter produced zero chunks for: {file_name}")

        # Assign deterministic IDs — critical for RRF dedup and index stability.
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"]    = _make_chunk_id(
                chunk.page_content, file_name, idx
            )
            chunk.metadata["chunk_index"] = idx   # positional context

        logger.info(
            "  -> Completed: %s → %d chunks", file_name, len(chunks)
        )
        return chunks

    except ChunkingError:
        raise   # re-raise structured errors as-is
    except Exception as exc:
        # Catch unexpected errors (corrupted PDF, permission denied, etc.)
        # Log and return empty list so one bad file doesn't kill the whole batch.
        logger.error("Failed to process %s: %s", file_name, exc)
        return []


# ── Public Pipeline Entry Point ───────────────────────────────────────────────
def load_and_chunk_pdfs() -> list[Document]:
    """
    Discover, parse, chunk, and annotate all PDF filings in DATA_DIR.

    Uses ThreadPoolExecutor for parallel I/O — PDF loading is I/O-bound,
    not CPU-bound, so threads (not processes) are the correct primitive.

    Returns:
        Flat list of annotated Document chunks from all discovered PDFs.

    Raises:
        FileNotFoundError: If no PDFs are found in DATA_DIR.
        RuntimeError:      If ALL files fail to process (partial failures are
                           tolerated and logged; total failure is not).
    """
    logger.info("[1/4] Starting Data Ingestion from PDFs (Parallel Mode)")
    logger.info("      Data directory: %s", DATA_DIR)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    pdf_files: list[str] = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in '{DATA_DIR}'. "
            "Ensure your 10-K filings are in the data/raw_pdfs/ directory."
        )

    logger.info(
        "Found %d SEC filings. Initiating parallel extraction...", len(pdf_files)
    )

    all_chunks: list[Document] = []
    failed_files: list[str]    = []

    # ThreadPoolExecutor — threads are correct for I/O-bound PDF loading.
    # cpu_count() * 4 is standard for I/O-bound tasks; cap at 32 to avoid
    # overwhelming the Colab / Drive connection pool.
    max_workers: int = min(32, (os.cpu_count() or 1) * 4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf: dict = {
            executor.submit(_process_single_pdf, pdf, text_splitter): pdf
            for pdf in pdf_files
        }

        for future in concurrent.futures.as_completed(future_to_pdf):
            source_pdf = future_to_pdf[future]
            try:
                chunks = future.result()
                if chunks:
                    all_chunks.extend(chunks)
                else:
                    failed_files.append(os.path.basename(source_pdf))
            except Exception as exc:
                failed_files.append(os.path.basename(source_pdf))
                logger.error(
                    "Unhandled exception for %s: %s", source_pdf, exc
                )

    # Fail loudly if every single file failed — silent empty returns are dangerous.
    if not all_chunks:
        raise RuntimeError(
            f"All {len(pdf_files)} PDFs failed to process. "
            f"Failed files: {failed_files}. Check logs for details."
        )

    if failed_files:
        logger.warning(
            "[1/4] Partial success. %d/%d files failed: %s",
            len(failed_files), len(pdf_files), failed_files,
        )

    logger.info(
        "[1/4] Complete. Generated %d semantic chunks from %d files.",
        len(all_chunks), len(pdf_files) - len(failed_files),
    )
    return all_chunks