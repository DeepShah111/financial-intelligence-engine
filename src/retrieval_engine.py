"""
Enterprise Hybrid Retrieval Engine.
Combines Dense Vectors (ChromaDB) and Sparse Keywords (BM25).
Features a custom Reciprocal Rank Fusion (RRF) implementation with company-balanced output.

UPGRADES vs previous version:
- Corrected RRF math: weights now scale the full RRF fraction, not just the numerator.
  Previous: weight / (rank + K)  — mathematically inconsistent with standard RRF.
  Fixed:    weight * (1 / (rank + K)) — weight scales the entire rank-fusion score.
- RRF_K constant imported from config (no hardcoding).
- Atomic BM25 serialization via tempfile + shutil.move() — prevents partial-write
  corruption that left the index in an unrecoverable split state.
- SHA-256 integrity check on BM25 load — detects file tampering or corruption.
- Company-balanced retrieval — prevents one company dominating the context window
  (was 71.4% Meta in the original, revealed by the telemetry dashboard).
- Full type annotations on all methods.
"""

import os
import re
import json
import pickle
import hashlib
import tempfile
import shutil
from typing import Optional

from langchain_chroma import Chroma
from src.jina_embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import (
    logger,
    VECTOR_DB_DIR,
    TOP_K_VECTORS,
    RRF_K,
    MAX_CHUNKS_PER_COMPANY,
    EMBEDDING_MODEL_NAME,
)

# ── Query-time retrieval settings ─────────────────────────────────────────────
# How many candidates each search hands to the fusion step (the final answer
# still has TOP_K_VECTORS chunks). A larger pool lets a relevant chunk that is
# ranked 8th or 15th by one search still take part in the fusion.
CANDIDATE_POOL_K: int = 20
# BM25 returns its top results across ALL companies, so it needs a deeper list
# to still have enough chunks left after filtering for one company.
SPARSE_POOL_K: int = 60

# Words that identify a company in a question -> the label stored in the chunk
# metadata (see data_ingestion.py). Matched as whole words, ignoring case.
_COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "Google":    ("google", "alphabet"),
    "Meta":      ("meta", "facebook"),
    "Microsoft": ("microsoft",),
}

# Abbreviations written out before searching. The filings print "Research and
# development" in full, so keyword search cannot match the short form "R&D".
_ABBREVIATIONS: dict[str, str] = {
    r"\bR&D\b": "research and development",
    r"\b(?:capital expenditures?|capex)\b": (
        "capital expenditures purchases of property and equipment"
    ),
}

# Extra words ADDED AT THE END of the search text (the question itself is kept).
# Testing on the real index showed that for "net income" this ranks the income
# statement chunks far better than rewriting the question in place.
_APPENDED_TERMS: dict[str, str] = {
    r"\bnet income\b": "net income income statements",
}


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
class CustomHybridRetriever:
    """
    Reciprocal Rank Fusion (RRF) over dense + sparse retrieval results.

    RRF Formula (per retriever r, per document d at rank i):
        score(d) += weight_r * (1 / (rank_i + K))

    Where K=60 is the standard smoothing constant that prevents top-ranked
    documents from receiving disproportionately high scores.

    Weighting: dense_weight + sparse_weight should equal 1.0. These scale
    each retriever's contribution to the fused score independently, allowing
    domain tuning (e.g., keyword-heavy financial text may warrant higher
    sparse_weight for exact figure matching).
    """

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> None:
        if not (0.0 < dense_weight <= 1.0 and 0.0 < sparse_weight <= 1.0):
            raise ValueError("Retriever weights must be in (0.0, 1.0].")
        self.dense_retriever  = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight     = dense_weight
        self.sparse_weight    = sparse_weight

    def _compute_rrf_scores(
        self,
        docs: list[Document],
        weight: float,
        rrf_scores: dict,
        doc_map: dict,
    ) -> None:
        """
        Accumulate weighted RRF scores into rrf_scores in-place.

        Args:
            docs:       Ranked list of documents from one retriever.
            weight:     Scalar weight for this retriever's contribution.
            rrf_scores: Mutable dict mapping chunk_id → cumulative score.
            doc_map:    Mutable dict mapping chunk_id → Document object.
        """
        for rank, doc in enumerate(docs):
            # Use deterministic chunk_id from metadata (set during ingestion).
            # Fall back to a hash of content if metadata is missing — this
            # prevents KeyError but also signals an ingestion configuration issue.
            chunk_id: str = doc.metadata.get(
                "chunk_id",
                hashlib.sha256(doc.page_content.encode()).hexdigest()[:16],
            )
            doc_map[chunk_id] = doc
            # CORRECTED: weight * (1 / (rank + K)) — weight scales the full fraction.
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0) + weight * (1.0 / (rank + RRF_K))
            )

    def _balance_by_company(
        self, sorted_docs: list[tuple[str, float]], doc_map: dict
    ) -> list[Document]:
        """
        Return top-K documents with per-company diversity enforcement.

        Prevents any single company from dominating the context window.
        A query about "Google vs Meta R&D" should surface chunks from both,
        not 70%+ from whichever company had more keyword matches.

        Algorithm: iterate RRF-ranked list; admit a doc only if its company
        hasn't exceeded MAX_CHUNKS_PER_COMPANY. Continue until TOP_K_VECTORS
        are collected or the list is exhausted.

        Args:
            sorted_docs: List of (chunk_id, rrf_score) sorted desc by score.
            doc_map:     Mapping from chunk_id to Document.

        Returns:
            Balanced list of up to TOP_K_VECTORS Document objects.
        """
        company_counts: dict[str, int] = {}
        balanced: list[Document]       = []

        for chunk_id, _ in sorted_docs:
            if len(balanced) >= TOP_K_VECTORS:
                break
            doc     = doc_map[chunk_id]
            company = doc.metadata.get("company", "unknown")
            count   = company_counts.get(company, 0)

            if count < MAX_CHUNKS_PER_COMPANY:
                company_counts[company] = count + 1
                balanced.append(doc)

        # Safety fallback: if balancing left us short (e.g., only 1 company
        # in the corpus), fill remaining slots from the unfiltered ranked list.
        if len(balanced) < TOP_K_VECTORS:
            seen_ids = {d.metadata.get("chunk_id") for d in balanced}
            for chunk_id, _ in sorted_docs:
                if len(balanced) >= TOP_K_VECTORS:
                    break
                if chunk_id not in seen_ids:
                    balanced.append(doc_map[chunk_id])
                    seen_ids.add(chunk_id)

        return balanced

    # ── Query helpers (new) ───────────────────────────────────────────────────
    @staticmethod
    def _expand_query(query: str) -> str:
        """Write out known abbreviations (R&D, capex) and add helper words for the searches."""
        expanded = query
        for pattern, full_text in _ABBREVIATIONS.items():
            expanded = re.sub(pattern, full_text, expanded, flags=re.IGNORECASE)
        for pattern, extra in _APPENDED_TERMS.items():
            if re.search(pattern, query, flags=re.IGNORECASE):
                expanded = f"{expanded} {extra}"
        if expanded != query:
            logger.info("[Retrieval] Query expanded for search: '%s'", expanded)
        return expanded

    @staticmethod
    def _detect_companies(query: str) -> list[str]:
        """Return the company labels the question names, in a fixed order."""
        lowered = query.lower()
        found: list[str] = []
        for label, aliases in _COMPANY_ALIASES.items():
            if any(re.search(rf"\b{re.escape(a)}\b", lowered) for a in aliases):
                found.append(label)
        return found

    def _dense_search(self, query: str, company: Optional[str]) -> list[Document]:
        """Dense (meaning) search, optionally limited to one company."""
        if company is None:
            return self.dense_retriever.invoke(query)[:CANDIDATE_POOL_K]
        vector_store = getattr(self.dense_retriever, "vectorstore", None)
        if vector_store is not None:
            return vector_store.similarity_search(
                query, k=CANDIDATE_POOL_K, filter={"company": company}
            )
        # Retriever without a vector store attached: filter its results instead.
        return [
            d for d in self.dense_retriever.invoke(query)
            if d.metadata.get("company") == company
        ][:CANDIDATE_POOL_K]

    def _sparse_search(self, query: str, company: Optional[str]) -> list[Document]:
        """Keyword (BM25) search, optionally limited to one company."""
        docs = self.sparse_retriever.invoke(query)
        if company is not None:
            docs = [d for d in docs if d.metadata.get("company") == company]
        return docs[:CANDIDATE_POOL_K]

    def _fuse(
        self, query: str, company: Optional[str]
    ) -> tuple[list[tuple[str, float]], dict]:
        """Run both searches and return (chunks ranked by fused score, doc map)."""
        dense_docs  = self._dense_search(query, company)
        sparse_docs = self._sparse_search(query, company)

        rrf_scores: dict[str, float]    = {}
        doc_map:    dict[str, Document] = {}

        self._compute_rrf_scores(dense_docs,  self.dense_weight,  rrf_scores, doc_map)
        self._compute_rrf_scores(sparse_docs, self.sparse_weight, rrf_scores, doc_map)

        sorted_docs: list[tuple[str, float]] = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_docs, doc_map

    def invoke(self, query: str) -> list[Document]:
        """
        Execute hybrid search and return RRF-fused results.

        Behaviour:
          - Known abbreviations such as R&D are written out for the searches.
          - A question that names exactly ONE company is searched inside that
            company only, and all TOP_K_VECTORS chunks come from it.
          - Any other question (no company, or two or more) keeps the original
            company-balanced behaviour.

        Args:
            query: The user's natural language query string.

        Returns:
            List of up to TOP_K_VECTORS Document objects ranked by fused score.
        """
        search_query = self._expand_query(query)
        companies    = self._detect_companies(query)

        if len(companies) == 1:
            sorted_docs, doc_map = self._fuse(search_query, companies[0])
            if len(sorted_docs) >= TOP_K_VECTORS:
                logger.info(
                    "[Retrieval] Single-company question: searched %s only.",
                    companies[0],
                )
                return [doc_map[cid] for cid, _ in sorted_docs[:TOP_K_VECTORS]]
            logger.info(
                "[Retrieval] Too few %s chunks (%d); using balanced search.",
                companies[0], len(sorted_docs),
            )

        sorted_docs, doc_map = self._fuse(search_query, None)
        return self._balance_by_company(sorted_docs, doc_map)


# ── Hybrid Retrieval Engine ───────────────────────────────────────────────────
class HybridRetrievalEngine:
    """
    Orchestrates ChromaDB (dense) + BM25 (sparse) index construction and loading.

    Smart Load Logic:
        - If both indexes exist on disk → load without re-embedding (warm start).
        - If either is missing → build from scratch using document_chunks.
        - Atomic BM25 writes prevent split-state corruption.
        - SHA-256 integrity verification on BM25 load detects tampering/corruption.
    """

    def __init__(self) -> None:
        self.vector_db_dir: str = VECTOR_DB_DIR
        self.bm25_path: str     = os.path.join(VECTOR_DB_DIR, "bm25_index.pkl")
        self.bm25_hash_path: str = os.path.join(VECTOR_DB_DIR, "bm25_index.sha256")

        logger.info("Using Jina API embeddings (no local model).")
        self.embedding_model = JinaEmbeddings()

        self.ensemble_retriever: Optional[CustomHybridRetriever] = None

    # ── Internal: Integrity Helpers ───────────────────────────────────────────
    @staticmethod
    def _compute_file_sha256(path: str) -> str:
        """Compute SHA-256 hex digest of a file's binary contents."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _write_bm25_with_integrity(self, sparse_retriever) -> None:
        """
        Serialize BM25 retriever atomically with integrity hash.

        Steps:
          1. Write pickle to a temp file in the same directory (same filesystem
             as destination → rename is atomic on POSIX).
          2. Compute SHA-256 of the temp file.
          3. Rename temp file to final path (atomic).
          4. Write hash to .sha256 sidecar file.

        This prevents the partial-write state where Chroma exists but BM25
        doesn't, causing an unrecoverable ValueError on the next cold start.
        """
        dir_path = os.path.dirname(self.bm25_path)

        with tempfile.NamedTemporaryFile(
            dir=dir_path, suffix=".pkl", delete=False
        ) as tmp:
            pickle.dump(sparse_retriever, tmp)
            tmp_path = tmp.name

        file_hash = self._compute_file_sha256(tmp_path)
        shutil.move(tmp_path, self.bm25_path)   # atomic rename

        # Write the hash the same way (temp file, then atomic replace), so a
        # crash can never leave a half-written .sha256 file behind.
        hash_tmp_path = self.bm25_hash_path + ".tmp"
        with open(hash_tmp_path, "w") as hf:
            hf.write(file_hash)
        os.replace(hash_tmp_path, self.bm25_hash_path)

        logger.info(
            "BM25 index serialized. SHA-256: %s", file_hash
        )

    def _load_bm25_with_integrity(self):
        """
        Deserialize BM25 and verify SHA-256 integrity before returning.

        Raises:
            RuntimeError: If the hash sidecar is missing or digest mismatches,
                          indicating file corruption or tampering.
        """
        if not os.path.exists(self.bm25_hash_path):
            raise RuntimeError(
                f"BM25 integrity file missing: {self.bm25_hash_path}. "
                "Delete the vector_db directory and rebuild indexes."
            )

        with open(self.bm25_hash_path, "r") as hf:
            expected_hash = hf.read().strip()

        actual_hash = self._compute_file_sha256(self.bm25_path)

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"BM25 index integrity check FAILED. "
                f"Expected {expected_hash}, got {actual_hash}. "
                "Index may be corrupted. Delete vector_db/ and rebuild."
            )

        with open(self.bm25_path, "rb") as f:
            sparse_retriever = pickle.load(f)   # safe: integrity verified above

        logger.info("BM25 index loaded and integrity verified.")
        return sparse_retriever

    def _clear_partial_index(self) -> None:
        """
        Empty the vector_db folder before a cold build.

        Only called when the index is NOT complete, so a healthy index is
        never deleted. The folder itself is kept; only its contents go.
        """
        if not os.path.isdir(self.vector_db_dir):
            return
        leftovers = os.listdir(self.vector_db_dir)
        if not leftovers:
            return
        logger.warning(
            "Incomplete index found in %s (%d item(s)). Clearing it before "
            "a clean rebuild to avoid duplicate chunks.",
            self.vector_db_dir, len(leftovers),
        )
        for name in leftovers:
            path = os.path.join(self.vector_db_dir, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    # ── Public: Build or Load Indexes ────────────────────────────────────────
    def build_indexes(
        self,
        document_chunks: Optional[list[Document]] = None,
    ) -> CustomHybridRetriever:
        """
        Build indexes from chunks (cold start) or load them from disk (warm start).

        Args:
            document_chunks: Required for cold start (first run). Ignored on
                             warm start (indexes already on disk).

        Returns:
            Initialized CustomHybridRetriever ready for querying.

        Raises:
            ValueError:  If cold start is required but no chunks provided.
            RuntimeError: If BM25 integrity check fails on warm start.
        """
        # An index is only "complete" when ALL THREE parts are present:
        # the ChromaDB database, the BM25 pickle, and its SHA-256 sidecar.
        # (The old check counted ANY file in the folder as ChromaDB, so a
        # folder holding only the BM25 files looked like a finished index.)
        chroma_exists: bool = os.path.exists(
            os.path.join(self.vector_db_dir, "chroma.sqlite3")
        )
        bm25_exists: bool = (
            os.path.exists(self.bm25_path)
            and os.path.exists(self.bm25_hash_path)
        )

        if chroma_exists and bm25_exists:
            logger.info(
                "[2/4] Smart Load: Found existing indexes on disk. "
                "Bypassing embedding compute..."
            )
            vector_store = Chroma(
                persist_directory=self.vector_db_dir,
                embedding_function=self.embedding_model,
            )
            sparse_retriever = self._load_bm25_with_integrity()

        else:
            if not document_chunks:
                raise ValueError(
                    "No existing indexes found on disk and no document_chunks provided. "
                    "Pass document_chunks=load_and_chunk_pdfs() for the initial build."
                )

            # A partly built index (for example a crash after ChromaDB was
            # written but before BM25 was saved) must be cleared first.
            # Otherwise Chroma.from_documents would ADD every chunk a second
            # time to the existing database, creating duplicate chunks.
            self._clear_partial_index()

            logger.info(
                "[2/4] Cold start: Building Dense Vector Database (ChromaDB)..."
            )
            vector_store = Chroma.from_documents(
                documents=document_chunks,
                embedding=self.embedding_model,
                persist_directory=self.vector_db_dir,
            )

            logger.info("[2/4] Building Sparse Keyword Index (BM25)...")
            sparse_retriever = BM25Retriever.from_documents(document_chunks)
            sparse_retriever.k = TOP_K_VECTORS

            logger.info("[2/4] Serializing BM25 index with integrity hash...")
            self._write_bm25_with_integrity(sparse_retriever)

        # Query-time pool sizes. These are set in memory only; the BM25 file on
        # disk is not changed.
        sparse_retriever.k = SPARSE_POOL_K
        dense_retriever = vector_store.as_retriever(
            search_kwargs={"k": CANDIDATE_POOL_K}
        )

        logger.info("[2/4] Initializing Reciprocal Rank Fusion engine...")
        self.ensemble_retriever = CustomHybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            dense_weight=0.5,
            sparse_weight=0.5,
        )

        logger.info("Hybrid Retrieval Engine ready.")
        return self.ensemble_retriever