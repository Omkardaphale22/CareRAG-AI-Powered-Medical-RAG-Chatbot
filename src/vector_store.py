"""
Vector store implementation using FAISS with sentence-transformers

Upgrades over v1
----------------
* search() can over-fetch (fetch_k), drop duplicate chunks, and re-rank with
  MMR so a large top_k returns *diverse* evidence instead of 12 near-copies.
* Fixed an index bug: FAISS returns -1 when fewer than k vectors exist, which
  silently pointed at the *last* document.
* expand_hits() grows every hit with its neighbouring chunks (same document)
  and merges them into longer passages. This enlarges the context handed to
  the LLM without re-building the index.
"""
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:  # langchain >= 0.2 split the splitters into their own tiny package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - older installs
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.data_ingestion import MedicalDocument


def _merge_overlap(a: str, b: str, max_overlap: int = 300) -> str:
    """Join two consecutive chunks, removing the text they share (chunk overlap)."""
    limit = min(len(a), len(b), max_overlap)
    for k in range(limit, 9, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + "\n" + b


class VectorStore:
    """FAISS-based vector store for document retrieval"""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index (exact search; cosine similarity via inner product)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict[str, Any]] = []
        self._lookup: Optional[Dict[Tuple[str, int], int]] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _chunk_documents(self, documents: List[MedicalDocument]) -> List[Dict[str, Any]]:
        """Split documents into chunks with metadata"""
        chunked_docs = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc.content)

            for i, chunk in enumerate(chunks):
                chunked_docs.append(
                    {
                        "doc_id": doc.doc_id,
                        "chunk_id": f"{doc.doc_id}_chunk_{i}",
                        "content": chunk,
                        "title": doc.title,
                        "source": doc.source,
                        "year": doc.year,
                        "url": doc.url,
                        "metadata": doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                )

        return chunked_docs

    def add_documents(self, documents: List[MedicalDocument]):
        """Add documents to the vector store (chunks that already exist are skipped)"""
        print(f"Chunking {len(documents)} documents...")
        chunked_docs = self._chunk_documents(documents)

        existing = {d["chunk_id"] for d in self.documents}
        chunked_docs = [d for d in chunked_docs if d["chunk_id"] not in existing]
        print(f"Created {len(chunked_docs)} new chunks")
        if not chunked_docs:
            return

        print("Generating embeddings...")
        texts = [doc["content"] for doc in chunked_docs]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )

        print("Adding to FAISS index...")
        self.index.add(embeddings.astype("float32"))
        self.documents.extend(chunked_docs)
        self._lookup = None

        print(f"Vector store now contains {len(self.documents)} chunks")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        fetch_k: Optional[int] = None,
        diversify: bool = False,
        mmr_lambda: float = 0.7,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar chunks.

        top_k       how many chunks to return
        fetch_k     how many candidates to pull from FAISS before de-duplication
                    / MMR (defaults to top_k)
        diversify   re-rank candidates with Maximal Marginal Relevance
        mmr_lambda  1.0 = pure relevance, 0.0 = pure diversity
        """
        total = self.index.ntotal
        if total == 0 or not self.documents:
            return []

        fetch_k = min(max(fetch_k or top_k, top_k), total)

        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, fetch_k)

        # Filter by threshold, drop invalid ids (-1) and duplicate text
        candidates: List[Tuple[int, float]] = []
        seen_text = set()
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(self.documents) or score < score_threshold:
                continue
            key = self.documents[idx]["content"].strip()
            if key in seen_text:
                continue
            seen_text.add(key)
            candidates.append((idx, float(score)))

        if not candidates:
            return []

        if diversify and len(candidates) > top_k:
            candidates = self._mmr(query_embedding[0], candidates, top_k, mmr_lambda)
        else:
            candidates = candidates[:top_k]

        return [(self.documents[idx], score) for idx, score in candidates]

    def _mmr(
        self,
        query_vec: np.ndarray,
        candidates: List[Tuple[int, float]],
        top_k: int,
        lam: float,
    ) -> List[Tuple[int, float]]:
        """Maximal Marginal Relevance selection (falls back to plain top-k on failure)."""
        try:
            vecs = np.vstack([self.index.reconstruct(idx) for idx, _ in candidates])
        except Exception:
            return candidates[:top_k]

        relevance = np.array([score for _, score in candidates], dtype="float32")
        selected: List[int] = [0]  # candidates are already sorted by relevance
        remaining = list(range(1, len(candidates)))

        while remaining and len(selected) < top_k:
            sim_to_selected = vecs[remaining] @ vecs[selected].T  # (remaining, selected)
            redundancy = sim_to_selected.max(axis=1)
            mmr_scores = lam * relevance[remaining] - (1 - lam) * redundancy
            best = remaining[int(np.argmax(mmr_scores))]
            selected.append(best)
            remaining.remove(best)

        selected.sort(key=lambda i: -relevance[i])
        return [candidates[i] for i in selected]

    # ------------------------------------------------------------------
    # Context expansion (bigger context window without re-indexing)
    # ------------------------------------------------------------------
    def _get_lookup(self) -> Dict[Tuple[str, int], int]:
        if self._lookup is None:
            self._lookup = {
                (d["doc_id"], int(d.get("chunk_index", 0))): pos
                for pos, d in enumerate(self.documents)
            }
        return self._lookup

    def expand_hits(
        self,
        hits: List[Tuple[Dict[str, Any], float]],
        window: int = 1,
        max_chars: int = 60000,
    ) -> List[Dict[str, Any]]:
        """
        Turn retrieved chunks into longer passages.

        Every hit is widened with up to `window` neighbouring chunks from the
        same document; overlapping/adjacent ranges are merged. Returns passages
        sorted by relevance, trimmed so the total text stays under `max_chars`.

        Each passage: {"doc": <chunk dict with merged "content">, "score": float,
                       "chunk_range": (first, last), "hit_count": int}
        """
        if not hits:
            return []

        lookup = self._get_lookup()
        by_doc: Dict[str, List[Tuple[int, float]]] = {}
        for doc, score in hits:
            by_doc.setdefault(doc["doc_id"], []).append((int(doc.get("chunk_index", 0)), score))

        passages: List[Dict[str, Any]] = []
        for doc_id, items in by_doc.items():
            wanted = set()
            for ci, _ in items:
                for j in range(ci - window, ci + window + 1):
                    if (doc_id, j) in lookup:
                        wanted.add(j)

            # group into runs of consecutive chunk indexes
            runs: List[List[int]] = []
            for j in sorted(wanted):
                if runs and j == runs[-1][-1] + 1:
                    runs[-1].append(j)
                else:
                    runs.append([j])

            for run in runs:
                run_hits = [s for ci, s in items if run[0] <= ci <= run[-1]]
                if not run_hits:
                    continue
                chunks = [self.documents[lookup[(doc_id, j)]] for j in run]
                content = chunks[0]["content"]
                for nxt in chunks[1:]:
                    content = _merge_overlap(content, nxt["content"])
                merged = dict(chunks[0])
                merged["content"] = content
                passages.append(
                    {
                        "doc": merged,
                        "score": max(run_hits),
                        "chunk_range": (run[0], run[-1]),
                        "hit_count": len(run_hits),
                    }
                )

        passages.sort(key=lambda p: -p["score"])

        selected, total = [], 0
        for p in passages:
            length = len(p["doc"]["content"])
            if selected and total + length > max_chars:
                continue
            selected.append(p)
            total += length
        return selected

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, save_dir: str):
        """Save vector store to disk"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_path / "faiss.index"))

        with open(save_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        config = {
            "embedding_model_name": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "dimension": self.dimension,
            "num_documents": len(self.documents),
        }
        with open(save_path / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        print(f"Vector store saved to {save_path}")

    def load(self, load_dir: str):
        """Load vector store from disk"""
        load_path = Path(load_dir)

        if not load_path.exists():
            raise FileNotFoundError(f"Vector store directory not found: {load_dir}")

        self.index = faiss.read_index(str(load_path / "faiss.index"))

        with open(load_path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
        self._lookup = None

        print(f"Vector store loaded from {load_path}")
        print(f"Loaded {len(self.documents)} document chunks")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        unique_docs = len(set(doc["doc_id"] for doc in self.documents))

        return {
            "total_chunks": len(self.documents),
            "unique_documents": unique_docs,
            "dimension": self.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
