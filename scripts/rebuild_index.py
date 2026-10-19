"""
Rebuild the FAISS index from the saved corpus (data/processed/*.json) using the
CURRENT chunk settings (CHUNK_SIZE / CHUNK_OVERLAP in .env).

Why: documents indexed with the old 500-character chunks keep those small
chunks. The app already widens each hit with its neighbouring chunks, but if you
want truly larger chunks run this once:

    python scripts/rebuild_index.py
    python scripts/rebuild_index.py --chunk-size 1200 --chunk-overlap 200

The old index is backed up to data/vector_store_backup_<timestamp>/.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.data_ingestion import MedicalDocument  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


def load_corpus(processed_dir: Path):
    docs, seen = [], set()
    for file in sorted(processed_dir.glob("*.json")):
        try:
            items = json.loads(file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  skip {file.name}: {exc}")
            continue
        if isinstance(items, dict):
            items = items.get("documents") or [items]
        for item in items:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            doc = MedicalDocument(
                content=item["content"],
                title=item.get("title", file.stem),
                source=item.get("source", file.name),
                doc_id=item.get("doc_id"),
                year=item.get("year"),
                url=item.get("url"),
                metadata=item.get("metadata", {}),
            )
            if doc.doc_id in seen:  # identical documents saved twice -> index once
                continue
            seen.add(doc.doc_id)
            docs.append(doc)
    return docs


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    docs = load_corpus(Path(args.processed_dir))
    if not docs:
        sys.exit(f"No documents found in {args.processed_dir}")
    print(f"Loaded {len(docs)} unique documents from {args.processed_dir}")

    store_path = Path(settings.vector_store_path)
    if store_path.exists():
        backup = store_path.parent / f"vector_store_backup_{int(time.time())}"
        shutil.copytree(store_path, backup)
        print(f"Backed up old index to {backup}")

    store = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    store.add_documents(docs)
    store.save(settings.vector_store_path)
    print("Done:", store.get_stats())


if __name__ == "__main__":
    main()
