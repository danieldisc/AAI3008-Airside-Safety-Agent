from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    INDEX_DIR,
    list_manual_pdfs,
)


def load_pdf_documents(pdf_paths: list[Path]):
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["manual"] = pdf_path.name
        documents.extend(pages)
    return documents


def build_index(output_dir: Path) -> None:
    pdf_paths = list_manual_pdfs()
    if not pdf_paths:
        raise FileNotFoundError("No PDF manuals found under ./manuals")

    print(f"Found {len(pdf_paths)} manuals")
    documents = load_pdf_documents(pdf_paths)
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    records = [
        {
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
        if chunk.page_content and chunk.page_content.strip()
    ]
    texts = [item["text"] for item in records]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform(texts)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "matrix": matrix,
            "records": records,
        },
        output_dir / "tfidf_index.joblib",
    )
    with (output_dir / "index_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "tfidf",
                "chunks": len(records),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            f,
            indent=2,
        )
    print(f"Saved index to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from manuals")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INDEX_DIR,
        help="Where to save the FAISS index",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(args.output_dir)
