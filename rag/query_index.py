from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity

from rag.config import DEFAULT_TOP_K, INDEX_DIR, citation_from_metadata


def query_index(index_dir: Path, query: str, top_k: int):
    index_file = index_dir / "tfidf_index.joblib"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Index not found at {index_dir}. Build it first with: python -m rag.build_index"
        )

    artifact = joblib.load(index_file)
    vectorizer = artifact["vectorizer"]
    matrix = artifact["matrix"]
    records = artifact["records"]

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix).ravel()
    ranked_indices = similarities.argsort()[::-1][:top_k]

    payload = []
    for rank, index in enumerate(ranked_indices, start=1):
        record = records[int(index)]
        payload.append(
            {
                "rank": rank,
                "score": float(similarities[int(index)]),
                "citation": citation_from_metadata(record["metadata"]),
                "source": record["metadata"],
                "text": record["text"].strip(),
            }
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query FAISS manual index")
    parser.add_argument("query", type=str, help="Natural language query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hits = query_index(args.index_dir, args.query, args.top_k)
    print(json.dumps(hits, indent=2, ensure_ascii=False))
