from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANUALS_DIR = PROJECT_ROOT / "manuals"
INDEX_DIR = PROJECT_ROOT / "rag_index"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5


def list_manual_pdfs() -> list[Path]:
    return sorted(MANUALS_DIR.glob("*.pdf"))


def citation_from_metadata(metadata: dict) -> str:
    source = Path(str(metadata.get("source", "unknown"))).name
    page = metadata.get("page")
    if page is None:
        return source
    return f"{source} (page {int(page) + 1})"
