"""Behavior tests against real (recorded) arXiv traffic.

Record fresh cassettes with:

    uv run pytest tests/integration_tests --record-mode=rewrite
"""

import pytest

from langchain_arxiv import ArxivRetriever


@pytest.mark.vcr
def test_fetch_by_id_returns_that_paper() -> None:
    retriever = ArxivRetriever(k=5)
    docs = retriever.invoke("1706.03762")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["title"] == "Attention Is All You Need"
    assert doc.metadata["arxiv_id"].startswith("1706.03762")
    assert doc.metadata["source_format"] == "abstract"
    assert doc.page_content == doc.metadata["summary"]


@pytest.mark.vcr
def test_full_text_html_source() -> None:
    # KAN (2024): has a native arXiv HTML rendering.
    retriever = ArxivRetriever(full_text=True, k=1)
    docs = retriever.invoke("2404.19756")
    (doc,) = docs
    assert doc.metadata["source_format"] == "html"
    assert doc.metadata["truncated"] is False
    assert doc.metadata["content_length"] == len(doc.page_content) > 50_000
    assert "Kolmogorov" in doc.page_content


@pytest.mark.vcr
def test_full_text_pdf_fallback() -> None:
    # Shor (1995): pre-dates native HTML, so full text must come from the PDF.
    retriever = ArxivRetriever(full_text=True, k=1)
    docs = retriever.invoke("quant-ph/9508027")
    (doc,) = docs
    assert doc.metadata["source_format"] == "pdf"
    assert doc.metadata["content_length"] > 10_000
    assert "factoring" in doc.page_content.lower()
