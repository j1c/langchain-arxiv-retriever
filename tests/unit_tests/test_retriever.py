"""Hermetic unit tests: the arXiv client and HTTP session are stubbed."""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import arxiv
import pytest
import requests
from pydantic import ValidationError

from langchain_arxiv import (
    ArxivAPIError,
    ArxivRetriever,
    ArxivRetrieverError,
    FullTextFetchError,
)
from langchain_arxiv.retriever import _is_arxiv_id_list

ENTRY_ID = "http://arxiv.org/abs/2305.05665v2"
PDF_URL = "http://arxiv.org/pdf/2305.05665v2"
HTML_URL = "https://arxiv.org/html/2305.05665v2"

EXPECTED_METADATA_KEYS = {
    "entry_id",
    "arxiv_id",
    "version",
    "title",
    "authors",
    "summary",
    "published",
    "updated",
    "primary_category",
    "categories",
    "doi",
    "journal_ref",
    "comment",
    "pdf_url",
    "content_length",
    "token_estimate",
    "truncated",
    "source_format",
}


def make_result(entry_id: str = ENTRY_ID, with_pdf_link: bool = True) -> arxiv.Result:
    links = [arxiv.Result.Link(PDF_URL, title="pdf")] if with_pdf_link else []
    return arxiv.Result(
        entry_id=entry_id,
        updated=datetime(2023, 5, 31, tzinfo=timezone.utc),
        published=datetime(2023, 5, 9, tzinfo=timezone.utc),
        title="ImageBind: One Embedding Space To Bind Them All",
        authors=[arxiv.Result.Author("Rohit Girdhar"), arxiv.Result.Author("Ishan Misra")],
        summary="We present ImageBind, an approach to learn a joint embedding.",
        comment="",
        journal_ref="",
        doi="",
        primary_category="cs.CV",
        categories=["cs.CV", "cs.AI"],
        links=links,
    )


class FakeClient:
    """Stands in for arxiv.Client; records searches and serves canned results."""

    def __init__(self, results: list[arxiv.Result] | None = None, error: Exception | None = None):
        self.results_list = results if results is not None else [make_result()]
        self.error = error
        self.searches: list[arxiv.Search] = []

    def results(self, search: arxiv.Search) -> Any:
        self.searches.append(search)
        if self.error is not None:
            raise self.error
        limit = search.max_results or len(self.results_list)
        return iter(self.results_list[:limit])


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Maps URLs to responses (or exceptions); records requested URLs."""

    def __init__(self, responses: dict[str, FakeResponse | Exception] | None = None):
        self.responses = responses or {}
        self.requested: list[str] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requested.append(url)
        outcome = self.responses.get(url, FakeResponse(status_code=404))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def build_retriever(client: FakeClient, session: FakeSession | None = None, **kwargs: Any) -> ArxivRetriever:
    retriever = ArxivRetriever(**kwargs)
    retriever._client = client
    retriever._session = session or FakeSession()
    return retriever


class TestIdDetection:
    @pytest.mark.parametrize(
        "query",
        [
            "2305.05665",
            "2305.05665v2",
            "0704.0001",
            "math/0211159",
            "math.GT/0309136",
            "quant-ph/0201082v1",
            "2305.05665 1706.03762",
        ],
    )
    def test_id_queries(self, query: str) -> None:
        assert _is_arxiv_id_list(query)

    @pytest.mark.parametrize(
        "query",
        [
            "attention is all you need",
            "1234.5678",  # month 34 is invalid
            "ti:2305.05665",
            "grokking 2305.05665",  # mixed tokens fall back to search
            "",
            "   ",
        ],
    )
    def test_search_queries(self, query: str) -> None:
        assert not _is_arxiv_id_list(query)


class TestAbstractMode:
    def test_page_content_is_abstract(self) -> None:
        retriever = build_retriever(FakeClient())
        (doc,) = retriever.invoke("imagebind")
        assert doc.page_content == "We present ImageBind, an approach to learn a joint embedding."
        assert doc.metadata["source_format"] == "abstract"

    def test_metadata_schema(self) -> None:
        retriever = build_retriever(FakeClient())
        (doc,) = retriever.invoke("imagebind")
        metadata = doc.metadata
        assert set(metadata) == EXPECTED_METADATA_KEYS
        assert metadata["entry_id"] == ENTRY_ID
        assert metadata["arxiv_id"] == "2305.05665v2"
        assert metadata["version"] == 2
        assert metadata["authors"] == ["Rohit Girdhar", "Ishan Misra"]
        assert metadata["published"] == "2023-05-09"
        assert metadata["updated"] == "2023-05-31"
        assert metadata["categories"] == ["cs.CV", "cs.AI"]
        assert metadata["pdf_url"] == PDF_URL
        # Empty strings from the API normalize to None.
        assert metadata["doi"] is None
        assert metadata["journal_ref"] is None
        assert metadata["comment"] is None
        assert metadata["truncated"] is False
        assert metadata["content_length"] == len(doc.page_content)
        assert metadata["token_estimate"] == len(doc.page_content) // 4
        assert doc.id == ENTRY_ID

    def test_no_network_calls_beyond_search(self) -> None:
        session = FakeSession()
        retriever = build_retriever(FakeClient(), session=session)
        retriever.invoke("imagebind")
        assert session.requested == []

    def test_zero_results_returns_empty_list(self) -> None:
        retriever = build_retriever(FakeClient(results=[]))
        assert retriever.invoke("noresultsquery") == []


class TestKParameter:
    def test_constructor_default_flows_to_search(self) -> None:
        client = FakeClient()
        build_retriever(client).invoke("q")
        assert client.searches[0].max_results == 3

    def test_constructor_k(self) -> None:
        client = FakeClient()
        build_retriever(client, k=7).invoke("q")
        assert client.searches[0].max_results == 7

    def test_invoke_k_overrides_constructor(self) -> None:
        client = FakeClient(results=[make_result(), make_result(), make_result()])
        docs = build_retriever(client, k=3).invoke("q", k=1)
        assert client.searches[0].max_results == 1
        assert len(docs) == 1

    def test_invoke_k_must_be_positive(self) -> None:
        retriever = build_retriever(FakeClient())
        with pytest.raises(ValueError, match="k must be >= 1"):
            retriever.invoke("q", k=0)

    def test_constructor_k_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ArxivRetriever(k=0)


class TestQueryRouting:
    def test_id_query_uses_id_list(self) -> None:
        client = FakeClient()
        build_retriever(client).invoke("2305.05665v2 1706.03762")
        search = client.searches[0]
        assert search.id_list == ["2305.05665v2", "1706.03762"]
        assert search.query == ""

    def test_field_syntax_passes_through_verbatim(self) -> None:
        client = FakeClient()
        build_retriever(client).invoke('ti:"checker-board" AND cat:cs.LG')
        search = client.searches[0]
        assert search.query == 'ti:"checker-board" AND cat:cs.LG'
        assert search.id_list == []


class TestErrorContract:
    def test_arxiv_error_wrapped(self) -> None:
        error = arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 500)
        retriever = build_retriever(FakeClient(error=error))
        with pytest.raises(ArxivAPIError) as excinfo:
            retriever.invoke("q")
        assert excinfo.value.__cause__ is error
        assert isinstance(excinfo.value, ArxivRetrieverError)

    def test_connection_error_wrapped(self) -> None:
        retriever = build_retriever(FakeClient(error=requests.ConnectionError("boom")))
        with pytest.raises(ArxivAPIError):
            retriever.invoke("q")

    def test_errors_never_become_documents(self) -> None:
        retriever = build_retriever(FakeClient(error=requests.ConnectionError("boom")))
        try:
            docs = retriever.invoke("q")
        except ArxivAPIError:
            docs = None
        assert docs is None


class TestFullTextMode:
    def test_html_path(self) -> None:
        session = FakeSession(
            {HTML_URL: FakeResponse(content=b"<html><body><p>Full body text.</p></body></html>")}
        )
        retriever = build_retriever(FakeClient(), session=session, full_text=True)
        (doc,) = retriever.invoke("imagebind")
        assert doc.page_content == "Full body text."
        assert doc.metadata["source_format"] == "html"
        assert doc.metadata["summary"].startswith("We present ImageBind")

    def test_pdf_fallback_when_html_missing(self, minimal_pdf: Callable[[str], bytes]) -> None:
        session = FakeSession({PDF_URL: FakeResponse(content=minimal_pdf("Fallback pdf body"))})
        retriever = build_retriever(FakeClient(), session=session, full_text=True)
        (doc,) = retriever.invoke("imagebind")
        assert "Fallback pdf body" in doc.page_content
        assert doc.metadata["source_format"] == "pdf"
        assert session.requested == [HTML_URL, PDF_URL]

    def test_pdf_fallback_when_html_empty(self, minimal_pdf: Callable[[str], bytes]) -> None:
        session = FakeSession(
            {
                HTML_URL: FakeResponse(content=b"<html><body></body></html>"),
                PDF_URL: FakeResponse(content=minimal_pdf("Pdf body")),
            }
        )
        retriever = build_retriever(FakeClient(), session=session, full_text=True)
        (doc,) = retriever.invoke("imagebind")
        assert doc.metadata["source_format"] == "pdf"

    def test_both_paths_failing_raises(self) -> None:
        session = FakeSession(
            {
                HTML_URL: FakeResponse(status_code=404),
                PDF_URL: FakeResponse(status_code=500),
            }
        )
        retriever = build_retriever(FakeClient(), session=session, full_text=True)
        with pytest.raises(FullTextFetchError):
            retriever.invoke("imagebind")

    def test_unparseable_pdf_raises(self) -> None:
        session = FakeSession({PDF_URL: FakeResponse(content=b"not a pdf at all")})
        retriever = build_retriever(FakeClient(), session=session, full_text=True)
        with pytest.raises(FullTextFetchError):
            retriever.invoke("imagebind")

    def test_no_pdf_link_raises(self) -> None:
        client = FakeClient(results=[make_result(with_pdf_link=False)])
        retriever = build_retriever(client, session=FakeSession(), full_text=True)
        with pytest.raises(FullTextFetchError, match="no PDF link"):
            retriever.invoke("imagebind")


class TestSizeContract:
    def test_no_cap_by_default(self) -> None:
        long_summary_result = make_result()
        long_summary_result.summary = "x" * 50_000
        retriever = build_retriever(FakeClient(results=[long_summary_result]))
        (doc,) = retriever.invoke("q")
        assert len(doc.page_content) == 50_000
        assert doc.metadata["truncated"] is False

    def test_cap_is_loud(self) -> None:
        retriever = build_retriever(FakeClient(), max_content_chars=10)
        (doc,) = retriever.invoke("q")
        assert doc.page_content == "We present"
        assert doc.metadata["truncated"] is True
        assert doc.metadata["content_length"] == 10
        assert doc.metadata["token_estimate"] == 2

    def test_cap_applies_to_full_text(self) -> None:
        session = FakeSession(
            {HTML_URL: FakeResponse(content=b"<html><body><p>0123456789ABCDEF</p></body></html>")}
        )
        retriever = build_retriever(
            FakeClient(), session=session, full_text=True, max_content_chars=5
        )
        (doc,) = retriever.invoke("q")
        assert doc.page_content == "01234"
        assert doc.metadata["truncated"] is True


class TestModeConsistency:
    def test_same_metadata_keys_in_both_modes(self) -> None:
        abstract_retriever = build_retriever(FakeClient())
        session = FakeSession(
            {HTML_URL: FakeResponse(content=b"<html><body><p>Body</p></body></html>")}
        )
        full_retriever = build_retriever(FakeClient(), session=session, full_text=True)
        (abstract_doc,) = abstract_retriever.invoke("q")
        (full_doc,) = full_retriever.invoke("q")
        assert set(abstract_doc.metadata) == set(full_doc.metadata) == EXPECTED_METADATA_KEYS


class TestAsync:
    async def test_ainvoke(self) -> None:
        retriever = build_retriever(FakeClient())
        docs = await retriever.ainvoke("imagebind")
        assert len(docs) == 1
        assert docs[0].metadata["source_format"] == "abstract"

    async def test_ainvoke_with_k(self) -> None:
        client = FakeClient(results=[make_result(), make_result()])
        retriever = build_retriever(client)
        docs = await retriever.ainvoke("imagebind", k=1)
        assert client.searches[0].max_results == 1
        assert len(docs) == 1
