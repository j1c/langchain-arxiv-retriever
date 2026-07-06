"""LangChain retriever for the arXiv API."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

import arxiv
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import run_in_executor
from pydantic import Field, PrivateAttr
from pypdf.errors import PyPdfError

from langchain_arxiv._extract import html_to_text, pdf_to_text
from langchain_arxiv.exceptions import ArxivAPIError, FullTextFetchError

try:
    _VERSION = version("langchain-arxiv-retriever")
except PackageNotFoundError:  # pragma: no cover - only hit in odd dev setups
    _VERSION = "0.0.0"

_USER_AGENT = f"langchain-arxiv-retriever/{_VERSION}"
_DOWNLOAD_TIMEOUT_SECONDS = 60.0

# arXiv identifier formats; see https://info.arxiv.org/help/arxiv_identifier.html
# New style (since 2007-04): YYMM.NNNNN with an optional version suffix.
_NEW_STYLE_ID = re.compile(r"^\d{2}(0[1-9]|1[0-2])\.\d{4,5}(v\d+)?$")
# Old style (pre 2007-04): archive[.SC]/YYMMNNN with an optional version suffix.
_OLD_STYLE_ID = re.compile(r"^[a-z][a-z-]*(\.[A-Z]{2})?/\d{2}(0[1-9]|1[0-2])\d{3}(v\d+)?$")

_VERSION_SUFFIX = re.compile(r"v(\d+)$")


def _is_arxiv_id_list(query: str) -> bool:
    """Decide whether a query should be treated as a list of arXiv IDs.

    Parameters
    ----------
    query : str
        The raw query string passed to the retriever.

    Returns
    -------
    bool
        True if every whitespace-separated token is an arXiv identifier
        (new style ``2305.05665v2`` or old style ``math/0211159``), in
        which case the query is fetched by ID instead of searched.
    """
    tokens = query.split()
    return bool(tokens) and all(
        _NEW_STYLE_ID.match(token) or _OLD_STYLE_ID.match(token) for token in tokens
    )


def _paper_metadata(result: arxiv.Result) -> dict[str, Any]:
    """Build the paper-identity portion of the metadata schema.

    Parameters
    ----------
    result : arxiv.Result
        A search result from the arXiv API client.

    Returns
    -------
    dict[str, Any]
        Paper-identity metadata: every schema key except the content
        descriptors, which depend on the retrieval mode.
    """
    arxiv_id = result.get_short_id()
    version_match = _VERSION_SUFFIX.search(arxiv_id)
    return {
        "entry_id": result.entry_id,
        "arxiv_id": arxiv_id,
        "version": int(version_match.group(1)) if version_match else None,
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "summary": result.summary,
        "published": result.published.date().isoformat(),
        "updated": result.updated.date().isoformat(),
        "primary_category": result.primary_category,
        "categories": list(result.categories),
        "doi": result.doi or None,
        "journal_ref": result.journal_ref or None,
        "comment": result.comment or None,
        "pdf_url": result.pdf_url,
    }


class ArxivRetriever(BaseRetriever):
    """Retriever for papers on `arXiv <https://arxiv.org>`__.

    Returns one ``Document`` per paper, in arXiv's relevance order. By
    default ``page_content`` is the paper's abstract; with
    ``full_text=True`` it is the paper's full text, extracted from arXiv's
    native HTML rendering when one exists (most papers submitted since
    December 2023) and from the PDF via ``pypdf`` otherwise.

    Parameters
    ----------
    k : int, default 3
        Number of papers to return. Can be overridden per call:
        ``retriever.invoke(query, k=10)``.
    full_text : bool, default False
        If True, ``page_content`` is the paper's full text (native arXiv
        HTML when available, PDF otherwise). If False, the abstract.
    max_content_chars : int, optional
        Hard cap on ``page_content`` length. When the cap fires, the cut
        is recorded in ``metadata["truncated"]``. Default: no cap.

    Raises
    ------
    ArxivAPIError
        If the arXiv search API request fails.
    FullTextFetchError
        In full-text mode, if neither the native HTML nor the PDF yields
        text for a paper.

    Notes
    -----
    * Queries are passed to the arXiv API verbatim. Field syntax
      (``ti:``, ``abs:``, ``au:``, ``cat:``, ``all:``) and boolean
      operators work as documented in the `arXiv API manual
      <https://info.arxiv.org/help/api/user-manual.html>`__. arXiv's
      search is lexical, not semantic; keyword-style queries retrieve
      better than natural-language questions.
    * A query consisting only of arXiv identifiers (for example
      ``"2305.05665"`` or ``"math/0211159v1"``) is fetched by ID
      instead of searched.
    * Content is never truncated silently. Set ``max_content_chars``
      to cap it; when the cap fires, ``metadata["truncated"]`` is
      ``True``.
    * An error is never returned as a ``Document``. A query with zero
      matches returns an empty list.
    * The retriever is stateless: nothing is cached and nothing is
      written to disk. Search requests observe arXiv's requested
      courtesy rate (one request per three seconds) via the ``arxiv``
      client.

    The metadata schema is identical in both modes: ``entry_id``,
    ``arxiv_id``, ``version``, ``title``, ``authors``, ``summary``,
    ``published``, ``updated``, ``primary_category``, ``categories``,
    ``doi``, ``journal_ref``, ``comment``, ``pdf_url``, plus the content
    descriptors ``content_length``, ``token_estimate`` (a rough
    ``chars / 4`` heuristic), ``truncated``, and ``source_format`` (one of
    ``"abstract"``, ``"html"``, ``"pdf"``).

    Examples
    --------
    >>> from langchain_arxiv import ArxivRetriever
    >>> retriever = ArxivRetriever()
    >>> docs = retriever.invoke('ti:"attention is all you need"')
    >>> docs = retriever.invoke("2305.05665")  # arXiv IDs fetch directly
    >>> docs = retriever.invoke("grokking", k=10)  # per-call k override

    Whole papers instead of abstracts:

    >>> full = ArxivRetriever(full_text=True, k=2)
    >>> docs = full.invoke("kolmogorov-arnold networks")
    >>> docs[0].metadata["source_format"]
    'html'
    """

    k: int = Field(default=3, ge=1)
    """Number of papers to return. Override per call: ``retriever.invoke(query, k=10)``."""

    full_text: bool = False
    """If ``True``, ``page_content`` is the paper's full text (native arXiv
    HTML when available, PDF otherwise). If ``False``, the abstract."""

    max_content_chars: Optional[int] = Field(default=None, ge=1)
    """Optional hard cap on ``page_content`` length. When the cap fires, the
    cut is recorded in ``metadata["truncated"]``. Default: no cap."""

    _client: arxiv.Client = PrivateAttr()
    _session: requests.Session = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        """Initialize the private arXiv client and HTTP session.

        Parameters
        ----------
        context : Any
            Pydantic post-init context; forwarded to the parent hook.
        """
        super().model_post_init(context)
        self._client = arxiv.Client()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def _search(self, query: str, k: int) -> list[arxiv.Result]:
        """Run an arXiv API search (or ID fetch) and collect the results.

        Parameters
        ----------
        query : str
            Search query, or whitespace-separated arXiv IDs.
        k : int
            Maximum number of results to fetch.

        Returns
        -------
        list[arxiv.Result]
            Up to ``k`` results; empty when nothing matches.

        Raises
        ------
        ArxivAPIError
            If the underlying API request fails.
        """
        if _is_arxiv_id_list(query):
            search = arxiv.Search(id_list=query.split(), max_results=k)
        else:
            search = arxiv.Search(query=query, max_results=k)
        # The client's page_size is how many entries each API request fetches;
        # left at its default of 100 a k=3 query would pull 100 entries. The
        # client stays shared so arXiv's courtesy delay spans invokes. (Under
        # concurrent invokes this assignment races, but page size only affects
        # fetch granularity, never which results are returned.)
        self._client.page_size = min(k, 100)
        try:
            return list(self._client.results(search))
        except (arxiv.ArxivError, requests.RequestException) as exc:
            raise ArxivAPIError(
                f"arXiv API request failed for query {query!r}: {exc}"
            ) from exc

    def _fetch_full_text(self, result: arxiv.Result) -> tuple[str, str]:
        """Fetch a paper's full text.

        Tries arXiv's native HTML rendering first, then falls back to the
        PDF.

        Parameters
        ----------
        result : arxiv.Result
            The paper to fetch.

        Returns
        -------
        tuple[str, str]
            The extracted text and its source format, ``"html"`` or
            ``"pdf"``.

        Raises
        ------
        FullTextFetchError
            If neither the HTML nor the PDF path yields text.
        """
        arxiv_id = result.get_short_id()
        html_failure: Exception | None = None
        try:
            response = self._session.get(
                f"https://arxiv.org/html/{arxiv_id}",
                timeout=_DOWNLOAD_TIMEOUT_SECONDS,
            )
            if response.status_code == requests.codes.ok:
                text = html_to_text(response.content.decode("utf-8", errors="replace"))
                if text:
                    return text, "html"
        except requests.RequestException as exc:
            html_failure = exc

        if not result.pdf_url:
            raise FullTextFetchError(
                f"No full text available for {arxiv_id}: no native HTML "
                "rendering and the API result carries no PDF link."
            ) from html_failure
        try:
            response = self._session.get(
                result.pdf_url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            text = pdf_to_text(response.content)
        except (requests.RequestException, PyPdfError, ValueError) as exc:
            raise FullTextFetchError(
                f"Failed to fetch full text for {arxiv_id}: no native HTML "
                f"rendering, and the PDF fallback failed: {exc}"
            ) from exc
        if not text:
            raise FullTextFetchError(
                f"Failed to extract full text for {arxiv_id}: no native HTML "
                "rendering, and the PDF contains no extractable text."
            )
        return text, "pdf"

    def _to_document(self, result: arxiv.Result) -> Document:
        """Convert an API result into a ``Document`` for the current mode.

        Parameters
        ----------
        result : arxiv.Result
            The paper to convert.

        Returns
        -------
        Document
            ``page_content`` per the retrieval mode, with the full
            metadata schema attached and any content cap applied loudly.

        Raises
        ------
        FullTextFetchError
            In full-text mode, when the paper's text cannot be fetched.
        """
        if self.full_text:
            content, source_format = self._fetch_full_text(result)
        else:
            content, source_format = result.summary, "abstract"

        truncated = False
        if self.max_content_chars is not None and len(content) > self.max_content_chars:
            content = content[: self.max_content_chars]
            truncated = True

        metadata = _paper_metadata(result)
        metadata.update(
            {
                "content_length": len(content),
                "token_estimate": len(content) // 4,
                "truncated": truncated,
                "source_format": source_format,
            }
        )
        return Document(id=metadata["entry_id"], page_content=content, metadata=metadata)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: Optional[int] = None,
    ) -> list[Document]:
        """Retrieve documents for a query (sync path).

        Parameters
        ----------
        query : str
            Search query, or whitespace-separated arXiv IDs.
        run_manager : CallbackManagerForRetrieverRun
            Callback manager supplied by ``BaseRetriever.invoke``.
        k : int, optional
            Per-call override of the constructor ``k``.

        Returns
        -------
        list[Document]
            One ``Document`` per paper; empty when nothing matches.

        Raises
        ------
        ValueError
            If ``k`` is given and is less than 1.
        ArxivAPIError
            If the arXiv API request fails.
        FullTextFetchError
            In full-text mode, when a paper's text cannot be fetched.
        """
        if k is not None and k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        results = self._search(query, self.k if k is None else k)
        return [self._to_document(result) for result in results]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        k: Optional[int] = None,
    ) -> list[Document]:
        """Retrieve documents for a query (async path).

        Runs the sync implementation in a thread executor — the
        underlying arXiv client is sync-only — keeping the event loop
        free and preserving the per-call ``k`` override.

        Parameters
        ----------
        query : str
            Search query, or whitespace-separated arXiv IDs.
        run_manager : AsyncCallbackManagerForRetrieverRun
            Callback manager supplied by ``BaseRetriever.ainvoke``.
        k : int, optional
            Per-call override of the constructor ``k``.

        Returns
        -------
        list[Document]
            One ``Document`` per paper; empty when nothing matches.

        Raises
        ------
        ValueError
            If ``k`` is given and is less than 1.
        ArxivAPIError
            If the arXiv API request fails.
        FullTextFetchError
            In full-text mode, when a paper's text cannot be fetched.
        """
        return await run_in_executor(
            None,
            self._get_relevant_documents,
            query,
            run_manager=run_manager.get_sync(),
            k=k,
        )
