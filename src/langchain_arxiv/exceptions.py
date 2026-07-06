"""Typed exceptions for langchain-arxiv-retriever.

The retriever's error contract: failures raise, always. An error is never
encoded as a ``Document`` in the result list. Callers who want resilience
can use the standard Runnable machinery (``.with_retry()``,
``.with_fallbacks()``).
"""

from __future__ import annotations


class ArxivRetrieverError(Exception):
    """Base exception for all langchain-arxiv-retriever failures."""


class ArxivAPIError(ArxivRetrieverError):
    """The arXiv search API request failed.

    Wraps the underlying cause (an ``arxiv.ArxivError`` or a
    ``requests`` exception) as ``__cause__``.
    """


class FullTextFetchError(ArxivRetrieverError):
    """Fetching or extracting a paper's full text failed.

    Raised in full-text mode when neither the native arXiv HTML nor the
    PDF yields text for a paper. The underlying cause, when there is one,
    is available as ``__cause__``.
    """
