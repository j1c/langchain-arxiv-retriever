"""LangChain's standard retriever test suite, replayed from VCR cassettes.

Record fresh cassettes (hits the live arXiv API, sparingly) with:

    uv run pytest tests/integration_tests --record-mode=rewrite
"""

import pytest
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_arxiv import ArxivRetriever


@pytest.mark.vcr
class TestArxivRetrieverStandard(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[ArxivRetriever]:
        return ArxivRetriever

    @property
    def retriever_query_example(self) -> str:
        return "transformer language models"
