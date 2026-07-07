# Example Usage

These examples show `ArxivRetriever` composed into LLM applications. They
use [`init_chat_model`](https://docs.langchain.com/oss/python/langchain/models)
so you can swap in any chat model provider.

```bash
pip install langchain-arxiv-retriever langchain langchain-openai
```

## RAG over abstracts

The classic retrieval chain: stuff the retrieved abstracts into a prompt as
context. Abstract mode keeps this fast — no PDFs are downloaded.

```python
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_arxiv import ArxivRetriever

llm = init_chat_model("openai:gpt-5.4-mini")
retriever = ArxivRetriever(k=5)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.
Cite papers by their arXiv ID.

Context: {context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata['arxiv_id']}] {doc.metadata['title']}\n{doc.page_content}"
        for doc in docs
    )


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("ImageBind joint embedding")
```

## Rewriting natural-language questions into arXiv queries

arXiv search is lexical, not semantic — conversational questions retrieve
poorly. The retriever deliberately passes queries through verbatim, so the
fix belongs in the chain: have the LLM translate the question into arXiv
query syntax first.

```python
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_arxiv import ArxivRetriever

llm = init_chat_model("openai:gpt-5.4-mini")
retriever = ArxivRetriever(k=5)

rewrite_prompt = ChatPromptTemplate.from_template(
    """Convert the question into one concise arXiv search query.
Prefer a few precise keywords over full sentences. Use arXiv field
syntax where it helps: ti:"..." for title terms, abs:"..." for
abstract terms, au:"..." for authors, cat:cs.LG for categories;
combine with AND / OR / ANDNOT.
Return only the query, nothing else.

Question: {question}"""
)

rewrite = rewrite_prompt | llm | StrOutputParser()
search = rewrite | retriever

docs = search.invoke(
    {"question": "why do transformers suddenly generalize after long training?"}
)
```

Combined with the RAG chain above, the rewriter replaces the bare retriever:

```python
chain = (
    {"context": search | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## Building structured arXiv queries with an LLM

The freeform rewrite above is the lightest touch. For tighter control,
have the model emit a *structured* query and assemble the arXiv syntax in
code: the fields stay machine-checkable, and the assembly is
deterministic instead of hoping the model quotes and parenthesizes
correctly.

```python
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_arxiv import ArxivRetriever


class ArxivQuery(BaseModel):
    """A structured arXiv search query."""

    keywords: list[str] = Field(
        description="2-5 precise technical keywords or short phrases"
    )
    categories: list[str] = Field(
        default_factory=list,
        description='arXiv category codes, e.g. ["cs.LG", "stat.ML"]',
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Author names, only if the question names them",
    )


def to_query_string(query: ArxivQuery) -> str:
    groups = []
    if query.keywords:
        groups.append("(" + " OR ".join(f'abs:"{k}"' for k in query.keywords) + ")")
    if query.categories:
        groups.append("(" + " OR ".join(f"cat:{c}" for c in query.categories) + ")")
    if query.authors:
        groups.append("(" + " OR ".join(f'au:"{a}"' for a in query.authors) + ")")
    return " AND ".join(groups)


structured_llm = init_chat_model("openai:gpt-5.4-mini").with_structured_output(
    ArxivQuery
)
retriever = ArxivRetriever(k=10)


def search(question: str):
    query = structured_llm.invoke(
        "Extract an arXiv search query from this question: " + question
    )
    return retriever.invoke(to_query_string(query))


docs = search("recent work on state space models for long-context language modeling")
```

For a question like the one above, the model might produce
`keywords=["state space models", "long context"]` and
`categories=["cs.CL", "cs.LG"]`, which assembles to:

```text
(abs:"state space models" OR abs:"long context") AND (cat:cs.CL OR cat:cs.LG)
```

## Reranking arXiv results

arXiv ranks results lexically. When you need semantic precision, use the
strategy this retriever was designed to compose with: **over-fetch, then
rerank locally** against the original question. Here the reranker is a
plain embeddings cosine similarity — swap in a dedicated reranker (e.g.
Cohere Rerank) at the same seam if you have one.

```python
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document

from langchain_arxiv import ArxivRetriever

embeddings = init_embeddings("openai:text-embedding-3-small")
retriever = ArxivRetriever(k=25)  # over-fetch; arXiv's lexical ranking is pass one


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def rerank(question: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    query_vector = embeddings.embed_query(question)
    doc_vectors = embeddings.embed_documents(
        [f"{doc.metadata['title']}\n{doc.page_content}" for doc in docs]
    )
    scored = sorted(
        zip(docs, doc_vectors),
        key=lambda pair: cosine(query_vector, pair[1]),
        reverse=True,
    )
    return [doc for doc, _ in scored[:top_n]]


question = "how do transformers learn in-context without weight updates?"
docs = rerank(question, retriever.invoke(question))
```

The two query-side examples and this one compose: rewrite (or structure)
the query for **recall**, then rerank the over-fetched results for
**precision** — all in the chain, while the retriever itself stays
deterministic.

## Using the retriever as agent tools

For agents, expose two tools built on two instances — a cheap search over
abstracts and a full-text fetch for the paper the model selects. The
docstrings teach the model to write keyword queries, so query quality
becomes the agent's job:

```python
from langchain.agents import create_agent
from langchain.tools import tool

from langchain_arxiv import ArxivRetriever

search_retriever = ArxivRetriever(k=5)
full_text_retriever = ArxivRetriever(full_text=True, max_content_chars=40_000)


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv and return matching papers' IDs, titles, and abstracts.

    Write concise keyword queries, not questions. arXiv field syntax is
    supported: ti:"..." (title), abs:"..." (abstract), au:"..." (author),
    cat:cs.LG (category), combined with AND / OR / ANDNOT.
    """
    docs = search_retriever.invoke(query)
    if not docs:
        return "No results."
    return "\n\n".join(
        f"{doc.metadata['arxiv_id']} — {doc.metadata['title']}\n{doc.page_content}"
        for doc in docs
    )


@tool
def read_arxiv_paper(arxiv_id: str) -> str:
    """Fetch the full text of one paper by its arXiv ID (e.g. '2404.19756')."""
    (doc,) = full_text_retriever.invoke(arxiv_id, k=1)
    text = doc.page_content
    if doc.metadata["truncated"]:
        text += "\n\n[Text truncated at 40,000 characters.]"
    return text


agent = create_agent("openai:gpt-5.4-mini", [search_arxiv, read_arxiv_paper])

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Find the paper that introduced Kolmogorov-Arnold "
                "networks and summarize what it claims about scaling laws.",
            }
        ]
    }
)
```

The `max_content_chars` cap keeps a full paper from blowing out the agent's
context window — and because truncation is recorded in
`metadata["truncated"]`, the tool can tell the model the text was cut
instead of letting it assume it read everything.

## Switching modes at runtime

An instance's mode is fixed at construction, and instances are stateless
and cheap — needing both usually means making two. If a single chain must
switch modes per call, use LangChain's standard
[`configurable_fields`](https://docs.langchain.com) mechanism, which routes
through the `RunnableConfig` that chains already propagate:

```python
from langchain_core.runnables import ConfigurableField

from langchain_arxiv import ArxivRetriever

retriever = ArxivRetriever().configurable_fields(
    full_text=ConfigurableField(id="full_text")
)

retriever.invoke("2404.19756")  # abstract
retriever.invoke("2404.19756", config={"configurable": {"full_text": True}})  # whole paper
```
