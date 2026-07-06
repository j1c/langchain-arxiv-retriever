# langchain-arxiv-retriever

A [LangChain](https://github.com/langchain-ai/langchain) retriever for the [arXiv](https://arxiv.org) API.

```bash
pip install langchain-arxiv-retriever
```

```python
from langchain_arxiv import ArxivRetriever

retriever = ArxivRetriever()
docs = retriever.invoke('ti:"attention is all you need"')
```

One `Document` per paper, in arXiv's relevance order. Three knobs, one metadata schema, typed errors, nothing silent.

**Documentation:** <https://j1c.github.io/langchain-arxiv-retriever/>

## Usage

### Abstracts (default)

`page_content` is the paper's abstract — fast, no downloads:

```python
retriever = ArxivRetriever()          # k=3 papers by default
docs = retriever.invoke("grokking transformers", k=10)   # per-call override
```

### Full text

`page_content` is the whole paper. Text comes from arXiv's native HTML rendering when available and from the PDF otherwise; `metadata["source_format"]` tells you which (`"html"` or `"pdf"`):

```python
full = ArxivRetriever(full_text=True, k=2)
docs = full.invoke("2404.19756")
docs[0].metadata["content_length"]   # 140369 — never silently cut
```

Need both modes? Instances are stateless and cheap — make two. For runtime switching inside a chain, use LangChain's standard [`configurable_fields`](https://docs.langchain.com) mechanism on `full_text`.

### Fetch by arXiv ID

A query consisting only of arXiv identifiers is fetched directly instead of searched:

```python
retriever.invoke("1706.03762")            # new-style ID
retriever.invoke("quant-ph/9508027")      # old-style ID
retriever.invoke("2305.05665 2404.19756") # several at once
```

### Query semantics

Queries go to the arXiv API **verbatim**. Everything in the [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html) works: `ti:`, `abs:`, `au:`, `cat:`, `all:`, `AND`/`OR`/`ANDNOT`, quoted phrases.

Be aware that arXiv search is **lexical, not semantic**. Keyword-style queries retrieve well; conversational questions don't.

## Metadata schema

Identical keys in both modes, snake_case, JSON-serializable:

| Key | Type | Notes |
|---|---|---|
| `entry_id` | `str` | Canonical abs URL, e.g. `http://arxiv.org/abs/2305.05665v2` (also the `Document.id`) |
| `arxiv_id` | `str` | Short ID with version, e.g. `2305.05665v2` |
| `version` | `int \| None` | Parsed from the ID |
| `title`, `summary` | `str` | `summary` is the abstract, in both modes |
| `authors` | `list[str]` | |
| `published`, `updated` | `str` | ISO dates: first submission / this version |
| `primary_category` | `str` | e.g. `cs.LG` |
| `categories` | `list[str]` | |
| `doi`, `journal_ref`, `comment` | `str \| None` | `None` when absent |
| `pdf_url` | `str \| None` | |
| `content_length` | `int` | Length of `page_content` in chars |
| `token_estimate` | `int` | `content_length // 4` |
| `truncated` | `bool` | `True` only if `max_content_chars` fired |
| `source_format` | `str` | `"abstract"`, `"html"`, or `"pdf"` |


## Why not `langchain-community`'s `ArxivRetriever`?

This package exists because [langchain-community](https://github.com/langchain-ai/langchain-community) has been [deprecated and is no longer maintained](https://github.com/langchain-ai/langchain-community/issues/674), and uses deprecated functions from [arxiv-py](https://github.com/lukasschmidinger/arxiv-py). Lastly, the community implementation has accumulated footguns:

| | `langchain-community` | `langchain-arxiv-retriever` |
|---|---|---|
| "Full documents" mode | Silently truncated at **4,000 chars** (`doc_content_chars_max` default) — about a page and a half | Whole text by default; truncation only if you opt in, and then it's flagged in metadata |
| Full-text source | PDF only, parsed with PyMuPDF (**AGPL**) | arXiv's native HTML when it exists (most papers since Dec 2023), `pypdf` (BSD) fallback — permissive licenses only |
| arXiv field syntax (`ti:`, `cat:`, …) | Strips `:` and `-` from queries in full-text mode, breaking it | Queries pass through verbatim |
| Query length | Silently cut at 300 chars | Untouched |
| Metadata | Different keys, casing, and value types per mode | One snake_case schema, both modes |
| Errors | Returned as a fake `Document` (`"Arxiv exception: …"`) that flows into your prompt | Raised, always, as typed exceptions |
| API traffic | Fetches 100 results per request regardless of `k`; re-downloads PDFs to disk on every call | Fetches what `k` needs; downloads stay in memory; nothing written to disk |
