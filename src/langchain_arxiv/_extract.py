"""Full-text extraction from arXiv HTML and PDF sources.

Private module: these helpers are implementation details of
``ArxivRetriever`` and may change without notice.
"""

from __future__ import annotations

import io
import re
from html.parser import HTMLParser

from pypdf import PdfReader

_SKIP_TAGS = {
    "button",
    "footer",
    "header",
    "nav",
    "noscript",
    "script",
    "style",
    "svg",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}

_INTRALINE_WS = re.compile(r"[ \t\r\f\v]+")


class _HTMLTextExtractor(HTMLParser):
    """Extracts readable text from arXiv's LaTeXML-generated HTML.

    Page furniture (navigation, scripts, headers/footers) is skipped.
    ``<math>`` elements are rendered as their ``alttext`` attribute — the
    original LaTeX source, which LaTeXML preserves — rather than as the
    concatenated text of their MathML tree.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._math_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track skip/math regions and emit line breaks for opened block tags."""
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "math":
            if self._math_depth == 0:
                alttext = dict(attrs).get("alttext")
                if alttext:
                    self._chunks.append(f" {alttext} ")
            self._math_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Unwind skip/math tracking and emit line breaks for closed block tags."""
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "math":
            self._math_depth = max(0, self._math_depth - 1)
        elif tag in _BLOCK_TAGS and not self._skip_depth:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        """Accumulate text found outside skipped and math regions."""
        if self._skip_depth or self._math_depth:
            return
        self._chunks.append(data)

    def text(self) -> str:
        """Assemble the extracted text.

        Returns
        -------
        str
            Accumulated text with intra-line whitespace collapsed and
            runs of blank lines reduced to single separators.
        """
        lines = [
            _INTRALINE_WS.sub(" ", line).strip()
            for line in "".join(self._chunks).splitlines()
        ]
        collapsed: list[str] = []
        for line in lines:
            if line:
                collapsed.append(line)
            elif collapsed and collapsed[-1]:
                collapsed.append("")
        return "\n".join(collapsed).strip()


def html_to_text(html: str) -> str:
    """Extract plain text from an arXiv HTML page.

    Parameters
    ----------
    html : str
        The page markup.

    Returns
    -------
    str
        Readable text; empty when the page holds no content outside
        skipped elements.
    """
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    extractor.close()
    return extractor.text()


def pdf_to_text(data: bytes) -> str:
    """Extract plain text from PDF bytes, page by page.

    Parameters
    ----------
    data : bytes
        The PDF file contents.

    Returns
    -------
    str
        Text of all pages joined by newlines; empty when the PDF has no
        extractable text layer.

    Raises
    ------
    pypdf.errors.PyPdfError
        If the bytes cannot be parsed as a PDF.
    """
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
