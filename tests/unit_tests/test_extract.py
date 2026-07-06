from collections.abc import Callable

from langchain_arxiv._extract import html_to_text, pdf_to_text

SAMPLE_HTML = """
<html>
<head><title>Page furniture</title><style>.x { color: red; }</style></head>
<body>
<nav>Skip to content | arXiv menu</nav>
<header>arXiv:2404.00001v1</header>
<article>
<h1>A Study of Things &amp; Stuff</h1>
<p>First paragraph with an inline
<math alttext="\\frac{a}{b}"><mrow><mi>a</mi><mo>/</mo><mi>b</mi></mrow></math>
formula.</p>
<div><script>var tracked = true;</script><p>Second   paragraph.</p></div>
</article>
<footer>Copyright arXiv</footer>
</body>
</html>
"""


def test_html_extracts_content_blocks() -> None:
    text = html_to_text(SAMPLE_HTML)
    assert "A Study of Things & Stuff" in text
    assert "First paragraph with an inline" in text
    assert "Second paragraph." in text


def test_html_skips_page_furniture() -> None:
    text = html_to_text(SAMPLE_HTML)
    assert "Skip to content" not in text
    assert "Copyright arXiv" not in text
    assert "arXiv:2404.00001v1" not in text
    assert "var tracked" not in text
    assert "color: red" not in text


def test_html_math_renders_as_latex_alttext() -> None:
    text = html_to_text(SAMPLE_HTML)
    assert "\\frac{a}{b}" in text
    assert "<mrow>" not in text
    # MathML child text must not leak alongside the alttext.
    assert "a / b" not in text


def test_html_block_tags_break_lines() -> None:
    text = html_to_text(SAMPLE_HTML)
    heading_line = next(line for line in text.splitlines() if "A Study" in line)
    assert heading_line == "A Study of Things & Stuff"


def test_empty_html_gives_empty_text() -> None:
    assert html_to_text("<html><body><nav>only chrome</nav></body></html>") == ""


def test_pdf_to_text(minimal_pdf: Callable[[str], bytes]) -> None:
    text = pdf_to_text(minimal_pdf("Hello arXiv full text"))
    assert "Hello arXiv full text" in text
