"""Step 1 of ingestion: parse an uploaded PDF or DOCX file into structured text blocks.

Output is a list of blocks (heading / paragraph / list_item / table_row) rather than
one flat string, so structure-based chunking can split on headings and semantic
chunking can keep paragraph boundaries.
"""

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..errors import ParseError, UnsupportedFileTypeError
from .text_utils import normalize_text

logger = logging.getLogger(__name__)

BlockKind = Literal["heading", "paragraph", "list_item", "table_row"]

SUPPORTED_EXTENSIONS = {"pdf", "docx"}
MIN_TEXT_CHARS = 20


@dataclass
class Block:
    kind: BlockKind
    text: str
    level: int = 0  # heading level (1 = top level); 0 for non-headings


@dataclass
class ParsedDocument:
    file_type: str
    blocks: list[Block] = field(default_factory=list)
    page_count: int | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


# --------------------------------------------------------------------------- file type


def detect_file_type(filename: str | None, data: bytes) -> str:
    """Accept only PDF and DOCX, checking both the extension and the file signature."""
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext == "doc":
        raise UnsupportedFileTypeError(
            "Legacy Word '.doc' files are not supported. Save the file as '.docx' and upload again."
        )
    if ext not in SUPPORTED_EXTENSIONS:
        shown = f"'.{ext}'" if ext else "(no extension)"
        raise UnsupportedFileTypeError(
            f"Unsupported file type {shown}. Only PDF (.pdf) and Word (.docx) files are accepted."
        )
    if ext == "pdf" and b"%PDF-" not in data[:1024]:
        raise UnsupportedFileTypeError("The file has a .pdf extension but is not a valid PDF.")
    if ext == "docx" and not data.startswith(b"PK\x03\x04"):
        raise UnsupportedFileTypeError("The file has a .docx extension but is not a valid DOCX.")
    return ext


def parse_document(file_type: str, data: bytes) -> ParsedDocument:
    if file_type == "pdf":
        parsed = _parse_pdf(data)
    elif file_type == "docx":
        parsed = _parse_docx(data)
    else:  # pragma: no cover - guarded by detect_file_type
        raise UnsupportedFileTypeError(f"Unsupported file type '{file_type}'.")

    if parsed.char_count < MIN_TEXT_CHARS:
        raise ParseError(
            "No extractable text found in the file. Scanned/image-only PDFs need OCR, "
            "which is not supported."
        )
    return parsed


# --------------------------------------------------------------------------- PDF

_PAGE_NOISE_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"^(\d+(\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.)\s+(?=\S)")
_LIST_RE = re.compile(r"^([-•*–▪●◦]|\(?[a-z0-9]{1,3}[.)])\s+")
_SMALL_WORDS = {
    "a", "an", "and", "or", "of", "the", "to", "for", "in", "on", "at", "by", "with",
    "from", "per", "vs", "&", "-", "|",
}


def _looks_like_heading(line: str) -> bool:
    """Heuristic for PDF text, which has no styles: short, no final punctuation,
    and mostly Capitalised / ALL CAPS words (optionally numbered: '2.1 Eligibility')."""
    if len(line) > 90 or line[-1] in ".!?;,":
        return False
    words = _NUMBERED_RE.sub("", line).split()
    if not words or len(words) > 12:
        return False
    alpha_words = [w for w in words if w[0].isalpha() and w.lower() not in _SMALL_WORDS]
    if not alpha_words:
        return False
    if line.isupper():
        return True
    capitalised = sum(1 for w in alpha_words if w[0].isupper())
    return capitalised / len(alpha_words) >= 0.6


def _blocks_from_lines(lines: list[str]) -> list[Block]:
    """Rebuild paragraphs from PDF text lines (PDF extraction breaks lines mid-sentence)."""
    blocks: list[Block] = []
    current: Block | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.text.strip():
            blocks.append(current)
        current = None

    for raw in lines:
        line = normalize_text(raw)
        if not line:
            flush()  # blank line = paragraph break
            continue
        if _PAGE_NOISE_RE.match(line):
            continue
        previous_ended = current is None or current.text.rstrip()[-1:] in ".!?:"
        if previous_ended and _looks_like_heading(line):
            flush()
            # "2." -> level 1, "2.1" -> level 2, unnumbered -> level 1
            level = line.split()[0].rstrip(".").count(".") + 1 if _NUMBERED_RE.match(line) else 1
            blocks.append(Block("heading", line, level=min(level, 6)))
            continue
        if _LIST_RE.match(line):
            flush()
            current = Block("list_item", line)
            continue
        if current is None:
            current = Block("paragraph", line)
        elif current.text.endswith("-") and line[:1].islower():
            current.text = current.text[:-1] + line  # de-hyphenate "remo-\nte"
        else:
            current.text = f"{current.text} {line}"
    flush()
    return blocks


def _parse_pdf(data: bytes) -> ParsedDocument:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise ParseError("The PDF is password-protected and cannot be read.") from exc
        lines: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            lines.extend(page_text.splitlines())
            lines.append("")  # page break acts as paragraph break
        return ParsedDocument("pdf", _blocks_from_lines(lines), page_count=len(reader.pages))
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf raises many different errors
        raise ParseError(f"Could not read the PDF file: {exc}") from exc


# --------------------------------------------------------------------------- DOCX


def _docx_heading_level(style_name: str) -> int | None:
    name = style_name.lower()
    if name == "title":
        return 1
    if name.startswith("heading"):
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 1
    return None


def _is_docx_list(paragraph, style_name: str) -> bool:
    if "list" in style_name.lower():
        return True
    p_pr = paragraph._p.pPr  # noqa: SLF001 - python-docx has no public API for numbering
    return p_pr is not None and p_pr.numPr is not None


def _is_bold_heading(paragraph, text: str) -> bool:
    """Paragraphs without a heading style but fully bold, short, no final period."""
    runs = [r for r in paragraph.runs if r.text.strip()]
    return bool(runs) and all(r.bold for r in runs) and len(text) <= 90 and text[-1] not in ".!?;,"


def _table_rows_as_blocks(table) -> list[Block]:
    """Render each row as 'Header: value; Header: value' so a row keeps its meaning alone."""
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = normalize_text(cell.text)
            if not cells or text != cells[-1]:  # merged cells repeat their text
                cells.append(text)
        rows.append(cells)
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    blocks: list[Block] = []
    if not body:
        return [Block("table_row", " | ".join(c for c in header if c))]
    for cells in body:
        if len(cells) == len(header):
            parts = [f"{h}: {c}" if h else c for h, c in zip(header, cells) if c]
        else:
            parts = [c for c in cells if c]
        if parts:
            blocks.append(Block("table_row", "; ".join(parts)))
    return blocks


def _parse_docx(data: bytes) -> ParsedDocument:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Could not read the DOCX file: {exc}") from exc

    blocks: list[Block] = []
    # Walk the body in document order so tables stay in place between paragraphs.
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, doc)
            text = normalize_text(paragraph.text)
            if not text:
                continue
            try:
                style_name = paragraph.style.name if paragraph.style is not None else ""
            except Exception:  # noqa: BLE001
                style_name = ""
            level = _docx_heading_level(style_name or "")
            if level is not None or _is_bold_heading(paragraph, text):
                blocks.append(Block("heading", text, level=level or 2))
            elif _is_docx_list(paragraph, style_name or ""):
                blocks.append(Block("list_item", text))
            else:
                blocks.append(Block("paragraph", text))
        elif child.tag == qn("w:tbl"):
            blocks.extend(_table_rows_as_blocks(Table(child, doc)))
    return ParsedDocument("docx", blocks)
