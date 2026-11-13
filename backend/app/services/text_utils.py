"""Small text helpers shared by parsing and chunking."""

import re
import unicodedata

_WS_RE = re.compile(r"[ \t  -​　]+")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Tokens that end with a period but do not end a sentence.
_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.", "no.", "nos.",
    "approx.", "dept.", "inc.", "ltd.", "co.", "corp.", "st.", "fig.", "art.", "sec.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.",
    "nov.", "dec.", "p.m.", "a.m.", "u.s.",
}
_INITIAL_RE = re.compile(r"(?:^|\s)[A-Z]\.$")  # "John F." / single-letter initials
_OPENING_PUNCT = "\"'([{“‘«"


def normalize_text(text: str) -> str:
    """NFC-normalise (important for Vietnamese) and collapse horizontal whitespace."""
    text = unicodedata.normalize("NFC", text)
    return _WS_RE.sub(" ", text).strip()


def _starts_like_sentence(piece: str) -> bool:
    stripped = piece.lstrip(_OPENING_PUNCT)
    if not stripped:
        return False
    first = stripped[0]
    return first.isupper() or first.isdigit()


def _ends_with_abbreviation(sentence: str) -> bool:
    last_token = sentence.rsplit(" ", 1)[-1].lower()
    return last_token in _ABBREVIATIONS or bool(_INITIAL_RE.search(sentence))


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences.

    Regex split on ., ! or ? followed by whitespace, then re-join false splits
    (abbreviations such as "e.g." or a next piece that starts in lowercase).
    """
    text = normalize_text(text)
    if not text:
        return []
    sentences: list[str] = []
    for piece in _SENTENCE_END_RE.split(text):
        piece = piece.strip()
        if not piece:
            continue
        if sentences and (_ends_with_abbreviation(sentences[-1]) or not _starts_like_sentence(piece)):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences
