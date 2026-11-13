"""Step 2 of ingestion: chunking.

Two strategies, both stored so they can be compared on the same documents:

* semantic_chunk  - split where the meaning changes (similarity between adjacent
                    pieces drops sharply). Own implementation, following the
                    assignment's algorithm:
                      1. split text into small pieces (sentences / list items / table rows / headings)
                      2. embed each piece (optionally with neighbours = "buffer")
                      3. cosine similarity between each piece and the next one
                      4. walk the pieces in order: high similarity -> same chunk;
                         low similarity or chunk too long -> new chunk
* structure_chunk - split by document structure: each heading starts a section,
                    sections are packed paragraph by paragraph up to a max length.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .parsing import Block
from .text_utils import split_sentences

EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass
class Chunk:
    index: int
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class _Unit:
    text: str
    block_index: int
    kind: str


# --------------------------------------------------------------------------- helpers


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """Split an over-long sentence on word boundaries (last resort)."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def _units_from_blocks(blocks: list[Block], max_chars: int) -> list[_Unit]:
    units: list[_Unit] = []
    for bi, block in enumerate(blocks):
        pieces = split_sentences(block.text) if block.kind == "paragraph" else [block.text]
        for piece in pieces:
            for part in _hard_wrap(piece, max_chars):
                units.append(_Unit(part, bi, block.kind))
    return units


def _join_units(units: list[_Unit]) -> str:
    """Sentences of one paragraph are joined by a space, different blocks by a newline."""
    out = ""
    for i, unit in enumerate(units):
        if i == 0:
            out = unit.text
        elif unit.block_index == units[i - 1].block_index:
            out += " " + unit.text
        else:
            out += "\n" + unit.text
    return out


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# --------------------------------------------------------------------------- semantic


@dataclass
class SemanticChunkingResult:
    chunks: list[Chunk]
    stats: dict


def semantic_chunk(
    blocks: list[Block],
    embed_fn: EmbedFn,
    *,
    threshold_type: Literal["percentile", "absolute"] = "percentile",
    breakpoint_percentile: float = 25.0,
    similarity_threshold: float = 0.55,
    buffer_size: int = 1,
    max_chars: int = 1200,
    min_chars: int = 150,
) -> SemanticChunkingResult:
    # 1. Split into small pieces.
    units = _units_from_blocks(blocks, max_chars)
    n = len(units)
    if n == 0:
        return SemanticChunkingResult([], {"units": 0})
    if n == 1:
        chunk = Chunk(0, units[0].text, {"unit_start": 0, "unit_end": 0, "break_reason": "start"})
        return SemanticChunkingResult([chunk], {"units": 1, "chunks": 1})

    # 2. Embed each piece. With buffer_size=b, piece i is embedded together with its
    #    b neighbours on each side; this smooths out very short sentences/headings.
    windows = [
        " ".join(u.text for u in units[max(0, i - buffer_size): i + buffer_size + 1])
        for i in range(n)
    ]
    vectors = _normalize_rows(np.asarray(embed_fn(windows), dtype=np.float32))

    # 3. Cosine similarity between each piece and the next one (vectors are unit length).
    sims = np.einsum("ij,ij->i", vectors[:-1], vectors[1:])  # sims[i] = sim(piece i, piece i+1)

    if threshold_type == "percentile" and len(sims) >= 3:
        threshold = float(np.percentile(sims, breakpoint_percentile))
        threshold_source = f"p{breakpoint_percentile:g} of adjacent similarities"
    else:
        threshold = float(similarity_threshold)
        threshold_source = "absolute"

    # 4. Walk the pieces in order.
    groups: list[list[int]] = [[0]]
    group_starts: list[tuple[str, float | None]] = [("start", None)]
    current_len = len(units[0].text)
    for i in range(1, n):
        sim = float(sims[i - 1])
        added_len = len(units[i].text) + 1
        if current_len + added_len > max_chars:
            reason = "max_length"
        elif sim < threshold and current_len >= min_chars:
            reason = "low_similarity"
        else:
            groups[-1].append(i)
            current_len += added_len
            continue

        current = groups[-1]
        if units[current[-1]].kind == "heading":
            if len(current) == 1:
                # Never leave a heading alone in a chunk: keep it with what follows.
                current.append(i)
                current_len += added_len
                continue
            # A heading belongs to the text after it, so move it into the new chunk.
            new_group = [current.pop(), i]
        else:
            new_group = [i]
        groups.append(new_group)
        group_starts.append((reason, sim))
        current_len = sum(len(units[j].text) + 1 for j in new_group) - 1

    # Merge a tiny trailing chunk into the previous one.
    if len(groups) > 1:
        last_len = sum(len(units[j].text) + 1 for j in groups[-1])
        prev_len = sum(len(units[j].text) + 1 for j in groups[-2])
        if last_len < min_chars and last_len + prev_len <= max_chars:
            groups[-2].extend(groups.pop())
            group_starts.pop()

    chunks: list[Chunk] = []
    for idx, (group, (reason, boundary_sim)) in enumerate(zip(groups, group_starts)):
        group_units = [units[j] for j in group]
        chunks.append(
            Chunk(
                index=idx,
                content=_join_units(group_units),
                metadata={
                    "unit_start": group[0],
                    "unit_end": group[-1],
                    "units": len(group),
                    # Why this chunk started: "start", "low_similarity" or "max_length".
                    "break_reason": reason,
                    # Similarity between the previous piece and this chunk's first piece.
                    "boundary_similarity": round(boundary_sim, 4) if boundary_sim is not None else None,
                    "threshold": round(threshold, 4),
                    "headings": [u.text for u in group_units if u.kind == "heading"],
                },
            )
        )

    reasons = [r for r, _ in group_starts]
    stats = {
        "units": n,
        "chunks": len(chunks),
        "threshold_type": threshold_type,
        "threshold": round(threshold, 4),
        "threshold_source": threshold_source,
        "buffer_size": buffer_size,
        "max_chars": max_chars,
        "min_chars": min_chars,
        "breaks_low_similarity": reasons.count("low_similarity"),
        "breaks_max_length": reasons.count("max_length"),
        "similarity_mean": round(float(sims.mean()), 4),
        "similarity_min": round(float(sims.min()), 4),
        "similarity_max": round(float(sims.max()), 4),
        "adjacent_similarities": [round(float(s), 4) for s in sims],
    }
    return SemanticChunkingResult(chunks, stats)


# --------------------------------------------------------------------------- structure


def structure_chunk(blocks: list[Block], *, max_chars: int = 1200, min_chars: int = 150) -> list[Chunk]:
    # Group blocks into sections: every heading starts a new section
    # (consecutive headings, e.g. title + subtitle, stay together).
    sections: list[list[Block]] = []
    for block in blocks:
        if block.kind == "heading" and (not sections or any(b.kind != "heading" for b in sections[-1])):
            sections.append([block])
        elif not sections:
            sections.append([block])
        else:
            sections[-1].append(block)

    # Pack each section paragraph by paragraph; never cross a section boundary.
    raw: list[tuple[str, dict]] = []
    for s_idx, section in enumerate(sections):
        headings = [b.text for b in section if b.kind == "heading"]
        pieces: list[str] = []
        for block in section:
            if len(block.text) <= max_chars:
                pieces.append(block.text)
            else:  # an over-long paragraph: fall back to sentences
                for sentence in split_sentences(block.text):
                    pieces.extend(_hard_wrap(sentence, max_chars))

        parts: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > max_chars:
                parts.append(current)
                current = piece
            else:
                current = f"{current}\n{piece}" if current else piece
        if current:
            parts.append(current)

        for p_idx, part in enumerate(parts):
            raw.append((part, {
                "section_index": s_idx,
                "section_heading": headings[-1] if headings else None,
                "part": p_idx + 1,
                "parts": len(parts),
            }))

    # Merge tiny chunks (e.g. a title-only section) into the next one.
    merged: list[tuple[str, dict]] = []
    i = 0
    while i < len(raw):
        text, meta = raw[i]
        while len(text) < min_chars and i + 1 < len(raw) and len(text) + 1 + len(raw[i + 1][0]) <= max_chars:
            i += 1
            text = f"{text}\n{raw[i][0]}"
            meta = {**raw[i][1], "merged_small_section": True}
        merged.append((text, meta))
        i += 1
    if len(merged) > 1 and len(merged[-1][0]) < min_chars and \
            len(merged[-2][0]) + 1 + len(merged[-1][0]) <= max_chars:
        last_text, _ = merged.pop()
        prev_text, prev_meta = merged.pop()
        merged.append((f"{prev_text}\n{last_text}", {**prev_meta, "merged_small_section": True}))

    return [Chunk(index=i, content=text, metadata=meta) for i, (text, meta) in enumerate(merged)]
