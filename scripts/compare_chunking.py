"""Compare semantic vs structure-based chunking on the same documents and questions.

Usage (after scripts/seed_sample_docs.py):
    python scripts/compare_chunking.py

For each question we know which document and which phrase contain the answer. The
script runs /api/search for both chunk sets and reports hit@1, hit@3, MRR@5 and scores,
plus chunk statistics. Output: reports/chunking_comparison.md
"""

import argparse
import statistics
from datetime import datetime

from common import DEFAULT_BASE_URL, REPORTS_DIR, Api, find_doc, md_escape, shorten

TENANT = "company-a"
METHODS = ("semantic", "structure")
TOP_K = 5

# (question, acceptable document titles, phrase that the right chunk must contain)
QUESTIONS = [
    ("How many days per week can I work from home?", ["Remote Work Policy"], "two (2) days per week"),
    ("Do I need my manager's approval before working remotely?", ["Remote Work Policy"], "approved by the line manager"),
    ("What hours do I have to be online when working remotely?", ["Remote Work Policy"], "10:00 to 16:00"),
    ("How much does the company pay for my home internet?", ["Remote Work Policy"], "SGD 30 per month"),
    ("How many days can I work remotely from overseas?", ["Remote Work Policy"], "ten (10) working days"),
    ("How many days of annual leave do I get per year?", ["Annual Leave Policy"], "eighteen (18)"),
    ("Until when can I use leave days carried over from last year?", ["Annual Leave Policy"], "30 June"),
    ("Can I get paid for annual leave I did not take?", ["Annual Leave Policy"], "cannot be exchanged for cash"),
    ("Can I fly business class to London?", ["Business Travel and Expense Policy"], "six (6) hours or longer"),
    ("What is the hotel limit per night in Southeast Asia?", ["Business Travel and Expense Policy"], "SGD 200"),
    ("How long do I have to submit an expense claim after a trip?", ["Business Travel and Expense Policy"], "thirty (30) days"),
    ("Is alcohol reimbursed on business trips?", ["Business Travel and Expense Policy"], "Alcohol is not reimbursed"),
    ("How often must I change my password?", ["Information Security Policy"], "every 180 days"),
    ("What should I do if my laptop is stolen?", ["Information Security Policy", "Remote Work Policy"], "reported to the IT Service Desk"),
    ("Can I use the Wi-Fi in a coffee shop for work?", ["Information Security Policy"], "public Wi-Fi only through the company VPN"),
]


def is_right(chunk: dict, titles: list[str], phrase: str) -> bool:
    return chunk["document_title"] in titles and phrase.lower() in chunk["content"].lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()
    api = Api(args.base_url)

    docs = api.documents(TENANT)
    current = [d for d in docs if d["effective_status"] == "current"]
    scope = [d["id"] for d in current]

    per_question: list[dict] = []
    for question, titles, phrase in QUESTIONS:
        row = {"question": question, "phrase": phrase}
        for method in METHODS:
            results = api.search(TENANT, question, scope, TOP_K, method)["results"]
            rank = next((c["rank"] for c in results if is_right(c, titles, phrase)), None)
            row[method] = {
                "rank": rank,
                "top1_score": results[0]["score"] if results else None,
                "right_score": next((c["score"] for c in results if c["rank"] == rank), None),
                "top1": results[0] if results else None,
                "top1_chars": len(results[0]["content"]) if results else None,
            }
        per_question.append(row)
        print(f"{shorten(question, 60):<62} semantic rank={row['semantic']['rank']}  "
              f"structure rank={row['structure']['rank']}")

    def metrics(method: str) -> dict:
        ranks = [r[method]["rank"] for r in per_question]
        right_scores = [r[method]["right_score"] for r in per_question if r[method]["right_score"] is not None]
        return {
            "hit1": sum(1 for x in ranks if x == 1),
            "hit3": sum(1 for x in ranks if x is not None and x <= 3),
            "mrr": sum(1 / x for x in ranks if x) / len(ranks),
            "top1": statistics.mean(r[method]["top1_score"] for r in per_question if r[method]["top1_score"] is not None),
            "right": statistics.mean(right_scores) if right_scores else float("nan"),
            "top1_chars": statistics.mean(r[method]["top1_chars"] for r in per_question if r[method]["top1_chars"]),
        }

    m = {method: metrics(method) for method in METHODS}
    n = len(per_question)

    lines = [
        "# Semantic vs structure-based chunking", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/compare_chunking.py` "
        f"(tenant `{TENANT}`, {len(current)} current documents, top_k={TOP_K}).", "",
        "## Retrieval quality", "",
        "| Metric | Semantic | Structure-based |", "|---|---|---|",
        f"| Hit@1 (right chunk ranked first) | {m['semantic']['hit1']}/{n} | {m['structure']['hit1']}/{n} |",
        f"| Hit@3 | {m['semantic']['hit3']}/{n} | {m['structure']['hit3']}/{n} |",
        f"| MRR@{TOP_K} | {m['semantic']['mrr']:.3f} | {m['structure']['mrr']:.3f} |",
        f"| Mean top-1 score | {m['semantic']['top1']:.4f} | {m['structure']['top1']:.4f} |",
        f"| Mean score of the right chunk | {m['semantic']['right']:.4f} | {m['structure']['right']:.4f} |",
        f"| Mean length of top-1 chunk (chars) | {m['semantic']['top1_chars']:.0f} | {m['structure']['top1_chars']:.0f} |",
        "",
        "## Per question (rank of the right chunk, '-' = not in top 5; score of the right chunk)", "",
        "| Question | Semantic | Structure |", "|---|---|---|",
    ]
    for r in per_question:
        cells = []
        for method in METHODS:
            x = r[method]
            cells.append(f"{x['rank'] or '-'} ({x['right_score']:.3f})" if x["rank"] else "-")
        lines.append(f"| {md_escape(r['question'])} | {cells[0]} | {cells[1]} |")

    lines += ["", "## Chunk statistics per document", "",
              "| Document | Semantic: chunks / avg / min / max chars | Structure: chunks / avg / min / max chars | Semantic threshold |",
              "|---|---|---|---|"]
    chunk_sets: dict[str, dict] = {}
    for d in current:
        data = api.chunks(TENANT, d["id"], "all")
        chunk_sets[d["id"]] = data
        cells = []
        for method in METHODS:
            sizes = [c["char_count"] for c in data["chunks"] if c["chunking_method"] == method]
            cells.append(f"{len(sizes)} / {statistics.mean(sizes):.0f} / {min(sizes)} / {max(sizes)}" if sizes else "-")
        sem = data["document"]["ingestion_stats"].get("semantic", {})
        lines.append(f"| {md_escape(d['title'])} v{d['version']} | {cells[0]} | {cells[1]} | "
                     f"{sem.get('threshold')} ({sem.get('breaks_low_similarity')} similarity breaks, "
                     f"{sem.get('breaks_max_length')} length breaks) |")

    remote = find_doc(current, "Remote Work Policy")
    lines += ["", f"## How '{remote['title']}' was split", ""]
    for method in METHODS:
        lines += [f"### {method}", "", "| # | Chars | Why it starts here | First words |", "|---|---|---|---|"]
        for c in chunk_sets[remote["id"]]["chunks"]:
            if c["chunking_method"] != method:
                continue
            meta = c["metadata"]
            why = (f"{meta.get('break_reason')} (sim {meta.get('boundary_similarity')})" if method == "semantic"
                   else f"section: {meta.get('section_heading')}")
            lines.append(f"| {c['chunk_index']} | {c['char_count']} | {md_escape(str(why))} | "
                         f"{md_escape(shorten(c['content'], 90))} |")
        lines.append("")

    lines += ["## Observations (write your own)", "",
              "- Which strategy ranks the right chunk first more often, and why?",
              "- Are semantic chunks merging several sections (or cutting one section in two)?",
              "- Do larger chunks get higher or lower scores? (a long chunk dilutes its embedding)", ""]

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "chunking_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nHit@1 semantic {m['semantic']['hit1']}/{n}, structure {m['structure']['hit1']}/{n}")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
