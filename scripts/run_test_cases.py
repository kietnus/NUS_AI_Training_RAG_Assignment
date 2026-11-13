"""Run test cases TC1-TC8 (plus ingestion checks) against the running API.

Usage (after scripts/seed_sample_docs.py):
    python scripts/run_test_cases.py                       # semantic chunks
    python scripts/run_test_cases.py --method structure    # structure-based chunks

Writes reports/test_results_<method>.md with the evidence (top chunks, scores, answers,
citations) and automatic checks. Items marked REVIEW need a human judgement
(e.g. "does the cited chunk really support the claim?") - add your diagnosis to the README.
"""

import argparse
import re
from datetime import datetime

from common import DEFAULT_BASE_URL, REPORTS_DIR, SAMPLE_DIR, Api, find_doc, md_escape, shorten

TENANT_A = "company-a"
TENANT_B = "company-b"

STOPWORDS = {
    "a", "an", "the", "i", "can", "do", "does", "is", "are", "am", "how", "what", "which", "who",
    "many", "much", "per", "of", "for", "to", "in", "on", "at", "my", "me", "there", "any", "from",
    "get", "be", "it", "and", "or", "with", "when", "not", "allowed", "whole",
}


class Report:
    def __init__(self):
        self.sections: list[str] = []
        self.summary: list[tuple[str, str, str]] = []

    def add(self, tc: str, title: str, checks: list[tuple[str, bool | None]], body: str) -> None:
        if any(ok is False for _, ok in checks):
            verdict = "FAIL"
        elif any(ok is None for _, ok in checks):
            verdict = "REVIEW"
        else:
            verdict = "PASS"
        self.summary.append((tc, title, verdict))
        lines = [f"## {tc} - {title}", "", f"**Automatic result: {verdict}**", ""]
        for desc, ok in checks:
            mark = {True: "✅", False: "❌", None: "🔎"}[ok]
            lines.append(f"- {mark} {desc}")
        lines += ["", body, ""]
        self.sections.append("\n".join(lines))
        print(f"{tc:<5} {verdict:<7} {title}")


# --------------------------------------------------------------------------- helpers


def chunk_table(chunks: list[dict], n: int = 3) -> str:
    if not chunks:
        return "_No chunks returned._"
    lines = ["| Rank | Document | Version | Score | Chunk | Content |", "|---|---|---|---|---|---|"]
    for c in chunks[:n]:
        lines.append(f"| {c['rank']} | {md_escape(c['document_title'])} | {c['version']} | "
                     f"{c['score']:.4f} | #{c['chunk_index']} | {md_escape(shorten(c['content'], 240))} |")
    return "\n".join(lines)


def answer_block(r: dict) -> str:
    lines = [
        f"- **Status:** `{r['status']}`",
        f"- **Answer:** {r['answer']}",
        f"- **Best score:** {r['top_score']} (MIN_SCORE {r['min_score']})",
    ]
    if r.get("reason"):
        lines.append(f"- **Reason:** {r['reason']}")
    if r.get("dropped_citations"):
        lines.append(f"- **Removed invalid citations:** {r['dropped_citations']}")
    for c in r["citations"]:
        lines.append(f"  - [{c['marker']}] {c['document_title']} v{c['version']} chunk #{c['chunk_index']} - "
                     f"quote: \"{c['quote'] or ''}\" - quote found in chunk: {c['quote_verified']}")
    lines.append(f"- query_log_id: `{r.get('query_log_id')}`")
    return "\n".join(lines)


def contains(text: str, *keywords: str) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def citations_valid(r: dict) -> bool:
    by_rank = {c["rank"]: c for c in r["chunks"]}
    return all(c["marker"] in by_rank and by_rank[c["marker"]]["chunk_id"] == c["chunk_id"]
               for c in r["citations"])


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1}


def question_block(tenant: str, question: str, scope: list[str]) -> str:
    return f"- **Tenant:** `{tenant}`\n- **Scope:** {', '.join(scope)}\n- **Question:** \"{question}\""


# --------------------------------------------------------------------------- test cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--method", choices=["semantic", "structure"], default="semantic")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    api, method, k = Api(args.base_url), args.method, args.top_k
    report = Report()

    docs_a = api.documents(TENANT_A)
    docs_b = api.documents(TENANT_B)
    remote_a = find_doc(docs_a, "Remote Work Policy")
    leave_v1 = find_doc(docs_a, "Annual Leave Policy", "1.0")
    leave_v2 = find_doc(docs_a, "Annual Leave Policy", "2.0")
    travel = find_doc(docs_a, "Business Travel and Expense Policy")
    remote_b = find_doc(docs_b, "Remote Work Policy")
    current_a = [d for d in docs_a if d["effective_status"] == "current"]
    current_a_ids = [d["id"] for d in current_a]
    scope_a = [f"{d['title']} v{d['version']}" for d in current_a]
    ids_a = {d["id"] for d in docs_a}

    # TC1 ------------------------------------------------------------------
    q1 = "How many days per week can I work from home?"
    r1 = api.ask(TENANT_A, q1, current_a_ids, k, method)
    top1 = r1["chunks"][0] if r1["chunks"] else None
    report.add("TC1", "Question worded differently from the document", [
        ("Top-1 chunk is from 'Remote Work Policy'", bool(top1) and top1["document_id"] == remote_a["id"]),
        ("Top-1 chunk contains the answer ('two (2) days per week')",
         bool(top1) and contains(top1["content"], "two (2) days per week")),
        ("Status is 'answered'", r1["status"] == "answered"),
        ("Answer states 2 days", contains(r1["answer"], "2 days", "two (2)", "two days", "2 remote")),
        ("Every citation points to a retrieved chunk", bool(r1["citations"]) and citations_valid(r1)),
    ], "\n".join([question_block(TENANT_A, q1, scope_a), "", chunk_table(r1["chunks"], k), "", answer_block(r1)]))

    # TC2 ------------------------------------------------------------------
    s2 = api.search(TENANT_A, q1, current_a_ids, 3, method)
    best = s2["results"][0] if s2["results"] else None
    q_words = content_words(q1)
    overlap_lines = []
    for c in s2["results"]:
        shared = sorted(q_words & content_words(c["content"]))
        missing = sorted(q_words - content_words(c["content"]))
        overlap_lines.append(f"- Rank {c['rank']}: shared words {shared or '-'}; question words NOT in chunk {missing or '-'}")
    report.add("TC2", "/api/search with the TC1 question (top 3)", [
        ("Correct chunk is in the top 3",
         any(contains(c["content"], "two (2) days per week") for c in s2["results"])),
        ("Explain why semantic search matches although the words differ (see README)", None),
    ], "\n".join([question_block(TENANT_A, q1, scope_a), "", chunk_table(s2["results"], 3), "",
                  "Keyword overlap between question and chunk:", *overlap_lines,
                  "", f"Top-1 score: {best['score'] if best else '-'}"]))

    # TC3 ------------------------------------------------------------------
    checks3, body3 = [], []
    for q3 in ("What is the dress code for client meetings?",
               "How many weeks of maternity leave can I take?"):
        r3 = api.ask(TENANT_A, q3, current_a_ids, k, method)
        checks3.append((f"\"{q3}\" -> not_enough_information", r3["status"] == "not_enough_information"))
        body3 += [question_block(TENANT_A, q3, scope_a), "", chunk_table(r3["chunks"], 3), "", answer_block(r3), ""]
    report.add("TC3", "Question not covered by any document", checks3, "\n".join(body3))

    # TC4 ------------------------------------------------------------------
    q4 = "How many days of annual leave do I get per year?"
    scope4 = [leave_v1["id"], leave_v2["id"]]  # BOTH versions selected on purpose
    r4 = api.ask(TENANT_A, q4, scope4, k, method)
    report.add("TC4", "Expired version and current version both uploaded", [
        (f"v1.0 is not current (effective_status = {leave_v1['effective_status']})",
         leave_v1["effective_status"] != "current"),
        ("No retrieved chunk comes from v1.0", all(c["document_id"] != leave_v1["id"] for c in r4["chunks"])),
        ("Retrieved chunks come from v2.0", bool(r4["chunks"]) and all(c["document_id"] == leave_v2["id"] for c in r4["chunks"])),
        ("Answer says 18 days (v2.0), not 14 (v1.0)",
         contains(r4["answer"], "18", "eighteen") and not contains(r4["answer"], "14 ", "fourteen")),
    ], "\n".join([
        question_block(TENANT_A, q4, ["Annual Leave Policy v1.0 (superseded)", "Annual Leave Policy v2.0"]),
        f"- v1.0: valid {leave_v1['valid_from']} → {leave_v1['valid_to']}, status `{leave_v1['status']}`, "
        f"superseded_by `{leave_v1['superseded_by']}`",
        f"- v2.0: valid {leave_v2['valid_from']} → {leave_v2['valid_to']}, status `{leave_v2['status']}`",
        "", chunk_table(r4["chunks"], k), "", answer_block(r4)]))

    # TC5 ------------------------------------------------------------------
    r5 = api.ask(TENANT_B, q1, [remote_b["id"], remote_a["id"]], k, method)  # A's id sent on purpose
    r5x = api.ask(TENANT_A, q1, [remote_b["id"]], k, method)
    report.add("TC5", "Switch to another tenant", [
        ("company-b document list contains no company-a document", not ({d["id"] for d in docs_b} & ids_a)),
        ("As company-b (with a company-a document id in the request!), no chunk from company-a",
         all(c["document_id"] not in ids_a for c in r5["chunks"])),
        ("company-b answer says 4 days", contains(r5["answer"], "4 days", "four (4)", "four days")),
        ("As company-a selecting company-b's document: no chunks at all", not r5x["chunks"]),
    ], "\n".join([
        question_block(TENANT_B, q1, ["Remote Work Policy v1.0 (company-b)", "Remote Work Policy v2.0 (company-a id!)"]),
        "", chunk_table(r5["chunks"], k), "", answer_block(r5), "",
        question_block(TENANT_A, q1, ["Remote Work Policy v1.0 (company-b id)"]), "", answer_block(r5x)]))

    # TC6 ------------------------------------------------------------------
    r6 = api.ask(TENANT_A, q4, [travel["id"]], k, method)
    report.add("TC6", "Select only document A (Travel), ask about document B (Annual Leave)", [
        ("All retrieved chunks are from the Travel policy", all(c["document_id"] == travel["id"] for c in r6["chunks"])),
        ("No chunk from the Annual Leave policy",
         all(c["document_id"] not in (leave_v1["id"], leave_v2["id"]) for c in r6["chunks"])),
        ("Status is not_enough_information", r6["status"] == "not_enough_information"),
    ], "\n".join([question_block(TENANT_A, q4, ["Business Travel and Expense Policy v1.0"]), "",
                  chunk_table(r6["chunks"], 3), "", answer_block(r6)]))

    # TC7 ------------------------------------------------------------------
    q7 = "What is the hotel limit per night when travelling in Southeast Asia?"
    r7 = api.ask(TENANT_A, q7, current_a_ids, k, method)
    checks7, rows7 = [], ["| Case | Citation | Points to retrieved chunk | Quote found in chunk | Cited chunk (excerpt) | Supports the claim? |",
                          "|---|---|---|---|---|---|"]
    for name, r in (("TC1", r1), ("TC4", r4), ("TC5", r5), ("TC7", r7)):
        if r["status"] != "answered":
            continue
        by_rank = {c["rank"]: c for c in r["chunks"]}
        for c in r["citations"]:
            chunk = by_rank.get(c["marker"])
            points = chunk is not None and chunk["chunk_id"] == c["chunk_id"]
            rows7.append(f"| {name} | [{c['marker']}] {md_escape(c['document_title'])} v{c['version']} | "
                         f"{'yes' if points else 'NO'} | {c['quote_verified']} | "
                         f"{md_escape(shorten(chunk['content'] if chunk else '', 160))} | _fill in_ |")
        checks7.append((f"{name}: every citation points to a retrieved chunk", citations_valid(r)))
        checks7.append((f"{name}: every quote is found in its cited chunk",
                        all(c["quote_verified"] is True for c in r["citations"]) if r["citations"] else None))
    checks7.append(("Manually confirm each cited chunk supports its claim (last column)", None))
    report.add("TC7", "Check citations", checks7, "\n".join([
        question_block(TENANT_A, q7, scope_a), "", answer_block(r7), "", *rows7]))

    # TC8 ------------------------------------------------------------------
    candidates = [
        ("Is there an allowance for working remotely?", "Remote Work Policy", "internet allowance"),
        ("Which employees are not allowed to work remotely?", "Remote Work Policy", "not eligible"),
        ("Can I work remotely from Vietnam for a whole month?", "Remote Work Policy", "ten (10) working days"),
    ]
    checks8, body8 = [], []
    for q8, title, keyword in candidates:
        s8 = api.search(TENANT_A, q8, current_a_ids, 3, method)
        top = s8["results"][0] if s8["results"] else None
        right_rank = next((c["rank"] for c in s8["results"] if c["document_title"] == title
                           and contains(c["content"], keyword)), None)
        failed = top is None or not (top["document_title"] == title and contains(top["content"], keyword))
        checks8.append((f"\"{q8}\": top-1 is {'NOT ' if failed else ''}the right chunk "
                        f"(right chunk rank: {right_rank or '>3'})", None))
        body8 += [f"### \"{q8}\"", f"Right answer: chunk of '{title}' containing \"{keyword}\".", "",
                  chunk_table(s8["results"], 3), ""]
        if failed:
            r8 = api.ask(TENANT_A, q8, current_a_ids, k, method)
            body8 += ["Does the LLM recover when the right chunk is lower in the top-k?", "", answer_block(r8), ""]
    checks8.append(("Pick the case where the most similar chunk is not the right one and explain why in the README", None))
    report.add("TC8", "Failure mode of similarity", checks8, "\n".join(body8))

    # Ingestion checks ---------------------------------------------------------
    txt = SAMPLE_DIR / "unsupported_example.txt"
    r_txt = api.upload(TENANT_A, txt, {"title": "Txt", "version": "1", "valid_from": "2025-01-01",
                                       "valid_to": "2025-12-31"})
    r_meta = api.upload(TENANT_A, SAMPLE_DIR / "information_security_policy.pdf",
                        {"title": "No Dates Policy", "version": "1"})
    r_tenant = api.client.get("/api/documents")
    after = {d["title"] for d in api.documents(TENANT_A)}
    report.add("ING", "Ingestion validation", [
        (f"Unsupported .txt file rejected with 415 (got {r_txt.status_code})", r_txt.status_code == 415),
        (f"Missing validity dates rejected with 422 (got {r_meta.status_code})", r_meta.status_code == 422),
        ("Rejected document was not indexed", "No Dates Policy" not in after),
        (f"Missing X-Tenant-Id rejected with 400 (got {r_tenant.status_code})", r_tenant.status_code == 400),
    ], "\n".join([f"- `.txt` upload → `{r_txt.text}`", f"- missing metadata → `{r_meta.text}`",
                  f"- no tenant header → `{r_tenant.text}`"]))

    # Write report -------------------------------------------------------------
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"test_results_{method}.md"
    header = [
        f"# Test results ({method} chunking, top_k={k})", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/run_test_cases.py`.", "",
        "| Test | Scenario | Automatic result |", "|---|---|---|",
        *[f"| {tc} | {title} | {verdict} |" for tc, title, verdict in report.summary], "",
        "PASS/FAIL = automatic checks; REVIEW = needs a human judgement (write the diagnosis in the README).", "",
    ]
    out.write_text("\n".join(header + report.sections), encoding="utf-8")
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
