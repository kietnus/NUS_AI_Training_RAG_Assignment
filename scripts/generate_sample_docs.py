"""Generate the sample policy documents (PDF + DOCX) and their upload manifest.

Usage (from the project root, with the backend virtualenv active):
    python scripts/generate_sample_docs.py

The set is designed for the assignment's test cases:
  * Annual Leave Policy v1.0 (old) and v2.0 (current)          -> TC4
  * a Remote Work Policy that belongs to another tenant         -> TC5
  * documents say "remote work", never "work from home"         -> TC1 / TC2
  * a Travel policy section about "remote project sites" with a
    "site allowance" that is NOT the remote-work allowance      -> TC8
  * no document covers dress code / maternity leave             -> TC3
Text is plain ASCII so the PDF core fonts can render it.
"""

import json
from pathlib import Path

from docx import Document
from fpdf import FPDF
from fpdf.enums import XPos, YPos

OUT_DIR = Path(__file__).resolve().parents[1] / "sample_docs"

DOCS = [
    {
        "file": "remote_work_policy_v2.pdf",
        "tenant": "company-a",
        "title": "Remote Work Policy",
        "version": "2.0",
        "valid_from": "2025-01-01",
        "valid_to": "2027-12-31",
        "subtitle": "Company A - Human Resources. Version 2.0, valid from 1 January 2025 to 31 December 2027.",
        "sections": [
            ("1. Purpose and Scope", [
                "This policy sets out the rules for remote work at Company A. It applies to all "
                "permanent and contract employees based in Singapore.",
            ]),
            ("2. Eligibility", [
                "Employees who have completed their probation period are eligible for remote work. "
                "Employees on probation must work on site during their first three months. Roles that "
                "require physical presence, such as laboratory and front-desk staff, are not eligible "
                "for remote work.",
            ]),
            ("3. Number of Remote Days", [
                "Eligible employees may work remotely for up to two (2) days per week. Unused remote "
                "days cannot be carried over to the following week. Team leads may set common office "
                "days for their teams, usually Tuesday and Thursday.",
            ]),
            ("4. Requesting Remote Days", [
                "Remote days must be requested through the HR Portal at least three (3) working days in "
                "advance and approved by the line manager. For urgent personal reasons, the line manager "
                "may approve a remote day on the same day.",
            ]),
            ("5. Working Hours and Availability", [
                "Employees working remotely must be online during core hours from 10:00 to 16:00 "
                "Singapore time. They must answer messages within one hour during core hours and join "
                "scheduled meetings with their camera on.",
            ]),
            ("6. Equipment and Internet Allowance", [
                "The company provides a laptop and a headset for remote work. Employees receive an "
                "internet allowance of SGD 30 per month, paid together with the monthly salary. Office "
                "furniture for use at home is not reimbursed.",
            ]),
            ("7. Security Requirements", [
                "Company systems must only be accessed through the company VPN. Confidential documents "
                "must not be printed at home. Lost or stolen devices must be reported to the IT Service "
                "Desk within 24 hours.",
            ]),
            ("8. Working Remotely from Overseas", [
                "Working remotely from another country requires prior written approval from HR. It is "
                "limited to a maximum of ten (10) working days per calendar year.",
            ]),
        ],
    },
    {
        "file": "annual_leave_policy_v1.docx",
        "tenant": "company-a",
        "title": "Annual Leave Policy",
        "version": "1.0",
        "valid_from": "2022-01-01",
        # Issued with a long validity, but replaced early by v2.0. Its dates are still "valid",
        # so only the versioning rule (status = expired) keeps it out of retrieval (TC4).
        "valid_to": "2026-12-31",
        "subtitle": "Company A - Human Resources. Version 1.0, valid from 1 January 2022.",
        "sections": [
            ("1. Annual Leave Entitlement", [
                "Full-time employees are entitled to fourteen (14) days of paid annual leave per "
                "calendar year.",
            ]),
            ("2. Carry Forward", [
                "Up to five (5) days of unused annual leave may be carried forward to the next year and "
                "must be used by 31 March.",
            ]),
            ("3. Applying for Leave", [
                "Annual leave must be requested at least two (2) weeks in advance by email to the line "
                "manager.",
            ]),
            ("4. Other Types of Leave", [
                "Sick leave and parental leave are covered by separate policies.",
            ]),
        ],
    },
    {
        "file": "annual_leave_policy_v2.pdf",
        "tenant": "company-a",
        "title": "Annual Leave Policy",
        "version": "2.0",
        "valid_from": "2025-01-01",
        "valid_to": "2027-12-31",
        "subtitle": "Company A - Human Resources. Version 2.0, valid from 1 January 2025 to 31 December "
                    "2027. Replaces version 1.0.",
        "sections": [
            ("1. Purpose", [
                "This policy explains how much paid annual leave employees receive and how to use it.",
            ]),
            ("2. Annual Leave Entitlement", [
                "Full-time employees are entitled to eighteen (18) days of paid annual leave per "
                "calendar year. Employees receive one (1) additional day for every two (2) completed "
                "years of service, up to a maximum of twenty-four (24) days per year.",
                "Part-time employees receive annual leave on a pro-rated basis according to their "
                "contracted hours.",
            ]),
            ("3. Carry Forward", [
                "Up to seven (7) days of unused annual leave may be carried forward to the next year. "
                "Carried-forward days must be used by 30 June, after which they are forfeited.",
            ]),
            ("4. Applying for Leave", [
                "Annual leave must be applied for through the HR Portal at least five (5) working days "
                "in advance. Leave of more than ten (10) consecutive working days requires approval from "
                "the department head.",
            ]),
            ("5. Leave Encashment", [
                "Unused annual leave cannot be exchanged for cash, except for the remaining balance when "
                "an employee leaves the company.",
            ]),
            ("6. Other Types of Leave", [
                "Sick leave, parental leave and compassionate leave are covered by separate policies and "
                "are not part of the annual leave entitlement.",
            ]),
        ],
    },
    {
        "file": "business_travel_expense_policy.docx",
        "tenant": "company-a",
        "title": "Business Travel and Expense Policy",
        "version": "1.0",
        "valid_from": "2024-07-01",
        "valid_to": "2027-12-31",
        "subtitle": "Company A - Finance Department. Version 1.0, valid from 1 July 2024.",
        "sections": [
            ("1. Scope", [
                "This policy covers business travel and work-related expenses for all employees of "
                "Company A.",
            ]),
            ("2. Travel Approval", [
                "All business trips must be approved in advance by the department head. Trips outside "
                "Southeast Asia also require approval from the Finance Director.",
            ]),
            ("3. Flights", [
                "Employees fly economy class for flights shorter than six (6) hours. Business class is "
                "allowed for flights of six (6) hours or longer with approval from the Finance Director.",
            ]),
            ("4. Hotel and Meal Limits", [
                "The following limits apply per person. Amounts above the limit are not reimbursed "
                "unless approved in advance by the Finance Director.",
                ("table", [
                    ["Item", "Singapore", "Southeast Asia", "Rest of the world"],
                    ["Hotel per night", "SGD 250", "SGD 200", "SGD 350"],
                    ["Meals per day", "SGD 60", "SGD 50", "SGD 80"],
                ]),
            ]),
            ("5. Local Transport", [
                "Public transport should be used where practical. Taxis and ride-hailing are reimbursed "
                "for trips between 22:00 and 07:00, or when carrying heavy equipment.",
            ]),
            ("6. Travel to Remote Sites", [
                "Employees who travel to remote project sites, such as offshore platforms or rural field "
                "stations, receive a site allowance of SGD 50 per day in addition to the meal limits.",
            ]),
            ("7. Submitting Expense Claims", [
                "Expense claims must be submitted in the Finance System within thirty (30) days after the "
                "trip, together with the original receipts. Late claims require approval from the Finance "
                "Director.",
                ("bullets", [
                    "Alcohol is not reimbursed.",
                    "Personal entertainment, such as movies or spa services, is not reimbursed.",
                    "Lost receipts must be replaced by a signed declaration from the employee.",
                ]),
            ]),
        ],
    },
    {
        "file": "information_security_policy.pdf",
        "tenant": "company-a",
        "title": "Information Security Policy",
        "version": "1.0",
        "valid_from": "2024-01-01",
        "valid_to": "2027-12-31",
        "subtitle": "Company A - IT Department. Version 1.0, valid from 1 January 2024.",
        "sections": [
            ("1. Passwords", [
                "Passwords must be at least twelve (12) characters long and must be changed every 180 "
                "days. Passwords must never be shared, including with IT staff.",
            ]),
            ("2. Multi-Factor Authentication", [
                "Multi-factor authentication is required for email, the HR Portal and the Finance System.",
            ]),
            ("3. Phishing", [
                "Suspicious emails must be reported with the Report Phishing button in the email client. "
                "Do not click links or open attachments in suspicious emails.",
            ]),
            ("4. Company Devices", [
                "Company laptops are encrypted and must not be lent to family members or friends. Lost or "
                "stolen devices must be reported to the IT Service Desk immediately, and no later than 24 "
                "hours after the loss.",
            ]),
            ("5. Public Networks", [
                "When working outside the office, connect to public Wi-Fi only through the company VPN.",
            ]),
            ("6. Data Classification", [
                "Confidential data must not be stored on personal devices or in personal cloud storage.",
            ]),
        ],
    },
    {
        "file": "company_b_remote_work_policy.docx",
        "tenant": "company-b",
        "title": "Remote Work Policy",
        "version": "1.0",
        "valid_from": "2025-01-01",
        "valid_to": "2027-12-31",
        "subtitle": "Company B - People Team. Version 1.0, valid from 1 January 2025.",
        "sections": [
            ("1. Scope", [
                "This policy applies to all employees of Company B.",
            ]),
            ("2. Number of Remote Days", [
                "Company B employees may work remotely for up to four (4) days per week. Fully remote "
                "arrangements are possible with approval from the Chief Executive Officer.",
            ]),
            ("3. Internet Allowance", [
                "Employees receive an internet allowance of SGD 50 per month.",
            ]),
            ("4. Requests", [
                "Remote days are agreed directly with the team lead; no formal request is needed.",
            ]),
        ],
    },
]

UNSUPPORTED_FILE = "unsupported_example.txt"


def write_docx(spec: dict, path: Path) -> None:
    doc = Document()
    doc.add_heading(spec["title"], level=0)
    doc.add_paragraph(spec["subtitle"])
    for heading, items in spec["sections"]:
        doc.add_heading(heading, level=1)
        for item in items:
            if isinstance(item, str):
                doc.add_paragraph(item)
            elif item[0] == "bullets":
                for bullet in item[1]:
                    doc.add_paragraph(bullet, style="List Bullet")
            elif item[0] == "table":
                rows = item[1]
                table = doc.add_table(rows=len(rows), cols=len(rows[0]))
                table.style = "Table Grid"
                for r, row in enumerate(rows):
                    for c, value in enumerate(row):
                        table.cell(r, c).text = value
    doc.save(path)


def write_pdf(spec: dict, path: Path) -> None:
    pdf = FPDF(format="A4")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    # Left-aligned (not justified) text keeps word spacing simple for PDF text extraction.
    nl = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT, "align": "L"}

    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, spec["title"], **nl)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, spec["subtitle"], **nl)
    pdf.ln(4)
    for heading, items in spec["sections"]:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, heading, **nl)
        pdf.set_font("Helvetica", "", 11)
        for item in items:
            if isinstance(item, str):
                pdf.multi_cell(0, 6, item, **nl)
            elif item[0] == "bullets":
                for bullet in item[1]:
                    pdf.multi_cell(0, 6, f"- {bullet}", **nl)
            elif item[0] == "table":
                header, *rows = item[1]
                for row in rows:
                    pdf.multi_cell(0, 6, "; ".join(f"{h}: {v}" for h, v in zip(header, row)) + ".", **nl)
            pdf.ln(2)
    pdf.output(str(path))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in DOCS:
        path = OUT_DIR / spec["file"]
        (write_pdf if path.suffix == ".pdf" else write_docx)(spec, path)
        manifest.append({k: spec[k] for k in ("file", "tenant", "title", "version", "valid_from", "valid_to")})
        print(f"wrote {path.relative_to(OUT_DIR.parent)}")

    (OUT_DIR / UNSUPPORTED_FILE).write_text(
        "This plain-text file is used to check that unsupported file types are rejected.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote sample_docs/{UNSUPPORTED_FILE} and sample_docs/manifest.json")


if __name__ == "__main__":
    main()
