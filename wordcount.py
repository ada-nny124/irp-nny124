from docx import Document
import re

FILE = "deliverables/nny124-final-report.docx"

doc = Document(FILE)

# Entire sections that should not count
SKIP_SECTIONS = {
    "acknowledgements",
    "acknowledgments",
    "ai acknowledgement",
    "ai acknowledgment",
    "ai acknowledgement statement",
    "table of contents",
    "list of figures",
    "list of tables",
    "list of symbols",
    "list of abbreviations",
    "references",
    "bibliography",
}

words = []
counting = False
skip_section = False

for p in doc.paragraphs:
    text = p.text.strip()
    style = p.style.name.lower() if p.style else ""

    if not text:
        continue

    text_lower = text.lower()

    # Start counting at Abstract (exclude title to list of figure-table)
    if text_lower == "abstract":
        counting = True
        skip_section = False
        continue  # heading itself does not count

    if not counting:
        continue

    # Detect headings
    if style.startswith("heading"):

        # References = stop counting
        if text_lower in {"references", "bibliography"}:
            skip_section = True
            continue

        # Any appendix = stop counting
        if text_lower.startswith("appendix"):
            skip_section = True
            continue

        # Other excluded sections
        if text_lower in SKIP_SECTIONS:
            skip_section = True
            continue

        # Normal new section
        skip_section = False

        # Section/subsection titles don't count
        continue

    if skip_section:
        continue

    # Figure/table captions don't count
    if "caption" in style:
        continue

    # Backup in case caption style wasn't applied
    if re.match(r"^(figure|table)\s+\d+[\.:]", text, re.I):
        continue

    # Count normal prose
    words.extend(
        re.findall(r"\b[\w’'-]+\b", text)
    )

print(f"Counted word total: {len(words):,}")