#!/usr/bin/env python3
"""Build the searchable index from the English seminar archive's two-column layout.

Requires PyMuPDF: python -m pip install pymupdf
Run with --check to detect a stale index without writing files.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from pathlib import Path
import re
from urllib.parse import urlsplit

import pymupdf


DATE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{2}")
ARCHIVE_DIR = Path(__file__).resolve().parents[1]


def extract_entries(pdf_path):
    """Keep archive order and repeated dates; associate links by card position."""
    entries = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, 1):
            lines = []
            for block_number, block in enumerate(page.get_text("dict")["blocks"]):
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line["spans"]).strip()
                    lines.append((block_number, line["bbox"][0], line["bbox"][1], text))
            dates = sorted((y, text) for _, x, y, text in lines if x < 150 and DATE.fullmatch(text))
            links = [link for link in page.get_links() if link.get("uri")]
            for i, (y, date) in enumerate(dates):
                # Link rectangles sit slightly above the corresponding text baseline.
                lower = y - 8
                upper = dates[i + 1][0] - 8 if i + 1 < len(dates) else 790
                paragraphs = {}
                for block, x, line_y, text in lines:
                    if x >= 160 and lower <= line_y < upper and text:
                        if not text.startswith("Additional source links:"):
                            paragraphs.setdefault(block, []).append(text)
                body = [" ".join(parts) for parts in paragraphs.values()]
                if not body:
                    raise ValueError(f"No seminar text at {date}, PDF page {page_number}")
                recordings, sources = [], []
                for link in links:
                    if not lower <= link["from"].y0 < upper:
                        continue
                    uri = link["uri"]
                    if urlsplit(uri).scheme not in {"http", "https"}:
                        raise ValueError(f"Unsupported link at {date}: {uri}")
                    if link["from"].x0 < 160:
                        if uri not in recordings:
                            recordings.append(uri)
                    elif urlsplit(uri).hostname != "staff.yandex-team.ru" and uri not in sources:
                        sources.append(uri)
                entries.append({
                    "date": datetime.strptime(date, "%d.%m.%y").date().isoformat(),
                    "page": page_number,
                    "body": body,
                    "recordings": recordings,
                    "sources": sources,
                })
        advertised = re.search(r"(\d+) seminars", document[0].get_text())
        if advertised is None or int(advertised[1]) != len(entries):
            raise ValueError("Extracted count does not match the archive cover; inspect its layout")
    return entries


def escape_text(text):
    # Preserve the displayed text without interpreting paper-title punctuation as Markdown.
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_\[\]])", r"\\\1", text)


def render_index(pdf_path, entries):
    counts = Counter(entry["date"][:4] for entry in entries)
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    lines = [
        "# Research seminar index", "",
        f"**{len(entries)} seminar entries** · " + " · ".join(f"{year}: {count}" for year, count in counts.items()), "",
        f"Source: [English recording archive]({pdf_path.name}). Recordings are in Russian.",
        "Search this page by topic, paper title, speaker, or date (YYYY-MM-DD).",
        "Entry descriptions and links reproduce the archive; they are not summaries of the recordings.",
        "Open the linked PDF page for access codes. Repeated dates identify separate entries.",
        "Links may require access; their availability has not been checked.", "",
        " · ".join(f"[{year}](#{year})" for year in counts), "",
    ]
    last_year = None
    occurrences = Counter()
    for entry in entries:
        year = entry["date"][:4]
        if year != last_year:
            lines.extend([f"## {year}", ""])
            last_year = year
        occurrences[entry["date"]] += 1
        number = occurrences[entry["date"]]
        suffix = f" (entry {number})" if number > 1 else ""
        lines.extend([f"### {entry['date']}{suffix}", ""])
        for paragraph in entry["body"]:
            lines.extend([escape_text(paragraph), ""])
        links = [f"[PDF entry / access code]({pdf_path.name}#page={entry['page']})"]
        for i, uri in enumerate(entry["recordings"], 1):
            label = "Recording" if len(entry["recordings"]) == 1 else f"Recording {i}"
            links.append(f"[{label}](<{uri}>)")
        lines.extend([" · ".join(links), ""])
        if not entry["recordings"]:
            lines.extend(["No recording link is listed for this entry in the archive.", ""])
        if entry["sources"]:
            lines.extend(["Source links from this entry: " + " · ".join(
                f"[{i}](<{uri}>)" for i, uri in enumerate(entry["sources"], 1)), ""])
    lines.extend([
        "## Updating this index", "",
        "Generated from the PDF with [build_index.py](scripts/build_index.py).",
        "Speaker-profile hyperlinks are omitted; names remain searchable in the entry text.",
        "After replacing the archive, run from the repository root:", "",
        "```bash",
        "python -m pip install pymupdf",
        "python research_seminars/scripts/build_index.py",
        "python research_seminars/scripts/build_index.py --check",
        "```", "",
        "The extractor expects this archive's two-column layout. If the layout or filename changes,",
        "update the extractor and verify entry boundaries and links before publishing the new index.", "",
        f"Source PDF SHA-256: `{digest}`", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the existing index without writing")
    args = parser.parse_args()
    pdf_path = ARCHIVE_DIR / "diffusion_research_seminars_09_2026.pdf"
    output = ARCHIVE_DIR / "INDEX.md"
    entries = extract_entries(pdf_path)
    rendered = render_index(pdf_path, entries)
    if args.check:
        if not output.exists() or output.read_text() != rendered:
            parser.exit(1, "Seminar index is missing or stale; run build_index.py to regenerate it.\n")
        print(f"Index matches the PDF: {len(entries)} entries.")
    else:
        output.write_text(rendered)
        print(f"Wrote {output}: {len(entries)} entries.")


if __name__ == "__main__":
    main()
