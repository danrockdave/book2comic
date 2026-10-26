"""Extract the book text from the PDF into a structured JSON.

Output structure (book.json):
{
  "stories": [
    {"index": 1, "title": "O Bruxo", "sections": [
        {"number": "I", "paragraphs": [
            {"id": "p0001", "segments": [
                {"id": "p0001.0", "kind": "narration" | "speech", "text": "..."}
            ]}
        ]}
    ]}
  ]
}

Paragraph and segment ids are deterministic, so annotation files can refer to
them without carrying the book text.

Usage:
    python extract.py livro.pdf book.json
"""

import json
import re
import sys

import pymupdf

TITLE_SIZE = 34
SECTION_SIZE = 26
DROP_CAP_SIZE = 60
HEADER_MAX_Y = 45
FOOTER_MIN_Y = 790
INDENT_MIN_X = 80

DASHES = "–—-"
# A dialogue dash is a dash preceded by whitespace and followed by whitespace
# or punctuation (e.g. "disse Nohorn –, esclareço").
SPEECH_SPLIT = re.compile(r"\s+[–—]\s*|\s+-\s+")


def iter_lines(doc):
    """Yield (page, size, x0, y0, y1, text) for every text line, in reading order."""
    for page_no, page in enumerate(doc):
        lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"])
                if not text.strip():
                    continue
                x0, y0, _, y1 = line["bbox"]
                if y0 < HEADER_MAX_Y or y0 > FOOTER_MIN_Y:
                    continue
                size = max(span["size"] for span in line["spans"])
                lines.append((page_no, size, x0, y0, y1, text))
        lines.sort(key=lambda item: (round(item[3]), item[2]))
        # Justified lines with wide gaps can be split into several pieces: merge
        # pieces that share the same baseline.
        merged = []
        for item in lines:
            last = merged[-1] if merged else None
            if last and abs(last[3] - item[3]) < 3 and abs(last[1] - item[1]) < 1:
                merged[-1] = (last[0], max(last[1], item[1]), last[2], last[3],
                              max(last[4], item[4]), last[5].rstrip() + " " + item[5].lstrip())
            else:
                merged.append(item)
        yield from merged


def join_line(current, text):
    text = text.strip()
    if not current:
        return text
    # Words broken at line end keep the hyphen (Portuguese compounds: "aproximou-se").
    if current.endswith("-"):
        return current + text
    return current + " " + text


def split_segments(paragraph_text):
    """Split a paragraph into narration/speech segments."""
    text = re.sub(r"\s+", " ", paragraph_text).strip()
    if not text or text[0] not in DASHES:
        return [("narration", text)]
    body = text.lstrip(DASHES).strip()
    parts = SPEECH_SPLIT.split(body)
    segments = []
    for i, part in enumerate(parts):
        part = part.strip().lstrip(",;").strip()
        if not part:
            continue
        kind = "speech" if i % 2 == 0 else "narration"
        segments.append((kind, part))
    return segments


def extract(pdf_path):
    doc = pymupdf.open(pdf_path)
    stories = []
    story = None
    section = None
    paragraph = ""
    pending_cap = ""
    cap_bottom = -1.0
    cap_page = -1
    para_counter = 0

    def flush():
        nonlocal paragraph, para_counter
        if paragraph.strip() and section is not None:
            para_counter += 1
            pid = f"p{para_counter:04d}"
            segments = [
                {"id": f"{pid}.{i}", "kind": kind, "text": text}
                for i, (kind, text) in enumerate(split_segments(paragraph))
            ]
            section["paragraphs"].append({"id": pid, "segments": segments})
        paragraph = ""

    for page_no, size, x0, y0, y1, text in iter_lines(doc):
        stripped = text.strip()
        if size >= DROP_CAP_SIZE:
            flush()
            cap_bottom = y1
            cap_page = page_no
            # Drop caps may come with a dialogue dash ("-S", "— G", "-...A").
            pending_cap = stripped.replace(" ", "")
            continue
        if size >= TITLE_SIZE and not re.fullmatch(r"[IVXL]+", stripped):
            flush()
            para_counter = 0
            story = {"index": len(stories) + 1, "title": stripped, "sections": []}
            stories.append(story)
            section = None
            continue
        if size >= SECTION_SIZE and re.fullmatch(r"[IVXL]+", stripped):
            flush()
            if story is None:
                continue
            section = {"number": stripped, "paragraphs": []}
            story["sections"].append(section)
            continue
        if section is None:
            continue
        if pending_cap:
            flush()
            cap = pending_cap
            pending_cap = ""
            if cap[0] in DASHES:
                cap = "– " + cap.lstrip(DASHES)
            paragraph = cap + stripped
            continue
        # Lines beside a drop cap are indented but still belong to the first paragraph.
        beside_cap = page_no == cap_page and y0 < cap_bottom - 2
        if x0 > INDENT_MIN_X and not beside_cap:
            flush()
        paragraph = join_line(paragraph, stripped)
    flush()
    return {"stories": stories}


def main():
    if len(sys.argv) != 3:
        print("Uso: python extract.py livro.pdf book.json")
        sys.exit(1)
    book = extract(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(book, f, ensure_ascii=False, indent=1)
    for s in book["stories"]:
        n = sum(len(sec["paragraphs"]) for sec in s["sections"])
        print(f"{s['index']:02d} {s['title']}: {len(s['sections'])} seções, {n} parágrafos")


if __name__ == "__main__":
    main()
