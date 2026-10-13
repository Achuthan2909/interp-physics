"""The formatting rules. Each check takes a Document and the spec and
returns a list of Findings; an empty list means the rule passed."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .document import PT_PER_IN, Document, Line, Page, normalize


@dataclass
class Finding:
    page: int | None
    message: str


@dataclass
class Check:
    rule: str
    title: str
    run: Callable[[Document, dict], list[Finding]]


# --------------------------------------------------------------------------
#  Geometry helpers
# --------------------------------------------------------------------------

def _pt(inches: float) -> float:
    return inches * PT_PER_IN


def _text_area(page: Page, spec: dict) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) edges of the text block, in points."""
    m = spec["margins"]
    return (
        _pt(m["left_in"]),
        _pt(m["top_in"]),
        page.width - _pt(m["right_in"]),
        page.height - _pt(m["bottom_in"]),
    )


def _text_center(page: Page, spec: dict) -> float:
    left, _, right, _ = _text_area(page, spec)
    return (left + right) / 2


_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def roman_to_int(s: str) -> int | None:
    if not re.fullmatch(r"[ivxlcdm]+", s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMAN[a]
        total += -v if b != " " and _ROMAN[b] > v else v
    return total if int_to_roman(total) == s else None


def int_to_roman(n: int) -> str:
    out = ""
    for value, sym in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
                       (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
                       (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= value:
            out += sym
            n -= value
    return out


@dataclass
class PageNumber:
    position: str  # "header" or "footer"
    style: str  # "arabic", "roman" or "other"
    value: int | None
    text: str
    line: Line


def _margin_lines(page: Page, spec: dict) -> tuple[list[Line], list[Line]]:
    _, top, _, bottom = _text_area(page, spec)
    header = [l for l in page.lines if l.center_y < top]
    footer = [l for l in page.lines if l.center_y > bottom]
    return header, footer


def detect_page_number(page: Page, spec: dict) -> PageNumber | None:
    """The printed page number: a lone token in the header or footer zone."""
    header, footer = _margin_lines(page, spec)
    for position, lines in (("header", header), ("footer", footer)):
        if len(lines) != 1:
            continue
        text = lines[0].text.strip()
        if " " in text:
            continue
        if text.isdigit():
            return PageNumber(position, "arabic", int(text), text, lines[0])
        value = roman_to_int(text)
        if value is not None:
            return PageNumber(position, "roman", value, text, lines[0])
        if len(text) <= 6:
            return PageNumber(position, "other", None, text, lines[0])
    return None



# --------------------------------------------------------------------------
#  Checks
# --------------------------------------------------------------------------

def check_page_size(doc: Document, spec: dict) -> list[Finding]:
    s = spec["page"]
    want_w, want_h, tol = _pt(s["width_in"]), _pt(s["height_in"]), s["tolerance_pt"]
    return [
        Finding(p.number, f"page is {p.width / PT_PER_IN:.2f} x "
                          f"{p.height / PT_PER_IN:.2f} in, expected "
                          f"{s['width_in']} x {s['height_in']} in")
        for p in doc.pages
        if abs(p.width - want_w) > tol or abs(p.height - want_h) > tol
    ]


def check_margins(doc: Document, spec: dict) -> list[Finding]:
    tol = _pt(spec["margins"]["tolerance_in"])
    findings = []
    for page in doc.pages:
        left, top, right, bottom = _text_area(page, spec)
        number = detect_page_number(page, spec)
        boxes = page.boxes
        if number is not None:
            l = number.line
            boxes = [b for b in boxes
                     if not (b.top >= l.top - 1 and b.bottom <= l.bottom + 1
                             and b.x0 >= l.x0 - 1 and b.x1 <= l.x1 + 1)]
        if not boxes:
            continue
        overflow = {
            "left": left - min(b.x0 for b in boxes),
            "top": top - min(b.top for b in boxes),
            "right": max(b.x1 for b in boxes) - right,
            "bottom": max(b.bottom for b in boxes) - bottom,
        }
        for side, amount in overflow.items():
            if amount > tol:
                findings.append(Finding(
                    page.number,
                    f"content extends "
                    f"{amount / PT_PER_IN:.2f} in into the {side} margin"))
    return findings


def check_page_numbers(doc: Document, spec: dict) -> list[Finding]:
    s = spec["page_numbers"]
    unnumbered = s["unnumbered_pages"]
    center_tol = _pt(s["center_tolerance_in"])
    findings = []
    numbers = [detect_page_number(p, spec) for p in doc.pages]

    for page, num in zip(doc.pages[:unnumbered], numbers):
        if num is not None:
            findings.append(Finding(page.number, f"should not show a page number but shows '{num.text}'"))

    in_main = False
    for page, num in zip(doc.pages[unnumbered:], numbers[unnumbered:]):
        if num is None:
            findings.append(Finding(page.number, "has no page number"))
            continue
        if num.style == "arabic":
            in_main = True
        if num.style == "other":
            findings.append(Finding(page.number, f"'{num.text}' is not a valid page number"))
            continue
        expected_style, expected_pos = ("arabic", "header") if in_main else ("roman", "footer")
        if num.style != expected_style:
            findings.append(Finding(page.number, f"'{num.text}' should be a {expected_style} numeral"))
        if num.position != expected_pos:
            where = "front matter" if expected_style == "roman" else "main text"
            findings.append(Finding(page.number, f"{where} page numbers belong in the "
                                    f"{expected_pos}, "
                                    f"found in the {num.position}"))
        if abs(num.line.center_x - _text_center(page, spec)) > center_tol:
            findings.append(Finding(page.number, "page number is not centered"))

    findings += _check_sequence(doc, numbers, unnumbered)
    return findings


def _check_sequence(doc: Document, numbers, unnumbered: int) -> list[Finding]:
    """Roman numbers continue from the unnumbered pages (iv, v, ...);
    arabic numbers start at 1 and have no gaps."""
    findings = []
    expected_roman = unnumbered + 1
    expected_arabic = 1
    for page, num in zip(doc.pages[unnumbered:], numbers[unnumbered:]):
        if num is None or num.value is None:
            continue
        if num.style == "roman":
            if num.value != expected_roman:
                findings.append(Finding(page.number, f"expected "
                                        f"'{int_to_roman(expected_roman)}', found '{num.text}'"))
            expected_roman = num.value + 1
        else:
            if num.value != expected_arabic:
                findings.append(Finding(page.number, f"expected "
                                        f"'{expected_arabic}', found '{num.text}'"))
            expected_arabic = num.value + 1
    return findings


def check_fonts_embedded(doc: Document, spec: dict) -> list[Finding]:
    return [
        Finding(None, f"font '{f.name}' is not embedded "
                              f"(first used on page {min(f.pages)})")
        for f in sorted(doc.fonts.values(), key=lambda f: min(f.pages))
        if not f.embedded
    ]


def _body_font_stats(doc: Document, spec: dict) -> Counter:
    unnumbered = spec["page_numbers"]["unnumbered_pages"]
    stats = Counter()
    for page in doc.pages[unnumbered:]:
        stats.update(page.font_chars)
    return stats


def check_body_font(doc: Document, spec: dict) -> list[Finding]:
    by_font = Counter()
    for (font, _), n in _body_font_stats(doc, spec).items():
        by_font[font] += n
    if not by_font:
        return [Finding(None, "no text found after the first pages")]
    font, n = by_font.most_common(1)[0]
    patterns = spec["fonts"]["body_font_patterns"]
    if any(p in font.lower() for p in patterns):
        return []
    share = 100 * n / sum(by_font.values())
    return [Finding(None, f"body text is set in '{font}' ({share:.0f}% of "
                          "characters); expected a Times font")]


def check_body_size(doc: Document, spec: dict) -> list[Finding]:
    by_size = Counter()
    for (_, size), n in _body_font_stats(doc, spec).items():
        by_size[size] += n
    if not by_size:
        return [Finding(None, "no text found after the first pages")]
    size, n = by_size.most_common(1)[0]
    want = spec["fonts"]["body_size_pt"]
    if abs(size - want) <= spec["fonts"]["size_tolerance_pt"]:
        return []
    share = 100 * n / sum(by_size.values())
    return [Finding(None, f"body text is {size:g} pt ({share:.0f}% of characters); "
                          f"expected {want:g} pt")]


# Loose on purpose: a heading only has to be roughly centered to be
# recognised, so a wrong margin is reported once (by `margins`), not again
# as missing sections.
HEADING_CENTER_TOLERANCE_IN = 0.75


def _headings(doc: Document, spec: dict) -> list[tuple[int, str]]:
    """Centered lines that could be headings, as (page number, normalized text).
    Table-of-contents entries are excluded by their dot leaders."""
    out = []
    for page in doc.pages:
        center = _text_center(page, spec)
        _, top, _, bottom = _text_area(page, spec)
        for line in page.lines:
            if not (top - 2 <= line.center_y <= bottom):
                continue
            if abs(line.center_x - center) > _pt(HEADING_CENTER_TOLERANCE_IN):
                continue
            text = normalize(line.text)
            if "..." in text:
                continue
            out.append((page.number, text))
    return out


def check_structure(doc: Document, spec: dict) -> list[Finding]:
    s = spec["structure"]
    findings = []

    for index, key, what in ((0, "title_page_text", "title page"),
                             (1, "copyright_page_text", "copyright page"),
                             (2, "approval_page_text", "approval page")):
        if len(doc.pages) <= index or normalize(s[key]) not in normalize(doc.pages[index].text):
            findings.append(Finding(index + 1, f"should be the {what} "
                                    f"(containing '{s[key]}')"))

    headings = _headings(doc, spec)
    found: list[tuple[str, int]] = []
    for group in s["required_in_order"]:
        wanted = {normalize(g) for g in group}
        pages = [p for p, text in headings if text in wanted]
        name = " / ".join(group)
        if not pages:
            findings.append(Finding(None, f"missing required section: {name}"))
            continue
        found.append((name, pages[0]))

    for (a, pa), (b, pb) in zip(found, found[1:]):
        if pb < pa:
            findings.append(Finding(None, f"{b} (page {pb}) should come after "
                                        f"{a} (page {pa})"))

    chapter_pages = [p for p, t in headings if re.fullmatch(r"chapter[a-z]+", t)]
    bib_group = {normalize(g) for g in s["required_in_order"][3]}
    bib_pages = [p for p, t in headings if t in bib_group]
    prefix = normalize(s["appendix_prefix"])
    for page, text in headings:
        if not text.startswith(prefix):
            continue
        if chapter_pages and page < max(chapter_pages):
            findings.append(Finding(page, "appendix comes before the last chapter"))
        if bib_pages and page > bib_pages[0]:
            findings.append(Finding(page, "appendix comes after the bibliography; "
                                          "appendices go before it"))
    return findings


def _phrase_findings(doc: Document, phrases: list[str], what: str) -> list[Finding]:
    pages = [(p.number, normalize(p.text)) for p in doc.pages]
    findings = []
    for phrase in phrases:
        needle = normalize(phrase)
        hits = [n for n, text in pages if needle in text]
        if hits:
            where = ", ".join(str(h) for h in hits[:5]) + (" ..." if len(hits) > 5 else "")
            findings.append(Finding(None, f"{what} '{phrase}' on page(s) {where}"))
    return findings


def check_placeholders(doc: Document, spec: dict) -> list[Finding]:
    return _phrase_findings(doc, spec["placeholders"]["phrases"], "template sample text")


def check_template_notes(doc: Document, spec: dict) -> list[Finding]:
    return _phrase_findings(doc, spec["template_notes"]["phrases"], "template instruction")


CHECKS: list[Check] = [
    Check("page-size", "Pages are US Letter", check_page_size),
    Check("margins", "Content stays inside the margins", check_margins),
    Check("page-numbers", "Page numbers are placed and sequenced correctly", check_page_numbers),
    Check("fonts-embedded", "All fonts are embedded", check_fonts_embedded),
    Check("body-font", "Body text is set in Times", check_body_font),
    Check("body-size", "Body text is 12 pt", check_body_size),
    Check("structure", "Required pages are present and in order", check_structure),
    Check("placeholders", "Template sample text has been replaced", check_placeholders),
    Check("template-notes", "Template instruction notes have been removed", check_template_notes),
]
