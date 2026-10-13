"""Load a PDF into the small page model the checks work on.

All coordinates are in PDF points (1/72 in) measured from the top-left
corner of the page, as pdfplumber reports them.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import pdfplumber
from pdfminer.pdftypes import resolve1

PT_PER_IN = 72.0

# TeX's interword spaces are narrower than pdfplumber's default merge
# threshold, which runs words together; this keeps them apart.
X_TOLERANCE = 1.5


@dataclass
class Line:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class Box:
    kind: str
    x0: float
    x1: float
    top: float
    bottom: float


@dataclass
class Page:
    number: int  # 1-based physical page number
    width: float
    height: float
    lines: list[Line]
    boxes: list[Box]
    text: str
    font_chars: Counter = field(default_factory=Counter)  # (font, size) -> chars


@dataclass
class FontInfo:
    name: str
    embedded: bool
    pages: set[int]


@dataclass
class Document:
    path: str
    pages: list[Page]
    fonts: dict[str, FontInfo]


def normalize(text: str) -> str:
    """Case- and whitespace-insensitive form used for all text matching."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", "", text).casefold()


def base_font_name(name: str) -> str:
    """Drop the six-letter subset prefix, e.g. 'ABCDEF+Times' -> 'Times'."""
    return re.sub(r"^[A-Z]{6}\+", "", name or "")


def load(path: str) -> Document:
    pages: list[Page] = []
    fonts: dict[str, FontInfo] = {}
    with pdfplumber.open(path) as pdf:
        for index, p in enumerate(pdf.pages, start=1):
            pages.append(_load_page(index, p))
            for name, embedded in _page_fonts(p):
                info = fonts.setdefault(name, FontInfo(name, embedded, set()))
                info.embedded = info.embedded and embedded
                info.pages.add(index)
    return Document(path=path, pages=pages, fonts=fonts)


def _load_page(number: int, p) -> Page:
    lines = [
        Line(l["text"], l["x0"], l["x1"], l["top"], l["bottom"])
        for l in p.extract_text_lines(x_tolerance=X_TOLERANCE, return_chars=False)
        if l["text"].strip()
    ]
    boxes = [
        Box("text", c["x0"], c["x1"], c["top"], c["bottom"])
        for c in p.chars
        if c["text"].strip()
    ]
    for kind in ("images", "rects", "lines", "curves"):
        boxes += [
            Box(kind.rstrip("s"), o["x0"], o["x1"], o["top"], o["bottom"])
            for o in getattr(p, kind)
        ]
    font_chars = Counter(
        (base_font_name(c["fontname"]), round(c["size"], 1))
        for c in p.chars
        if c["text"].strip()
    )
    return Page(
        number=number,
        width=float(p.width),
        height=float(p.height),
        lines=lines,
        boxes=boxes,
        text=p.extract_text(x_tolerance=X_TOLERANCE) or "",
        font_chars=font_chars,
    )


def _page_fonts(p):
    """Yield (font name, embedded?) for every font the page uses."""
    yield from _resource_fonts(p.page_obj.resources, depth=0)


def _resource_fonts(resources, depth: int):
    resources = resolve1(resources) or {}
    for ref in (resolve1(resources.get("Font")) or {}).values():
        font = resolve1(ref)
        yield base_font_name(_name(font.get("BaseFont"))), _is_embedded(font)
    if depth < 4:
        for ref in (resolve1(resources.get("XObject")) or {}).values():
            xobj = resolve1(ref)
            attrs = getattr(xobj, "attrs", {})
            if _name(attrs.get("Subtype")) == "Form" and "Resources" in attrs:
                yield from _resource_fonts(attrs["Resources"], depth + 1)


def _is_embedded(font) -> bool:
    subtype = _name(font.get("Subtype"))
    if subtype == "Type3":
        return True  # glyphs are defined inside the PDF itself
    if subtype == "Type0":
        descendants = resolve1(font.get("DescendantFonts")) or []
        if not descendants:
            return False
        font = resolve1(descendants[0])
    descriptor = resolve1(font.get("FontDescriptor")) or {}
    return any(k in descriptor for k in ("FontFile", "FontFile2", "FontFile3"))


def _name(value) -> str:
    value = resolve1(value)
    return getattr(value, "name", value if isinstance(value, str) else "") or ""
