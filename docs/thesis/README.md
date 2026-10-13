# CDS Masters Thesis Template — LaTeX version

A LaTeX port of Boston University's *General Thesis Template* (Word, July 2026).
It reproduces the Word template's layout: page geometry, fonts, spacing, headings,
page numbering, table of contents, list styles and captions.

## Build

```
latexmk -pdf thesis.tex
# or: pdflatex thesis; bibtex thesis; pdflatex thesis; pdflatex thesis
```

Works on Overleaf. Upload the folder (or the zip) and set `thesis.tex` as the main file.

## Files

| File | What it is |
|---|---|
| `thesis.tex` | Main file: your title, name, degree and year, and the order of the pages |
| `buthesis.cls` | All the formatting. You normally don't need to edit it |
| `frontmatter/` | Approval page, epigraph, dedication, acknowledgments, abstract, preface, lists, abbreviations, glossary |
| `chapters/` | One file per chapter |
| `backmatter/` | Appendix, notes shown after the bibliography, and curriculum vitae |
| `references.bib` | Your bibliography entries. The three sample entries are placeholders |
| `checker/` | `buthesis-check`, which checks the finished PDF against the formatting rules |

## Checking your formatting

`buthesis-check` reads your finished PDF and reports anything that breaks the
formatting rules: page size, margins, page numbering, fonts, the order of the
required pages, and leftover template text or instruction notes.

```
pip install ./checker
buthesis-check thesis.pdf
```

If your thesis is in a GitHub repository created from this template, the
*Check thesis formatting* workflow builds and checks it on every push. See
[`checker/README.md`](checker/README.md) for the full list of rules.

## Formatting reproduced from the Word template

- US Letter paper. Margins: 1.5 in at top and left, 1 in at right and bottom.
- 12 pt Times, double spaced, with a 0.5 in first-line indent. Text is ragged right and not hyphenated, as in Word.
- Pages i–iii (title, copyright, approval) show no page number. From the epigraph onward, the preliminary pages use lower-case roman numerals centered in the footer.
- The main text uses arabic numerals centered in the header, starting at 1 on Chapter One. Chapter opening pages are numbered the same way.
- Chapter headings read "CHAPTER ONE", "CHAPTER TWO", and so on, bold and centered. Sections are bold and flush left, and subsections are italic.
- The table of contents and lists use dot leaders. Sections are indented under their chapter.
- Captions are 11 pt bold and sit below the table or figure: "Table 1. My First Table."
  Tables, figures and illustrations are numbered continuously through the thesis.
- The bibliography is single spaced with a hanging indent and 12 pt between entries. It comes after the appendices and before the CV.

## Everyday use

- **Chapter titles:** `\chapter{}` gives just "CHAPTER ONE", as in the template.
  `\chapter{INTRODUCTION}` puts the title on a second line under "CHAPTER ONE".
- **Other headings:** `\buheading{...}` makes a preliminary or back-matter heading and adds it to the table of contents. Use it for extra appendices, for example
  `\buheading{APPENDIX B: SURVEY INSTRUMENT}`.
- **Floats:** use `table`, `figure` and `illustration`. Each one gets its own list automatically.
- **Removing optional pages:** delete the matching `\input{...}` line in `thesis.tex`, or the `\listof...` line in `frontmatter/lists.tex`.
- **Instruction notes:** the template's italic instruction notes are shown by default. Change `\documentclass[notes]{buthesis}` to `[nonotes]` to hide them all for the final PDF.
- **Citation style:** `plainnat` sorts alphabetically by author. For numbered citations in order of appearance, switch to `unsrtnat`. Remove `\nocite{*}` once you cite your sources for real.

## Deliberate differences from the Word file

- In the Word file, "Section Two" is typed as plain text instead of in the Heading 2 style. Here it is a proper `\section` and appears bold, like "Section One".
- In the Word file, the figure caption and its List of Figures entry have no number. Here figures are numbered ("Figure 1.") to match the tables.
- The Word-specific instructions ("show all nonprinting characters", "section break") have been reworded for LaTeX.
