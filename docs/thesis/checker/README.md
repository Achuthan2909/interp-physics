# buthesis-check

Checks a thesis PDF against the formatting rules of the BU CDS thesis
template. It reads the PDF rather than the LaTeX, so it works the same for
theses written with the LaTeX template, the Word template, or anything else.

## Install and run

```
pip install ./checker          # from the root of the template
buthesis-check thesis.pdf
```

The command exits with status 1 if any check fails, and 0 otherwise.

Options:

- `--skip placeholders,template-notes` skips the listed rules.
- `--json` prints a machine-readable report.
- `--spec rules.toml` checks against a different rules file.

## What it checks

| Rule | Checks that |
|---|---|
| `page-size` | Every page is US Letter |
| `margins` | Text, images and table rules stay inside 1.5 in (top, left) and 1 in (right, bottom) margins; page numbers excepted |
| `page-numbers` | Pages i–iii have no number; front matter uses roman numerals centered in the footer from iv; main text uses arabic numerals centered in the header from 1; no gaps |
| `fonts-embedded` | Every font is embedded in the PDF |
| `body-font` | Most of the text is set in a Times face |
| `body-size` | Most of the text is 12 pt |
| `structure` | Title, copyright and approval pages come first; Abstract, Table of Contents, Chapter One, Bibliography and Curriculum Vitae are present and in order; appendices sit between the last chapter and the bibliography |
| `placeholders` | The template's sample text (names, "202X", sample paragraphs) has been replaced |
| `template-notes` | The template's instruction notes have been removed |

Page references in the report are PDF page numbers (1 = title page), not the
printed numbers.

The values behind each rule are in
[`buthesis_check/spec.toml`](buthesis_check/spec.toml). They mirror
`buthesis.cls`; if one changes, change the other and bump `version`.

## Tests

```
pip install "./checker[test]"
pytest checker/tests
```

The tests build the template once cleanly, then once per rule with a
deliberate formatting error, and check that exactly the expected rules fail.
They need `latexmk` and a TeX distribution; without them only the tests on the
committed `thesis.pdf` run.
