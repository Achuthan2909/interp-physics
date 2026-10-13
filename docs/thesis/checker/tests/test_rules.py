"""Build the template with one deliberate formatting error at a time and
check that the checker reports exactly the expected rules."""

import pytest

from buthesis_check.checks import int_to_roman, roman_to_int
from buthesis_check.cli import load_spec, run

from conftest import REPO

NONOTES = ("\\documentclass[notes]{buthesis}", "\\documentclass[nonotes]{buthesis}")
AFTER_CLASS = "\\documentclass[nonotes]{buthesis}\n"


def preamble(tex: str) -> tuple[str, str]:
    return (AFTER_CLASS, AFTER_CLASS + tex + "\n")


def failed_rules(pdf, skip=("placeholders",)):
    report = run(str(pdf), load_spec(None), set(skip))
    return {r["rule"] for r in report["results"] if r["status"] == "fail"}


# name -> (edits applied to thesis.tex, rules expected to fail)
CASES = {
    "clean": ([NONOTES], set()),
    # A4 is narrower, so the approval page's signature line overflows too
    "a4-paper": ([NONOTES, preamble(r"\geometry{a4paper}")], {"page-size", "margins"}),
    "narrow-left-margin": ([NONOTES, preamble(r"\geometry{left=1in}")],
                           {"margins", "page-numbers"}),
    "sans-serif-font": ([NONOTES, preamble(r"\usepackage{helvet}"
                                            r"\renewcommand{\familydefault}{\sfdefault}")],
                        {"body-font"}),
    "11pt-body": ([NONOTES, preamble(r"\renewcommand{\normalsize}{\fontsize{11}{13.6}\selectfont}"
                                      r"\AtBeginDocument{\normalsize}")],
                  {"body-size"}),
    "arabic-front-matter": ([NONOTES, (r"\prelims ", r"\prelims\pagenumbering{arabic} ")],
                            {"page-numbers"}),
    "missing-cv": ([NONOTES, (r"\input{backmatter/cv}", "")], {"structure"}),
    "appendix-after-bibliography": (
        [NONOTES,
         (r"\input{backmatter/appendix}", ""),
         (r"\input{backmatter/bibliography-notes}",
          r"\input{backmatter/bibliography-notes}" "\n" r"\input{backmatter/appendix}")],
        {"structure"}),
}


@pytest.mark.parametrize("name", CASES)
def test_rule_violations(thesis_builder, name):
    edits, expected = CASES[name]
    assert failed_rules(thesis_builder(name, edits)) == expected


def test_template_with_notes_flags_notes(thesis_builder):
    assert failed_rules(thesis_builder("with-notes", [])) == {"template-notes"}


def test_placeholders_detected(thesis_builder):
    pdf = thesis_builder("clean", CASES["clean"][0])
    assert failed_rules(pdf, skip=()) == {"placeholders"}


def test_committed_pdf_matches_rules():
    """The PDF shipped in the repo keeps its sample text and notes but must
    otherwise follow every rule."""
    assert failed_rules(REPO / "thesis.pdf", skip=()) == {"placeholders", "template-notes"}


@pytest.mark.parametrize("n", [1, 3, 4, 9, 14, 40, 49, 90, 400, 1994])
def test_roman_round_trip(n):
    assert roman_to_int(int_to_roman(n)) == n


@pytest.mark.parametrize("s", ["iiii", "vx", "abc", "", "xm"])
def test_roman_rejects_invalid(s):
    assert roman_to_int(s) is None
