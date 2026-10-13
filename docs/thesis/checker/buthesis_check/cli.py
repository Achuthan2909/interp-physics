"""Command-line entry point: buthesis-check thesis.pdf"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from importlib import resources

from . import __version__
from .checks import CHECKS
from .document import load

MAX_SHOWN = 10


def load_spec(path: str | None) -> dict:
    if path:
        with open(path, "rb") as f:
            return tomllib.load(f)
    return tomllib.loads(resources.files(__package__).joinpath("spec.toml").read_text())


def run(pdf_path: str, spec: dict, skip: set[str]) -> dict:
    doc = load(pdf_path)
    results = []
    for check in CHECKS:
        if check.rule in skip:
            results.append({"rule": check.rule, "title": check.title,
                            "status": "skip", "findings": []})
            continue
        findings = check.run(doc, spec)
        results.append({
            "rule": check.rule,
            "title": check.title,
            "status": "fail" if findings else "pass",
            "findings": [{"page": f.page, "message": f.message} for f in findings],
        })
    return {
        "checker_version": __version__,
        "spec_version": spec.get("version"),
        "file": pdf_path,
        "pages": len(doc.pages),
        "results": results,
    }


def page_ranges(pages: list[int]) -> str:
    """[4, 5, 6, 9] -> 'pages 4-6, 9'"""
    runs: list[list[int]] = []
    for p in sorted(pages):
        if runs and p == runs[-1][-1] + 1:
            runs[-1].append(p)
        else:
            runs.append([p])
    text = ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs)
    return ("pages " if len(pages) > 1 else "page ") + text


def group_findings(findings: list[dict]) -> list[str]:
    """Merge findings with the same message into one line with page ranges."""
    pages_by_message: dict[str, list[int]] = {}
    for f in findings:
        pages_by_message.setdefault(f["message"], [])
        if f["page"] is not None:
            pages_by_message[f["message"]].append(f["page"])
    return [f"{page_ranges(pages)}: {message}" if pages else message
            for message, pages in pages_by_message.items()]


def format_text(report: dict) -> str:
    out = [f"buthesis-check {report['checker_version']} (rules {report['spec_version']}): "
           f"{report['file']}, {report['pages']} pages", ""]
    width = max(len(r["rule"]) for r in report["results"])
    for r in report["results"]:
        out.append(f"{r['status'].upper():4}  {r['rule']:{width}}  {r['title']}")
        lines = group_findings(r["findings"])
        for line in lines[:MAX_SHOWN]:
            out.append(f"{'':{width + 8}}- {line}")
        if len(lines) > MAX_SHOWN:
            out.append(f"{'':{width + 8}}  ... and {len(lines) - MAX_SHOWN} more")
    failed = sum(r["status"] == "fail" for r in report["results"])
    ran = sum(r["status"] != "skip" for r in report["results"])
    out += ["", f"{failed} of {ran} checks failed." if failed else f"All {ran} checks passed."]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    rules = [c.rule for c in CHECKS]
    parser = argparse.ArgumentParser(
        prog="buthesis-check",
        description="Check a thesis PDF against the BU CDS thesis formatting rules.")
    parser.add_argument("pdf", help="the thesis PDF to check")
    parser.add_argument("--skip", default="", metavar="RULES",
                        help=f"comma-separated rules to skip: {', '.join(rules)}")
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    parser.add_argument("--spec", help="use a different rules file (TOML)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(rules)
    if unknown:
        parser.error(f"unknown rule(s): {', '.join(sorted(unknown))}")

    try:
        report = run(args.pdf, load_spec(args.spec), skip)
    except Exception as exc:  # unreadable or malformed PDF
        print(f"buthesis-check: cannot read {args.pdf}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2) if args.json else format_text(report))
    return 1 if any(r["status"] == "fail" for r in report["results"]) else 0


if __name__ == "__main__":
    sys.exit(main())
