import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TEX_SOURCES = ["thesis.tex", "buthesis.cls", "references.bib",
               "frontmatter", "chapters", "backmatter", "figures"]


def build_thesis(workdir: Path, edits: list[tuple[str, str]]) -> Path:
    """Copy the template into workdir, apply (find, replace) edits to
    thesis.tex and build it. Returns the PDF path."""
    for name in TEX_SOURCES:
        src = REPO / name
        if src.is_dir():
            shutil.copytree(src, workdir / name)
        elif src.exists():
            shutil.copy2(src, workdir / name)
    main = workdir / "thesis.tex"
    tex = main.read_text()
    for find, replace in edits:
        assert find in tex, f"edit target not found in thesis.tex: {find!r}"
        tex = tex.replace(find, replace, 1)
    main.write_text(tex)
    result = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "thesis.tex"],
        cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        log = (workdir / "thesis.log")
        tail = log.read_text(errors="replace")[-3000:] if log.exists() else result.stdout[-3000:]
        pytest.fail(f"LaTeX build failed:\n{tail}")
    return workdir / "thesis.pdf"


@pytest.fixture(scope="session")
def thesis_builder(tmp_path_factory):
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk not installed")
    cache: dict[str, Path] = {}

    def build(name: str, edits: list[tuple[str, str]]) -> Path:
        if name not in cache:
            cache[name] = build_thesis(tmp_path_factory.mktemp(name), edits)
        return cache[name]

    return build
