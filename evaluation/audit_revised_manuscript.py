"""Static consistency audit for the evidence-backed revised manuscript draft."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "docs" / "paper_revision"
MASTER = PAPER_ROOT / "GNN_IEEE_REVISED_SINGLE.tex"
REPORT = ROOT / "docs" / "PAPER_MANUSCRIPT_AUDIT_20260826.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def expand(path: Path, stack: list[Path], sources: set[Path]) -> str:
    if path in stack:
        raise ValueError(f"Recursive LaTeX input: {path}")
    stack.append(path)
    sources.add(path)
    text = path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        relative = match.group(1)
        child = (path.parent / relative).with_suffix(".tex")
        if not child.exists():
            raise FileNotFoundError(f"Missing LaTeX input {relative} from {path}")
        return expand(child, stack.copy(), sources)

    return re.sub(r"\\input\{([^}]+?)(?:\.tex)?\}", replace, text)


def balanced_braces(text: str) -> tuple[bool, int]:
    depth = 0
    escaped = False
    for character in text:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False, depth
    return depth == 0, depth


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    source_files: set[Path] = set()
    try:
        expanded = expand(MASTER, [], source_files)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        expanded = MASTER.read_text(encoding="utf-8") if MASTER.exists() else ""
    source = strip_comments(expanded)
    balanced, depth = balanced_braces(source)
    if not balanced:
        errors.append(f"Unbalanced braces; final depth={depth}")

    labels = re.findall(r"\\label\{([^}]+)\}", source)
    references = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", source)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    undefined = sorted(set(references) - set(labels))
    if duplicate_labels:
        errors.append("Duplicate labels: " + ", ".join(duplicate_labels))
    if undefined:
        errors.append("Undefined references: " + ", ".join(undefined))

    required_sections = (
        "Introduction",
        "Problem Formulation",
        "Adopted BP Association and Probabilistic Update",
        "Causal Group-Innovation Prior Correction",
        "Experimental Protocol",
        "Results and Discussion",
        "Conclusion",
    )
    for section in required_sections:
        if f"\\section{{{section}}}" not in source:
            errors.append(f"Missing required section: {section}")

    forbidden = (
        "Structural Embedding Attention Network (SEAN) is developed",
        "Graph-guided Adaptive Spectral Filter (GASF) is introduced",
        "uses group-structure supervision",
        "three-random-seed",
        "three-seed protocol completes",
        "[TBD]",
    )
    for phrase in forbidden:
        if phrase in source:
            errors.append(f"Forbidden/stale claim remains: {phrase}")

    citation_keys = sorted(
        {
            key.strip()
            for citation in re.findall(r"\\cite\{([^}]+)\}", source)
            for key in citation.split(",")
        }
    )
    class_match = re.search(r"\\documentclass(?:\[[^]]*\])?\{([^}]+)\}", source)
    if class_match:
        requested_class = class_match.group(1)
        class_path = PAPER_ROOT / f"{requested_class}.cls"
        if class_path.exists():
            provided_match = re.search(
                r"\\ProvidesClass\{([^}]+)\}",
                class_path.read_text(encoding="utf-8", errors="replace"),
            )
            if provided_match and provided_match.group(1) != requested_class:
                warnings.append(
                    f"{class_path.name} is requested as {requested_class} but declares "
                    f"\\ProvidesClass{{{provided_match.group(1)}}}; confirm the journal template."
                )
    bib = PAPER_ROOT / "ref.bib"
    if not bib.exists():
        warnings.append(
            "ref.bib is absent from docs/paper_revision and from the supplied paper directory; citation keys cannot be compiled or validated."
        )
    else:
        bib_text = bib.read_text(encoding="utf-8")
        bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))
        missing_keys = sorted(set(citation_keys) - bib_keys)
        if missing_keys:
            errors.append("Citation keys missing from ref.bib: " + ", ".join(missing_keys))

    if shutil.which("pdflatex") is None:
        warnings.append("pdflatex is not installed or not on PATH; compilation was not run.")
    if not (PAPER_ROOT / "IEEEtaes.cls").exists():
        warnings.append(
            "IEEEtaes.cls is not stored with the draft; compilation depends on an external TeX installation providing it."
        )

    tex_files = sorted(source_files)
    lines = [
        "# Revised Manuscript Static Audit (2026-08-26)",
        "",
        f"Static status: **{'PASS' if not errors else 'FAIL'}**",
        "",
        f"- Expanded source files: {len(tex_files)}",
        f"- Labels/references: {len(labels)}/{len(references)}",
        f"- Citation keys: {len(citation_keys)}",
        f"- Balanced braces: {balanced}",
        "",
        "## Source hashes",
        "",
    ]
    lines.extend(
        f"- `{path.relative_to(ROOT).as_posix()}`: `{sha256(path)}`" for path in tex_files
    )
    lines.extend(("", "## Errors", ""))
    lines.extend(f"- {message}" for message in errors)
    if not errors:
        lines.append("- None.")
    lines.extend(("", "## Warnings / external blockers", ""))
    lines.extend(f"- {message}" for message in warnings)
    if not warnings:
        lines.append("- None.")
    lines.extend(("", "## Citation keys used", "", ", ".join(citation_keys), ""))
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"errors={len(errors)} warnings={len(warnings)} report={REPORT}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
