"""Validate or recursively inline a LaTeX manuscript into its canonical file."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "paper_revision" / "GNN_IEEE_REVISED_SINGLE.tex"
DEFAULT_OUTPUT = ROOT / "docs" / "paper_revision" / "GNN_IEEE_REVISED_SINGLE.tex"
INPUT_PATTERN = re.compile(r"\\input\{([^}]+)\}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inline(path: Path, stack: tuple[Path, ...], sources: list[Path]) -> str:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Recursive LaTeX input detected: {chain}")
    if not path.exists():
        raise FileNotFoundError(path)
    sources.append(path)
    text = path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        relative = Path(match.group(1))
        if relative.suffix == "":
            relative = relative.with_suffix(".tex")
        child = (path.parent / relative).resolve()
        body = inline(child, (*stack, path), sources)
        display = child.relative_to(DEFAULT_SOURCE.parent.resolve()).as_posix()
        return (
            f"% ===== BEGIN INLINED SOURCE: {display} =====\n"
            f"{body.rstrip()}\n"
            f"% ===== END INLINED SOURCE: {display} ====="
        )

    return INPUT_PATTERN.sub(replace, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sources: list[Path] = []
    merged = inline(args.source, (), sources)
    remaining = INPUT_PATTERN.findall(merged)
    if remaining:
        raise RuntimeError(f"Unresolved LaTeX inputs: {remaining}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(merged.rstrip() + "\n", encoding="utf-8", newline="\n")

    print(f"output={args.output.resolve()}")
    print(f"sources={len(sources)}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"sha256={sha256(args.output)}")
    for source in sources:
        print(f"inlined={source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
