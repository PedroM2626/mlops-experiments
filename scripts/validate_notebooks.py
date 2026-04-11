#!/usr/bin/env python3
"""Validate notebook structure and basic Python syntax in code cells.

- Scans all .ipynb files under the repository.
- Sanitizes notebook magics and shell/help lines (!, %, ?) before compilation.
- Marks notebooks in known external folders as EXT (skipped from syntax checks).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


EXTERNAL_PATH_MARKERS = (
    "databricks forecast",
    "ibm-experiments",
)


@dataclass
class NotebookResult:
    path: Path
    status: str  # PASS | FAIL | EXT | ERROR
    message: str = ""


def is_external_notebook(path: Path) -> bool:
    lowered = str(path).lower().replace("\\", "/")
    return any(marker in lowered for marker in EXTERNAL_PATH_MARKERS)


def sanitize_line_for_syntax(line: str) -> str:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    if not stripped:
        return line

    first = stripped[0]
    if first in ("!", "%", "?"):
        return f"{indent}pass\n"

    return line


def sanitize_cell_source(source: Iterable[str] | str) -> str:
    if isinstance(source, str):
        lines = source.splitlines(keepends=True)
    else:
        lines = list(source)

    sanitized = [sanitize_line_for_syntax(line) for line in lines]
    return "".join(sanitized)


def validate_notebook(path: Path) -> NotebookResult:
    try:
        raw = path.read_text(encoding="utf-8")
        nb = json.loads(raw)
    except Exception as exc:
        return NotebookResult(path=path, status="ERROR", message=f"json-read: {exc}")

    if is_external_notebook(path):
        return NotebookResult(path=path, status="EXT", message="external notebook")

    cells = nb.get("cells", [])
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        source = sanitize_cell_source(cell.get("source", ""))
        if not source.strip():
            continue

        try:
            compile(source, f"{path}::cell_{idx}", "exec")
        except SyntaxError as exc:
            return NotebookResult(
                path=path,
                status="FAIL",
                message=f"cell {idx}: line {exc.lineno}: {exc.msg}",
            )
        except Exception as exc:
            return NotebookResult(path=path, status="ERROR", message=f"cell {idx}: {exc}")

    return NotebookResult(path=path, status="PASS")


def collect_notebooks(repo_root: Path) -> List[Path]:
    return sorted(p for p in repo_root.rglob("*.ipynb") if ".git" not in p.parts)


def print_results(results: List[NotebookResult], repo_root: Path) -> Tuple[int, int, int, int]:
    pass_n = fail_n = ext_n = err_n = 0

    for result in results:
        rel = result.path.relative_to(repo_root)
        if result.status == "PASS":
            pass_n += 1
            print(f"PASS {rel}")
        elif result.status == "FAIL":
            fail_n += 1
            print(f"FAIL {rel} :: {result.message}")
        elif result.status == "EXT":
            ext_n += 1
            print(f"EXT  {rel} :: {result.message}")
        else:
            err_n += 1
            print(f"ERROR {rel} :: {result.message}")

    print(
        "\nSummary: "
        f"PASS={pass_n} FAIL={fail_n} EXT={ext_n} ERROR={err_n} TOTAL={len(results)}"
    )

    return pass_n, fail_n, ext_n, err_n


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate notebooks in this repository")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root path",
    )
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    notebooks = collect_notebooks(repo_root)
    if not notebooks:
        print("No notebooks found.")
        return 0

    results = [validate_notebook(path) for path in notebooks]
    _, fail_n, _, err_n = print_results(results, repo_root)

    return 1 if (fail_n + err_n) else 0


if __name__ == "__main__":
    raise SystemExit(main())
