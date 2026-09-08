"""Validacao estrutural dos notebooks do repositorio.

Uso:
    python scripts/validate_notebooks.py            # resumo + exit 1 se houver JSON quebrado
    python scripts/validate_notebooks.py --strict   # tambem falha se houver notebook sem outputs executados

Regras:
- Todo `*.ipynb` (fora .venv/.git/mlruns/artifacts) deve ser JSON valido no
  formato nbformat>=4 com ao menos 1 celula de codigo.
- Notebooks externos (originais cloud: `ibm-experiments/`, `databricks-forecast/`,
  ou com `EXT` no nome/primeira celula) sao marcados como EXT e dispensados da
  exigencia de outputs (rodam fora do repo, com credenciais cloud).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".venv311", ".git", ".states", ".kilo", ".qoder",
             "mlruns", "node_modules", "__pycache__"}
EXTERNAL_DIRS = {"ibm-experiments", "databricks-forecast"}


def is_external(path: Path, nb: dict) -> bool:
    if any(part in EXTERNAL_DIRS for part in path.parts):
        return True
    if "EXT" in path.stem.upper():
        return True
    try:
        first = (nb.get("cells") or [{}])[0]
        src = "".join(first.get("source") or []).upper()
        return "EXT" in src.split("\n")[0][:60]
    except Exception:
        return False


def check_notebook(path: Path) -> dict:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"path": str(path), "ok": False, "ext": False, "error": f"JSON invalido: {e}"}
    if not isinstance(nb, dict) or nb.get("nbformat", 0) < 4:
        return {"path": str(path), "ok": False, "ext": False, "error": "nbformat ausente ou < 4"}
    cells = nb.get("cells") or []
    n_code = sum(1 for c in cells if c.get("cell_type") == "code")
    if n_code == 0:
        if any(c.get("cell_type") == "markdown" for c in cells):
            return {"path": str(path), "ok": True, "ext": False,
                    "docs_only": True, "cells": len(cells), "code": 0,
                    "with_outputs": 0}
        return {"path": str(path), "ok": False, "ext": False, "error": "sem celulas"}
    ext = is_external(path, nb)
    n_out = sum(1 for c in cells
                if c.get("cell_type") == "code" and c.get("outputs"))
    return {"path": str(path), "ok": True, "ext": ext,
            "cells": len(cells), "code": n_code, "with_outputs": n_out}


def iter_notebooks(root: Path):
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        # poda antes de descer (evita .venv/mlruns com milhares de arquivos)
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and d != ".ipynb_checkpoints" and not d.startswith(".")
        )
        for fn in sorted(filenames):
            if fn.endswith(".ipynb"):
                yield Path(dirpath) / fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="falha tambem se notebook nao-EXT nao tiver outputs executados")
    args = ap.parse_args(argv)

    results = [check_notebook(p) for p in iter_notebooks(REPO_ROOT)]
    broken = [r for r in results if not r["ok"]]
    docs_only = [r for r in results if r.get("docs_only")]
    no_out = [r for r in results
              if r["ok"] and not r["ext"] and not r.get("docs_only") and r["with_outputs"] == 0]
    n_ext = sum(1 for r in results if r.get("ext"))

    print(f"notebooks: {len(results)} | externos (EXT): {n_ext} | "
          f"docs-only: {len(docs_only)} | quebrados: {len(broken)} | sem outputs: {len(no_out)}")
    for r in broken:
        print(f"  BROKEN  {r['path']}: {r['error']}")
    if args.strict:
        for r in no_out:
            print(f"  NO-OUT  {r['path']}")

    if broken or (args.strict and no_out):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
