"""Unit tests for scripts/validate_notebooks.py, the repo's own structural gate."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / "validate_notebooks.py"

spec = importlib.util.spec_from_file_location("validate_notebooks", MODULE)
vn = importlib.util.module_from_spec(spec)
sys.modules.setdefault("validate_notebooks", vn)
spec.loader.exec_module(vn)


def notebook_md_then_code(tmp_path: Path, first_line: str, name: str = "nb.ipynb"):
    nb = {"cells": [{"cell_type": "markdown", "metadata": {},
                     "source": [first_line]},
                    {"cell_type": "code", "metadata": {}, "execution_count": 1,
                     "outputs": [{"output_type": "stream", "name": "stdout",
                                  "text": ["ok\n"]}],
                     "source": ["print(1)\n"]}],
            "metadata": {"kernelspec": {"name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / name
    p.write_text(json.dumps(nb), encoding="utf-8")
    return p


def test_notebook_with_outputs_is_ok(tmp_path):
    p = notebook_md_then_code(tmp_path, "# Title")
    r = vn.check_notebook(p)
    assert r["ok"] and not r["ext"] and r["with_outputs"] == 1


def test_broken_json_is_reported(tmp_path):
    p = tmp_path / "bad.ipynb"
    p.write_text("{not json", encoding="utf-8")
    r = vn.check_notebook(p)
    assert not r["ok"] and "invalid JSON" in r["error"]


def test_nbformat_below_four_is_rejected(tmp_path):
    p = tmp_path / "old.ipynb"
    p.write_text(json.dumps({"cells": [], "nbformat": 3}), encoding="utf-8")
    r = vn.check_notebook(p)
    assert not r["ok"] and "nbformat" in r["error"]


def test_notebook_without_any_cell_is_rejected(tmp_path):
    p = tmp_path / "empty.ipynb"
    p.write_text(json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5}),
                 encoding="utf-8")
    assert not vn.check_notebook(p)["ok"]


def test_markdown_only_notebook_is_flagged_docs_only(tmp_path):
    p = tmp_path / "docsonly.ipynb"
    p.write_text(json.dumps({"cells": [{"cell_type": "markdown", "metadata": {},
                                       "source": ["# only prose"]}],
                             "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    r = vn.check_notebook(p)
    assert r["ok"] and r.get("docs_only")


def test_cloud_directories_are_always_external(tmp_path):
    p = notebook_md_then_code(tmp_path, "# Something",
                              name="x.ipynb")
    folded = tmp_path / "ibm-experiments" / "x.ipynb"
    folded.parent.mkdir()
    folded.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    assert vn.is_external(folded, json.loads(folded.read_text(encoding="utf-8")))


def test_ext_marker_needs_a_word_boundary(tmp_path):
    """A notebook titled "Hierarchical Text Classification" is a local run.

    The marker used to be a plain substring test, so "Text", "Next" and
    "Context" marked ordinary notebooks as cloud-external and silently exempted
    them from the output requirement.
    """
    nb = {"cells": [{"cell_type": "markdown", "metadata": {},
                     "source": ["# Hierarchical Text Classification — 20 Newsgroups"]}],
          "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "hierarchical_classification.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    assert not vn.is_external(p, nb)

    marked = {"cells": [{"cell_type": "markdown", "metadata": {},
                         "source": ["EXT — runs on the Databricks cluster"]}],
              "nbformat": 4, "nbformat_minor": 5}
    q = tmp_path / "cloud_one.ipynb"
    q.write_text(json.dumps(marked), encoding="utf-8")
    assert vn.is_external(q, marked)


def test_ext_in_the_file_name_still_counts(tmp_path):
    p = notebook_md_then_code(tmp_path, "# Title", name="forecast_EXT_cloud.ipynb")
    assert vn.is_external(p, json.loads(p.read_text(encoding="utf-8")))


def test_iter_notebooks_skips_venv_and_hidden_dirs(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / "mlruns").mkdir()
    notebook_md_then_code(tmp_path, "# real", name="a.ipynb")
    notebook_md_then_code(tmp_path / ".venv", "# junk", name="b.ipynb")
    notebook_md_then_code(tmp_path / "mlruns", "# junk", name="c.ipynb")
    found = sorted(p.name for p in vn.iter_notebooks(tmp_path))
    assert found == ["a.ipynb"], found


def test_summary_counts_match_a_clean_repo(tmp_path, capsys):
    notebook_md_then_code(tmp_path, "# ok", name="good.ipynb")
    code = vn.main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "broken: 0" in out
