"""Guard tests for the cross-references that hold the repository together.

These failed silently before: a notebook was renamed, a README kept citing the
old name, and the dashboard pointed at paths that no longer exist. Everything
here is a static check over tracked files, so it is fast and needs no data.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def tracked(*globs: str) -> list[str]:
    cmd = ["git", "ls-files", "-z"] + (list(globs) or ["."])
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True).stdout
    return [f.decode("utf-8") for f in out.split(b"\0") if f]


LINK = re.compile(r"\[[^\]]*\]\((?!https?:|mailto:|#)(<[^>]+>|[^)\s]+)")


@pytest.mark.parametrize("rel", tracked("*.md"), ids=lambda x: x)
def test_relative_markdown_links_resolve(rel: str):
    path = ROOT / rel
    text = path.read_text(encoding="utf-8", errors="replace")
    base = path.parent
    for m in LINK.finditer(text):
        target = m.group(1)
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        target = target.split("#")[0]
        if not target or "<" in target or "{{" in target:
            continue
        assert (base / target).exists(), f"{rel}: broken link -> {m.group(1)}"


DASH_PATH = re.compile(r"(?:script|notebook|path|readme|file)\s*:\s*"
                       r"[\"']((?:experiments|docs|mlops|dashboard)/[^\"']+)[\"']")


def test_dashboard_references_exist():
    text = (ROOT / "dashboard" / "js" / "app.js").read_text(encoding="utf-8")
    refs = sorted(set(DASH_PATH.findall(text)))
    assert len(refs) > 40, "the dashboard data file was not parsed as expected"
    missing = [r for r in refs if not (ROOT / r).exists()]
    assert not missing, f"dashboard references files that are not in git: {missing}"


def test_dashboard_covers_every_experiment_readme():
    """Each documented experiment group must be reachable from the dashboard."""
    text = (ROOT / "dashboard" / "js" / "app.js").read_text(encoding="utf-8")
    refs = set(DASH_PATH.findall(text))
    for readme in tracked("experiments/*/README.md"):
        group = Path(readme).parent.as_posix()
        assert any(r.startswith(group + "/") for r in refs), \
            f"dashboard has no entry for {group}/"


def test_root_index_links_every_group_readme():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\((experiments/[^/)]+/README\.md)", text))
    for readme in [f for f in tracked("experiments/*/README.md")
                   if f.count("/") == 2]:
        assert readme in linked, f"{readme} is not linked from the root index"


SKIP_DOC = ("experiments/ibm-experiments/", "experiments/databricks-forecast/",
            "experiments/artifacts/", "experiments/sales-forecast/tests/")


def test_every_experiment_code_file_is_documented():
    """No notebook or script may be orphaned.

    Everything committed under experiments/ must be named either by some
    README in the repository or by the dashboard data file; this is what caught
    the single-cell anomaly variant and the imputation study going undocumented.
    """
    docs = "\n".join((ROOT / f).read_text(encoding="utf-8", errors="replace")
                     for f in tracked("*.md"))
    docs += (ROOT / "dashboard" / "js" / "app.js").read_text(encoding="utf-8",
                                                             errors="replace")
    orphans = []
    for f in tracked("experiments/*"):
        if not f.endswith((".ipynb", ".py")) or any(f.startswith(p) for p in SKIP_DOC):
            continue
        if Path(f).name == "__init__.py":       # package markers, not artifacts
            continue
        if Path(f).name not in docs:
            orphans.append(f)
    assert not orphans, f"undocumented code files: {orphans}"


NON_ASCII = re.compile(r"[^\x00-\x7F]")
PT_WORDS = re.compile(
    r"\b(não|nao|são|sao|está|estao|estão|análise|analise|resultados|objetivo|"
    r"também|tambem|versão|versao|leitura|gráfico|grafico|janela|previsão|"
    r"previsao|classificação|classificacao|avaliação|avaliacao|métrica|metrica|"
    r"hierarquico|hierarquica|visualizacao|estrategica|classificacao|dados|"
    r"código|codigo|arquivo|pasta|experimento|amostra|conjunto|treino)\b", re.I)
EXEMPT = re.compile(r"(datasets/|artifacts/|ibm-experiments/assets/|"
                    r"README_mlops|strategic_visualization|\.lock)")


@pytest.mark.parametrize("rel", [f for f in tracked("*.md", "*.py", "*.js", "*.html",
                                                    "*.css", "*.yml", "*.yaml", "*.txt")
                                 if not EXEMPT.search(f)],
                         ids=lambda x: x)
def test_no_portuguese_prose_left_in_documentation(rel: str):
    """Prose is English; only real data (corpora, dataset dumps, model assets)
    may still be Portuguese, and those paths are exempt above."""
    text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.split("\n") if PT_WORDS.search(ln)]
    assert not lines, f"{rel}: Portuguese prose remains: {lines[:2]}"
