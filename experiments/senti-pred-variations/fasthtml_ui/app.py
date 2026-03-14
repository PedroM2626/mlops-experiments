from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fasthtml.common import *


APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
PYTHON_EXE = os.environ.get("PYTHON_EXE") or sys.executable


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "logistic": {
        "name": "Logistic Senti Pred",
        "path": BASE_DIR / "logistic-senti-pred",
        "description": "Pipeline scripts (EDA -> preprocessing -> modeling -> evaluation).",
        "commands_full": [
            [PYTHON_EXE, "src/scripts/01_eda.py"],
            [PYTHON_EXE, "src/scripts/02_preprocessing.py"],
            [PYTHON_EXE, "src/scripts/03_modeling.py"],
            [PYTHON_EXE, "src/scripts/04_evaluation.py"],
        ],
        "commands_smoke": [[PYTHON_EXE, "src/scripts/01_eda.py"]],
        "metrics_candidates": ["reports/metrics/model_metrics.json"],
    },
    "multinomialnb": {
        "name": "MultinomialNB Senti Pred",
        "path": BASE_DIR / "multinomialnb-Senti-Pred",
        "description": "Pipeline scripts (EDA -> preprocessing -> modeling -> evaluation).",
        "commands_full": [
            [PYTHON_EXE, "src/scripts/01_eda.py"],
            [PYTHON_EXE, "src/scripts/02_preprocessing.py"],
            [PYTHON_EXE, "src/scripts/03_modeling.py"],
            [PYTHON_EXE, "src/scripts/04_evaluation.py"],
        ],
        "commands_smoke": [[PYTHON_EXE, "src/scripts/01_eda.py"]],
        "metrics_candidates": ["reports/metrics/model_metrics.json"],
    },
    "random_forest": {
        "name": "Random Forest Senti Pred",
        "path": BASE_DIR / "random_forest-Senti-Pred",
        "description": "Pipeline scripts (EDA -> preprocessing -> modeling -> evaluation).",
        "commands_full": [
            [PYTHON_EXE, "src/scripts/01_eda.py"],
            [PYTHON_EXE, "src/scripts/02_preprocessing.py"],
            [PYTHON_EXE, "src/scripts/03_modeling.py"],
            [PYTHON_EXE, "src/scripts/04_evaluation.py"],
        ],
        "commands_smoke": [[PYTHON_EXE, "src/scripts/01_eda.py"]],
        "metrics_candidates": ["reports/metrics/model_metrics.json"],
    },
    "old_upgrade": {
        "name": "Old Senti Pred Upgrade",
        "path": BASE_DIR / "old_senti-pred_upgrade",
        "description": "Legacy upgraded pipeline with multiple baselines.",
        "commands_full": [
            [PYTHON_EXE, "src/scripts/01_eda.py"],
            [PYTHON_EXE, "src/scripts/02_preprocessing.py"],
            [PYTHON_EXE, "src/scripts/03_modeling.py"],
            [PYTHON_EXE, "src/scripts/04_evaluation.py"],
        ],
        "commands_smoke": [[PYTHON_EXE, "src/scripts/01_eda.py"]],
        "metrics_candidates": ["reports/metrics/model_metrics.json"],
    },
    "flaml": {
        "name": "FLAML Senti Pred",
        "path": BASE_DIR / "flaml-Senti-Pred",
        "description": "AutoML run using FLAML and TF-IDF features.",
        "commands_full": [[PYTHON_EXE, "src/models/train.py"]],
        "commands_smoke": [[PYTHON_EXE, "src/data/preprocess.py"]],
        "metrics_candidates": ["reports/metrics/flaml_optimized_metrics.json"],
    },
    "autogluon": {
        "name": "AutoGluon Senti Pred",
        "path": BASE_DIR,
        "description": "AutoGluon script focused on best-quality text classification.",
        "commands_full": [[PYTHON_EXE, "autogluon_senti_pred.py"]],
        "commands_smoke": [[PYTHON_EXE, "autogluon_senti_pred.py"]],
        "metrics_candidates": ["ag_leaderboard_optimized.csv"],
    },
}


RUNS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def rel_to_base(path: Path) -> str:
    return str(path.relative_to(BASE_DIR)).replace("\\", "/")


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def list_visualizations(exp_path: Path) -> list[Path]:
    vis_dir = exp_path / "reports" / "visualizacoes"
    if not vis_dir.exists():
        return []
    files = [p for p in vis_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}]
    return sorted(files, key=lambda p: p.name.lower())


def find_metrics_file(config: dict[str, Any]) -> Path | None:
    exp_path: Path = config["path"]
    for candidate in config.get("metrics_candidates", []):
        p = exp_path / candidate
        if p.exists() and p.is_file():
            return p
    return None


def load_metrics_summary(config: dict[str, Any]) -> dict[str, Any] | None:
    metrics_file = find_metrics_file(config)
    if metrics_file is None:
        return None
    try:
        if metrics_file.suffix.lower() == ".json":
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
            summary: dict[str, Any] = {
                "file": rel_to_base(metrics_file),
                "kind": "json",
                "best_model": payload.get("best_model") or payload.get("best_estimator"),
            }
            if "results" in payload and isinstance(payload["results"], dict):
                first_key = next(iter(payload["results"].keys()), None)
                if first_key:
                    info = payload["results"][first_key]
                    summary["best_model"] = summary["best_model"] or first_key
                    summary["accuracy"] = info.get("accuracy")
                    summary["f1_macro"] = info.get("f1_macro")
            if "accuracy" in payload:
                summary["accuracy"] = payload.get("accuracy")
            report = payload.get("report")
            if isinstance(report, dict) and "macro avg" in report:
                summary["f1_macro"] = report["macro avg"].get("f1-score")
            return summary
        return {"file": rel_to_base(metrics_file), "kind": metrics_file.suffix.lower().lstrip(".")}
    except Exception as exc:
        return {"file": rel_to_base(metrics_file), "error": str(exc)}


def _append_log(run_id: str, line: str) -> None:
    with LOCK:
        run = RUNS.get(run_id)
        if run is None:
            return
        run["logs"].append(line)
        if len(run["logs"]) > 2500:
            run["logs"] = run["logs"][-2500:]


def _mark_status(run_id: str, status: str, error: str | None = None) -> None:
    with LOCK:
        run = RUNS.get(run_id)
        if run is None:
            return
        run["status"] = status
        run["ended_at"] = time.time()
        run["finished_at_iso"] = now_iso()
        if error:
            run["error"] = error


def select_commands(config: dict[str, Any], mode: str) -> list[list[str]]:
    if mode == "smoke":
        return config.get("commands_smoke") or config.get("commands_full", [])
    return config.get("commands_full", [])


def execute_run(run_id: str, exp_key: str, mode: str) -> None:
    config = EXPERIMENTS[exp_key]
    exp_path: Path = config["path"]
    commands: list[list[str]] = select_commands(config, mode)
    _append_log(run_id, f"[{now_iso()}] Starting {mode} run for '{config['name']}'")

    try:
        for idx, command in enumerate(commands, start=1):
            if not exp_path.exists():
                raise FileNotFoundError(f"Experiment path not found: {exp_path}")

            cmd_str = " ".join(command)
            _append_log(run_id, f"[{now_iso()}] [{idx}/{len(commands)}] $ {cmd_str}")

            process = subprocess.Popen(
                command,
                cwd=str(exp_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                _append_log(run_id, line.rstrip())

            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Command failed with exit code {return_code}: {cmd_str}")

        _mark_status(run_id, "success")
        _append_log(run_id, f"[{now_iso()}] Run completed successfully.")
    except Exception as exc:
        _mark_status(run_id, "failed", str(exc))
        _append_log(run_id, f"[{now_iso()}] ERROR: {exc}")


def build_layout(*content: Any):
    nav = Div(
        A("Inicio", href="/", cls="nav-link"),
        A("Analise", href="/analysis", cls="nav-link"),
        A("Runs", href="/runs", cls="nav-link"),
        cls="nav",
    )

    return Div(
        Style(
            """
            :root {
                --bg: #f6f4ef;
                --panel: #ffffff;
                --ink: #1f2a37;
                --muted: #5a6573;
                --primary: #0f766e;
                --primary-2: #115e59;
                --line: #d7dce2;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                background: radial-gradient(circle at top right, #dce7e5 0%, var(--bg) 52%);
                color: var(--ink);
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 24px;
            }
            .hero {
                background: linear-gradient(120deg, #0f766e 0%, #134e4a 80%);
                color: #f2fffd;
                padding: 24px;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(15, 118, 110, 0.22);
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 16px;
                margin-top: 20px;
            }
            .card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            }
            .muted { color: var(--muted); }
            .btn {
                border: 0;
                background: var(--primary);
                color: #fff;
                padding: 10px 14px;
                border-radius: 10px;
                cursor: pointer;
                font-weight: 600;
            }
            .btn:hover { background: var(--primary-2); }
            .btn.link {
                text-decoration: none;
                display: inline-block;
            }
            .nav {
                margin: 16px 0 8px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
            .nav-link {
                text-decoration: none;
                color: var(--primary-2);
                background: #e8f2f1;
                border: 1px solid #c7dcda;
                border-radius: 8px;
                padding: 8px 10px;
                font-weight: 600;
            }
            .status-running { color: #9a6700; font-weight: 700; }
            .status-success { color: #137333; font-weight: 700; }
            .status-failed { color: #b42318; font-weight: 700; }
            .table-wrap {
                overflow: auto;
                border: 1px solid var(--line);
                border-radius: 10px;
                background: #fff;
            }
            table { width: 100%; border-collapse: collapse; }
            th, td {
                text-align: left;
                padding: 10px;
                border-bottom: 1px solid var(--line);
                vertical-align: top;
            }
            pre {
                background: #0b1f1d;
                color: #d5f6f2;
                padding: 14px;
                border-radius: 10px;
                overflow: auto;
                max-height: 420px;
                font-size: 12px;
            }
            .vis-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                gap: 12px;
            }
            .vis-item img {
                width: 100%;
                height: 200px;
                object-fit: contain;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #fafafa;
            }
            @media (max-width: 720px) {
                .container { padding: 16px; }
                .hero { padding: 18px; }
                .vis-item img { height: 160px; }
            }
            """
        ),
        Div(
            Div(H1("Senti Pred Variations Lab"), P("FastHTML cockpit to run and analyze experiments."), cls="hero"),
            nav,
            *content,
            cls="container",
        ),
    )


app, rt = fast_app()


@rt("/")
def homepage():
    cards = []
    for exp_key, config in EXPERIMENTS.items():
        metrics = load_metrics_summary(config)
        metrics_block = P("No metrics file found yet.", cls="muted")
        if metrics:
            parts = [f"file={metrics.get('file', '-')}"]
            if metrics.get("best_model"):
                parts.append(f"best={metrics['best_model']}")
            if isinstance(metrics.get("accuracy"), (float, int)):
                parts.append(f"acc={metrics['accuracy']:.4f}")
            if isinstance(metrics.get("f1_macro"), (float, int)):
                parts.append(f"f1_macro={metrics['f1_macro']:.4f}")
            metrics_block = P(" | ".join(parts), cls="muted")

        cards.append(
            Div(
                H3(config["name"]),
                P(config["description"], cls="muted"),
                metrics_block,
                Div(
                    Form(
                        Button("Run full", cls="btn"),
                        action=f"/runs/start/{exp_key}/full",
                        method="post",
                    ),
                    Form(
                        Button("Smoke test", cls="btn"),
                        action=f"/runs/start/{exp_key}/smoke",
                        method="post",
                    ),
                    style="display:flex;gap:8px;flex-wrap:wrap;",
                ),
                cls="card",
            )
        )

    return build_layout(
        H2("Experiments"),
        P("Run full pipelines from this page and inspect output in Runs/Analysis."),
        Div(*cards, cls="grid"),
    )


@rt("/runs/start/{exp_key}/{mode}", methods=["POST"])
def start_run(exp_key: str, mode: str):
    if exp_key not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Unknown experiment key")
    if mode not in {"full", "smoke"}:
        raise HTTPException(status_code=400, detail="Mode must be 'full' or 'smoke'")

    run_id = uuid.uuid4().hex[:10]
    now = time.time()
    with LOCK:
        RUNS[run_id] = {
            "id": run_id,
            "experiment_key": exp_key,
            "experiment_name": EXPERIMENTS[exp_key]["name"],
            "mode": mode,
            "status": "running",
            "created_at": now,
            "created_at_iso": now_iso(),
            "ended_at": None,
            "finished_at_iso": None,
            "logs": [],
            "error": None,
        }

    t = threading.Thread(target=execute_run, args=(run_id, exp_key, mode), daemon=True)
    t.start()
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


def run_status_class(status: str) -> str:
    return {
        "running": "status-running",
        "success": "status-success",
        "failed": "status-failed",
    }.get(status, "")


def render_runs_table() -> Any:
    with LOCK:
        ordered = sorted(RUNS.values(), key=lambda r: r["created_at"], reverse=True)

    rows = []
    for run in ordered:
        elapsed = (run["ended_at"] or time.time()) - run["created_at"]
        rows.append(
            Tr(
                Td(run["id"]),
                Td(run["experiment_name"]),
                Td(run.get("mode", "full")),
                Td(run["created_at_iso"]),
                Td(fmt_seconds(elapsed)),
                Td(Span(run["status"], cls=run_status_class(run["status"]))),
                Td(A("Open", href=f"/runs/{run['id']}")),
            )
        )

    if not rows:
        rows.append(Tr(Td("No runs yet", colspan="7", cls="muted")))

    return Div(
        Div(
            Table(
                Thead(Tr(Th("Run ID"), Th("Experiment"), Th("Mode"), Th("Started"), Th("Duration"), Th("Status"), Th("Details"))),
                Tbody(*rows),
            ),
            cls="table-wrap",
        ),
        hx_get="/runs/table",
        hx_trigger="load, every 3s",
        hx_swap="outerHTML",
    )


@rt("/runs")
def runs_page():
    return build_layout(H2("Runs"), P("Live status refreshes every 3 seconds."), render_runs_table())


@rt("/runs/table")
def runs_table_partial():
    return render_runs_table()


@rt("/runs/{run_id}")
def run_detail(run_id: str):
    with LOCK:
        run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    err = Div(P(f"Error: {run['error']}", cls="status-failed"), cls="card") if run.get("error") else ""

    return build_layout(
        H2(f"Run {run_id}"),
        Div(
            P(f"Experiment: {run['experiment_name']}"),
            P(f"Mode: {run.get('mode', 'full')}"),
            P(f"Status: {run['status']}", cls=run_status_class(run["status"])),
            P(f"Started: {run['created_at_iso']}"),
            P(f"Finished: {run.get('finished_at_iso') or '-'}"),
            cls="card",
        ),
        err,
        H3("Logs"),
        Div(
            Pre("Loading logs..."),
            hx_get=f"/runs/{run_id}/logs",
            hx_trigger="load, every 2s",
            hx_swap="innerHTML",
        ),
    )


@rt("/runs/{run_id}/logs")
def run_logs_partial(run_id: str):
    with LOCK:
        run = RUNS.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        text = "\n".join(run["logs"]) or "No logs yet"
    return Pre(text)


@rt("/analysis")
def analysis_page():
    exp_sections = []

    for exp_key, config in EXPERIMENTS.items():
        metrics = load_metrics_summary(config)
        metrics_rows = []
        if metrics:
            for k in ("file", "best_model", "accuracy", "f1_macro", "kind", "error"):
                if k not in metrics:
                    continue
                value = metrics[k]
                if isinstance(value, float):
                    value = f"{value:.6f}"
                metrics_rows.append(Tr(Td(k), Td(str(value))))
        else:
            metrics_rows.append(Tr(Td("metrics"), Td("not found")))

        visuals = list_visualizations(config["path"])
        vis_cards = []
        for visual in visuals:
            rel = rel_to_base(visual)
            file_url = f"/artifact?path={quote(rel)}"
            vis_cards.append(
                Div(
                    A(Img(src=file_url, alt=visual.name), href=file_url, target="_blank"),
                    P(visual.name, cls="muted"),
                    cls="vis-item",
                )
            )

        if not vis_cards:
            vis_cards = [P("No visualization files found.", cls="muted")]

        exp_sections.append(
            Div(
                H3(config["name"]),
                P(rel_to_base(config["path"]), cls="muted"),
                Div(Table(Tbody(*metrics_rows)), cls="table-wrap"),
                H4("Visualizations"),
                Div(*vis_cards, cls="vis-grid"),
                cls="card",
            )
        )

    return build_layout(H2("Analysis"), P("Metrics and generated artifacts by experiment."), Div(*exp_sections, cls="grid"))


@rt("/artifact")
def artifact_file(path: str):
    candidate = (BASE_DIR / path).resolve()
    if BASE_DIR.resolve() not in candidate.parents and candidate != BASE_DIR.resolve():
        raise HTTPException(status_code=403, detail="Invalid artifact path")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(candidate)


if __name__ == "__main__":
    serve(host="127.0.0.1", port=8502)
