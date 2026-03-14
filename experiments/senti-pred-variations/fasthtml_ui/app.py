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
from urllib.parse import quote, urlencode
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fasthtml.common import *


APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
REPO_ROOT = BASE_DIR.parents[1]
PYTHON_EXE = os.environ.get("PYTHON_EXE") or sys.executable
PY311_FALLBACK_EXE = REPO_ROOT / ".venv311" / "Scripts" / "python.exe"
AUTOG_PYTHON_EXE = os.environ.get("AUTOG_PYTHON_EXE") or (str(PY311_FALLBACK_EXE) if PY311_FALLBACK_EXE.exists() else PYTHON_EXE)


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
        "commands_full": [[AUTOG_PYTHON_EXE, "autogluon_senti_pred.py"]],
        "commands_smoke": [[AUTOG_PYTHON_EXE, "-c", "from autogluon.tabular import TabularPredictor; print('AutoGluon import OK')"]],
        "metrics_candidates": ["ag_leaderboard_optimized.csv"],
    },
}


RUNS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()
RUN_TERMINATION_TIMEOUT = 5


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


def status_options() -> list[str]:
    return ["all", "running", "canceling", "success", "failed", "canceled"]


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


def _get_run_copy(run_id: str) -> dict[str, Any] | None:
    with LOCK:
        run = RUNS.get(run_id)
        if run is None:
            return None
        return dict(run)


def _is_cancel_requested(run_id: str) -> bool:
    with LOCK:
        run = RUNS.get(run_id)
        return bool(run and run.get("cancel_requested"))


def _set_process(run_id: str, process: subprocess.Popen[str] | None) -> None:
    with LOCK:
        run = RUNS.get(run_id)
        if run is not None:
            run["process"] = process


def _stream_process_output(run_id: str, process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    try:
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            _append_log(run_id, line.rstrip())
    finally:
        try:
            process.stdout.close()
        except Exception:
            pass


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


def request_cancel(run_id: str) -> bool:
    process: subprocess.Popen[str] | None = None
    with LOCK:
        run = RUNS.get(run_id)
        if run is None or run.get("status") not in {"running", "canceling"}:
            return False
        run["cancel_requested"] = True
        if run["status"] == "running":
            run["status"] = "canceling"
        process = run.get("process")

    _append_log(run_id, f"[{now_iso()}] Cancellation requested by user.")

    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=RUN_TERMINATION_TIMEOUT)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    return True


def filtered_runs(experiment: str, status: str) -> list[dict[str, Any]]:
    with LOCK:
        ordered = sorted(RUNS.values(), key=lambda r: r["created_at"], reverse=True)
    rows = []
    for run in ordered:
        if experiment and experiment != "all" and run.get("experiment_key") != experiment:
            continue
        if status and status != "all" and run.get("status") != status:
            continue
        rows.append(run)
    return rows


def runs_query(experiment: str, status: str) -> str:
    params = {}
    if experiment and experiment != "all":
        params["experiment"] = experiment
    if status and status != "all":
        params["status"] = status
    query = urlencode(params)
    return f"?{query}" if query else ""


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
            if _is_cancel_requested(run_id):
                _mark_status(run_id, "canceled")
                _append_log(run_id, f"[{now_iso()}] Run canceled before next command.")
                return
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
            _set_process(run_id, process)

            reader = threading.Thread(target=_stream_process_output, args=(run_id, process), daemon=True)
            reader.start()

            return_code = process.wait()
            reader.join(timeout=3)
            _set_process(run_id, None)

            if _is_cancel_requested(run_id):
                _mark_status(run_id, "canceled")
                _append_log(run_id, f"[{now_iso()}] Run canceled during command execution.")
                return
            if return_code != 0:
                raise RuntimeError(f"Command failed with exit code {return_code}: {cmd_str}")

        _mark_status(run_id, "success")
        _append_log(run_id, f"[{now_iso()}] Run completed successfully.")
    except Exception as exc:
        if _is_cancel_requested(run_id):
            _mark_status(run_id, "canceled")
            _append_log(run_id, f"[{now_iso()}] Run canceled.")
            return
        _mark_status(run_id, "failed", str(exc))
        _append_log(run_id, f"[{now_iso()}] ERROR: {exc}")
    finally:
        _set_process(run_id, None)


def button_class(kind: str = "primary") -> str:
    return f"btn btn-{kind}"


def filter_controls(experiment: str, status: str):
    return Div(
        Form(
            Div(
                Label("Experiment", fr="experiment", cls="label"),
                Select(
                    Option("All", value="all", selected=(experiment in {"", "all"})),
                    *[
                        Option(cfg["name"], value=key, selected=(experiment == key))
                        for key, cfg in EXPERIMENTS.items()
                    ],
                    name="experiment",
                    id="experiment",
                ),
                cls="field",
            ),
            Div(
                Label("Status", fr="status", cls="label"),
                Select(
                    *[
                        Option(opt.title(), value=opt, selected=(status == opt or (opt == "all" and status == "")))
                        for opt in status_options()
                    ],
                    name="status",
                    id="status",
                ),
                cls="field",
            ),
            Div(
                Button("Apply filters", cls=button_class("primary")),
                A("Clear", href="/runs", cls="btn btn-ghost"),
                cls="actions",
            ),
            method="get",
            action="/runs",
            cls="filters",
        ),
        cls="card filter-card",
    )


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
                --bg: #0a0f1a;
                --bg-soft: #121a2a;
                --panel: rgba(17, 24, 39, 0.84);
                --panel-strong: rgba(15, 23, 42, 0.96);
                --ink: #ecf3ff;
                --muted: #9fb0cb;
                --primary: #5eead4;
                --primary-2: #2dd4bf;
                --accent: #7c3aed;
                --danger: #fb7185;
                --warning: #fbbf24;
                --success: #34d399;
                --line: rgba(148, 163, 184, 0.22);
                --shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                background:
                    radial-gradient(circle at top left, rgba(94, 234, 212, 0.16), transparent 24%),
                    radial-gradient(circle at top right, rgba(124, 58, 237, 0.18), transparent 28%),
                    linear-gradient(180deg, #060913 0%, var(--bg) 100%);
                color: var(--ink);
                min-height: 100vh;
            }
            .container {
                max-width: 1280px;
                margin: 0 auto;
                padding: 24px;
            }
            .hero {
                background:
                    linear-gradient(135deg, rgba(45, 212, 191, 0.15) 0%, rgba(124, 58, 237, 0.28) 100%),
                    var(--panel-strong);
                color: #f7fbff;
                padding: 28px;
                border: 1px solid rgba(94, 234, 212, 0.16);
                border-radius: 24px;
                box-shadow: var(--shadow);
                position: relative;
                overflow: hidden;
            }
            .hero:before {
                content: "";
                position: absolute;
                inset: -20% auto auto -10%;
                width: 240px;
                height: 240px;
                background: radial-gradient(circle, rgba(94, 234, 212, 0.25), transparent 70%);
                pointer-events: none;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 18px;
                margin-top: 20px;
            }
            .card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 18px;
                box-shadow: var(--shadow);
                backdrop-filter: blur(10px);
            }
            .muted { color: var(--muted); }
            .btn {
                border: 1px solid transparent;
                color: #061018;
                padding: 10px 14px;
                border-radius: 12px;
                cursor: pointer;
                font-weight: 600;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                transition: 160ms ease;
                min-height: 42px;
            }
            .btn:hover { transform: translateY(-1px); }
            .btn-primary {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
                box-shadow: 0 10px 24px rgba(45, 212, 191, 0.24);
            }
            .btn-secondary {
                background: linear-gradient(135deg, #93c5fd 0%, #60a5fa 100%);
                box-shadow: 0 10px 24px rgba(96, 165, 250, 0.2);
            }
            .btn-danger {
                background: linear-gradient(135deg, #fb7185 0%, #e11d48 100%);
                color: #fff;
                box-shadow: 0 10px 24px rgba(225, 29, 72, 0.22);
            }
            .btn-ghost {
                background: rgba(148, 163, 184, 0.08);
                color: var(--ink);
                border-color: rgba(148, 163, 184, 0.18);
            }
            .nav {
                margin: 18px 0 10px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
            .nav-link {
                text-decoration: none;
                color: var(--ink);
                background: rgba(148, 163, 184, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 999px;
                padding: 10px 14px;
                font-weight: 600;
            }
            .nav-link:hover {
                border-color: rgba(94, 234, 212, 0.4);
                color: var(--primary);
            }
            .status-running { color: var(--warning); font-weight: 700; }
            .status-canceling { color: #fda4af; font-weight: 700; }
            .status-canceled { color: #cbd5e1; font-weight: 700; }
            .status-success { color: var(--success); font-weight: 700; }
            .status-failed { color: var(--danger); font-weight: 700; }
            .table-wrap {
                overflow: auto;
                border: 1px solid var(--line);
                border-radius: 16px;
                background: rgba(8, 13, 24, 0.75);
            }
            table { width: 100%; border-collapse: collapse; }
            th, td {
                text-align: left;
                padding: 10px;
                border-bottom: 1px solid var(--line);
                vertical-align: top;
            }
            th {
                color: #dce7fb;
                background: rgba(148, 163, 184, 0.06);
            }
            pre {
                background: #040814;
                color: #d7fff7;
                padding: 14px;
                border-radius: 14px;
                overflow: auto;
                max-height: 420px;
                font-size: 12px;
                border: 1px solid rgba(94, 234, 212, 0.12);
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
                border-radius: 12px;
                background: rgba(255,255,255,0.03);
            }
            .actions {
                display: flex;
                gap: 8px;
                align-items: end;
                flex-wrap: wrap;
            }
            .filters {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 14px;
                align-items: end;
            }
            .filter-card {
                margin: 14px 0 8px;
            }
            .field {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .label {
                color: #dbeafe;
                font-size: 13px;
                letter-spacing: 0.02em;
            }
            input, select {
                width: 100%;
                background: rgba(15, 23, 42, 0.92);
                color: var(--ink);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                padding: 11px 12px;
            }
            .split {
                display: flex;
                justify-content: space-between;
                gap: 12px;
                align-items: center;
                flex-wrap: wrap;
            }
            .kpis {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin: 10px 0 0;
            }
            .pill {
                display: inline-flex;
                gap: 6px;
                align-items: center;
                padding: 7px 10px;
                border-radius: 999px;
                background: rgba(148, 163, 184, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.16);
                color: #dce7fb;
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
                Div(
                    H3(config["name"]),
                    Span("ready", cls="pill"),
                    cls="split",
                ),
                P(config["description"], cls="muted"),
                metrics_block,
                Div(
                    Form(
                        Button("Run full", cls=button_class("primary")),
                        action=f"/runs/start/{exp_key}/full",
                        method="post",
                    ),
                    Form(
                        Button("Smoke test", cls=button_class("secondary")),
                        action=f"/runs/start/{exp_key}/smoke",
                        method="post",
                    ),
                    cls="actions",
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
            "cancel_requested": False,
            "process": None,
        }

    t = threading.Thread(target=execute_run, args=(run_id, exp_key, mode), daemon=True)
    t.start()
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


def run_status_class(status: str) -> str:
    return {
        "running": "status-running",
        "canceling": "status-canceling",
        "canceled": "status-canceled",
        "success": "status-success",
        "failed": "status-failed",
    }.get(status, "")


def render_runs_table(experiment: str, status: str) -> Any:
    ordered = filtered_runs(experiment, status)
    rows = []
    for run in ordered:
        elapsed = (run["ended_at"] or time.time()) - run["created_at"]
        redirect_to = f"/runs{runs_query(experiment, status)}"
        action_cell: Any = A("Open", href=f"/runs/{run['id']}")
        if run["status"] in {"running", "canceling"}:
            action_cell = Div(
                A("Open", href=f"/runs/{run['id']}"),
                Form(
                    Input(type="hidden", name="redirect_to", value=redirect_to),
                    Button("Cancel", cls=button_class("danger")),
                    action=f"/runs/cancel/{run['id']}",
                    method="post",
                ),
                cls="actions",
            )
        rows.append(
            Tr(
                Td(run["id"]),
                Td(run["experiment_name"]),
                Td(run.get("mode", "full")),
                Td(run["created_at_iso"]),
                Td(fmt_seconds(elapsed)),
                Td(Span(run["status"], cls=run_status_class(run["status"]))),
                Td(action_cell),
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
        hx_get=f"/runs/table{runs_query(experiment, status)}",
        hx_trigger="load, every 3s",
        hx_swap="outerHTML",
    )


@rt("/runs")
def runs_page(experiment: str = "all", status: str = "all"):
    active_runs = len(filtered_runs(experiment, "running"))
    visible_runs = len(filtered_runs(experiment, status))
    return build_layout(
        H2("Runs"),
        P("Filterable live view with auto-refresh every 3 seconds."),
        filter_controls(experiment, status),
        Div(
            Span(f"Visible runs: {visible_runs}", cls="pill"),
            Span(f"Running now: {active_runs}", cls="pill"),
            cls="kpis",
        ),
        render_runs_table(experiment, status),
    )


@rt("/runs/table")
def runs_table_partial(experiment: str = "all", status: str = "all"):
    return render_runs_table(experiment, status)


@rt("/runs/cancel/{run_id}", methods=["POST"])
def cancel_run(run_id: str, redirect_to: str = "/runs"):
    run = _get_run_copy(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    request_cancel(run_id)
    return RedirectResponse(url=redirect_to, status_code=303)


@rt("/runs/{run_id}")
def run_detail(run_id: str):
    with LOCK:
        run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    err = Div(P(f"Error: {run['error']}", cls="status-failed"), cls="card") if run.get("error") else ""
    cancel_button = ""
    if run["status"] in {"running", "canceling"}:
        cancel_button = Form(
            Input(type="hidden", name="redirect_to", value=f"/runs/{run_id}"),
            Button("Cancel run", cls=button_class("danger")),
            action=f"/runs/cancel/{run_id}",
            method="post",
        )

    return build_layout(
        H2(f"Run {run_id}"),
        Div(
            Div(
                Div(
                    P(f"Experiment: {run['experiment_name']}"),
                    P(f"Mode: {run.get('mode', 'full')}"),
                    P(f"Status: {run['status']}", cls=run_status_class(run["status"])),
                    P(f"Started: {run['created_at_iso']}"),
                    P(f"Finished: {run.get('finished_at_iso') or '-'}"),
                ),
                cancel_button,
                cls="split",
            ),
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
