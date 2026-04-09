from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Iterable


@dataclass(frozen=True)
class RunContext:
    base_dir: Path
    experiment_name: str
    timestamp: str
    git_sha: str
    run_id: str
    artifact_dir: Path


def get_git_sha(base_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        sha = result.stdout.strip()
        return sha or "nogit"
    except Exception:
        return "nogit"


def create_run_context(base_dir: Path, experiment_name: str) -> RunContext:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = get_git_sha(base_dir)
    run_id = f"{timestamp}_{git_sha}"
    artifact_dir = base_dir / "artifacts" / f"{experiment_name}_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        base_dir=base_dir,
        experiment_name=experiment_name,
        timestamp=timestamp,
        git_sha=git_sha,
        run_id=run_id,
        artifact_dir=artifact_dir,
    )


def log_reproducibility(mlflow_module, context: RunContext, seed: int) -> None:
    mlflow_module.log_param("seed", seed)
    mlflow_module.log_param("run_timestamp", context.timestamp)
    mlflow_module.log_param("git_sha", context.git_sha)

    freeze_path = context.artifact_dir / "pip_freeze.txt"
    freeze_result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    freeze_path.write_text(freeze_result.stdout, encoding="utf-8")
    mlflow_module.log_artifact(str(freeze_path))


def first_existing_path(candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Nenhum caminho candidato existente foi encontrado.")
