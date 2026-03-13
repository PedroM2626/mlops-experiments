from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import time
import os
from pathlib import Path

import matplotlib.pyplot as plt
import reflex as rx


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "experiments" / "flexible_ensemble_pyramid.py"
ASSETS_DIR = ROOT_DIR / "assets"
TOPOLOGY_IMAGE = ASSETS_DIR / "ensemble_topology.png"


class TrainingState(rx.State):
    """State for running the flexible pyramid and streaming progress."""

    layers: int = 6
    seed: int = 2007
    min_models: int = 2
    max_models: int = 6
    epsilon: float = 0.2
    patience: int = 3
    metric: str = "f1"
    strategy: str = "dense"
    jitter: bool = True
    tfidf_max: int = 50000
    tfidf_ngrams: int = 2

    is_training: bool = False
    status_text: str = "Aguardando execucao"
    current_layer: int = 0
    completed_layers: int = 0
    logs: list[str] = []
    model_rows: list[dict[str, str]] = []

    nodes: list[dict[str, str | float]] = []
    edges: list[dict[str, str]] = []
    graph_image_url: str = "/ensemble_topology.png"
    training_pid: int = 0
    stop_requested: bool = False
    is_smoke_test: bool = False
    sample_train_rows: int = 0
    sample_val_rows: int = 0
    training_start_ts: float = 0.0
    elapsed_seconds: int = 0
    eta_seconds: int = 0
    ensemble_counters: dict[str, int] = {}

    last_node_id: str = "input"

    @rx.var
    def logs_text(self) -> str:
        return "\n".join(self.logs[-300:])

    @rx.var
    def layer_progress_percent(self) -> int:
        total_layers = max(1, int(self.layers))
        return int(min(100, (self.completed_layers / total_layers) * 100))

    @rx.var
    def layer_progress_width(self) -> str:
        return f"{self.layer_progress_percent}%"

    @rx.var
    def layer_progress_text(self) -> str:
        return f"{self.completed_layers}/{self.layers} camadas concluidas ({self.layer_progress_percent}%)"

    @rx.var
    def flow_nodes_text(self) -> str:
        return f"Nos: {len(self.nodes)}"

    @rx.var
    def flow_edges_text(self) -> str:
        return f"Conexoes: {len(self.edges)}"

    @rx.var
    def elapsed_text(self) -> str:
        mins = self.elapsed_seconds // 60
        secs = self.elapsed_seconds % 60
        return f"Decorrido: {mins:02}:{secs:02}"

    @rx.var
    def eta_text(self) -> str:
        if not self.is_training:
            return "ETA: --:--"
        mins = self.eta_seconds // 60
        secs = self.eta_seconds % 60
        return f"ETA: {mins:02}:{secs:02}"

    def _update_time_estimates(self):
        if self.training_start_ts <= 0:
            return

        self.elapsed_seconds = max(0, int(time.time() - self.training_start_ts))
        if self.completed_layers <= 0:
            self.eta_seconds = 0
            return

        avg_layer_seconds = self.elapsed_seconds / max(1, self.completed_layers)
        remaining_layers = max(0, int(self.layers) - int(self.completed_layers))
        self.eta_seconds = int(avg_layer_seconds * remaining_layers)

    def update_layers(self, value: str):
        self.layers = max(1, int(value or 1))

    def update_seed(self, value: str):
        self.seed = int(value or 2007)

    def update_min_models(self, value: str):
        self.min_models = max(1, int(value or 1))

    def update_max_models(self, value: str):
        self.max_models = max(self.min_models, int(value or self.min_models))

    def update_epsilon(self, value: str):
        self.epsilon = min(1.0, max(0.0, float(value or 0.2)))

    def update_patience(self, value: str):
        self.patience = max(1, int(value or 1))

    def update_metric(self, value: str):
        if value in ("f1", "accuracy"):
            self.metric = value

    def update_strategy(self, value: str):
        if value in ("dense", "residual", "simple"):
            self.strategy = value

    def update_jitter(self, value: bool):
        self.jitter = bool(value)

    def update_tfidf_max(self, value: str):
        self.tfidf_max = max(1000, int(value or 1000))

    def update_tfidf_ngrams(self, value: str):
        parsed = int(value or 2)
        self.tfidf_ngrams = 1 if parsed <= 1 else 2

    def _reset_session(self):
        self.logs = []
        self.model_rows = []
        self.current_layer = 0
        self.completed_layers = 0
        self.status_text = "Inicializando treino"
        self.training_pid = 0
        self.stop_requested = False
        self.training_start_ts = 0.0
        self.elapsed_seconds = 0
        self.eta_seconds = 0
        self.ensemble_counters = {}
        self.nodes = [
            {
                "id": "input",
                "label": "TF-IDF",
                "kind": "input",
                "x": 0.0,
                "y": 0.0,
            }
        ]
        self.edges = []
        self.last_node_id = "input"
        self._render_topology()

    def _add_edge(self, source: str, target: str, kind: str = "main"):
        if source == target:
            return
        if not any(e.get("source") == source and e.get("target") == target for e in self.edges):
            self.edges.append({"source": source, "target": target, "kind": kind})

    def _has_node(self, node_id: str) -> bool:
        return any(str(n.get("id")) == node_id for n in self.nodes)

    def _ensure_layer_node(self, layer: int) -> str:
        node_id = f"layer_{layer}"
        if any(n["id"] == node_id for n in self.nodes):
            return node_id

        strategy_badge = ""
        if self.strategy == "dense":
            strategy_badge = "[dense]"
        elif self.strategy == "residual":
            strategy_badge = "[residual]"

        self.nodes.append(
            {
                "id": node_id,
                "label": f"Camada {layer}",
                "kind": "layer",
                "badge": strategy_badge,
                "x": float(layer),
                "y": 0.0,
            }
        )

        if layer == 1:
            self._add_edge("input", node_id, "main")
        elif self.strategy == "dense":
            for prev_layer in range(1, layer):
                prev_merge = f"layer_{prev_layer}_merge"
                if self._has_node(prev_merge):
                    self._add_edge(prev_merge, node_id, "dense")
            prev_direct = f"layer_{layer-1}"
            if self._has_node(prev_direct):
                self._add_edge(prev_direct, node_id, "main")
        elif self.strategy == "residual":
            prev_merge = f"layer_{layer-1}_merge"
            if self._has_node(prev_merge):
                self._add_edge(prev_merge, node_id, "main")
            else:
                self._add_edge(self.last_node_id, node_id, "main")
            self._add_edge("input", node_id, "skip")
        else:
            self._add_edge(self.last_node_id, node_id, "main")

        self.last_node_id = node_id
        return node_id

    def _add_selected_models(self, layer: int, models: list[str]):
        layer_node_id = self._ensure_layer_node(layer)
        spread = max(1, len(models))

        for idx, model_name in enumerate(models):
            node_id = f"layer_{layer}_model_{idx}"
            y_value = (idx - (spread - 1) / 2.0) * 0.9
            if not self._has_node(node_id):
                self.nodes.append(
                    {
                        "id": node_id,
                        "label": model_name,
                        "kind": "model",
                        "x": float(layer) + 0.55,
                        "y": y_value,
                    }
                )
            self._add_edge(layer_node_id, node_id, "branch")

        if models:
            merge_id = f"layer_{layer}_merge"
            if not self._has_node(merge_id):
                self.nodes.append(
                    {
                        "id": merge_id,
                        "label": "Meta-features",
                        "kind": "merge",
                        "x": float(layer) + 1.0,
                        "y": 0.0,
                    }
                )
            for idx, _ in enumerate(models):
                self._add_edge(f"layer_{layer}_model_{idx}", merge_id, "merge")
            self.last_node_id = merge_id

    def _add_ensemble_node(self, layer: int, model_name: str):
        layer_key = str(layer)
        idx = int(self.ensemble_counters.get(layer_key, 0))
        node_id = f"layer_{layer}_ens_{idx}"
        self.ensemble_counters[layer_key] = idx + 1

        kind = "voting" if "voting" in model_name.lower() else "stacking"
        label = model_name if len(model_name) <= 24 else model_name[:24] + "..."
        y_pos = -1.2 - (idx * 0.45)

        if not self._has_node(node_id):
            self.nodes.append(
                {
                    "id": node_id,
                    "label": label,
                    "kind": kind,
                    "x": float(layer) + 1.35,
                    "y": y_pos,
                }
            )

        merge_id = f"layer_{layer}_merge"
        if self._has_node(merge_id):
            self._add_edge(merge_id, node_id, "ensemble")
        else:
            self._add_edge(f"layer_{layer}", node_id, "ensemble")

    def _render_topology(self):
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#0b1220")
        ax.set_facecolor("#111827")

        node_index = {node["id"]: node for node in self.nodes}
        for edge in self.edges:
            source = node_index.get(edge["source"])
            target = node_index.get(edge["target"])
            if not source or not target:
                continue

            edge_kind = str(edge.get("kind", "main"))
            line_color = "#3b4a63"
            line_style = "-"
            line_width = 1.6
            if edge_kind == "dense":
                line_color = "#06b6d4"
                line_style = "--"
                line_width = 1.2
            elif edge_kind == "skip":
                line_color = "#f43f5e"
                line_style = ":"
                line_width = 1.5
            elif edge_kind in ("merge", "branch"):
                line_color = "#64748b"
            elif edge_kind == "ensemble":
                line_color = "#a78bfa"
                line_style = "--"

            ax.plot(
                [source["x"], target["x"]],
                [source["y"], target["y"]],
                color=line_color,
                linewidth=line_width,
                linestyle=line_style,
                alpha=0.9,
                zorder=1,
            )

        color_map = {
            "input": "#38bdf8",
            "layer": "#22c55e",
            "model": "#fb923c",
            "merge": "#a78bfa",
            "voting": "#f59e0b",
            "stacking": "#ec4899",
        }

        for node in self.nodes:
            color = color_map.get(str(node["kind"]), "#455a64")
            ax.scatter(
                float(node["x"]),
                float(node["y"]),
                s=400,
                color=color,
                edgecolor="#0b1220",
                linewidth=1.5,
                zorder=3,
            )
            ax.text(
                float(node["x"]),
                float(node["y"]) - 0.25,
                str(node["label"]),
                fontsize=8,
                ha="center",
                va="top",
                color="#e5e7eb",
            )

            badge = str(node.get("badge", ""))
            if badge:
                _badge_colors = {"[dense]": "#06b6d4", "[residual]": "#f43f5e"}
                badge_color = _badge_colors.get(badge, "#94a3b8")
                ax.text(
                    float(node["x"]),
                    float(node["y"]) + 0.28,
                    badge,
                    fontsize=7.5,
                    ha="center",
                    va="bottom",
                    color=badge_color,
                    fontweight="bold",
                    zorder=4,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor=badge_color,
                        alpha=0.18,
                        edgecolor=badge_color,
                        linewidth=0.9,
                    ),
                )

        ax.set_title("Fluxo de Treinamento: Entrada, Camadas e Ramificacoes", fontsize=13, color="#f8fafc")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        fig.tight_layout()
        fig.savefig(TOPOLOGY_IMAGE, dpi=150)
        plt.close(fig)

        self.graph_image_url = f"/ensemble_topology.png?ts={int(time.time() * 1000)}"

    def _handle_line(self, line: str):
        self.logs.append(line)
        self._update_time_estimates()

        complete_match = re.search(r"\[INFO\]\s+Layer\s+(\d+)\s+Done", line)
        if complete_match:
            self.completed_layers = max(self.completed_layers, int(complete_match.group(1)))

        early_stop_match = re.search(r"Stopping at Layer\s+(\d+)", line)
        if early_stop_match:
            self.completed_layers = max(self.completed_layers, int(early_stop_match.group(1)))

        if "COMPLETE | Time" in line:
            self.completed_layers = max(self.completed_layers, self.current_layer)

        layer_match = re.search(r"\[LAYER\s+(\d+)\]", line)
        if layer_match:
            self.current_layer = int(layer_match.group(1))
            self.status_text = f"Treinando camada {self.current_layer}"
            self._ensure_layer_node(self.current_layer)
            self._render_topology()
            return

        selected_match = re.search(r"Selected set \((\d+) models\): \[(.*)\]", line)
        if selected_match and self.current_layer > 0:
            raw_models = selected_match.group(2)
            models = [m.strip().strip("'") for m in raw_models.split(",") if m.strip()]
            self._add_selected_models(self.current_layer, models)
            self._render_topology()
            return

        model_match = re.search(
            r"Layer\s+(\d+)\s+\|\s+([^|]+)\|\s+F1:\s+([0-9.]+)\s+\|\s+Acc:\s+([0-9.]+)",
            line,
        )
        if model_match:
            layer_num = model_match.group(1)
            model_name = model_match.group(2).strip()
            self.model_rows.append(
                {
                    "layer": layer_num,
                    "model": model_name,
                    "f1": model_match.group(3),
                    "acc": model_match.group(4),
                }
            )
            self.model_rows = self.model_rows[-30:]

            if "voting" in model_name.lower() or "stack" in model_name.lower():
                self._add_ensemble_node(int(layer_num), model_name)
                self._render_topology()

    @rx.event(background=True)
    async def run_training(self):
        if self.is_training:
            return

        async with self:
            self.is_training = True
            self._reset_session()

        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--layers",
            str(self.layers),
            "--seed",
            str(self.seed),
            "--min_models",
            str(self.min_models),
            "--max_models",
            str(self.max_models),
            "--epsilon",
            str(self.epsilon),
            "--patience",
            str(self.patience),
            "--metric",
            self.metric,
            "--tfidf_max",
            str(self.tfidf_max),
            "--tfidf_ngrams",
            str(self.tfidf_ngrams),
            "--jitter",
            str(self.jitter),
            "--strategy",
            self.strategy,
            "--max_train_rows",
            str(self.sample_train_rows),
            "--max_val_rows",
            str(self.sample_val_rows),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async with self:
            self.training_pid = int(proc.pid)
            run_type = "Smoke Test" if self.is_smoke_test else "Treino"
            self.status_text = f"{run_type} em andamento (PID {proc.pid})"
            self.training_start_ts = time.time()
            self.elapsed_seconds = 0
            self.eta_seconds = 0

        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            async with self:
                if self.stop_requested:
                    self.logs.append("[UI] Solicitacao de parada recebida. Encerrando processo...")
                    break
                self._handle_line(line)

        if proc.returncode is None and self.stop_requested:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, text=True)
            else:
                proc.terminate()

        exit_code = await proc.wait()
        async with self:
            self.is_training = False
            self.training_pid = 0
            self._update_time_estimates()
            self.eta_seconds = 0
            if self.stop_requested:
                self.status_text = "Treino interrompido pelo usuario"
            else:
                self.status_text = "Treino finalizado" if exit_code == 0 else f"Falha no treino (codigo {exit_code})"

    def run_smoke_test(self):
        if self.is_training:
            return

        self.layers = 2
        self.min_models = 1
        self.max_models = 2
        self.epsilon = 0.35
        self.patience = 1
        self.metric = "f1"
        self.strategy = "simple"
        self.jitter = False
        self.tfidf_max = 1500
        self.tfidf_ngrams = 1
        self.sample_train_rows = 2500
        self.sample_val_rows = 1200
        self.is_smoke_test = True
        return TrainingState.run_training

    def run_full_training(self):
        self.sample_train_rows = 0
        self.sample_val_rows = 0
        self.is_smoke_test = False
        return TrainingState.run_training

    def stop_training(self):
        if not self.is_training:
            return

        self.stop_requested = True
        self.status_text = "Parando treino..."

        if self.training_pid > 0 and os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(self.training_pid), "/T", "/F"], capture_output=True, text=True)
        elif self.training_pid > 0:
            try:
                os.kill(self.training_pid, 15)
            except OSError:
                pass


def control_panel() -> rx.Component:
    return rx.box(
        rx.heading("Controles do Treino", size="4", color="#f8fafc"),
        rx.grid(
            rx.vstack(
                rx.text("Camadas", color="#cbd5e1"),
                rx.input(value=TrainingState.layers, on_change=TrainingState.update_layers, type="number"),
                align="start",
                spacing="1",
            ),
            rx.vstack(
                rx.text("Seed", color="#cbd5e1"),
                rx.input(value=TrainingState.seed, on_change=TrainingState.update_seed, type="number"),
                align="start",
                spacing="1",
            ),
            rx.vstack(
                rx.text("Modelos Min/Max", color="#cbd5e1"),
                rx.hstack(
                    rx.input(value=TrainingState.min_models, on_change=TrainingState.update_min_models, type="number"),
                    rx.input(value=TrainingState.max_models, on_change=TrainingState.update_max_models, type="number"),
                ),
                align="start",
                spacing="1",
            ),
            rx.vstack(
                rx.text("Epsilon RL", color="#cbd5e1"),
                rx.input(value=TrainingState.epsilon, on_change=TrainingState.update_epsilon, type="number", step="0.05"),
                align="start",
                spacing="1",
            ),
            columns="2",
            spacing="3",
            width="100%",
        ),
        rx.hstack(
            rx.vstack(
                rx.text("Metric", color="#cbd5e1"),
                rx.select(["f1", "accuracy"], value=TrainingState.metric, on_change=TrainingState.update_metric),
                align="start",
            ),
            rx.vstack(
                rx.text("Strategy", color="#cbd5e1"),
                rx.select(["dense", "residual", "simple"], value=TrainingState.strategy, on_change=TrainingState.update_strategy),
                align="start",
            ),
            rx.vstack(
                rx.text("TF-IDF max", color="#cbd5e1"),
                rx.input(value=TrainingState.tfidf_max, on_change=TrainingState.update_tfidf_max, type="number"),
                align="start",
            ),
            rx.vstack(
                rx.text("N-grams (1 ou 2)", color="#cbd5e1"),
                rx.input(value=TrainingState.tfidf_ngrams, on_change=TrainingState.update_tfidf_ngrams, type="number"),
                align="start",
            ),
            spacing="4",
            width="100%",
        ),
        rx.hstack(
            rx.text("Jitter", color="#cbd5e1"),
            rx.switch(checked=TrainingState.jitter, on_change=TrainingState.update_jitter),
            rx.spacer(),
            rx.button(
                "Parar Treino",
                on_click=TrainingState.stop_training,
                disabled=~TrainingState.is_training,
                color_scheme="red",
                variant="soft",
            ),
            rx.button(
                "Iniciar Treino",
                on_click=TrainingState.run_full_training,
                loading=TrainingState.is_training,
                disabled=TrainingState.is_training,
                color_scheme="green",
            ),
            rx.button(
                "Teste Rapido",
                on_click=TrainingState.run_smoke_test,
                loading=TrainingState.is_training,
                disabled=TrainingState.is_training,
                color_scheme="blue",
                variant="soft",
            ),
            width="100%",
        ),
        padding="1.2rem",
        border="1px solid #334155",
        border_radius="14px",
        background="#111827",
        width="100%",
    )


def topology_panel() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.heading("Topologia da Piramide", size="4", color="#f8fafc"),
            rx.badge(TrainingState.status_text, color_scheme="blue"),
            rx.badge(rx.cond(TrainingState.is_training, "Treinando", "Parado"), color_scheme="indigo"),
            justify="between",
            width="100%",
        ),
        rx.text("Fluxo visual em tempo real: conexao linear entre camadas e ramificacoes em arvore para modelos por camada.", color="#cbd5e1"),
        rx.hstack(
            rx.badge(TrainingState.elapsed_text, color_scheme="gray"),
            rx.badge(TrainingState.eta_text, color_scheme="teal"),
            rx.badge(TrainingState.flow_nodes_text, color_scheme="orange"),
            rx.badge(TrainingState.flow_edges_text, color_scheme="purple"),
            width="100%",
            spacing="2",
        ),
        rx.vstack(
            rx.hstack(
                rx.text("Progresso por Camada", weight="medium", color="#e2e8f0"),
                rx.spacer(),
                rx.text(TrainingState.layer_progress_text, size="2", color="#cbd5e1"),
                width="100%",
            ),
            rx.box(
                rx.box(
                    width=TrainingState.layer_progress_width,
                    height="100%",
                    background="linear-gradient(90deg, #0f9d58 0%, #59c98a 100%)",
                    border_radius="999px",
                    transition="width 240ms ease",
                ),
                width="100%",
                height="14px",
                background="#243244",
                border_radius="999px",
                overflow="hidden",
            ),
            spacing="1",
            width="100%",
        ),
        rx.image(src=TrainingState.graph_image_url, width="100%", height="420px", object_fit="contain"),
        rx.text("Nos: azul entrada, verde camada, laranja modelos, roxo meta-features, amarelo voting, rosa stacking", size="2", color="#94a3b8"),
        rx.text("Arestas: ciano tracejado=dense, vermelho pontilhado=residual skip, roxo tracejado=ensemble", size="2", color="#94a3b8"),
        padding="1.2rem",
        border="1px solid #334155",
        border_radius="14px",
        background="#111827",
        width="100%",
    )


def metrics_panel() -> rx.Component:
    return rx.box(
        rx.heading("Modelos Avaliados", size="4", color="#f8fafc"),
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Layer"),
                    rx.table.column_header_cell("Model"),
                    rx.table.column_header_cell("F1"),
                    rx.table.column_header_cell("Acc"),
                )
            ),
            rx.table.body(
                rx.foreach(
                    TrainingState.model_rows,
                    lambda row: rx.table.row(
                        rx.table.cell(row["layer"]),
                        rx.table.cell(row["model"]),
                        rx.table.cell(row["f1"]),
                        rx.table.cell(row["acc"]),
                    ),
                )
            ),
            variant="surface",
            size="1",
            width="100%",
        ),
        padding="1.2rem",
        border="1px solid #334155",
        border_radius="14px",
        background="#111827",
        width="100%",
        overflow_x="auto",
    )


def logs_panel() -> rx.Component:
    return rx.box(
        rx.heading("Log de Treinamento", size="4", color="#f8fafc"),
        rx.text_area(
            value=TrainingState.logs_text,
            read_only=True,
            min_height="320px",
            width="100%",
        ),
        padding="1.2rem",
        border="1px solid #334155",
        border_radius="14px",
        background="#111827",
        width="100%",
    )


def index() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading("Flexible Ensemble Pyramid Studio", size="7", color="#f8fafc"),
            rx.text("Execute, acompanhe e visualize a arquitetura do treino de forma interativa.", color="#cbd5e1"),
            spacing="2",
            align="start",
            width="100%",
        ),
        rx.box(height="14px"),
        control_panel(),
        rx.box(height="14px"),
        rx.hstack(
            rx.box(topology_panel(), width="65%"),
            rx.box(logs_panel(), width="35%"),
            width="100%",
            align="start",
            spacing="4",
        ),
        rx.box(height="14px"),
        metrics_panel(),
        padding="2rem",
        background="radial-gradient(circle at 20% 20%, #1f2937 0%, #0f172a 45%, #020617 100%)",
        min_height="100vh",
    )


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        radius="medium",
        accent_color="cyan",
    )
)
app.add_page(index, title="Flexible Ensemble Reflex UI")
