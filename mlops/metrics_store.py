"""Persistencia das metricas de producao (latencia, custo, drift) em SQLite."""
import sqlite3
import json
import time
from contextlib import contextmanager
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    weeks INTEGER,
    n_predictions INTEGER,
    latency_ms REAL,
    cost_usd REAL,
    status TEXT
);
CREATE TABLE IF NOT EXISTS drift_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    n_features INTEGER,
    drifted_features INTEGER,
    max_psi REAL,
    max_share_change REAL,
    triggered INTEGER,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS retrain_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    reason TEXT,
    run_id TEXT,
    val_mae REAL,
    status TEXT
);
"""


def _connect():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as c:
        c.executescript(SCHEMA)


def log_prediction(weeks, n_predictions, latency_ms, status="ok"):
    cost = (n_predictions / 1000.0) * config.COST_PER_1000_PREDICTIONS
    with get_conn() as c:
        c.execute(
            "INSERT INTO predictions (ts, weeks, n_predictions, latency_ms, cost_usd, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), weeks, n_predictions, latency_ms, cost, status),
        )
    return cost


def log_drift(n_features, drifted_features, max_psi, max_share_change, triggered, payload):
    with get_conn() as c:
        c.execute(
            "INSERT INTO drift_checks (ts, n_features, drifted_features, max_psi, max_share_change, triggered, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), n_features, drifted_features, max_psi, max_share_change, int(triggered),
             json.dumps(payload, default=str)),
        )


def last_retrain_ts():
    """TS (epoch) do retrain mais recente; 0.0 se nunca houve."""
    init_db()
    with get_conn() as c:
        row = c.execute("SELECT MAX(ts) FROM retrain_events").fetchone()
    return row[0] or 0.0


def log_retrain(reason, run_id, val_mae, status):
    with get_conn() as c:
        c.execute(
            "INSERT INTO retrain_events (ts, reason, run_id, val_mae, status) VALUES (?, ?, ?, ?, ?)",
            (time.time(), reason, run_id, val_mae, status),
        )


def get_summary(seconds=3600):
    now = time.time()
    cutoff = now - seconds
    with get_conn() as c:
        preds = c.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(n_predictions),0) total_preds, "
            "COALESCE(SUM(cost_usd),0) cost, COALESCE(AVG(latency_ms),0) avg_lat, "
            "COALESCE(MAX(latency_ms),0) max_lat FROM predictions WHERE ts >= ?",
            (cutoff,)).fetchone()
        drift = c.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(triggered),0) triggered, COALESCE(MAX(max_psi),0) max_psi "
            "FROM drift_checks WHERE ts >= ?", (cutoff,)).fetchone()
        retrains = c.execute(
            "SELECT COUNT(*) n FROM retrain_events WHERE ts >= ?", (cutoff,)).fetchone()
    return {
        "window_seconds": seconds,
        "predictions": {"calls": preds["n"], "total_predictions": preds["total_preds"],
                        "cost_usd": round(preds["cost"], 6), "avg_latency_ms": round(preds["avg_lat"], 2),
                        "max_latency_ms": round(preds["max_lat"], 2)},
        "drift": {"checks": drift["n"], "triggered": drift["triggered"], "max_psi": round(drift["max_psi"], 4)},
        "retrains": retrains["n"],
    }


def recent_predictions(limit=20):
    with get_conn() as c:
        rows = c.execute(
            "SELECT ts, weeks, n_predictions, latency_ms, cost_usd, status "
            "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def recent_drift(limit=10):
    with get_conn() as c:
        rows = c.execute(
            "SELECT ts, n_features, drifted_features, max_psi, max_share_change, triggered "
            "FROM drift_checks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
