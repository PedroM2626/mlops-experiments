"""Testes do metrics_store com SQLite temporário (não toca o DB de produção)."""

import mlops.config as config
import mlops.metrics_store as ms


def test_prediction_drift_retrain_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_mlops.db")
    ms.init_db()
    cost = ms.log_prediction(weeks=5, n_predictions=1000, latency_ms=12.5)
    assert cost == 0.0009
    ms.log_drift(10, 2, 0.31, 0.10, True, {"max_psi": 0.31})
    ms.log_retrain("test", "run123", 1.42, "ok")
    s = ms.get_summary(3600)
    assert s["predictions"]["calls"] == 1
    assert s["drift"]["triggered"] == 1
    assert s["retrains"] == 1
    assert len(ms.recent_predictions(5)) == 1
    assert len(ms.recent_drift(5)) == 1
    assert ms.last_retrain_ts() > 0
