"""Testes do lifespan do serve sem carregar dados/modelo reais.

Importar `mlops.serve` e monkeypatchar `_ensure_predictor`/`_kick_precompute`
+ DB temporário: o bloco lifespan executa sem tocar no registry nem nos
parquets de vendas.
"""
import asyncio

import mlops.config as config
import mlops.serve as serve


def test_no_legacy_on_event_handlers():
    assert serve.app.router.on_startup == []
    assert serve.app.router.lifespan_context is not None


def test_lifespan_runs_with_monkeypatched_internals(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "serve_test.db")
    calls = []
    monkeypatch.setattr(serve, "_ensure_predictor",
                        lambda: calls.append("predictor") or None)
    monkeypatch.setattr(serve, "_kick_precompute",
                        lambda: calls.append("precompute"))

    async def _run():
        async with serve.app.router.lifespan_context(serve.app):
            pass

    asyncio.run(_run())
    assert calls == ["predictor", "precompute"]
    assert (tmp_path / "serve_test.db").exists()
