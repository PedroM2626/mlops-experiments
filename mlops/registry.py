"""Helpers do Model Registry com aliases (MLflow 3.x) + fallback a stages.

Os fluxos legados usavam `get_latest_versions(stages=[...])` /
`transition_model_version_stage`, ambos deprecated no MLflow 3.x. O caminho
moderno sao aliases (`models:/<nome>@<alias>`). Para nao quebrar deploys
existentes (registry com stage Production mas sem alias), a resolucao tenta
o alias primeiro e cai para o stage.
"""
from __future__ import annotations


def promote_version(client, name: str, version: str, alias: str, stage: str) -> dict:
    """Aponta `alias` para `version`; tenta tambem o stage legado (best-effort)."""
    client.set_registered_model_alias(name=name, alias=alias, version=version)
    stage_ok, stage_err = True, None
    try:
        client.transition_model_version_stage(
            name=name, version=version, stage=stage, archive_existing_versions=True)
    except Exception as e:  # noqa: BLE001 - stage pode nao existir no MLflow futuro
        stage_ok, stage_err = False, str(e)[:120]
    return {"alias": alias, "version": str(version), "stage_ok": stage_ok,
            "stage_error": stage_err}


def resolve_production_version(client, name: str, alias: str, stage: str):
    """Retorna o ModelVersion de producao: alias primeiro, stage como fallback."""
    try:
        return client.get_model_version_by_alias(name=name, alias=alias)
    except Exception:
        pass
    versions = client.search_model_versions(f"name='{name}'")
    prod = [v for v in versions if v.current_stage == stage]
    if not prod:
        raise RuntimeError(f"sem alias '{alias}' nem stage '{stage}' para '{name}'")
    return sorted(prod, key=lambda x: x.last_updated_timestamp)[-1]


def latest_unstaged_version(client, name: str):
    """Versao mais recente ainda sem stage (a auto-registrada pelo log_model)."""
    versions = client.search_model_versions(f"name='{name}'")
    fresh = [v for v in versions if v.current_stage in (None, "None", "")]
    pool = fresh or list(versions)
    if not pool:
        raise RuntimeError(f"nenhuma versao de '{name}'")
    return sorted(pool, key=lambda x: x.last_updated_timestamp)[-1]
