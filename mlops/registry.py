"""Model Registry helpers with aliases (MLflow 3.x) + fallback to stages.

The legacy flows used `get_latest_versions(stages=[...])` /
`transition_model_version_stage`, both deprecated in MLflow 3.x. The modern
path is aliases (`models:/<name>@<alias>`). To avoid breaking existing deploys
(registry with stage Production but no alias), resolution tries
the alias first and falls back to the stage.
"""
from __future__ import annotations


def promote_version(client, name: str, version: str, alias: str, stage: str) -> dict:
    """Points `alias` at `version`; also tries the legacy stage (best-effort)."""
    client.set_registered_model_alias(name=name, alias=alias, version=version)
    stage_ok, stage_err = True, None
    try:
        client.transition_model_version_stage(
            name=name, version=version, stage=stage, archive_existing_versions=True)
    except Exception as e:  # noqa: BLE001 - the stage may not exist in a future MLflow
        stage_ok, stage_err = False, str(e)[:120]
    return {"alias": alias, "version": str(version), "stage_ok": stage_ok,
            "stage_error": stage_err}


def resolve_production_version(client, name: str, alias: str, stage: str):
    """Returns the production ModelVersion: alias first, stage as fallback."""
    try:
        return client.get_model_version_by_alias(name=name, alias=alias)
    except Exception:
        pass
    versions = client.search_model_versions(f"name='{name}'")
    prod = [v for v in versions if v.current_stage == stage]
    if not prod:
        raise RuntimeError(f"no alias '{alias}' and no stage '{stage}' for '{name}'")
    return sorted(prod, key=lambda x: x.last_updated_timestamp)[-1]


def latest_unstaged_version(client, name: str):
    """Most recent version still without a stage (the one auto-registered by log_model)."""
    versions = client.search_model_versions(f"name='{name}'")
    fresh = [v for v in versions if v.current_stage in (None, "None", "")]
    pool = fresh or list(versions)
    if not pool:
        raise RuntimeError(f"no version of '{name}'")
    return sorted(pool, key=lambda x: x.last_updated_timestamp)[-1]
