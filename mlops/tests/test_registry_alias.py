"""Registry tests with aliases (MLflow 3.x) in a temporary tracking store.

All offline (file store in tmp_path): creates a model + version from a local
source, promotes via the helper and resolves it back. No training, no network.
"""
import mlflow
from mlflow.tracking import MlflowClient

from mlops.registry import (
    latest_unstaged_version, promote_version, resolve_production_version)


def _client(tmp_path):
    uri = tmp_path.as_uri()
    mlflow.set_tracking_uri(uri)
    return MlflowClient(uri), tmp_path


def test_promote_and_resolve_via_alias(tmp_path):
    client, _ = _client(tmp_path)
    client.create_registered_model("demo_model")
    src = tmp_path / "art"
    src.mkdir()
    (src / "MLmodel").write_text("artifact_path: model\n", encoding="utf-8")
    v = client.create_model_version(name="demo_model", source=str(src), run_id=None)
    res = promote_version(client, "demo_model", v.version, "production", "Production")
    assert res["alias"] == "production"
    got = resolve_production_version(client, "demo_model", "production", "Production")
    assert got.version == v.version


def test_resolve_falls_back_to_stage_without_alias(tmp_path):
    client, _ = _client(tmp_path)
    client.create_registered_model("stage_model")
    src = tmp_path / "art"
    src.mkdir()
    (src / "MLmodel").write_text("artifact_path: model\n", encoding="utf-8")
    v = client.create_model_version(name="stage_model", source=str(src), run_id=None)
    client.transition_model_version_stage(
        name="stage_model", version=v.version, stage="Production",
        archive_existing_versions=True)
    got = resolve_production_version(client, "stage_model", "production", "Production")
    assert got.version == v.version


def test_latest_unstaged_picks_newest(tmp_path):
    client, _ = _client(tmp_path)
    client.create_registered_model("m")
    src = tmp_path / "art"
    src.mkdir()
    (src / "MLmodel").write_text("artifact_path: model\n", encoding="utf-8")
    v1 = client.create_model_version(name="m", source=str(src), run_id=None)
    v2 = client.create_model_version(name="m", source=str(src), run_id=None)
    assert latest_unstaged_version(client, "m").version == v2.version
    assert v1.version != v2.version
