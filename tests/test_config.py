import pytest
from pydantic import ValidationError

from worldmodel_harness.config import ModelSettings, load_api_key, load_settings


def test_repository_settings(monkeypatch):
    from pathlib import Path

    monkeypatch.delenv("WORLDMODEL_MODEL_CONFIG", raising=False)
    settings = load_settings(Path(__file__).resolve().parents[1])
    assert settings.name == "openrouter/nvidia/nemotron-3-super-120b-a12b:free"


def test_free_profile_is_explicit_and_cannot_fall_back(monkeypatch):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("WORLDMODEL_MODEL_CONFIG", "configs/model-free-test.toml")
    settings = load_settings(root)
    assert settings.name == "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    assert settings.max_tokens == 8192
    monkeypatch.setenv("WORLDMODEL_MODEL_CONFIG", "configs/nonexistent.toml")
    with pytest.raises(FileNotFoundError):
        load_settings(root)


def test_smoke_and_repository_default_profiles_are_free(monkeypatch):
    from pathlib import Path

    monkeypatch.delenv("WORLDMODEL_MODEL_CONFIG", raising=False)
    root = Path(__file__).resolve().parents[1]
    assert load_settings(root, default_profile="configs/model-free-test.toml").name.endswith(
        ":free"
    )
    assert load_settings(root).name == "openrouter/nvidia/nemotron-3-super-120b-a12b:free"


def test_dotenv_is_root_relative_and_secret_is_redacted(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=fixture-secret\n")
    key = load_api_key(tmp_path, ModelSettings())
    assert key.get_secret_value() == "fixture-secret"
    assert "fixture-secret" not in repr(key)


def test_explicit_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-secret")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=file-secret\n")
    assert load_api_key(tmp_path, ModelSettings()).get_secret_value() == "environment-secret"


def test_missing_key_fails_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing OPENROUTER_API_KEY"):
        load_api_key(tmp_path, ModelSettings())


@pytest.mark.parametrize(
    "options",
    [
        {"name": "qwen/qwen3.8-27b"},
        {"api_base": "https://untrusted.example"},
        {"max_tokens": 262144},
        {"timeout_seconds": 0},
        {"unexpected": True},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValidationError):
        ModelSettings(**options)
