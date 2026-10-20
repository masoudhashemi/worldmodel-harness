"""Validated public settings; credentials are loaded separately and never serialized."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    client: str = "litellm"
    name: str = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    dotenv_path: str = ".env"
    max_tokens: int = Field(default=2048, gt=0)
    context_window_tokens: int = Field(default=262144, gt=0)
    temperature: float = Field(default=0.0, ge=0, le=2)
    timeout_seconds: float = Field(default=60.0, gt=0)
    reasoning_effort: Literal["xhigh", "medium", "low", "none"] | None = None
    json_mode: bool = False

    @model_validator(mode="after")
    def validate_routing(self):
        if self.client != "litellm":
            raise ValueError("This runtime requires LiteLLM routing")
        if self.name.startswith("openrouter/"):
            if self.api_base != "https://openrouter.ai/api/v1":
                raise ValueError("OpenRouter models require https://openrouter.ai/api/v1")
        elif "elementai.com" in self.api_base or self.name.startswith("openai/"):
            pass
        else:
            raise ValueError("This runtime requires LiteLLM OpenRouter or Element AI routing")
        if self.max_tokens >= self.context_window_tokens:
            raise ValueError("Output budget must be smaller than context budget")
        return self


def load_settings(root: Path, *, default_profile: str = "configs/model.toml") -> ModelSettings:
    # Explicit profile selection applies to acting, learning and adaptation clients.
    # Invalid/missing profiles fail closed; never fall back to a paid model.
    profile = Path(os.environ.get("WORLDMODEL_MODEL_CONFIG", default_profile))
    with (root / profile).open("rb") as stream:
        document = tomllib.load(stream)
    return ModelSettings.model_validate(document["model"])


def load_api_key(root: Path, settings: ModelSettings) -> SecretStr:
    # No shell sourcing, interpolation, global environment changes, or secret logging.
    value = os.environ.get(settings.api_key_env)
    if not value:
        value = dotenv_values(root / settings.dotenv_path, interpolate=False).get(
            settings.api_key_env
        )
    if not value or not value.strip():
        raise ValueError(f"Missing {settings.api_key_env} in environment or project .env")
    return SecretStr(value.strip())
