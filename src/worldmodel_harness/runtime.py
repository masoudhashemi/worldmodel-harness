"""Thin model-client adapter. Stirrup owns the agent loop."""

from pydantic import SecretStr
from stirrup.clients.litellm_client import LiteLLMClient

from .config import ModelSettings


def build_client(settings: ModelSettings, api_key: SecretStr) -> LiteLLMClient:
    return LiteLLMClient(
        model=settings.name,
        api_key=api_key.get_secret_value(),
        max_tokens=settings.max_tokens,
        context_window_tokens=settings.context_window_tokens,
        kwargs={
            "api_base": settings.api_base,
            "temperature": settings.temperature,
            "timeout": settings.timeout_seconds,
            "num_retries": 0,
            **({"response_format": {"type": "json_object"}} if settings.json_mode else {}),
            **(
                {"extra_body": {"reasoning": {"effort": settings.reasoning_effort}}}
                if settings.reasoning_effort is not None
                else {}
            ),
        },
    )
