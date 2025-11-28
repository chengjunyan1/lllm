from __future__ import annotations

from typing import Callable, Dict

from lllm.providers.base import BaseProvider
from lllm.providers.openai import OpenAIProvider

ProviderBuilder = Callable[[Dict], BaseProvider]

_PROVIDER_BUILDERS: Dict[str, ProviderBuilder] = {
    "openai": lambda cfg: OpenAIProvider(cfg),
}


def register_provider(name: str, builder: ProviderBuilder, *, overwrite: bool = False) -> None:
    name = name.lower()
    if name in _PROVIDER_BUILDERS and not overwrite:
        raise ValueError(f"Provider '{name}' already registered")
    _PROVIDER_BUILDERS[name] = builder


def build_provider(config: Dict) -> BaseProvider:
    provider_name = config.get("provider", "openai").lower()
    try:
        builder = _PROVIDER_BUILDERS[provider_name]
    except KeyError as exc:
        raise KeyError(
            f"Provider '{provider_name}' not registered. Available: {sorted(_PROVIDER_BUILDERS)}"
        ) from exc
    provider_config = config.get("provider_config", config)
    return builder(provider_config)


__all__ = ["register_provider", "build_provider", "ProviderBuilder"]
