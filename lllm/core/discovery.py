from __future__ import annotations

import importlib.util
import inspect
import types
import warnings
from pathlib import Path
from typing import Iterable, Optional

from lllm.core.config import auto_discovery_disabled, load_config
from lllm.core.models import Prompt, register_prompt
from lllm.proxies.base import BaseProxy, register_proxy

IGNORED_FILES = {"__init__.py", "__pycache__"}
PROMPT_SECTION = "prompts"
PROXY_SECTION = "proxies"

_DISCOVERY_DONE = False
_DEFAULT_AUTO_DISCOVER = True


def auto_discover(config_path: Optional[str | Path] = None, *, force: bool = False) -> None:
    global _DISCOVERY_DONE
    if _DISCOVERY_DONE and not force:
        return
    if auto_discovery_disabled():
        _DISCOVERY_DONE = True
        return
    config = load_config(config_path)
    if not config:
        _DISCOVERY_DONE = True
        return
    base_dir = Path(config["_config_path"]).parent
    try:
        _discover_prompts(config.get(PROMPT_SECTION, {}), base_dir)
        _discover_proxies(config.get(PROXY_SECTION, {}), base_dir)
    finally:
        _DISCOVERY_DONE = True


def configure_auto_discover(enabled: bool) -> None:
    """
    Set the default behavior for future ``auto_discover_if_enabled`` calls when
    they do not supply an explicit flag.
    """
    global _DEFAULT_AUTO_DISCOVER
    _DEFAULT_AUTO_DISCOVER = bool(enabled)


def _should_auto_discover(flag: Optional[bool]) -> bool:
    if flag is None:
        return _DEFAULT_AUTO_DISCOVER
    return bool(flag)


def auto_discover_if_enabled(
    flag: Optional[bool] = None,
    config_path: Optional[str | Path] = None,
    *,
    force: bool = False,
) -> None:
    """
    Wrapper that respects the configured default/explicit flag before delegating
    to :func:`auto_discover`.
    """
    if not _should_auto_discover(flag):
        return
    auto_discover(config_path, force=force)


def _discover_prompts(section: dict, base_dir: Path) -> None:
    folders = _normalize_paths(section.get("folders") or [], base_dir)
    for folder in folders:
        for module, namespace in _load_modules_from_folder(folder, prefix="prompts"):
            _register_prompts_from_module(module, namespace)


def _discover_proxies(section: dict, base_dir: Path) -> None:
    folders = _normalize_paths(section.get("folders") or [], base_dir)
    for folder in folders:
        for module, namespace in _load_modules_from_folder(folder, prefix="proxies"):
            _register_proxies_from_module(module, namespace)


def _normalize_paths(entries: Iterable[str], base_dir: Path) -> list[Path]:
    normalized = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if path.exists():
            normalized.append(path)
        else:
            warnings.warn(f"LLLM discovery skipped missing path: {path}", RuntimeWarning)
    return normalized


def _load_modules_from_folder(folder: Path, prefix: str) -> Iterable[tuple[types.ModuleType, str]]:
    for file in sorted(folder.glob("*.py")):
        if file.name in IGNORED_FILES or file.name.startswith("_"):
            continue
        namespace = f"{prefix}.{folder.name}.{file.stem}"
        try:
            module = _load_module_from_file(file, namespace)
        except Exception as exc:  # pragma: no cover - best effort discovery
            warnings.warn(f"LLLM discovery failed to load {file}: {exc}", RuntimeWarning)
            continue
        yield module, file.stem


def _load_module_from_file(file_path: Path, namespace: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(namespace, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _register_prompts_from_module(module: types.ModuleType, namespace: str) -> None:
    for _, attr in vars(module).items():
        if isinstance(attr, Prompt):
            prompt = attr
            if "/" not in prompt.path:
                prompt.path = f"{namespace}/{prompt.path}".strip("/")
            register_prompt(prompt)


def _register_proxies_from_module(module: types.ModuleType, namespace: str) -> None:
    for _, cls in vars(module).items():
        if inspect.isclass(cls) and issubclass(cls, BaseProxy) and cls is not BaseProxy:
            proxy_name = getattr(cls, "_proxy_path", f"{namespace}/{cls.__name__}")
            register_proxy(proxy_name, cls, overwrite=True)
