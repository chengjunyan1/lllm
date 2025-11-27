from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10
    import tomli as tomllib  # type: ignore


LLLM_CONFIG_ENV = "LLLM_CONFIG"
LLLM_CONFIG_DISABLE_ENV = "LLLM_AUTO_DISCOVER"
CONFIG_FILENAMES = ("lllm.toml", ".lllm.toml", "LLLM.toml")
CONFIG_SUBDIRS = ("", "template")


def _resolve_candidate(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "lllm.toml"
    if candidate.is_file():
        return candidate.resolve()
    return None


def find_config_file(start_path: Optional[str | os.PathLike[str]] = None) -> Optional[Path]:
    """
    Locate the nearest lllm.toml by checking:
      1. The LLLM_CONFIG environment variable (file or directory)
      2. The provided start_path and its parents
      3. The current working directory and its parents
    """
    env_path = _resolve_candidate(os.environ.get(LLLM_CONFIG_ENV))
    if env_path:
        return env_path

    search_roots: List[Path] = []
    if start_path is not None:
        search_roots.append(Path(start_path).resolve())
    search_roots.append(Path.cwd())

    for root in search_roots:
        for path in [root, *root.parents]:
            for subdir in CONFIG_SUBDIRS:
                base = path if subdir == "" else path / subdir
                for name in CONFIG_FILENAMES:
                    candidate = base / name
                    if candidate.is_file():
                        return candidate.resolve()
    return None


def load_config(path: Optional[str | os.PathLike[str]] = None) -> Optional[Dict]:
    config_path = _resolve_candidate(path) or find_config_file(path)
    if not config_path:
        return None
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    data["_config_path"] = config_path
    return data


def auto_discovery_disabled() -> bool:
    return os.environ.get(LLLM_CONFIG_DISABLE_ENV, "1").lower() in {"0", "false", "no"}
