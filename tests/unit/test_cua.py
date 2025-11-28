import importlib

import pytest

from lllm.tools import cua


def test_ensure_playwright_missing(monkeypatch):
    monkeypatch.setattr(cua, "_PLAYWRIGHT_LOADER", None)
    monkeypatch.setattr(cua, "_PLAYWRIGHT_TIMEOUT", None)

    def fake_import(name):
        raise ImportError("missing module")

    monkeypatch.setattr(cua.importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError):
        cua._ensure_playwright()


def test_load_async_azure_missing_client(monkeypatch):
    class DummyModule:
        pass

    real_import = importlib.import_module

    def fake_import(name):
        if name == "openai":
            return DummyModule()
        return real_import(name)

    monkeypatch.setattr(cua.importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError):
        cua._load_async_azure_openai()
