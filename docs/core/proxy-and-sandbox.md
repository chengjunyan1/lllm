# Proxies & Sandbox

LLLM treats tool access as a first-class capability. Proxies standardize how agents reach external APIs, while the sandbox module enables stateful notebook or browser-style execution. Together they make it possible to build advanced coding agents that mix reasoning, API calls, and code execution.

## BaseProxy & Endpoint Registration

`lllm/proxies.py` defines `BaseProxy`, a reflection-based registry for HTTP endpoints. Endpoints are declared via decorators:

```python
from lllm.proxies import BaseProxy, ProxyRegistrator

@ProxyRegistrator(path="finance/fmp", name="Financial Modeling Prep", description="Market data API")
class FMPProxy(BaseProxy):
    base_url = "https://financialmodelingprep.com/api/v3"
    api_key_name = "apikey"

    @BaseProxy.endpoint(
        category="search",
        endpoint="search-symbol",
        description="Search tickers",
        params={"query*": (str, "AAPL"), "limit": (int, 10)},
        response=[{"symbol": "AAPL", "name": "Apple"}],
    )
    def search_symbol(self, params):
        return params
```

At import time, decorated methods receive metadata (`endpoint_info`). The base class introspects every method, builds an index, and exposes:

- `registry` – endpoint metadata and docstrings.
- `index` – directory of categories/subcategories.
- `_entries` – callable map used by `__call__` to execute endpoints.

Helpers include:

- `endpoint_directory` and `api_directory` for generating human-readable instructions that prompts can embed.
- `auto_test()` to smoke-test every endpoint with mock parameters.
- `ProxyRegistrator` to attach user-friendly names and descriptions for documentation.

## Runtime Proxy

The `Proxy` runtime composes multiple `BaseProxy` subclasses:

```python
from lllm.proxies import Proxy

proxy = Proxy(activate_proxies=["finance/fmp", "news/gkg"], cutoff_date="2024-01-01")
result = proxy("finance/fmp/search/search-symbol", {"query": "AAPL"})
```

Features:

- **Selective activation** – `activate_proxies` filters which proxies load; missing proxies are skipped without crashing the agent.
- **Cutoff dates** – Pass `cutoff_date` to enforce data availability constraints. Each proxy can opt-in to date filtering via `dt_cutoff` metadata.
- **Deployment mode** – `deploy_mode=True` disables cutoffs for production runs.
- **Documentation helpers** – `proxy.api_catalog`, `proxy.api_directory`, and `proxy.retrieve_api_docs()` return rich text that can be inserted into prompts so the model understands available tooling.

The example template (`template/example/system/proxy/modules/*.py`) showcases large sets of proxies (finance, macro, search, Wolfram Alpha, etc.) managed through this infrastructure.

## Prompt Integration

Prompts can link directly to proxy-backed functions. A typical pattern:

1. The proxy exposes a python method (e.g., `search_symbol`).
2. A prompt defines a `Function` with the same signature.
3. `Prompt.link_function("search_symbol", proxy_instance.search_symbol)` wires the tool call.
4. The agent instructs the model to call `search_symbol` via function-calling. Tool responses are fed back through interrupt handlers, keeping the agent deterministic.

Because proxies carry detailed metadata, you can also programmatically generate tool-selection prompts (“Here’s the API catalog... choose the minimal endpoint”).

## Sandbox for Notebook Agents

`lllm/sandbox.py` implements a programmable Jupyter environment aimed at coding agents:

- `JupyterSession` launches kernels, keeps notebooks on disk, and inserts a guarded init cell that imports project code plus the proxy runtime.
- Sessions store metadata such as proxy activation, cutoff dates, and project roots so that generated notebooks remain reproducible.
- Helpers like `insert_cell`, `overwrite_cell`, `run_all_cells`, and `directory_tree` let an agent (or a supervising process) manipulate the notebook without manual intervention.
- Execution is tracked with timestamps and can be synchronized with the logging subsystem.

This sandbox doubles as the runtime for notebook-style proxies—agents can spin up a session, run code, and feed the results back through interrupt handlers.

## Computer-Use Agent (CUA)

For browser-centric automation, `lllm/tools/cua.py` provides a stub Computer Use Agent powered by Playwright. It:

- Normalizes keyboard/mouse actions, scrolls, and text entry.
- Captures screenshots (with caching on failure) and enforces display boundaries.
- Defines control signals (e.g., `Ctrl+W`) so the model can terminate sessions safely.
- Illustrates how to extend LLLM with long-running, tool-controlled workflows.

Although the CUA is optional, it demonstrates how proxies and sandboxed environments align: prompts produce tool calls, proxies/sandboxes execute them, and the agent ensures the conversation reaches a conclusion.
