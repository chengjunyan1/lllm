# Guide: Templates & CLI

The `lllm` CLI bootstraps reproducible agent projects using the templates shipped in this repository.

## CLI Overview

```bash
pip install git+https://github.com/ChengJunyan1/lllm.git
lllm --help
lllm create --name system            # uses template/init_template
lllm create --name lab --template example
```

`lllm create` copies the chosen template into the current directory, performing simple text substitutions:

- `__project_name__` → provided `--name`.
- `{{project_name}}` and `{{PROJECT_NAME}}` placeholders are replaced in file contents and paths.

Text files (`.py`, `.md`, `.toml`, `.yaml`, etc.) are rendered; binaries are copied verbatim.

## Templates

### `init_template`

A minimal scaffold intended for fresh projects:

- `lllm.toml` – points to `system/agent/prompts` and `system/proxy/modules`.
- `config/<project>/default.yaml` – stub configuration with logging, agent, and proxy settings.
- `system/agent/agent.py` – contains a `Vanilla` agent wired to `Prompts('vanilla')`.
- `system/system.py` – exposes `SimpleSystem` with `.call()`.

Use this template when you want a clean slate without bundled proxies or prompts.

### `example`

A richer showcase that demonstrates:

- Multiple proxy modules (finance, macro, knowledge-base, Wolfram Alpha, custom MCP).
- Prompt definitions with Pydantic formats (`template/example/system/agent/prompts/vanilla.py`).
- `system/system.py` illustrating how to manage experiment ids, streaming output, and async agents.

This template also includes configuration defaults for cutting-edge models (e.g., `o4-mini-2025-04-16`).

## Customizing the Scaffold

1. **Update configuration** – edit `config/<name>/default.yaml`. Key sections:
   - `agent_configs` – specify `model_name`, `system_prompt_path`, `temperature`, etc.
   - Retry and safety knobs (`max_exception_retry`, `max_interrupt_times`).
   - `activate_proxies` – list of proxy identifiers to preload (matching `_proxy_path`).
2. **Register prompts** – add `.py` files under `system/agent/prompts`. Auto-discovery registers any module-level `Prompt` objects.
3. **Add proxies** – drop new proxy modules under `system/proxy/modules`, decorate classes with `@ProxyRegistrator`, and configure API keys via environment variables.
4. **Version control** – templates avoid user-specific secrets by design. Store credentials as env vars or encrypted secrets.

## Publishing Docs Alongside Your System

This repository now includes a MkDocs configuration (see `mkdocs.yml`). Projects generated from the template can copy the `docs/` structure or point to this repository as a submodule, then run `mkdocs build` to publish GitHub Pages documentation for their custom agents.

## Troubleshooting

- **"Path already exists"** – the CLI refuses to overwrite existing folders; pick a new `--name` or delete the directory manually.
- **Auto-discovery misses modules** – ensure `lllm.toml` paths are correct relative to the project root and that files end with `.py` (not `._py`).
- **Proxy modules raise import errors** – verify third-party dependencies (e.g., `requests`, `tqdm`) are installed in your environment.
