<div align="center">
  <img src="assets/LLLM-logo.png" alt="LLLM Logo" width="600" style="margin-left:'auto' margin-right:'auto'"/>
  <br>
  <br>
  <h1>Low-Level Language Models (LLLM) </h1>
</div>

LLLM is a lightweight framework designed to facilitate the rapid prototyping of advanced agentic systems. 
Prioritizing minimalism, modularity, and type safety, it is specifically optimized for research in program synthesis and neuro-symbolic AI. 
While these fields require deep architectural customization, researchers often face the burden of managing low-level complexities such as exception handling, output parsing, and API error management. 
LLLM bridges this gap by offering necessary abstractions that balance high-level encapsulation with the simplicity required for flexible experimentation.



## Features

- **Modular Architecture**: Core abstractions, providers, tools, and memory are decoupled.
- **Type Safety**: Built on Pydantic for robust data validation and strict typing.
- **Multi-Provider Support**: First-class support for OpenAI, with an extensible provider interface.
- **Neuro-Symbolic Design**: Advanced prompt management with structured output, exception handling, and interrupt logic.
- **Jupyter Sandbox**: Secure code execution environment for program synthesis.

## Installation

```bash
pip install -e .
```

## Quick Start

### Basic Chat

```python
from lllm import AgentBase, Prompt, register_prompt

# Define a prompt
simple_prompt = Prompt(
    path="simple_chat",
    prompt="You are a helpful assistant. User says: {user_input}"
)
register_prompt(simple_prompt)

# Define an Agent
class SimpleAgent(AgentBase):
    agent_type = "simple"
    agent_group = ["assistant"]
    
    def call(self, task: str, **kwargs):
        dialog = self.agents["assistant"].init_dialog({"user_input": task})
        response, dialog, _ = self.agents["assistant"].call(dialog)
        return response.content

# Configure and Run
config = {
    "name": "simple_chat_agent",
    "log_dir": "./logs",
    "log_type": "localfile",
    "agent_configs": {
        "assistant": {
            "model_name": "gpt-4o-mini",
            "system_prompt_path": "simple_chat",
            "temperature": 0.7,
        }
    }
}

agent = SimpleAgent(config, ckpt_dir="./ckpt")
print(agent("Hello!"))
```

## Documentation

See `docs/` for detailed documentation.

## Examples

Check `examples/` for more usage scenarios:
- `examples/basic_chat.py`
- `examples/tool_use.py`
- `examples/proxy_catalog.py`
- `examples/jupyter_sandbox_smoke.py`

### Proxies & Tools

Built-in proxies (financial data, search, etc.) register automatically when their modules are imported. If you plan to call `Proxy()` directly, either:

1. Set up an `lllm.toml` with a `[proxies]` section so discovery imports your proxy folders on startup, or
2. Call `load_builtin_proxies()` to import the packaged modules, or manually import the proxies you care about (e.g., `from lllm.proxies.builtin import exa_proxy`).

This mirrors how prompts are auto-registered via `[prompts]` in `lllm.toml`.

Once proxies are loaded you can check what is available by calling `Proxy().available()`.

### Auto-Discovery Config

A starter `lllm.toml.example` lives in the repo root. Copy it next to your project entry point and edit the folder paths:

```bash
cp lllm.toml.example lllm.toml
```

The sample configuration points to `examples/autodiscovery/prompts/` and `examples/autodiscovery/proxies/`, giving you a working prompt (`examples/hello_world`) and proxy (`examples/sample`) to experiment with immediately.

## Testing & Offline Mocks

- Run the full suite: `pytest`.
- For an end-to-end agent/tool flow without real OpenAI requests, see `tests/integration/test_tool_use_mock_openai.py`. It uses the scripted client defined in `tests/helpers/mock_openai.py`, mirroring what a VCR fixture would capture.
- When you eventually want to replay real HTTP exchanges, swap the scripted responses with your recorded payloads—the rest of the provider stack already supports dependency injection via this mock client.

## Testing

Run tests with pytest:

```bash
pytest tests/
```
