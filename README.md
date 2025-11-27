# LLLM: Large Language Model Agent Framework

LLLM is a professional, production-ready framework for building advanced agentic systems. It emphasizes modularity, type safety, program synthesis, and neuro-symbolic capabilities.

## Features

- **Modular Architecture**: Core abstractions, providers, tools, and memory are decoupled.
- **Type Safety**: Built on Pydantic for robust data validation and strict typing.
- **Multi-Provider Support**: First-class support for OpenAI, with extensible provider interface.
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

## Testing

Run tests with pytest:

```bash
pytest tests/
```
