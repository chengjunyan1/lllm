# Providers

Providers abstract the underlying LLM APIs, allowing you to switch between models and services easily.

## `BaseProvider`

All providers inherit from `BaseProvider`. This abstract base class enforces a consistent interface for:

- `call`: Executing a completion request.
- `cost`: Calculating the cost of a request.

## Supported Providers

### OpenAI

The `OpenAIProvider` supports all OpenAI-compatible APIs (including Together AI, etc.).

```python
from lllm.providers.openai import OpenAIProvider

provider = OpenAIProvider(config)
```

### Adding a New Provider

To add a new provider (e.g., Anthropic), subclass `BaseProvider` and implement the required methods.
