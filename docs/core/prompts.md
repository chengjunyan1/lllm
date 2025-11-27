# Prompt Design

Prompts in LLLM are programmable objects defined in `lllm/models.py`. They encapsulate the raw prompt string, structured output schema, safety handlers, parser hints, and linked tools. By treating prompts as data, the platform can auto-register, introspect, and reuse them across agents.

## Anatomy of a Prompt

```python
from lllm.models import Prompt, Function
from pydantic import BaseModel

class TaskResponse(BaseModel):
    output: str

main_prompt = Prompt(
    path="vanilla/task_query",
    prompt="""
    Solve the task: {task}
    """,
    format=TaskResponse,
    xml_tags=["analysis", "answer"],
)
```

Key fields:

- `path` – unique identifier used by `Prompts('<root>')('task_query')` and by auto-discovery.
- `prompt` – string template formatted with keyword arguments before being inserted into the dialog.
- `format` – optional Pydantic model used for OpenAI structured responses (forces schema compliance when supported by the provider).
- `parser` – callable that converts `Message.content` into structured data. Defaults to `default_parser`, which understands XML tags, Markdown fences, and user-defined signal tags.
- `xml_tags` / `md_tags` / `signal_tags` – hints for the default parser so it can extract multi-block outputs (useful for multi-part reasoning or JSON-in-text patterns).
- `_functions` – list of `Function` definitions linked to actual Python callables at runtime via `Prompt.link_function`.
- `_mcp_servers` – list of MCP server descriptors for OpenAI’s Model Context Protocol (optional).
- `exception_prompt` / `interrupt_prompt` – inline handler prompts for error recovery and function-call responses.

## Parsers & Structured Output

LLLM provides multiple layers of structure enforcement:

1. **Pydantic formats** – If `format` is set and the target model supports `response_format`, the `LLMCaller` instructs the provider to emit JSON that conforms to the schema. Parsed output is the validated dict.
2. **XML / Markdown parsing** – Without structured output support, you can ask the model to wrap sections in XML tags (`<analysis>...</analysis>`) or Markdown fences (```json). The default parser extracts the requested tags and populates `Message.parsed`.
3. **Custom parsers** – Supply your own callable if you need domain-specific parsing (e.g., spreadsheets or DSLs). Raise `ParseError` to trigger the exception handler.

Tip: mix multiple hints—structured output where available, XML/MD tags everywhere else—to build resilient prompts.

## Handlers in Practice

```python
analysis_prompt = Prompt(
    path="research/system",
    prompt="""
    You are a research analyst...
    <analysis>{analysis_plan}</analysis>
    """,
    exception_prompt="""
    The previous answer could not be parsed:
    {error_message}
    Fix the output using the same XML structure.
    """,
    interrupt_prompt="""
    Tool results:
    {call_results}
    Continue reasoning with the same XML sections.
    """,
)
```

- Exception dialogs are pruned, so retry noise never leaks into the final transcript.
- Interrupt dialogs stay in place, making it obvious which tool responses influenced the final answer.
- The `Prompt.interrupt_handler_final` helper issues a final "wrap up" command once the agent stops calling tools.

## Function & MCP Integration

`Function` objects describe Python callables with JSON schemas:

```python
from lllm.models import Function

def search_symbols(query: str) -> str:
    ...

search_fn = Function(
    name="search_symbols",
    description="Lookup ticker metadata",
    properties={"query": {"type": "string"}},
    required=["query"],
)

main_prompt = Prompt(..., _functions=[search_fn])
main_prompt.link_function("search_symbols", search_symbols)
```

During an agent call, tool invocations become structured `FunctionCall` objects that carry `arguments`, `result`, and `result_str`. MCP servers are declared similarly to expose entire tool suites.

## Auto-Registration

Any module loaded through discovery can define prompts as module-level variables. Example (`template/example/system/agent/prompts/vanilla.py`):

```python
main_system_prompt = Prompt(path="system", prompt="...")
```

Once the file is imported, the prompt is registered at `prompts.vanilla.system`. Systems then access it via `Prompts('vanilla')('system')` with zero boilerplate.

## Best Practices

- Keep prompt paths short but descriptive (`analysis/system`, `analysis/task_query`).
- Co-locate prompts with their agents or modules for easier mental mapping.
- Always provide at least one structured cue (XML tags or Pydantic format) so the default parser can validate outputs.
- Store shared handler templates (e.g., generic JSON fixer) in a central module to enforce consistency across teams.
- Use the `allow_web_search` and `computer_use_config` flags only when the target model supports those features (see `const.py`).
