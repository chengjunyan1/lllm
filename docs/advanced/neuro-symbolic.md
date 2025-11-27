# Neuro-Symbolic Workflows

LLLM’s prompts and agent loop were designed so that researchers can combine symbolic constraints (XML tags, Markdown sections, structured parsers) with probabilistic reasoning. The recommended pattern is:

1. **Define schemas** – create `Prompt` objects that declare XML/Markdown tags or attach a Pydantic `response_format` model.
2. **Enable handlers** – customize `exception_prompt`/`interrupt_prompt` so the agent can repair malformed output or summarize tool results before continuing.
3. **Link tools** – register `Function` objects that wrap symbolic routines (e.g., theorem checkers, graph search) and attach them to prompts via `functions_list`.
4. **Parse deterministically** – leverage the `parser` callback on prompts to convert returned text into strongly typed Python objects.

Because the agent loop keeps retrying until the parser succeeds (or the retry budget is exhausted), you can layer multiple symbolic constraints without losing the rapid iteration benefits of free-form LLM prompting.
