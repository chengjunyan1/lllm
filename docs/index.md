# Low-Level Language Models (LLLM)

LLLM is a batteries-included toolkit for building deterministic LLM agents without hiding the low-level details. The platform grew out of practical agent work, so every subsystem is designed to be:

- **Simple** – few abstractions, everything can be inspected in Python.
- **Flexible** – prompts, proxies, and sandboxes are decoupled, so you can swap parts without rewriting the agent.
- **Deterministic** – the "agent call" state machine shepherds every dialog to the next state or raises an explicit error.
- **Observable** – logs, prompts, and proxy calls are stored in replayable collections for debugging and benchmarking.
- **Product-ready** – configuration-driven discovery, API proxying, and sandboxed execution let one repository power research notebooks and deployed agents.

## Design Philosophy

1. **Agent-first, not model-first.** LLM calls are wrapped inside `Agent.call`, which guarantees progress from the initial prompt to a concrete output state rather than returning the raw model string.
2. **Prompts are code.** Prompts ship with handlers, structured output validators, parsers, functions, and MCP servers so that complex conversations can be reasoned about and replayed.
3. **Infrastructure is reusable.** Proxies, sandboxes, and CLI scaffolds can be auto-discovered from simple `lllm.toml` declarations, so teams can compose new systems from existing parts.
4. **Errors are expected.** Exception and interrupt handlers are first-class citizens. They keep dialogs in a known shape and surface actionable failures instead of silent degradations.
5. **Transparency over magic.** Every helper lives in the `lllm/` package and is documented below; there are no hidden services.

## Using this Documentation

This site is structured for a GitHub Pages or GitBook workflow using [MkDocs](https://www.mkdocs.org/).

```bash
pip install mkdocs
mkdocs serve           # live-reload docs locally
mkdocs build           # generate the static site for GitHub Pages
```

When publishing with GitHub Pages, point the Pages build to the `mkdocs build` output (`site/`).

## Doc Map

- [Architecture](architecture/overview.md) – repository layout and runtime data flow.
- [Core Concepts](core/agent-call.md) – deep dives into agent calls, prompts, proxies, sandboxing, logging, and configuration.
- [Guides](guides/building-agents.md) – practical walkthroughs for creating systems and using the CLI scaffold.
- [Reference](reference/modules.md) – module-by-module summaries for quick lookup.

Start with the design philosophy above, then continue with the Architecture section to see how the pieces fit together.
