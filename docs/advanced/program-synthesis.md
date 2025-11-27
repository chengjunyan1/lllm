# Program Synthesis With LLLM

LLLM ships with a Jupyter-based sandbox and a (work-in-progress) Computer Use Agent so you can iteratively synthesize and execute code under tight supervision.

- Use `lllm.sandbox.JupyterSession` to spin up a reproducible notebook workspace. The inserted init cell exposes `CALL_API`, giving your prompts access to the same proxies/tools configured in `lllm.toml`.
- Capture tool output via prompt interrupt handlers so the agent can reason about code execution results before deciding whether to continue editing.
- For browser-driven or GUI-heavy workflows, the `lllm.tools.cua` module demonstrates how to normalize keyboard/mouse actions and surface screenshots back to the agent loop.

Combine these components with structured prompts (see the neuro-symbolic guide) to build agents that can write, run, and validate code with minimal boilerplate.
