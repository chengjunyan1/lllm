"""
Sample prompt module used by `lllm.toml.example`.

Auto-discovery imports every module under `examples/autodiscovery/prompts`
and registers any Prompt objects defined at module scope.
"""

from lllm.core.models import Prompt

hello_world = Prompt(
    path="examples/hello_world",
    prompt="You are a friendly assistant. Respond cheerfully to: {task}",
)
