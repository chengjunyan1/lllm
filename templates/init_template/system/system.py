from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from system.agent.agent import AgentType, build_agent


class SimpleSystem:
    """Kickstarts the default agent and provides a `.call` helper."""

    def __init__(self, config: Dict[str, Any], ckpt_dir: str | None = None, stream=None):
        self.config = config
        self.ckpt_dir = ckpt_dir or config.get('ckpt_dir', './ckpt')
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        self.agent = build_agent(config, self.ckpt_dir, stream, AgentType.VANILLA)

    def call(self, task: str, **kwargs) -> Any:
        """Run the single agent against a textual task."""
        return self.agent(task, **kwargs)


def build_system(config: Dict[str, Any], ckpt_dir: str | None = None, stream=None, **_) -> SimpleSystem:
    """Factory kept for parity with more advanced systems."""
    return SimpleSystem(config, ckpt_dir=ckpt_dir, stream=stream)
