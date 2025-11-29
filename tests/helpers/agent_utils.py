from typing import Any

from lllm.core.agent import Agent
from lllm.core.models import Prompt
from lllm.core.log import NoLog


def make_agent(system_prompt: Prompt, provider, log_config: dict, **agent_kwargs: Any) -> Agent:
    log_base = NoLog("tests", log_config)
    payload = {
        "name": agent_kwargs.pop("name", "assistant"),
        "system_prompt": system_prompt,
        "model": agent_kwargs.pop("model", "gpt-4o-mini"),
        "llm_provider": provider,
        "log_base": log_base,
    }
    payload.update(agent_kwargs)
    return Agent(**payload)
