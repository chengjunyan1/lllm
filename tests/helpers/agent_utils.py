from lllm.core.agent import Agent
from lllm.core.models import Prompt
from lllm.core.log import NoLog


def make_agent(system_prompt: Prompt, provider, log_config: dict) -> Agent:
    log_base = NoLog("tests", log_config)
    return Agent(
        name="assistant",
        system_prompt=system_prompt,
        model="gpt-4o-mini",
        llm_provider=provider,
        log_base=log_base,
    )
