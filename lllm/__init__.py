"""Top-level package for Low-Level Language Models (LLLM)."""

from .llm import (
    AGENT_REGISTRY,
    Agent,
    AgentBase,
    AsyncAgentBase,
    Dialog,
    LLMCaller,
    LLMResponder,
    PROMPT_REGISTRY,
    build_registered_agent,
    print_prompts,
    register_prompt,
)
from .log import ReplayableLogBase, build_log_base
from .models import Function, FunctionCall, MCP, Message, Prompt
from .proxies import (
    BaseProxy,
    PROXY_REGISTRY,
    ProxyRegistrator,
    get_proxy,
    list_proxies,
    register_proxy,
)
from .discovery import auto_discover

__all__ = [
    "Agent",
    "AgentBase",
    "AsyncAgentBase",
    "Dialog",
    "LLMCaller",
    "LLMResponder",
    "PROMPT_REGISTRY",
    "AGENT_REGISTRY",
    "build_registered_agent",
    "print_prompts",
    "register_prompt",
    "ReplayableLogBase",
    "build_log_base",
    "Function",
    "FunctionCall",
    "MCP",
    "Message",
    "Prompt",
    "BaseProxy",
    "ProxyRegistrator",
    "PROXY_REGISTRY",
    "register_proxy",
    "get_proxy",
    "list_proxies",
    "auto_discover",
]

__version__ = "0.1.0"

auto_discover()
