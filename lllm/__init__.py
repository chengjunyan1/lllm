from lllm.llm import (
    Agent,
    AgentBase,
    Prompts,
    Prompt,
    build_agent,
    register_agent_class,
    register_prompt,
)
from lllm.core.models import Message, Function, FunctionCall, MCP
from lllm.core.const import Roles, Modalities, Providers, Features, APITypes
from lllm.core.discovery import auto_discover
from lllm.proxies import BaseProxy, Proxy, register_proxy, ProxyRegistrator
from lllm.sandbox.jupyter import JupyterSandbox, JupyterSession

__version__ = "0.0.0"

# Trigger auto-discovery
auto_discover()
