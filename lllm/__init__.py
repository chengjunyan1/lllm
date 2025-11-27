from lllm.core.agent import Agent, AgentBase, build_agent, register_agent_class
from lllm.core.models import Prompt, Message, Function, FunctionCall, MCP, register_prompt
from lllm.core.const import Roles, Modalities, Providers, Features, APITypes
from lllm.core.discovery import auto_discover
from lllm.proxies.base import BaseProxy, Proxy, register_proxy
from lllm.sandbox.jupyter import JupyterSandbox, JupyterSession

__version__ = "0.1.0"

# Trigger auto-discovery
auto_discover()
