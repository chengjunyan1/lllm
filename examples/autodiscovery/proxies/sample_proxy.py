"""
Sample proxy module referenced by `lllm.toml.example`.

Importing this file registers a proxy under the path `examples/sample`.
"""

from lllm.proxies import BaseProxy, ProxyRegistrator


@ProxyRegistrator(
    path="examples/sample",
    name="Sample Local Proxy",
    description="Demonstrates how auto-discovered proxies register themselves.",
)
class SampleProxy(BaseProxy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ping(self, message: str = "pong") -> dict:
        """Echo endpoint that can be linked to a tool for testing."""
        return {"echo": message}
