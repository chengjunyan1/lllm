"""Example proxy demonstrating the structure for registering tools."""

from lllm.proxies import BaseProxy, ProxyRegistrator


@ProxyRegistrator(
    path="{{project_name}}/sample",
    name="Sample Proxy",
    description="Placeholder proxy. Replace with real tool integrations.",
)
class SampleProxy(BaseProxy):
    def __init__(self, cutoff_date: str | None = None, use_cache: bool = True, **kwargs):
        super().__init__(cutoff_date=cutoff_date, use_cache=use_cache, **kwargs)
        self.base_url = "https://api.example.com"
        self.api_key_name = "api_key"

    @BaseProxy.endpoint(
        category="Demo",
        endpoint="echo",
        name="Echo Endpoint",
        description="Returns whatever payload you send. Replace with real API calls.",
        params={"message*": (str, "hello world")},
        response={"message": "hello world"},
    )
    def echo(self, params: dict) -> dict:
        # This just returns the params to demonstrate shape. Replace with API logic.
        return params
