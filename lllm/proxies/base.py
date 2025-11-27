import inspect
import functools as ft
import datetime as dt
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
import lllm.utils as U

class BaseProxy:
    def __init__(self, activate_proxies: List[str] = [], cutoff_date: dt.datetime = None, deploy_mode: bool = False):
        self.activate_proxies = activate_proxies
        self.cutoff_date = cutoff_date
        self.deploy_mode = deploy_mode

    @staticmethod
    def endpoint(category: str, endpoint: str, description: str, params: dict, response: list,
                 name: str = None, sub_category: str = None, remove_keys: list = None,
                 dt_cutoff: tuple = None, method: str = 'GET'):
        def decorator(func):
            func.endpoint_info = {
                'category': category,
                'endpoint': endpoint,
                'name': name,
                'description': description,
                'sub_category': sub_category,
                'remove_keys': remove_keys,
                'params': params,
                'response': response,
                'dt_cutoff': dt_cutoff,
                'method': method
            }
            return func
        return decorator

    @staticmethod
    def postcall(func):
        func.is_postcall = True
        return func

    def auto_test(self):
        # This method can be used to automatically test all endpoints
        pass

class Proxy:
    def __init__(self, activate_proxies: List[str] = [], cutoff_date: dt.datetime = None, deploy_mode: bool = False):
        self.proxies = {}
        self.activate_proxies = activate_proxies
        self.cutoff_date = cutoff_date
        self.deploy_mode = deploy_mode
        # Proxy discovery logic should be injected or handled by a registry
        # For now, we assume proxies are registered elsewhere or passed in.
        # But the original code likely did auto-discovery.
        # We will rely on `lllm.core.discovery` to populate proxies, or manual registration.
        pass

    def register(self, name: str, proxy_cls: Any):
        self.proxies[name] = proxy_cls(self.activate_proxies, self.cutoff_date, self.deploy_mode)

    def __call__(self, endpoint: str, **kwargs):
        # Dispatch to the appropriate proxy
        # Format: "category.endpoint" or just "endpoint" if unique?
        # Original code likely had logic to route calls.
        # I'll implement a simple routing mechanism.
        parts = endpoint.split('.')
        if len(parts) == 2:
            category, func_name = parts
            if category in self.proxies:
                proxy = self.proxies[category]
                if hasattr(proxy, func_name):
                    return getattr(proxy, func_name)(**kwargs)
        raise ValueError(f"Endpoint {endpoint} not found")

PROXY_REGISTRY: Dict[str, Any] = {}

def register_proxy(name: str, proxy_cls: Any, overwrite: bool = False):
    if name in PROXY_REGISTRY and not overwrite:
        raise ValueError(f"Proxy {name} already registered")
    PROXY_REGISTRY[name] = proxy_cls


def ProxyRegistrator(path: str, name: str, description: str):
    def decorator(cls):
        cls._proxy_path = path
        cls._proxy_name = name
        cls._proxy_description = description
        register_proxy(path, cls, overwrite=True)
        return cls
    return decorator


