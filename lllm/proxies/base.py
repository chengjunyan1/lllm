import inspect
import functools as ft
import datetime as dt
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
import lllm.utils as U

class BaseProxy:
    """Base class for describing an API surface that agents can call as tools."""

    def __init__(
        self,
        *args,
        activate_proxies: Optional[List[str]] = None,
        cutoff_date: Optional[dt.datetime] = None,
        deploy_mode: bool = False,
        use_cache: bool = True,
    ):
        """
        Support both legacy signatures (cutoff_date, use_cache) and the newer keyword-driven one.
        """
        self.activate_proxies = activate_proxies[:] if activate_proxies else []
        self.cutoff_date = cutoff_date
        self.deploy_mode = deploy_mode
        self.use_cache = use_cache

        legacy_args = list(args)
        if legacy_args:
            first = legacy_args.pop(0)
            if isinstance(first, list):  # legacy Proxy runtime order
                self.activate_proxies = first
                if legacy_args:
                    self.cutoff_date = legacy_args.pop(0)
                if legacy_args:
                    self.deploy_mode = legacy_args.pop(0)
            else:
                self.cutoff_date = first
                if legacy_args:
                    self.use_cache = legacy_args.pop(0)

        if isinstance(self.cutoff_date, str):
            try:
                self.cutoff_date = dt.datetime.fromisoformat(self.cutoff_date)
            except ValueError:
                self.cutoff_date = None

    @staticmethod
    def endpoint(category: str, endpoint: str, description: str, params: dict, response: list,
                 name: str = None, sub_category: str = None, remove_keys: list = None,
                 dt_cutoff: tuple = None, method: str = 'GET'):
        """Decorator that records metadata about an API endpoint."""
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
    """
    Runtime registry that instantiates every discovered proxy and forwards calls.

    Agents rarely instantiate this directly; instead the sandbox or higher-level tooling
    wires it up so prompts can enumerate available endpoints for tool selection.
    """

    def __init__(self, activate_proxies: Optional[List[str]] = None, cutoff_date: dt.datetime = None, deploy_mode: bool = False):
        from lllm.core.discovery import auto_discover
        auto_discover()
        self.activate_proxies = activate_proxies or []
        self.cutoff_date = cutoff_date
        self.deploy_mode = deploy_mode
        self.proxies: Dict[str, BaseProxy] = {}
        self._load_registered_proxies()

    def _load_registered_proxies(self):
        for name, proxy_cls in PROXY_REGISTRY.items():
            if self.activate_proxies and name not in self.activate_proxies:
                continue
            try:
                instance = proxy_cls(
                    cutoff_date=self.cutoff_date,
                    activate_proxies=self.activate_proxies,
                    deploy_mode=self.deploy_mode,
                )
            except TypeError:
                # Fallback to legacy positional order if subclass has not been updated yet
                instance = proxy_cls(self.activate_proxies, self.cutoff_date, self.deploy_mode)
            self.proxies[name] = instance

    def register(self, name: str, proxy_cls: Any):
        """Register (or override) a proxy implementation at runtime."""
        if name in self.proxies:
            U.cprint(f'Proxy {name} already instantiated, overwriting instance', 'y')
        try:
            instance = proxy_cls(
                cutoff_date=self.cutoff_date,
                activate_proxies=self.activate_proxies,
                deploy_mode=self.deploy_mode,
            )
        except TypeError:
            instance = proxy_cls(self.activate_proxies, self.cutoff_date, self.deploy_mode)
        self.proxies[name] = instance

    def available(self) -> List[str]:
        """Return the sorted list of proxy identifiers currently loaded."""
        return sorted(self.proxies.keys())

    def _resolve(self, endpoint: str) -> tuple[str, str]:
        if '.' in endpoint:
            parts = endpoint.split('.', 1)
            return parts[0], parts[1]
        path_parts = endpoint.split('/')
        if len(path_parts) < 2:
            raise ValueError(f"Invalid endpoint '{endpoint}'. Use '<proxy>.<method>' or '<proxy>/<method>'.")
        return '/'.join(path_parts[:-1]), path_parts[-1]

    def __call__(self, endpoint: str, *args, **kwargs):
        """Dispatch ``proxy_path.endpoint_name`` or ``proxy_path/endpoint`` to the proxy."""
        proxy_name, func_name = self._resolve(endpoint)
        if proxy_name not in self.proxies:
            raise KeyError(f"Proxy '{proxy_name}' not registered. Available: {list(self.proxies.keys())}")
        proxy = self.proxies[proxy_name]
        if not hasattr(proxy, func_name):
            raise AttributeError(f"Proxy '{proxy_name}' has no endpoint '{func_name}'")
        handler = getattr(proxy, func_name)
        return handler(*args, **kwargs)

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
