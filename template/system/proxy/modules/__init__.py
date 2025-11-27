# Interface for the Information Engine 

import os
import importlib
import inspect
from lllm.proxies import BaseProxy, PROXY_REGISTRY, register_proxy

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file not in ['__init__.py', 'base_proxy.py']:
        module_name = file[:-3]
        module = importlib.import_module(f'.{module_name}', package=__name__)
        for name, member in inspect.getmembers(module, inspect.isclass):
            if name == 'BaseProxy':
                continue
            if issubclass(member, BaseProxy):
                proxy_name = getattr(member, '_proxy_path', module_name.split('_')[0])
                register_proxy(proxy_name, member, overwrite=True)

print(f'{len(PROXY_REGISTRY)} proxies registered: {list(PROXY_REGISTRY.keys())}')
