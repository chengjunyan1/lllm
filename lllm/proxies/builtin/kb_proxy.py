# Internal Knowledge Base Proxy

import os
import lllm.utils as U
from lllm.proxies.base import BaseProxy, ProxyRegistrator


@ProxyRegistrator(
    path='kb',
    name='Internal Knowledge Base',
    description=(
        "Internal Knowledge Base"
    )
)
class KBProxy(BaseProxy):
    def __init__(self, cutoff_date: str = None, cache: bool = True, **kwargs):
        super().__init__(cutoff_date=cutoff_date, use_cache=cache, **kwargs)
        raise NotImplementedError
