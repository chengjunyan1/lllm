# Internal Knowledge Base Proxy

import os
import analytica.utils as U
from lllm.proxies import BaseProxy, ProxyRegistrator


@ProxyRegistrator(
    path='kb',
    name='Internal Knowledge Base',
    description=(
        "Internal Knowledge Base"
    )
)
class KBProxy(BaseProxy):
    def __init__(self, cutoff_date: str = None, cache: bool = True):
        super().__init__(cutoff_date, cache)
        raise NotImplementedError

