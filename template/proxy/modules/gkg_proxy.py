# Golden Knowledge Graph Proxy
# https://docs.golden.com/reference/getting-started 


import os
import datetime as dt
import analytica.utils as U
from lllm.proxies import BaseProxy, ProxyRegistrator
import requests



@ProxyRegistrator(
    path='gkg',
    name='Golden Knowledge Graph API',
    description=(
        ""
    )
)
class GKGProxy(BaseProxy):
    """
    Golden Knowledge Graph API

    This API provides access to Golden Knowledge Graph API.
    """
    def __init__(self, cutoff_date: str = None, cache: bool = True):
        super().__init__(cutoff_date, cache)
        self.api_key_name = "api_key"
        self.api_key = os.getenv("GOLDEN_API_KEY")
        self.base_url = "https://golden.com/api/v2/public"
        self.enums = {}

        raise NotImplementedError


    # TODO: maybe need to manually cache the list for search, no search api provided


    