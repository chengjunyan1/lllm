# Prediction Markets Data Proxy
# https://trading-api.readme.io/reference/getting-started 
# https://docs.polymarket.com/#introduction

# https://github.com/Polymarket/polymarket-subgraph
# https://polymarketanalytics.com/ 


import os
import lllm.utils as U
import gdown
import datetime as dt
from lllm.proxies import BaseProxy, ProxyRegistrator


FILE_PATH = os.path.dirname(os.path.abspath(__file__)) # /home/junyanc/analytica/analytica/proxy/modules


def get_kalshi_candles_eod(series_ticker: str, market: dict, use_cache: bool = True): 
    ticker: str = market['ticker']
    start_date: str = market['open_time'][:10] # date format: 2024-01-01
    end_date: str = market['close_time'][:10]
    url = f"https://api.elections.kalshi.com/trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks"
    headers = {"accept": "application/json"}
    start_ts = int(dt.datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(dt.datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    period_interval = 1440 # 1 day
    params = {
        'start_ts': start_ts,
        'end_ts': end_ts,
        'period_interval': period_interval
    }
    return U.call_api(url, params, headers, use_cache)


def get_polymarket_eod(market, use_cache: bool = True): 
    clob_api_url = 'https://clob.polymarket.com'
    url = f"{clob_api_url}/prices-history"
    # market_id = market['conditionId']
    outcomes = eval(market['outcomes'])
    if 'clobTokenIds' not in market:
        raise ValueError(f"No clobTokenIds found for {market['question']}")
    token_ids = eval(market['clobTokenIds'])
    token_eods ={}
    for i in range(len(outcomes)):
        outcome = outcomes[i]
        token_id = token_ids[i]
        # start_ts = int(dts_to_dt(market['startDate']).timestamp())
        # end_ts = int(dts_to_dt(market['endDate']).timestamp())
        interval = 'max'
        fidelity = 1440 # 1 day
        params = {
            'market': token_id,
            # 'startTs': start_ts,
            # 'endTs': end_ts,
            'fidelity': fidelity,
            'interval': interval,
        }
        token_eods[outcome] = U.call_api(url, params, use_cache=use_cache)
    return token_eods


@ProxyRegistrator(
    path='pm',
    name='Prediction Market Search API',
    description=(
        "Search for prediction markets on Polymarket and Kalshi, get time series and metadata for the events and markets."
    )
)
class PMProxy(BaseProxy):
    def __init__(self, cutoff_date: str = None, cache: bool = True):
        super().__init__(cutoff_date, cache)
        raise NotImplementedError

        self.api_key = os.getenv("KALSHI_API_KEY_ID")
        self.base_url = "https://trading-api.kalshi.com/v2"

        kalshi_events_url = 'https://drive.google.com/uc?id=134nkl5At8ZdnEZCzI4cU0QN3iRdTev7N'
        kalshi_markets_url = 'https://drive.google.com/uc?id=1U9KHxCSQyYnN82ZPmO80bv-GXthr5oCs'
        polymarket_events_url = 'https://drive.google.com/uc?id=1JWLVA8Xg8IYFlPSeimoCQEgcXi_7Iy2b'
        polymarket_markets_url = 'https://drive.google.com/uc?id=1MC_id9Dl4V0VaIOq2x5sNQnXIYhLsQLm'

        kashi_events_dir = U.pjoin(FILE_PATH,'kalshi_events.json')
        kashi_markets_dir = U.pjoin(FILE_PATH,'kalshi_markets.json')
        polymarket_events_dir = U.pjoin(FILE_PATH,'polymarket_events.json')
        polymarket_markets_dir = U.pjoin(FILE_PATH,'polymarket_markets.json')

        if not U.pexists(kashi_events_dir):
            gdown.download(kalshi_events_url, kashi_events_dir, quiet=False)
        if not U.pexists(kashi_markets_dir):
            gdown.download(kalshi_markets_url, kashi_markets_dir, quiet=False)
        if not U.pexists(polymarket_events_dir):
            gdown.download(polymarket_events_url, polymarket_events_dir, quiet=False)
        if not U.pexists(polymarket_markets_dir):
            gdown.download(polymarket_markets_url, polymarket_markets_dir, quiet=False)

        self.kalshi_events = U.load_json(kashi_events_dir)
        self.kalshi_markets = U.load_json(kashi_markets_dir)
        self.polymarket_events = U.load_json(polymarket_events_dir)
        self.polymarket_markets = U.load_json(polymarket_markets_dir)


    def search_kalshi_events(self, query: str):
        raise NotImplementedError
    
    def search_polymarket_events(self, query: str):
        raise NotImplementedError

    @BaseProxy.endpoint(
        category='Prediction Market',
        endpoint='search_kalshi_events',
        name='Search Kalshi Events',
        description='Search for one or more keywords on Kalshi events',
        params={
            "query*": (str, "keyword"),
        },
        response={},
    )
    def search_events(self, params: dict):
        raise NotImplementedError

    def get_time_series(self, event: dict):
        raise NotImplementedError
    
    def get_metadata(self, event: dict):
        raise NotImplementedError
    
    