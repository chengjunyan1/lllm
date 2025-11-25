import os
import datetime as dt
import requests
import inspect
from tqdm import tqdm
import lllm.utils as U
import hashlib


def ProxyRegistrator(path, name, description):
    def decorator(cls):
        cls._proxy_path = path
        cls._proxy_name = name
        cls._proxy_description = description
        return cls
    return decorator


class BaseProxy():
    '''
    
    * means required
    $ means path parameter
    '''
    def __init__(self, cutoff_date: str = None, use_cache: bool = True):
        self.registry = {}  # holds endpoint registration organized by category
        self._postcall = {}
        self._entries = {}  # direct mapping of endpoint key -> method
        self.base_url = None  # Child class must initialize this
        self.api_key_name = None # usually 'apikey'
        self.api_key = None
        self.enums = {}
        self.index={}
        self.cutoff_date = cutoff_date
        self.use_cache = use_cache
        self.additional_docs = {}
        
        # Mainly Store three things:
        # 1. index: category -> sub_category -> endpoint_key
        # 2. registry: endpoint_key -> endpoint_info
        # 3. _entries: endpoint_key -> method
        for name, member in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            if hasattr(member, "endpoint_info"):
                info = member.endpoint_info
                cat = info['category'].lower().replace(' ', '-')
                sub_cat = info['sub_category'].lower().replace(' ', '-') if info['sub_category'] else None
                endpoint = info['endpoint'].lower().replace(' ', '-')
                if sub_cat is None:
                    ep_key = f"{cat}/{endpoint}"
                else:
                    ep_key = f"{cat}/{sub_cat}/{endpoint}"
                if info['method'] == 'POST':
                    ep_key += '-post'
                if cat not in self.index:
                    self.index[cat] = {}
                if sub_cat not in self.index[cat]:
                    self.index[cat][sub_cat] = []
                self.index[cat][sub_cat].append(ep_key)
                self.registry[ep_key] = {
                    'name': info['name'],
                    'description': info['description'],
                    'doc_string': inspect.getdoc(member),
                    'category': info['category'],
                    'sub_category': info['sub_category'],
                    'entry': getattr(self, name),
                    'params': info['params'],
                    'response': info['response'],
                    'dt_cutoff': info['dt_cutoff'],
                    'endpoint': endpoint
                }
                self._entries[ep_key] = getattr(self, name)

                if hasattr(member, "postcall_info"):
                    self._postcall[ep_key] = getattr(self, name)


    @property 
    def cutoff_date(self): # init as None
        return self._cutoff_date

    @cutoff_date.setter
    def cutoff_date(self, cutoff_date: str = None):
        if cutoff_date is None:
            self._cutoff_date = None
        else:
            self._cutoff_date = dt.datetime.strptime(cutoff_date, '%Y-%m-%d')

    def additional_doc(self, indent: str = ''):
        _prompt = ''
        for key in self.additional_docs:
            _prompt += f'{indent} - {key}:\n'
            _space = indent + '  '
            lines = [_space+i for i in self.additional_docs[key].strip().split('\n')]
            _prompt += f'{_space}---\n{'\n'.join(lines)}\n{_space}---\n'
        return _prompt

    def endpoint_directory(self, by_cat: bool = False, indent = ''):
        _api_path = self._proxy_path
        _prompt = ''
        for cat in self.index:
            if by_cat:
                _prompt += f'{indent} - {cat}\n'
            for sub_cat in self.index[cat]:
                if sub_cat:
                    if by_cat:
                        inner_indent = indent*2
                        _prompt += f'{inner_indent} - {sub_cat}\n'
                    else:
                        inner_indent = indent
                else:
                    inner_indent = indent
                inner_indent += indent
                sub_inner_indent = inner_indent + indent
                for path in self.index[cat][sub_cat]:
                    _data = self.registry[path]
                    if by_cat:
                        endpoint = _data['endpoint']
                    else:
                        endpoint = _api_path + '/' + path
                    _prompt += f'{inner_indent} - full path: {endpoint}\n'
                    if _data['name']:
                        _prompt += f'{sub_inner_indent} name: {_data["name"]}\n'
                    if not by_cat:
                        _prompt += f'{sub_inner_indent} category: {_data["category"]}\n'
                        if _data["sub_category"]:
                            _prompt += f'{sub_inner_indent} sub-category: {_data["sub_category"]}\n'
                    _prompt += f'{sub_inner_indent} description: {_data["description"]}\n'
                    _prompt+='\n'
                _prompt+='\n'
        return _prompt
    

    def __call__(self, ep_key: str, params: dict) -> dict:
        """
        Call a registered endpoint by its endpoint key.

        Args:
            endpoint (str): The endpoint key (e.g., "search-symbol").
            params (dict): Parameters to pass to the endpoint.

        Returns:
            dict: The JSON response from the API.
        """
        # INPUT PROCESSING
        if ep_key not in self._entries:
            raise ValueError(f"Endpoint '{ep_key}' is not registered.")
        # process params
        params = self._entries[ep_key](params)
        # remove decorators, i.e. * $
        # processed_params = {}
        # for key in params:
            # newkey = key.replace('*', '').replace('$', '')
        # processed_params[key] = params[key]
        # setup url, api key
        info = self._entries[ep_key].endpoint_info
        url = f"{self.base_url}/{info['endpoint']}"
        headers = {}
        if self.api_key_name is not None:
            if self.api_key_name.startswith('*'):
                headers = {self.api_key_name[1:]: self.api_key}
            else:
                params[self.api_key_name] = self.api_key
        # handle parameters
        for key in info['params']:
            _key = key.replace('$', '').replace('*', '').replace('#', '')
            _param_info = info['params'][key]
            if key.endswith('*'):
                if _key not in params:
                    raise ValueError(f"Required parameter '{_key}' is missing.")
            if _key not in params:
                continue
            # check type
            _type = _param_info[0]
            value = params[_key]
            if _type == 'date':
                _type = str # TODO: try to convert to date
            if not isinstance(value, _type):
                raise ValueError(f"Parameter '{_key}' must be of type {_type.__name__}.")
            # handle path parameters
            if key.startswith('$'):
                if _key in params:
                    value = params.pop(_key)
                    url = url.format(**{_key: value})
                    # url = url.replace(f"{{{_key}}}", value)
        # # remove redundant params
        # to_pop = []
        # all_keys = [key.replace('$', '').replace('*', '') for key in info['params']]
        # for key in processed_params:
        #     if key not in all_keys:
        #         to_pop.append(key)
        # for key in to_pop:
        #     processed_params.pop(key)
        
        # CALL THE API
        response = self._call_api(url, params, info, headers)

        # OUTPUT PROCESSING
        # remove keys
        remove_keys = info.get('remove_keys', None)
        if remove_keys is not None:
            def filter_item(item):
                if isinstance(item, dict):
                    for key in remove_keys:
                        item.pop(key, None)
                return item
            if isinstance(response, list):
                response = [filter_item(item) for item in response]
            elif isinstance(response, dict):
                response = filter_item(response)
        # filter by date cutoff
        dt_cutoff = info.get('dt_cutoff', None)
        if dt_cutoff is not None and self.cutoff_date is not None:
            response = filter_by_dt_cutoff(response, self.cutoff_date, dt_cutoff[0], dt_cutoff[1])
        # post-call processing
        if ep_key in self._postcall:
            response = self._postcall[ep_key](response)
        U.raise_error(response)
        return response

    def _call_api(self, url: str, params: dict, endpoint_info: dict, headers: dict) -> dict:
        """
        Helper method to call the API using the requests library.
        """
        method = endpoint_info['method']
        if method == 'GET':
            response = U.call_api(url, params, headers, self.use_cache)
        elif method == 'POST':
            _json, _data = {}, {}
            for key in params:
                if key.startswith('#'):
                    _json[key[1:]] = params[key]
                _json[key] = params[key]
            response = U.call_api_post(url, _json, headers, self.use_cache)
        return response
    

    @staticmethod
    def endpoint(category: str, endpoint: str, description: str, params: dict, response: list,
                 name: str = None, sub_category: str = None, remove_keys: list = None, 
                 dt_cutoff: tuple = None, method: str = 'GET'):
        """
        Decorator for registering an API endpoint.

        Args:
            category (str): The category name.
            endpoint (str): The endpoint key (e.g., "search-symbol").
            name (str): The display name of the endpoint.
            description (str): A one-sentence description of the endpoint.
            params (dict): The parameters for the endpoint.
            response (list): The expected response from the endpoint.
            sub_category (str, optional): The sub-category name.
            remove_keys (list, optional): A list of keys to remove from the returned JSON response.
            dt_cutoff (tuple, optional): A tuple of (dt_key, dt_format) for date filtering.
        """
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
    def postcall(endpoint: str):
        """
        Decorator for post-call processing of the response.

        Args:
            endpoint (str): The endpoint key.
        """
        def decorator(func):
            func.postcall_info = {
                'endpoint': endpoint,
            }
            return func
        return decorator
    
    def overview(self):
        U.cprint(f"Proxy: {self._proxy_name}", color='y')
        U.cprint(f"Description: {self._proxy_description}", color='y')
        # print(f"Total endpoints: {len(self._entries)}")
        total_endpoints = 0
        for cat in self.index:
            n_endpoints = sum(len(self.index[cat][sub_cat]) for sub_cat in self.index[cat])
            print(f"  {cat} ({n_endpoints} endpoints)")
            for sub_cat in self.index[cat]:
                print(f"    {sub_cat} ({len(self.index[cat][sub_cat])} endpoints)")
                total_endpoints += len(self.index[cat][sub_cat])
        print(f"Total endpoints: {total_endpoints}\n")

    def _check_repeated_endpoints(self):
        remap_endpoints = {} # endpoint -> [(cat, sub_cat), ...]
        for cat in self.index:
            for sub_cat in self.index[cat]:
                for endpoint in self.index[cat][sub_cat]:
                    if endpoint in self.registry:
                        if endpoint not in remap_endpoints:
                            remap_endpoints[endpoint] = []
                        remap_endpoints[endpoint].append(f"{cat} - {sub_cat}")
        for endpoint in remap_endpoints:
            if len(remap_endpoints[endpoint]) > 1:
                U.cprint(f"Endpoint {endpoint} is repeated in {', '.join(remap_endpoints[endpoint])}", color='r')
        
    
    def auto_test(self, skip_k = 0):
        U.cprint(f"Testing {self._proxy_name}...", color='y')
        self.overview()
        self._check_repeated_endpoints()
        results = {}
        tested = 0
        total = sum(len(self.index[cat][sub_cat]) for cat in self.index for sub_cat in self.index[cat])
        for cat in self.index:
            U.cprint(f"[Testing {cat}...]", color='y')
            cat_passed = 0
            for sub_cat in self.index[cat]:
                if sub_cat:
                    U.cprint(f"Testing {sub_cat}...", color='y')
                bar = tqdm(self.index[cat][sub_cat])
                passed = 0
                for endpoint in bar:
                    tested+=1
                    if tested <= skip_k:
                        continue
                    bar.set_postfix_str(f"Testing {endpoint}... {tested}/{total}")
                    info = self.registry[endpoint]
                    params = {}
                    for key in info['params']:
                        mock_input = info['params'][key][1]
                        if mock_input is None:
                            continue
                        key = key.replace('*', '').replace('$', '').replace('#', '')
                        params[key] = mock_input
                    path = f'{cat}/{sub_cat}/{endpoint}'
                    # try:
                    response = self(endpoint, params)
                    results[path] = True
                    print(str(response)[:200])
                    # except Exception as e:
                    #     results[path] = False
                    passed += results[path]
                bar.close()
                if sub_cat:
                    U.cprint(f"Testing {sub_cat}...done. Passed {passed}/{len(self.index[cat][sub_cat])} endpoints", color='w')
                cat_passed += passed
            total_endpoints = sum(len(self.index[cat][sub_cat]) for sub_cat in self.index[cat])
            U.cprint(f"[Testing {cat}...done] Passed {cat_passed}/{total_endpoints} endpoints", color='w')
            print()

        U.cprint(f"Testing {self._proxy_name} complete.\n", color='y')
        if all(results.values()):
            U.cprint("All endpoints passed the test!", color='g')
        else:
            for path in results:
                if not results[path]:
                    U.cprint(f"{path} failed the test.", color='r')
            n_passed = sum(results[path] for path in results)
            U.cprint(f"\n{n_passed}/{len(results)} endpoints passed the test.", color='y')




def filter_by_dt_cutoff(response: list|dict, cutoff_date: dt.datetime, 
                        path: str = 'date', dt_format: str = '%Y-%m-%d') -> list:
    
    def _dt_before_cutoff(dt_str):
        if dt_str == 'None':
            return False
        date_obj = dt.datetime.strptime(dt_str, dt_format)
        return date_obj < cutoff_date

    def _filter_response_list(response: list, dt_key: str = 'date'):
        filtered_response = []
        for item in response:
            if _dt_before_cutoff(str(item[dt_key])):
                filtered_response.append(item)
        return filtered_response

    if isinstance(response, list): # dict of lists
        return _filter_response_list(response, path)
    elif isinstance(response, dict): 
        assert path is not None, "Path must be provided if response is a dictionary"
        keys = path.split('/')
        _response = response
        if len(keys) > 1: # nested response, the list of dicts are nested in the dict
            for i, key in enumerate(keys[:-1]):
                _response = _response[key]
            last_key = keys[-1] # the last key is the dt_key
            if isinstance(_response[last_key], list):
                _response[last_key] = _filter_response_list(_response[last_key])
        else: # single dict response, not nested
            dt_str = str(_response[path])
            if not _dt_before_cutoff(dt_str):
                response = {
                    'error': f'No data found at time {dt_str}, please input the time before current datetime {cutoff_date}'
                }
        return response
    else:
        raise ValueError(f"Unsupported response type: {type(response)}")

