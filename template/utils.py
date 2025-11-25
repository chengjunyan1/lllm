import os
import json
import functools as ft
import time
import hashlib
import datetime as dt
import requests
import yaml
import pandas as pd
import shutil
from PIL import Image
from pathlib import Path
from itertools import islice
import base64
from lllm.utils import cprint,call_api,create_cache_key,save_cache_by_key,load_cache_by_key
import re
from io import BytesIO
import imgkit
from typing import Dict, List
import io
import plotly.io as pio
import random
import uuid
import string

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_env(env_file: str = '.env'):
    _file = pjoin(PROJECT_ROOT, env_file)
    if not pexists(_file):
        raise FileNotFoundError(f'Env file not found: {_file}')
    with open(_file, 'r') as f:
        for line in f:
            if line.strip().startswith('#'): continue
            if line.strip() == '': continue
            key, value = line.strip().split('=')
            # print(f'Loaded env: {key}')
            os.environ[key] = value


pjoin=os.path.join
psplit=os.path.split
pexists=os.path.exists
mkdirs=ft.partial(os.makedirs, exist_ok=True)
rmtree=ft.partial(shutil.rmtree, ignore_errors=True)


try:
    load_env()
except:
    pass

tmp_dir = os.getenv('TMP_DIR')
state_dir = pjoin(tmp_dir, '.state')
mkdirs(state_dir)


DEFAULT_CONFIG_PATH = pjoin(PROJECT_ROOT, 'configs', 'default.yaml')

assert pexists(DEFAULT_CONFIG_PATH), f'Default config file not found: {DEFAULT_CONFIG_PATH}'

def dt_now_str(format: str = '%Y%m%d_%H%M%S'):
    return dt.datetime.now().strftime(format)

def load_json(file,default={}):
    if not pexists(file):
        if default is None:
            raise FileNotFoundError(f'File {file} not found')
        return default
    with open(file, encoding='utf-8') as f:
        return json.load(f)
    
def save_json(file,data,indent=4): 
    mkdirs(os.path.dirname(file))
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

def load_yaml(file,default={}):
    if not pexists(file):
        if default is None:
            raise FileNotFoundError(f'File {file} not found')
        return default
    with open(file, encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_csv(file,default=None,skiprows=0):
    if not pexists(file):
        return default
    with open(file, encoding='utf-8') as f:
        return pd.read_csv(f,skiprows=skiprows)




def dts_to_dt(dts):
    formats =[
        '%Y-%m-%dT%H:%M:%SZ',   
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d',
    ]
    for format in formats:
        try:
            return dt.datetime.strptime(dts,format)
        except:
            pass
    raise ValueError(f'Invalid date string: {dts}')

def save_state(state_name: str, state: dict):
    state_file = pjoin(state_dir, f"{state_name}.json")
    save_json(state_file, state)

def load_state(state_name: str):
    state_file = pjoin(state_dir, f"{state_name}.json")
    if pexists(state_file):
        return load_json(state_file)
    return {}

def get_days(time_horizon):
    if time_horizon is None:
        return None
    elif isinstance(time_horizon, int):
        return time_horizon
    elif isinstance(time_horizon, str):
        time_horizon = time_horizon.lower()
        if time_horizon == '1d':
            return 1
        elif time_horizon == '1w':
            return 7
        elif time_horizon == '1m':
            return 30
        elif time_horizon == '3m':
            return 90
        elif time_horizon == '6m':
            return 180
        elif time_horizon == '1y':
            return 365
        else:
            raise ValueError(f'Invalid time horizon: {time_horizon}')
    else:
        raise ValueError(f'Invalid time horizon type: {type(time_horizon)}')
    
def to_str_span(span_days: int):
    str_span = {
        365: 'one year',
        180: 'half year',
        90: 'one quarter',
        30: 'one month',
        14: 'two weeks',
        7: 'one week',
        1: 'one day',
    }
    if span_days in str_span:
        return str_span[span_days]
    else:
        return f'{span_days} days'

def list2freq(lst: list, round_base: float = None, sort: bool = False):
    freq = {}
    for item in lst:
        if round_base is not None: # for float
            item = round(item, round_base)
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    if sort:
        return dict(sorted(freq.items(), key=lambda x: x[0], reverse=False))
    return freq


def iteratively_set_default(config: dict, default_config: dict):
    for key, value in default_config.items():
        if isinstance(value, dict):
            if key not in config:
                config[key] = {}
            iteratively_set_default(config[key], value)
        else:
            if key not in config:
                config[key] = value
    return config

def load_config(config_path: str):
    config = load_yaml(config_path, None)
    default_config = load_yaml(DEFAULT_CONFIG_PATH)
    config = iteratively_set_default(config, default_config)
    config['project_root'] = PROJECT_ROOT
    return config


def remove_ansi_codes(text: str) -> str:
  ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
  return ansi_escape.sub('', text)


def base64_to_image(base64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(base64_str)))



def process_html_output(html_content: str) -> Dict[str, str]:
    """
    Processes a mime_bundle:
    1. If 'text/html' contains a table, converts it to a plain text string.
    2. Else if 'text/html' exists, converts the HTML to a Base64 PNG image.
    3. Else if 'image/svg+xml' exists, converts SVG to a Base64 PNG image.
    Returns a dictionary with the processed output ('text/plain' or 'image/png')
    or an error/status message.
    """
    output_result = {}
    is_table_converted = False

    try:
        dfs: List[pd.DataFrame] = pd.read_html(io.StringIO(html_content))
        
        if dfs:  # If pandas found any tables
            plain_text_tables = []
            for i, df in enumerate(dfs):
                if not df.empty: # Ensure DataFrame is not empty
                    table_header = f"--- Table {i+1} ---\n"
                    plain_text_tables.append(table_header + df.to_string())
            
            if plain_text_tables: # If any non-empty tables were actually stringified
                output_result['table'] = "\n\n".join(plain_text_tables)
                is_table_converted = True
    except ValueError:
        pass # Fall through to image conversion
    except Exception as e:
        # print(f"Error during pandas table parsing: {e}. Attempting HTML to image conversion.")
        pass # Fall through to image conversion

    # If a table was converted, we are done with this mime_bundle
    if is_table_converted:
        return output_result

    # If no table was successfully converted from HTML, try to convert the HTML to an image
    try:
        options = {
            'format': 'png',
            'quiet': '',
            'load-error-handling': 'ignore', # Try to ignore resource load errors
            'javascript-delay': 2000 # Wait 2s for JS to execute (adjust as needed)
        }
        png_bytes = imgkit.from_string(html_content, False, options=options)
        base64_png = base64.b64encode(png_bytes).decode('ascii')
        output_result['base64'] = base64_png
        return output_result
    except Exception as e: # Other imgkit errors
        output_result['error_html_to_image'] = f"Error converting HTML to image with imgkit: {e}"
        # print(output_result['error_html_to_image'])

    # If no conversion was successful, return status or any accumulated errors
    if not output_result:
        output_result['status'] = "No applicable content (HTML table, general HTML, or SVG) found or converted."
    elif 'image/png' not in output_result and 'text/plain' not in output_result:
        # Consolidate errors if multiple attempts failed
        final_errors = []
        if output_result.get('error_html_to_image'): final_errors.append(output_result.get('error_html_to_image'))
        if output_result.get('error_svg_to_image'): final_errors.append(output_result.get('error_svg_to_image'))
        if final_errors:
             output_result = {'error': "; ".join(final_errors)}
        else: # Should not happen if this branch is reached
             output_result['status'] = "No content processed."

    return output_result


def plotly_json_to_base64_png(plotly_spec_dict: dict) -> str:
    png_bytes = pio.to_image(fig=plotly_spec_dict, format='png')
    base64_bytes = base64.b64encode(png_bytes)
    base64_string = base64_bytes.decode('ascii')
    return base64_string


def random_str(length: int = 10) -> str:
    return ''.join(random.sample(uuid.uuid4().hex, length))

def hash_str(seed: str, length: int = 10) -> str:
    return hashlib.sha256(seed.encode('utf-8')).hexdigest()[:length]

def idx2var(idx: int):
    k=idx//26
    r=idx%26
    tail = string.ascii_uppercase[r]
    return tail if k==0 else idx2var(k-1)+tail

def replace_special_chars(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '_', text)




#############################
