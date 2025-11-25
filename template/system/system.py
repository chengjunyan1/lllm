from system.proxy.proxy import Proxy
from system.agent.agent import build_agent, Report
import system.utils as U
from typing import List, Dict
import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from dataclasses import dataclass, field
import random
import traceback
from enum import Enum
from tqdm import tqdm
import concurrent.futures
import time

U.load_env()


class SystemBase:
    """
    Analytical System Base
    
     - Maker: Make the task
     - Proxy: Proxy of tools
     - Agent: Agent to solve the task
    """
    def __init__(self,config, stream, exp_name=None):
        if exp_name is None:
            exp_name = U.dt_now_str() + '_' + U.random_str(6)
        # set log dir to pass to agent
        self.config = config
        self.set_path(exp_name)
        self.agent = build_agent(config, self.ckpt_dir, stream)
        self._set_exp_name(exp_name)
        self.st = stream
        self.__exp_name_eval_backup = exp_name # for cross selection

 

    def call(self, task: str, **kwargs) -> Report:
        assert not self.agent.is_async, 'Agent is async, use async_call instead'
        report = self.agent(task, **kwargs)
        return report
    
    async def async_call(self, task: str, **kwargs) -> Report:
        assert self.agent.is_async, 'Agent is not async, use call instead'
        report = await self.agent(task, **kwargs)
        return report




class BasicSystem(SystemBase):
    pass




def build_system(config, exp_name=None, stream=None, deploy_mode=False, new_benchmark=False):
    system_type = config['system_type']
    if system_type == 'basic':
        system = BasicSystem(config, stream, exp_name, deploy_mode, new_benchmark)
    else:
        raise ValueError(f'Invalid system type: {system_type}')
    return system

