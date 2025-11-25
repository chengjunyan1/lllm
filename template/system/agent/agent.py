from lllm.llm import Agent, LLMCaller, LLMResponder, Dialog, Prompts, find_model_card
from lllm.const import APITypes
from lllm.llm import PROMPT_REGISTRY as PR
from lllm.log import build_log_base
from typing import Dict, Any, List
from enum import Enum
import datetime as dt
from lllm.utils import PrintSystem, StreamWrapper
import inspect
import numpy as np
import asyncio
import os
import shutil


class AgentType(Enum): # AE: Agentical Engine
    RANDOM = 'random'
    VANILLA = 'vanilla'


class AgentBase:
    agent_type: AgentType = None
    agent_group: List[str] = None # it maps to the agent_configs in config for better reuse 
    is_async: bool = False

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream = None): # use a name extension to distinguish different runs
        if stream is None:
            stream = PrintSystem()
        self.config = config
        assert self.agent_group is not None, f"Agent group is not set for {self.agent_type}"
        _agent_configs = config['agent_configs']
        self.agent_configs = {}
        for agent_name in self.agent_group:
            assert agent_name in _agent_configs, f"Agent {agent_name} not found in agent configs"
            self.agent_configs[agent_name] = _agent_configs[agent_name]
        self._stream = stream
        self._stream_backup = stream
        self.st = None # to be initialized when calling __call__
        self.ckpt_dir = ckpt_dir
        self._log_base = build_log_base(config)
        self.agents = {}
        self.llm_caller = LLMCaller(self.config)
        self.llm_responder = LLMResponder(self.config)
        for agent_name, model_config in self.agent_configs.items():
            model_config = model_config.copy()
            model_name = model_config.pop('model_name')
            self.model = find_model_card(model_name)
            system_prompt_path = model_config.pop('system_prompt_path')
            _api_type = APITypes(model_config.pop('api_type', 'completion'))
            if _api_type == APITypes.COMPLETION:
                _caller = self.llm_caller
            elif _api_type == APITypes.RESPONSE:
                _caller = self.llm_responder
            else:
                raise ValueError(f"Unsupported API type: {_api_type}")
            self.agents[agent_name] = Agent(
                name=agent_name,
                system_prompt=PR[system_prompt_path],  # TODO: directly from prompt
                model=model_name,
                llm_caller=_caller,
                model_args=model_config,
                log_base=self._log_base,
                max_exception_retry=self.config.get('max_exception_retry', 3),
                max_interrupt_times=self.config.get('max_interrupt_times', 5),
                max_llm_recall=self.config.get('max_llm_recall', 0),
            )
        assert self.agent_type is not None, "Agent type is not set"

        self.__additional_args = {}
        # check if any args in call besides query and **kwargs
        for arg in inspect.signature(self.call).parameters:
            if arg != 'query' and arg != '**kwargs':
                # get the default value, None if not provided
                self.__additional_args[arg] = inspect.signature(self.call).parameters[arg].default

    def set_st(self, session_name: str):
        self.st = StreamWrapper(self._stream, self._log_base, session_name)

    def restore_st(self):
    #     self.st = None
        pass

    def silent(self):
        self._stream = PrintSystem(silent=True)

    def restore(self):
        self._stream = self._stream_backup  

    def call(self, task: str, **kwargs):
        raise NotImplementedError("Subclass must implement this method")
    
    def __call__(self, task: str, session_name: str = None, **kwargs) -> str:
        if session_name is None:
            session_name = task.replace(' ', '+')+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        # unsafe in multi-threading, so we use a clone in multi-threading
        self.set_st(session_name)
        report = self.call(task, **kwargs)
        with self.st.expander('Prediction Overview', expanded=True):
            self.st.code(f'{report}')
        self.restore_st()
        return report
    

class AsyncAgentBase(AgentBase):
    is_async: bool = True

    async def call(self, task: str, **kwargs):
        raise NotImplementedError("Subclass must implement this method")

    async def __call__(self, task: str, session_name: str = None, **kwargs):
        if session_name is None:
            session_name = task.replace(' ', '+')+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        # unsafe in multi-threading, so we use a clone in multi-threading
        self.set_st(session_name)
        report = await self.call(task, **kwargs)
        with self.st.expander('Prediction Overview', expanded=True):
            self.st.code(f'{report}')
        self.restore_st()
        return report   

    


class Vanilla(AgentBase):
    agent_type: AgentType = AgentType.VANILLA
    agent_group: List[str] = ['vanilla']

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream, **kwargs):
        super().__init__(config, ckpt_dir, stream)
        self.agent: Agent = self.agents['vanilla']
        self.prompts = Prompts('vanilla')

    def call(self, task: str, **kwargs):
        # system prompt
        dialog: Dialog = self.agent.init_dialog({'role': self.role_prompt})
        _dialog = dialog
        self.st.write(f'{task}')

        _message = self.agent.send_message(
            _dialog,
            self.prompts('task_query'),
            {
                'task': task,
            },
        )
        self.st.write(f'{_message.content}')
        _response, _dialog, _ = self.agent.call(_dialog)
        parsed = _response.parsed
        _dialog.overview(remove_tail=True, stream=self.st)
        return parsed




AGENT_REGISTRY: Dict[AgentType, AgentBase] = {}

# add all classes in this file to AGENT_REGISTRY if it is a subclass or subsubclass or subsubsubclass... of AgentBase

# traverse all classes in this file

def traverse_agentbase_classes(cls):
    for subcls in cls.__subclasses__():
        # check if subcls is a subclass of AgentBase
        if issubclass(subcls, AgentBase):
            assert subcls.agent_type not in AGENT_REGISTRY, f"Agent {subcls.agent_type} already exists"
            AGENT_REGISTRY[subcls.agent_type] = subcls
        traverse_agentbase_classes(subcls)

traverse_agentbase_classes(AgentBase)


def build_agent(config: Dict[str, Any], ckpt_dir: str, stream, agent_type: AgentType = None) -> AgentBase:
    assert 'log_dir' in config, "log_dir is not set"
    if agent_type is None:
        agent_type = AgentType(config['agent_type'])
    elif isinstance(agent_type, str):
        agent_type = AgentType(agent_type)
    return AGENT_REGISTRY[agent_type](config, ckpt_dir, stream)





