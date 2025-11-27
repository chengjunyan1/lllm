import random
import time
import json
import uuid
import inspect
import datetime as dt
import numpy as np
from typing import List, Dict, Any, Tuple, Type, Optional
from dataclasses import dataclass, field
from enum import Enum

from lllm.core.models import Message, Prompt, FunctionCall, AgentException, Function
from lllm.core.const import Roles, APITypes, find_model_card
from lllm.core.dialog import Dialog
from lllm.core.log import ReplayableLogBase, build_log_base
from lllm.providers.base import BaseProvider
from lllm.providers.openai import OpenAIProvider
import lllm.utils as U

AGENT_REGISTRY: Dict[str, Type['AgentBase']] = {}

def _normalize_agent_type(agent_type):
    if isinstance(agent_type, Enum) or (isinstance(agent_type, type) and issubclass(agent_type, Enum)):
        return agent_type.value
    elif isinstance(agent_type, str):
        return agent_type
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

def register_agent_class(agent_cls: Type['AgentBase']) -> Type['AgentBase']:
    agent_type = _normalize_agent_type(getattr(agent_cls, 'agent_type', None))
    assert agent_type not in (None, ''), f"Agent class {agent_cls.__name__} must define `agent_type`"
    if agent_type in AGENT_REGISTRY and AGENT_REGISTRY[agent_type] is not agent_cls:
        raise ValueError(f"Agent type '{agent_type}' already registered with {AGENT_REGISTRY[agent_type].__name__}")
    AGENT_REGISTRY[agent_type] = agent_cls
    return agent_cls

def get_agent_class(agent_type: str) -> Type['AgentBase']:
    if agent_type not in AGENT_REGISTRY:
        raise KeyError(f"Agent type '{agent_type}' not found. Registered: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[agent_type]

def build_agent(config: Dict[str, Any], ckpt_dir: str, stream, agent_type: str = None, **kwargs) -> 'AgentBase':
    if agent_type is None:
        agent_type = config.get('agent_type')
    agent_type = _normalize_agent_type(agent_type)
    agent_cls = get_agent_class(agent_type)
    return agent_cls(config, ckpt_dir, stream, **kwargs)

class ClassificationError(Exception):
    def __init__(self, message: str, top_probs: Dict[str, float]):
        self.message = message
        self.top_probs = top_probs

@dataclass
class Agent:
    name: str # the role of the agent, or a name of the agent
    system_prompt: Prompt
    model: str # the latest snapshot of a model
    llm_provider: BaseProvider
    log_base: ReplayableLogBase   
    model_args: Dict[str, Any] = field(default_factory=dict) # additional args, like temperature, seed, etc.
    max_exception_retry: int = 3
    max_interrupt_times: int = 5
    max_llm_recall: int = 0

    def __post_init__(self):
        self.model_card = find_model_card(self.model)
        self.model_card.check_args(self.model_args)
        self.model = self.model_card.latest_snapshot.name

    def reload_system(self, system_prompt: Prompt):
        self.system_prompt = system_prompt
        
    # initialize the dialog
    def init_dialog(self, prompt_args: Dict[str, Any] = {}, session_name: str = None) -> Dialog:
        if session_name is None:
            session_name = dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(uuid.uuid4())[:6]
        system_message = Message(
            role=Roles.SYSTEM,
            content=self.system_prompt(**prompt_args),
            creator='system',
        )
        return Dialog([system_message], self.log_base, session_name)

    # send a message to the dialog manually
    def send_message(self, dialog: Dialog, prompt: Prompt, prompt_args: Dict[str, Any] = {}, 
                     creator: str = 'internal', extra: Dict[str, Any] = {}, role: Roles = Roles.USER):
        return dialog.send_message(prompt, prompt_args, creator=creator, extra=extra, role=role)

    # it performs the "Agent Call"
    def call(self, 
        dialog: Dialog, # it assumes the prompt is already loaded into the dialog as the top prompt by send_message
        extra: Dict[str, Any] = {}, # for tracking additional information, such as frontend replay info
        args: Dict[str, Any] = {}, # for tracking additional information, such as frontend replay info
        parser_args: Dict[str, Any] = {},   
    ) -> Tuple[Message, Dialog, List[FunctionCall]]:
        # Prompt: a function maps prompt args and dialog into the expected output 
        current_prompt = dialog.top_prompt
        interrupts = []
        for i in range(10000 if self.max_interrupt_times == 0 else self.max_interrupt_times+1): # +1 for the final response
            llm_recall = self.max_llm_recall 
            exception_retry = self.max_exception_retry 
            working_dialog = dialog.fork() # make a copy of the dialog, truncate all excception handling dialogs
            while True: # ensure the response is no exception
                _attempts = []
                try:
                    _model_args = self.model_args.copy()
                    _model_args.update(args)
                    
                    response = self.llm_provider.call(working_dialog, current_prompt, self.model, _model_args, 
                                                    parser_args=parser_args, responder=self.name, extra=extra)
                    working_dialog.append(response) 
                    if response._errors != []:
                        _attempts.append(response)
                        raise AgentException(response.error_message)
                    else: 
                        break
                except AgentException as e: # handle the exception from the agent
                    if exception_retry > 0:
                        exception_retry -= 1
                        U.cprint(f'{self.name} is handling an exception {e}, retry times: {self.max_exception_retry-exception_retry}/{self.max_exception_retry}','r')
                        working_dialog.send_message(current_prompt.exception_handler, {'error_message': str(e)}, creator='exception')
                        current_prompt = dialog.top_prompt
                        continue
                    else:
                        raise e
                except Exception as e: # handle the exception from the LLM
                    # Simplified error handling for now
                    wait_time = random.random()*15+1
                    if U.is_openai_rate_limit_error(e): # for safe
                        time.sleep(wait_time)
                    else:
                        if llm_recall > 0:
                            llm_recall -= 1
                            time.sleep(1) # wait for a while before retrying
                            continue
                        else:
                            raise e

            response._attempts = _attempts
            dialog.append(response) # update the dialog state
            # now handle the interruption
            if response.is_function_call:
                _func_names = [func_call.name for func_call in response.function_calls]
                U.cprint(f'{self.name} is calling function {_func_names}, interrupt times: {i+1}/{self.max_interrupt_times}','y')
                # handle the function call
                for function_call in response.function_calls:
                    if function_call.is_repeated(interrupts):
                        result_str = f'The function {function_call.name} with identical arguments {function_call.arguments} has been called earlier, please check the previous results and do not call it again. If you do not need to call more functions, just stop calling and provide the final response.'
                    else:
                        print(f'{self.name} is calling function {function_call.name} with arguments {function_call.arguments}')
                        function = current_prompt.functions[function_call.name]
                        function_call = function(function_call)
                        result_str = function_call.result_str
                        interrupts.append(function_call)
                    _role = Roles.TOOL if response.api_type == APITypes.COMPLETION else Roles.USER
                    dialog.send_message(current_prompt.interrupt_handler, {'call_results': result_str}, 
                                        role=_role, creator='function', extra={'tool_call_id': function_call.id})
                if i == self.max_interrupt_times-1:
                    dialog.send_message(current_prompt.interrupt_handler_final, role=Roles.USER, creator='function')
                current_prompt = dialog.top_prompt
            else: # the response is not a function call, it is the final response
                if i > 0:   
                    U.cprint(f'{self.name} stopped calling functions, total interrupt times: {i}/{self.max_interrupt_times}','y')
                return response, dialog, interrupts
        raise ValueError('Failed to call the agent')

    # a special agent call for classification
    def _classify(self, dialog: Dialog, classes: List[str], classifier_args: Dict[str, Any]):
        _, dialog, _ = self.call(dialog, args=classifier_args)
        response = dialog.tail.raw_response
        choice = response.choices[0]
        logprobs = choice.logprobs.content
        if not len(logprobs) == 1:
            raise ClassificationError(f'Failed to classify the proposition, not only one token ({len(logprobs)})', {})
        chosen_token_data = choice.logprobs.content[0]
        top_probs = {}
        for top_logprob_entry in chosen_token_data.top_logprobs:
            token = top_logprob_entry.token
            prob = np.exp(top_logprob_entry.logprob)
            top_probs[token] = prob
        U.cprint(top_probs, 'y')
        errors = []
        for token in classes:
            if token not in top_probs:
                errors.append(f"Token {token} not found in the top logprobs")
        if errors != []:
            raise ClassificationError(f'Failed to classify the proposition:\n{"\n".join(errors)}', {})
        return top_probs, dialog

    def classify(self, dialog: Dialog, classes: List[str], classifier_prompt: str = None, strength: int = 10):
        # binary classification by default
        _classifier_args = self.model_card.make_classifier(classes, strength)
        if classifier_prompt is None:
            _classes = ' or '.join([f'"{t}"' for t in classes])
            classifier_prompt = f"Please respond with one and only one word from {_classes}."
        dialog.send_message(classifier_prompt)
        _dialog = dialog.fork()
        llm_recall = self.max_llm_recall 
        exception_retry = self.max_exception_retry 
        while True:
            try:
                top_probs, _dialog = self._classify(_dialog, classes, _classifier_args)
                dialog.append(_dialog.tail) # truncate the error handlings
                return top_probs, dialog
            except ClassificationError as e:
                if exception_retry > 0:
                    _dialog.send_message(f'Please respond with one and only one word from {classes}')
                    exception_retry -= 1
                    print(f'{e}\nRetrying... times: {self.max_exception_retry-exception_retry}/{self.max_exception_retry}')
                else:
                    raise e
            except Exception as e:
                if llm_recall > 0:
                    llm_recall -= 1
                    time.sleep(1)
                    continue
                else:
                    raise e

class AgentBase:
    agent_type: str | Enum = None
    agent_group: List[str] = None
    is_async: bool = False

    def __init_subclass__(cls, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register:
            register_agent_class(cls)

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream = None):
        if stream is None:
            stream = U.PrintSystem()
        self.config = config
        assert self.agent_group is not None, f"Agent group is not set for {self.agent_type}"
        _agent_configs = config['agent_configs']
        self.agent_configs = {}
        for agent_name in self.agent_group:
            assert agent_name in _agent_configs, f"Agent {agent_name} not found in agent configs"
            self.agent_configs[agent_name] = _agent_configs[agent_name]
        self._stream = stream
        self._stream_backup = stream
        self.st = None
        self.ckpt_dir = ckpt_dir
        self._log_base = build_log_base(config)
        self.agents = {}
        
        # Initialize Provider
        # Assuming OpenAIProvider for now, but this should be configurable
        self.llm_provider = OpenAIProvider(config)

        for agent_name, model_config in self.agent_configs.items():
            model_config = model_config.copy()
            model_name = model_config.pop('model_name')
            self.model = find_model_card(model_name)
            system_prompt_path = model_config.pop('system_prompt_path')
            
            # We assume PROMPT_REGISTRY is available globally or passed. 
            # This is a bit of a dependency issue. 
            # Ideally, prompts should be loaded/registered before Agent initialization.
            # For now, we'll assume the user registers prompts before creating agents.
            from lllm.llm import PROMPT_REGISTRY # Backward compatibility or new location
            
            self.agents[agent_name] = Agent(
                name=agent_name,
                system_prompt=PROMPT_REGISTRY[system_prompt_path],
                model=model_name,
                llm_provider=self.llm_provider,
                model_args=model_config,
                log_base=self._log_base,
                max_exception_retry=self.config.get('max_exception_retry', 3),
                max_interrupt_times=self.config.get('max_interrupt_times', 5),
                max_llm_recall=self.config.get('max_llm_recall', 0),
            )

        self.__additional_args = {}
        sig = inspect.signature(self.call)
        for arg in sig.parameters:
            if arg not in {'task', '**kwargs'}:
                self.__additional_args[arg] = sig.parameters[arg].default

    def set_st(self, session_name: str):
        self.st = U.StreamWrapper(self._stream, self._log_base, session_name)

    def restore_st(self):
        pass

    def silent(self):
        self._stream = U.PrintSystem(silent=True)

    def restore(self):
        self._stream = self._stream_backup

    def call(self, task: str, **kwargs):
        raise NotImplementedError

    def __call__(self, task: str, session_name: str = None, **kwargs) -> str:
        if session_name is None:
            session_name = task.replace(' ', '+')+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.set_st(session_name)
        report = self.call(task, **kwargs)
        with self.st.expander('Prediction Overview', expanded=True):
            self.st.code(f'{report}')
        self.restore_st()
        return report
