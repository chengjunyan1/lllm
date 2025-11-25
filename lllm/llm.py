# Low-level LLM interface
# provide simple tools for building agents
# 1. LLM calls & dialogs
# 2. Prompt management
# 3. Replay system and frontend support

from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
import openai
from openai import RateLimitError
import json
import numpy as np
import os
import time
import random


from lllm.const import APITypes, find_model_card, Providers, Roles, Modalities, CompletionCost
from lllm.log import ReplayableLogBase
from lllm.const import RCollections
from lllm.models import Prompt, Message, ParseError, FunctionCall, Function, AgentException
from pydantic import BaseModel
import copy
import uuid
import datetime as dt
import lllm.utils as U

PROMPT_REGISTRY = {}


# https://cookbook.openai.com/examples/prompt_caching101
# https://openai.com/index/api-prompt-caching/ 

def register_prompt(prompt: Prompt):
    PROMPT_REGISTRY[prompt.path] = prompt


def print_prompts():
    U.cprint('--------------------------------')
    U.cprint(f'{len(PROMPT_REGISTRY)} prompts registered')
    for path, prompt_obj in PROMPT_REGISTRY.items():
        U.cprint(f' - {path}')
    U.cprint('--------------------------------')


class ClassificationError(Exception):
    def __init__(self, message: str, top_probs: Dict[str, float]):
        self.message = message
        self.top_probs = top_probs



@dataclass
class Dialog:
    """
    Whenever a dialog is created/forked, it should be associated with a session name
    
    By default, the dialog will use the parser and format of the last user message
    Optionally, you can set the parser and format of the next assistant message
    """
    _messages: List[Message]
    log_base: ReplayableLogBase
    session_name: str
    parent_dialog: str = None
    top_prompt: Prompt = None

    def __post_init__(self):
        self.dialog_id = uuid.uuid4().hex
        dialogs_sess = self.log_base.get_collection(RCollections.DIALOGS).create_session(self.session_name) # track the dialogs created in this session
        dialogs_sess.log(self.dialog_id, metadata={'parent_dialog': self.parent_dialog})
        self.sess = self.log_base.get_collection(RCollections.MESSAGES).create_session(f'{self.session_name}/{self.dialog_id}') # track the dialogs created in this session

    def append(self, message: Message): # ensure this is the only way to write the messages to make sure the trackability
        message.extra['dialog_id'] = self.dialog_id
        self._messages.append(message)
        try:
            self.sess.log(message.content, metadata=message.metadata_log)
        except Exception as e:
            try:
                _metadata = message.metadata_log.pop('parsed')
                print('WARNING: Failed to log message, metadata is not loggable, try to drop the parsed field')
                self.sess.log(message.content, metadata=_metadata)
            except Exception as e:
                print(f'WARNING: Failed to log message: {e}, log the message without metadata')
                self.sess.log(message.content)
                

    def to_dict(self):
        return {
            'messages': [message.to_dict() for message in self._messages],
            'session_name': self.session_name,
            'parent_dialog': self.parent_dialog,
            'top_prompt_path': self.top_prompt.path if self.top_prompt is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict, log_base: ReplayableLogBase, prompt_registry: Dict[str, Prompt]):
        top_prompt_path = d['top_prompt_path']
        if top_prompt_path is not None:
            assert top_prompt_path in prompt_registry, f"Prompt {top_prompt_path} not found in the registry, please register it first"
            top_prompt = prompt_registry[top_prompt_path]
        else:
            top_prompt = None
        return cls(
            _messages=[Message.from_dict(message) for message in d['messages']],
            log_base=log_base,
            session_name=d['session_name'],
            parent_dialog=d['parent_dialog'],
            top_prompt=top_prompt,
        )
    
    @property
    def messages(self):
        return self._messages
    
    def send_base64_image(self, image_base64: str, caption: str = None, creator: str = 'user', extra: Dict[str, Any] = {}, role: Roles = Roles.USER) -> Message:
        if caption is not None:
            extra['caption'] = caption
        message = Message(
            role=role,
            content=image_base64,
            creator=creator,
            modality=Modalities.IMAGE,
            extra=extra,
        )
        self.append(message)
        return message

    def send_message(self, prompt: Prompt | str, prompt_args: Dict[str, Any] = {}, creator: str = 'user', # or 'user', etc.
                     extra: Dict[str, Any] = {}, role: Roles = Roles.USER) -> Message:
        if isinstance(prompt, str):
            assert prompt_args == {}, f"Prompt args are not allowed for string prompt"
            prompt = Prompt(path='__temp_prompt_'+str(uuid.uuid4())[:6], prompt=prompt)
            content = prompt.prompt
        elif prompt_args == {}:
            content = prompt.prompt
        else:
            content = prompt(**prompt_args)
        message = Message(
            role=role,
            content=content,
            creator=creator,
            modality=Modalities.TEXT,
            extra=extra
        )
        self.append(message)
        self.top_prompt = prompt
        return message
    
    def fork(self) -> 'Dialog':
        _messages = [copy.deepcopy(message) for message in self._messages]
        _dialog = Dialog(_messages, self.log_base, self.session_name, self.dialog_id)
        _dialog.top_prompt = self.top_prompt
        return _dialog
    
    def overview(self, remove_tail: bool = False, max_length: int = 100, 
                 stream = None, divider: bool = False):
        _overview = ''
        for idx, message in enumerate(self.messages):
            if remove_tail and idx == len(self.messages)-1:
                break
            _overview += f'[{idx}. {message.creator} ({message.role.value})]: {message.overview(max_length)}\n\n'
        _overview = _overview.strip()
        cost = self.tail.cost
        if stream is not None:
            if divider:
                stream.divider()
            stream.write(U.html_collapse(f'Context overview', _overview), unsafe_allow_html=True)
            stream.write(str(cost))
        return _overview

    @property
    def tail(self): # last message in the dialog, use it to get last response from the LLM
        return self._messages[-1]
    
    @property
    def system(self):
        return self._messages[0]

    @property
    def openai(self): # create message history for openai api
        messages = []
        for message in self._messages:
            if message.from_llm_side:
                _api_type = APITypes(message.extra.get('api_type', APITypes.COMPLETION.value))
                if _api_type == APITypes.COMPLETION.value:
                    messages.append(message.raw_response.choices[0].message)
                elif _api_type == APITypes.RESPONSE.value:
                    messages.extend(message.raw_response.output)
            else:
                if message.role == Roles.TOOL:
                    assert 'tool_call_id' in message.extra, f"Tool call id is not found in the message extra"
                    messages.append({
                        "role": message.role.value,
                        "content": message.content,
                        "tool_call_id": message.extra['tool_call_id']
                    })
                else:
                    if message.modality == Modalities.IMAGE:
                        _content = []
                        if 'caption' in message.extra:
                            _content.append({ "type": "text", "text": message.extra['caption'] })
                        _content.append({ "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{message.content}" } })
                        messages.append({
                            "role": message.role.value,
                            "content": _content
                        })
                    elif message.modality == Modalities.TEXT:   
                        messages.append({
                            "role": message.role.value,
                            "content": message.content
                        })
                    else:
                        raise ValueError(f"Unsupported modality: {message.modality}")
        return messages

    @property
    def gemini(self):
        raise NotImplementedError("to_gemini not implemented")

    @property
    def anthropic(self):
        raise NotImplementedError("to_anthropic not implemented")

    def context_copy(self, n: int = 1) -> 'Dialog':
        _dialog = self.fork()
        if n > 0:
            _dialog._messages = _dialog._messages[:-n]
        return _dialog

    @property
    def cost(self) -> CompletionCost:
        return self.get_cost()

    def get_cost(self, model: str = None) -> CompletionCost:
        costs = [message.get_cost(model) for message in self._messages]
        return CompletionCost(
            prompt_tokens=sum([cost.prompt_tokens for cost in costs]),
            completion_tokens=sum([cost.completion_tokens for cost in costs]),
            cached_prompt_tokens=sum([cost.cached_prompt_tokens for cost in costs]),
            cost=sum([cost.cost for cost in costs])
        )
    


class LLMCaller:
    # It just call the LLM given the dialog, and return the response, parse the response according to the prompt on the top of the dialog
    def __init__(self, config: dict):
        self.random_seed = config.get('random_seed', 42)
        self.openai_client = openai.OpenAI(api_key=os.getenv('MY_OPENAI_KEY'))
        self.together_client = openai.OpenAI(api_key=os.getenv('TOGETHER_API_KEY'), base_url='https://api.together.xyz/v1')

    def call(self, 
             dialog: Dialog, 
             prompt: Prompt,
             model: str, 
             model_args: Dict[str, Any] = {}, 
             parser_args: Dict[str, Any] = {},
             responder: str = 'assistant', 
             extra: Dict[str, Any] = {}) -> Message:
        # notice that the LLM caller does not operate on the dialog
        model_card = find_model_card(model)
        assert isinstance(prompt, Prompt), f"Prompt {prompt} is not a Prompt object"
        if model_card.provider == Providers.OPENAI:
            response = self._call_openai(dialog, prompt, model, model_args, parser_args, responder, extra)
        elif model_card.provider == Providers.GEMINI:
            response = self._call_gemini(dialog, prompt, model, model_args, parser_args, responder, extra)
        elif model_card.provider == Providers.ANTHROPIC:
            response = self._call_anthropic(dialog, prompt, model, model_args, parser_args, responder, extra)
        else:
            raise ValueError(f"Provider {model_card.provider} not supported")
        return response

    def _call_openai(self, dialog: Dialog, prompt: Prompt, model: str, model_args: Dict[str, Any] = {}, 
                     parser_args: Dict[str, Any] = {}, responder: str = 'assistant', extra: Dict[str, Any] = {}) -> Message:
        funcs = [func.to_tool(Providers.OPENAI) for func in prompt.functions.values()]
        model_card = find_model_card(model)
        
        _client = self.openai_client
        if model_card.base_url is not None:
            if 'together' in model_card.base_url:
                _client = self.together_client
            else:
                raise ValueError(f"Unsupported base url: {model_card.base_url}")

        if prompt.format is None:
            call_fn = _client.chat.completions.create
            # model_args['logprobs'] = False
        else:
            call_fn = _client.beta.chat.completions.parse
            model_args['response_format'] = prompt.format
        if model_card.is_reasoning:
            model_args['temperature'] = 1
            # _model_output_limit = model_card.max_output_tokens
            # _max_completion_tokens = model_args.get('max_completion_tokens', _model_output_limit)
            # model_args['max_completion_tokens'] = min(_max_completion_tokens*2, _model_output_limit) # for reasoning 

        completion = call_fn(
            model=model,
            messages=dialog.openai,
            # seed=self.random_seed,
            tools=funcs,
            **model_args
        )
        choice = completion.choices[0]
        usage = json.loads(completion.usage.model_dump_json())

        if choice.finish_reason == 'tool_calls':
            role = Roles.TOOL_CALL
            logprobs = None
            parsed = None
            errors = []
            function_calls = [FunctionCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments)
            ) for tool_call in choice.message.tool_calls]   
            content = 'Tool calls:\n\n'+'\n'.join([f'{idx}. {tool_call.function.name}: {tool_call.function.arguments}' for idx, tool_call in enumerate(choice.message.tool_calls)])
        else:
            role = Roles.ASSISTANT
            errors   = []
            if prompt.format is None:
                content = choice.message.content
                logprobs = choice.logprobs.content if choice.logprobs is not None else None
                if logprobs is not None:
                    logprobs = [logprob.model_dump() for logprob in logprobs]
                try:
                    parsed = prompt.parser(content, **parser_args) if prompt.parser is not None else None
                except ParseError as e:
                    errors.append(e)
                    parsed = {'raw': content}
            else:
                if choice.message.refusal:
                    raise ValueError(choice.message.refusal)
                content = str(choice.message.parsed.json())
                logprobs = None
                parsed = json.loads(content)
            if 'response_format' in model_args and prompt.format is not None:
                # convert the format uninstantiated class to a json string
                model_args['response_format'] = prompt.format.model_json_schema()
            function_calls = []
        response = Message(
            role=role,
            raw_response=completion,
            creator=responder,
            function_calls=function_calls,
            content=content,
            logprobs=logprobs,
            model=model,
            model_args=model_args,
            usage=usage,
            parsed=parsed,
            extra=extra,
            _errors=errors
        )
        return response

    def _call_gemini(self, dialog: Dialog, prompt: Prompt, model: str, model_args: Dict[str, Any] = {}, 
                     parser_args: Dict[str, Any] = {}, responder: str = 'assistant', extra: Dict[str, Any] = {}) -> Message:
        raise NotImplementedError("call_gemini not implemented")
    
    def _call_anthropic(self, dialog: Dialog, prompt: Prompt, model: str, model_args: Dict[str, Any] = {}, 
                        parser_args: Dict[str, Any] = {}, responder: str = 'assistant', extra: Dict[str, Any] = {}) -> Message:
        raise NotImplementedError("call_anthropic not implemented")



class LLMResponder(LLMCaller):
    def call(self, 
             dialog: Dialog, 
             prompt: Prompt,
             model: str, 
             model_args: Dict[str, Any] = {}, 
             parser_args: Dict[str, Any] = {},
             responder: str = 'assistant', 
             extra: Dict[str, Any] = {}) -> Message:
        model_card = find_model_card(model)
        assert isinstance(prompt, Prompt), f"Prompt {prompt} is not a Prompt object"
        if model_card.provider == Providers.OPENAI:
            response = self._respond_openai(dialog, prompt, model, model_args, parser_args, responder, extra)
        else:
            raise ValueError(f"Provider {model_card.provider} not supported for LLMResponder")
        return response
    
    def _respond_openai(self, dialog: Dialog, prompt: Prompt, model: str, model_args: Dict[str, Any] = {}, 
            parser_args: Dict[str, Any] = {}, responder: str = 'assistant', extra: Dict[str, Any] = {}) -> Message:
        funcs = [func.to_tool(Providers.OPENAI, APITypes.RESPONSE) for func in prompt.functions.values()]
        mcps = [mcp.to_tool(Providers.OPENAI) for mcp in prompt.mcp_servers.values()]
        assert prompt.format is None, "Please use completion API for structured output."
        model_card = find_model_card(model)
        
        tools = funcs + mcps
        if prompt.allow_web_search and Features.WEB_SEARCH in model_card.features:
            tools.append({"type": "web_search_preview" })
        if prompt.computer_use_config and Features.COMPUTER_USE in model_card.features:
            tools.append({
                "type": "computer_use_preview",
                "display_width": prompt.computer_use_config.get('display_width', 1280),
                "display_height": prompt.computer_use_config.get('display_height', 800),
                "environment": prompt.computer_use_config.get("environment", "browser")
            })
        
        if 'max_completion_tokens' in model_args:
            max_output_tokens = model_args.pop('max_completion_tokens')
        else:
            max_output_tokens = 32000

        response = self.openai_client.responses.create(
            model=model,
            input=dialog.openai,
            tools=tools,
            tool_choice='auto',
            max_output_tokens=max_output_tokens,
            truncation=model_args.get('truncation','auto'),
            **model_args
        )
        usage = json.loads(response.usage.model_dump_json())

        errors = []
        if any(output.type == 'function_call' for output in response.output):
            role = Roles.TOOL_CALL
            parsed = None
            logprobs=None
            function_calls = [FunctionCall(
                id=output.call_id,
                name=output.name,
                arguments=json.loads(output.arguments)
            ) for output in response.output if output.type == 'function_call']   
            content = 'Tool calls:\n\n'+'\n'.join([f'{idx}. {tool_call.name}: {tool_call.arguments}' for idx, tool_call in enumerate(function_calls)])
        else:
            role = Roles.ASSISTANT
            function_calls = []
            content = response.output_text
            logprobs=None # TODO: Implement logprobs if needed
            try:
                parsed = prompt.parser(content, **parser_args) if prompt.parser is not None else None
            except ParseError as e:
                errors.append(e)
                parsed = {'raw': content}

        extra['reasoning'] = response.reasoning.model_dump_json()
        extra['api_type'] = APITypes.RESPONSE.value
        message = Message(
            role=role,
            raw_response=response,
            creator=responder,
            function_calls=function_calls,
            content=content,
            logprobs=logprobs, 
            model=model,
            model_args=model_args,
            usage=usage,
            parsed=parsed,
            extra=extra,
            _errors=errors
        )
        return message

# Maintain the dialog separately from the agent

@dataclass
class Prompts:
    root: str

    def __call__(self, name: str) -> Prompt:
        path = f'{self.root}/{name}'
        if path not in PROMPT_REGISTRY:
            raise ValueError(f'Prompt {path} not found in the registry')
        return PROMPT_REGISTRY[path]



general_classification_prompt = '''Please respond with one and only one word from the following options: {classes}.
You can be fussy or unsure, but you must select one of the options, do not refuse to answer.
'''

TRUE_TOKEN = 'True'
FALSE_TOKEN = 'False'
BINARY_CLASSES = [TRUE_TOKEN, FALSE_TOKEN]
BINARY_CLASSIFIER_PROMPT = f'''Please respond with one and only one word from {' or '.join([f'"{t}"' for t in BINARY_CLASSES])}.
You can be fussy, but you must select one, do not refuse to answer, and do not respond like "maybe" or "unsure" or "unknown".
'''



@dataclass
class Agent:
    name: str # the role of the agent, or a name of the agent
    system_prompt: Prompt
    model: str # the latest snapshot of a model
    llm_caller: LLMCaller
    log_base: ReplayableLogBase   
    model_args: Dict[str, Any] = field(default_factory=dict) # additional args, like temperature, seed, etc.
    max_exception_retry: int = 3
    max_interrupt_times: int = 5
    max_llm_recall: int = 0

    def __post_init__(self):
        self.model_card = find_model_card(self.model)
        self.model_card.check_args(self.model_args)
        self.model = self.model_card.latest_snapshot.name
        # self.prompts = Prompts(self.name)

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

    # it performs the "Agent Call", check the LLLM README.md for more details
    def call(self, 
        dialog: Dialog, # it assumes the prompt is already loaded into the dialog as the top prompt by send_message
        extra: Dict[str, Any] = {}, # for tracking additional information, such as frontend replay info
        args: Dict[str, Any] = {}, # for tracking additional information, such as frontend replay info
        parser_args: Dict[str, Any] = {},   
    ) -> Tuple[Message, Dialog, List[FunctionCall]]:
        # Prompt: a function maps prompt args and dialog into the expected output 
        current_prompt = dialog.top_prompt
        interrupts = []
        for i in range(1e10 if self.max_interrupt_times == 0 else self.max_interrupt_times+1): # +1 for the final response
            llm_recall = self.max_llm_recall 
            exception_retry = self.max_exception_retry 
            working_dialog = dialog.fork() # make a copy of the dialog, truncate all excception handling dialogs
            while True: # ensure the response is no exception
                _attempts = []
                try:
                    _model_args = self.model_args.copy()
                    _model_args.update(args)
                    # FIXME: remove this
                    # print(working_dialog.overview(max_length=1e10))
                    response = self.llm_caller.call(working_dialog, current_prompt, self.model, _model_args, 
                                                    parser_args=parser_args, responder=self.name, extra=extra)
                    working_dialog.append(response) 
                    if response._errors != []:
                        _attempts.append(response)
                        raise AgentException(response.error_message)
                    else: # 
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
                except RateLimitError as e: # handle the rate limit error from the LLM
                    wait_time = random.random()*15+1
                    time.sleep(wait_time) # wait for a while before retrying
                except Exception as e: # handle the exception from the LLM
                    # if its not found error
                    wait_time = random.random()*15+1
                    if U.is_openai_rate_limit_error(e): # for safe
                        time.sleep(wait_time)
                    else:
                        if llm_recall > 0:
                            llm_recall -= 1
                            time.sleep(1) # wait for a while before retrying
                            continue
                        else:
                            tmp_dir = os.environ.get('TMP_DIR', './tmp')
                            error_log_dir = U.pjoin(tmp_dir, '.error', 'llm_call')
                            U.mkdirs(error_log_dir)
                            log_name = f'{self.name}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:4]}'
                            error_log_path = U.pjoin(error_log_dir, f'{log_name}.txt')
                            with open(error_log_path, 'w', encoding='utf-8') as f:
                                json.dump({'timestamp': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'error_message': str(e)}, f)
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
        # print(chosen_token_data)
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
            classifier_prompt = general_classification_prompt.format(classes=_classes)
        dialog.send_message(classifier_prompt)
        _dialog = dialog.fork()
        llm_recall = self.max_llm_recall 
        exception_retry = self.max_exception_retry 
        while True:
            try:
                dialog_backup = _dialog.fork()
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
                    _dialog = dialog_backup # retry from the begining
                    llm_recall -= 1
                    continue
                else:
                    raise e
    
    def binary_classify(self, dialog: Dialog, strength: int = 10):
        probs, dialog = self.classify(dialog, BINARY_CLASSES, BINARY_CLASSIFIER_PROMPT, strength)
        p_true = probs[TRUE_TOKEN]
        p_false = probs[FALSE_TOKEN]
        p_true = p_true / (p_true + p_false)
        return p_true, dialog

                                     





