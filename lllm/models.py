# Basic reusable prompts
from pydantic import BaseModel
from dataclasses import dataclass, asdict, field
from typing import Callable, List, Dict, Any, Tuple
from lllm.const import Roles, Modalities, find_model_card, CompletionCost, LLM_SIDE_ROLES, Providers, APITypes, ParseError
from lllm.utils import find_xml_blocks, find_md_blocks
import functools as ft


class AgentException(Exception):
    def __init__(self, message: str, context: str = ""):
        self.message = message
        self.context = context
        super().__init__(self.message)


@dataclass
class FunctionCall:
    id: str # it should be the tool_call_id from the openai response for example
    name: str # the function name
    arguments: Dict[str, Any]
    result: Any = None
    result_str: str = None
    error_message: str = None

    @property
    def success(self):
        return self.error_message is None and self.result_str is not None

    def __str__(self):
        _str = f'Calling function: {self.name} with arguments: {self.arguments}\n'
        if self.success:
            _str += f'''Return:\n
---
{self.result_str}
---
'''
        return _str
    
    def equals(self, other: 'FunctionCall') -> bool:
        if self.name != other.name:
            return False
        for k, v in self.arguments.items():
            if k not in other.arguments:
                return False
            if other.arguments[k] != v:
                return False
        return True

    def is_repeated(self, function_calls: List['FunctionCall']) -> bool:
        for call in function_calls:
            if self.equals(call):
                return True
        return False

def default_function_call_processor(result: str, function_call: FunctionCall):
    return f'''Return of calling function {function_call.name} with arguments {function_call.arguments}:
---
{result}
---
'''

@dataclass
class Function:
    name: str
    description: str
    properties: Dict[str, Any]
    required: List[str] = field(default_factory=list)
    additional_properties: bool = False
    strict: bool = True
    function: Callable = None
    processor: Callable = default_function_call_processor

    def to_tool(self, provider: Providers):
        assert self.linked, f"Function {self.name} is not linked"
        if provider == Providers.OPENAI:
            return self.openai_tool
        elif provider == Providers.ANTHROPIC:
            return self.anthropic_tool
        elif provider == Providers.GEMINI:
            return self.gemini_tool 
        
    @property
    def openai_tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self.required,
                    "additionalProperties": self.additional_properties,
                },
                "strict": self.strict
            }
        }
    
    @property
    def anthropic_tool(self):
        raise NotImplementedError("Anthropic tools not implemented")
    
    @property
    def gemini_tool(self):
        raise NotImplementedError("Gemini tools not implemented")
    
    def link_function(self, function: Callable):
        self.function = function

    @property
    def linked(self):
        return self.function is not None
    
    def __call__(self, function_call: FunctionCall) -> FunctionCall:
        assert self.function is not None, "Function not linked"
        try:
            result = self.function(**function_call.arguments)
        except Exception as e:
            function_call.error_message = str(e)
            function_call.result_str = f'Error: {e}'
            return function_call
        function_call.result = result
        function_call.result_str = self.processor(result, function_call)
        return function_call



@dataclass
class MCP:
    server_label: str
    server_url: str
    require_approval: bool = 'never' 
    allowed_tools: List[str] = None

    def to_tool(self, provider: Providers):
        if provider in [Providers.OPENAI, Providers.DATABRICKS]:
            return self.openai_tool
        
    @property
    def openai_tool(self):
        _tool = {
            "type": "mcp",
            "server_label": self.server_label,
            "server_url": self.server_url,
            "require_approval": self.require_approval,
        }
        if self.allowed_tools is not None:
            _tool["allowed_tools"] = self.allowed_tools
        return _tool


@dataclass
class Message:
    role: Roles 
    content: str
    creator: str # for tracking the sender of the message
    raw_response: Any = None # the raw message data such as the openai response, for better caching
    function_calls: List[FunctionCall] = field(default_factory=list) # the function call data such as the openai response, for better caching
    modality: Modalities = Modalities.TEXT
    logprobs: List[float] = field(default_factory=list) # only for assistant messages
    parsed: Dict[str, Any] = field(default_factory=dict)
    model: str = None
    usage: Dict[str, float] = field(default_factory=dict)
    model_args: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict) # for tracking additional information, such as frontend replay info
    _errors: List[Exception] = field(default_factory=list)
    _attempts: List['Message'] = field(default_factory=list) # for debugging the response

    def __post_init__(self):
        if not self.is_function_call:   
            assert isinstance(self.content, str), f"Content must be a string, got {type(self.content)}"
        else:
            self.modality = Modalities.FUNCTION_CALL
    
    def parse(self, parser: Callable):
        assert self.modality == Modalities.TEXT, f"Parsing is only supported for text messages, got {self.modality}"
        self.parsed = parser(self)
        return self.parsed
    
    @property
    def error_message(self):
        return '\n'.join([str(e) for e in self._errors])
    
    def overview(self, max_length: int = 100) -> str:
        _content = self.content.replace('\n', '\\n ')
        if self.modality == Modalities.IMAGE:
            caption = self.extra.get('caption', None)
            if caption is not None:
                return f'Base64 encoded image: {caption} ({_content[:20]}...)'
            else:
                return f'Base64 encoded image ({_content[:20]}...)'
        else:
            if len(_content) <= max_length:
                return _content
            else:
                return _content[:max_length] + '...'

    @property
    def cost(self) -> CompletionCost:
        return self.get_cost(self.model)

    @property
    def api_type(self) -> APITypes:
        _api_type = self.extra.get('api_type', APITypes.COMPLETION.value)
        return APITypes(_api_type)

    def get_cost(self, model: str = None) -> CompletionCost:
        if model is None:
            model = self.model
        if self.from_llm_side:
            model_card = find_model_card(model)
            return model_card.cost(self.usage)
        else:
            return CompletionCost(0,0,0,0)
    
    @property
    def from_llm_side(self) -> bool:
        return self.role in LLM_SIDE_ROLES
    
    @property
    def is_function_call(self) -> bool:
        return len(self.function_calls) > 0
    
    @property
    def attempt_costs(self) -> List[CompletionCost]:
        return [m.cost for m in self._attempts]

    @property
    def input_tokens(self):
        return self.cost.input_tokens

    @property
    def output_tokens(self):
        return self.cost.output_tokens

    @property
    def metadata_log(self): # for logging the metadata
        _metadata = asdict(self)
        _metadata.pop('content')
        _metadata.pop('raw_response')
        _metadata.pop('function_calls')

        def filter_loggable_dict(d: dict):
            _d = {}
            if d is None:
                return None
            for k, v in d.items():
                if isinstance(v, dict):
                    _d[k] = filter_loggable_dict(v)
                elif isinstance(v, Callable):
                    _d[k] = v.__name__
                else:
                    _d[k] = v
            return _d

        _metadata['parsed'] = filter_loggable_dict(self.parsed)
        _metadata['modality'] = self.modality.value
        _metadata['role'] = self.role.value
        _metadata['_errors'] = [str(e) for e in self._errors]
        _metadata['_attempts'] = [m.metadata_log for m in self._attempts]
        return _metadata
    
    def to_dict(self):
        return {
            'role': self.role.value,
            'content': self.content,
            'creator': self.creator,
            'modality': self.modality.value,
            'parsed': self.parsed,
            'logprobs': self.logprobs,
            'model': self.model,
            'usage': self.usage,
            'model_args': self.model_args,
            'extra': self.extra,
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            role=Roles(d['role']),
            content=d['content'],
            creator=d['creator'],
            raw_response=d.get('raw_response', None),
            function_calls=d.get('function_calls', []),
            modality=Modalities(d['modality']),
            parsed=d['parsed'],
            logprobs=d.get('logprobs', []),
            model=d.get('model', None),
            usage=d.get('usage', {}),
            model_args=d.get('model_args', {}),
            extra=d.get('extra', {}),
        )
    
    def __str__(self):
        return f'[{self.creator} ({self.role.value})]:\n\n{self.content}'


# PROMPT_REGISTRY = {}

def default_parser(message: str, xml_tags: List[str] = [], md_tags: List[str] = [], signal_tags: List[str] = [], required_xml_tags: List[str] = [], required_md_tags: List[str] = []):
    xml_tag_blocks = {}
    md_tag_blocks = {}
    errors = []
    for tag in xml_tags:
        matches = find_xml_blocks(message, tag)
        if len(matches) == 0:
            errors.append(f"No {tag} tags found, it should be provided as <{tag}>...</{tag}>")
        xml_tag_blocks[tag] = matches
    for tag in md_tags:
        matches = find_md_blocks(message, tag)
        if len(matches) == 0:
            errors.append(f"No {tag} tags found, it should be provided as ```{tag} ... ```")
        md_tag_blocks[tag] = matches
    for tag in required_xml_tags:
        if tag not in xml_tag_blocks:
            errors.append(f"Required {tag} tag not found, it should be provided as <{tag}>...</{tag}>")
    for tag in required_md_tags:
        if tag not in md_tag_blocks:
            errors.append(f"Required {tag} tag not found, it should be provided as ```{tag} ... ```")
    if len(errors) > 0:
        raise ParseError(f"Parsing errors:\n{'\n'.join(errors)}")
    parsed = {
        'raw': message,
        'xml_tags': xml_tag_blocks,
        'md_tags': md_tag_blocks
    } 
    for tag in signal_tags: # <SIGNAL_TAG>
        parsed[tag] = f'<{tag}>' in message
    return parsed




general_exception_prompt_template='''There is an unexpected error from your response. 
Here is the error message:

---
{error_message}
---

Please fix the error. Remember to follow the instructions from the user message.
'''

general_interrupt_prompt_template='''The return of the function call is as follows:

---
{call_results}
---

You can choose to make more function calls, or you can provide your final response.
'''


@dataclass
class Prompt:
    '''
    The core properties are (see LLLM README.md for more details):
     - prompt: the string used to call LLM
     - functions: the only interruption handlers supported now
     - exception_prompt: the prompt used to handle the error message from this prompt
     - interrupt_prompt: the prompt used to handle the function call results from this prompt
     - parser
    The other properties are for convinience, mainly for building the parsers:
     - format: for openai structured output, an alternative to parser
     - xml_tags: for xml style tags
     - md_tags: for markdown style tags
     - signal_tags: for signal tags
    '''
    path: str
    prompt: str
    _functions: List[Function] = field(default_factory=list) 
    _mcp_servers: List[MCP] = field(default_factory=list) # for mcp servers, not used yet
    parser: Callable[[str], Dict[str, Any]] = None
    exception_prompt: str = general_exception_prompt_template # input: error message, output: error message
    interrupt_prompt: str = general_interrupt_prompt_template # input: call results, output: call results
    format: BaseModel = None # for openai structured output
    xml_tags: List[str] = field(default_factory=list) # for xml style tags
    md_tags: List[str] = field(default_factory=list) # for markdown style tags
    signal_tags: List[str] = field(default_factory=list) # for signal tags like <SIGNAL_TAG>
    required_xml_tags: List[str] = field(default_factory=list) # for xml style tags that are required
    required_md_tags: List[str] = field(default_factory=list) # for markdown style tags that are required
    allow_web_search: bool = False
    computer_use_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.parser is None:
            self.parser = ft.partial(
                default_parser, xml_tags=self.xml_tags, md_tags=self.md_tags, signal_tags=self.signal_tags, required_xml_tags=self.required_xml_tags, required_md_tags=self.required_md_tags)
        self.functions = {}
        for function in self._functions:
            assert isinstance(function, Function), f"Function {function} is not a Function object"
            self.functions[function.name] = function
        self.mcp_servers = {}
        for mcp in self._mcp_servers:
            assert isinstance(mcp, MCP), f"MCP {mcp} is not a MCP object"
            self.mcp_servers[mcp.server_label] = mcp
        if self.exception_prompt is not None:
            # it is a special prompt that is used to handle the error message from this prompt, it should expect the same things
            assert 'error_message' in self.exception_prompt, "Exception handler must contain 'error_message' in the prompt"
        if self.interrupt_prompt is not None:
            assert 'call_results' in self.interrupt_prompt, "Interrupt handler must contain 'call_results' in the prompt"


    def link_function(self, name: str, function: Callable):
        self.functions[name].link_function(function)

    @property
    def exception_handler(self): # to avoid the circular construction, we use a property here
        return Prompt(
            path=f'__{self.path}_exception_handler',
            prompt=self.exception_prompt,
            parser = self.parser,
            _functions = self._functions,
            _mcp_servers = self._mcp_servers,
            exception_prompt = self.exception_prompt, # its recursive!
            interrupt_prompt = self.interrupt_prompt, # its recursive!
            format = self.format,
            xml_tags = self.xml_tags,
            md_tags = self.md_tags,
            signal_tags = self.signal_tags,
            required_xml_tags = self.required_xml_tags,
            required_md_tags = self.required_md_tags,
            allow_web_search = self.allow_web_search,
            computer_use_config = self.computer_use_config,
        )

    @property
    def interrupt_handler(self):
        return Prompt(
            path=f'__{self.path}_interrupt_handler',
            prompt=self.interrupt_prompt,
            parser = self.parser,
            _functions = self._functions,
            _mcp_servers = self._mcp_servers,
            exception_prompt = self.exception_prompt, # its recursive!
            interrupt_prompt = self.interrupt_prompt, # its recursive!
            format = self.format,
            xml_tags = self.xml_tags,
            md_tags = self.md_tags,
            signal_tags = self.signal_tags,
            required_xml_tags = self.required_xml_tags,
            required_md_tags = self.required_md_tags,
            allow_web_search = self.allow_web_search,
            computer_use_config = self.computer_use_config,
        )
    
    @property
    def interrupt_handler_final(self):
        return Prompt(
            path=f'__{self.path}_interrupt_handler_final',
            prompt='Please provide your final response.',
            parser = self.parser,
            _mcp_servers = self._mcp_servers,
            exception_prompt = self.exception_prompt, # its recursive!
            format = self.format,
            xml_tags = self.xml_tags,
            md_tags = self.md_tags,
            signal_tags = self.signal_tags,
            required_xml_tags = self.required_xml_tags,
            required_md_tags = self.required_md_tags,
            allow_web_search = self.allow_web_search,
            computer_use_config = self.computer_use_config,
        )

    def __call__(self,**kwargs):
        for name, function in self.functions.items():
            assert function.linked, f"Function {name} is not linked"  
        if kwargs == {}:
            return self.prompt
        return self.prompt.format(**kwargs)
    




