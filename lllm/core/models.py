from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from lllm.core.const import Roles, Modalities, APITypes, CompletionCost, ModelCard, MODEL_CARDS, Providers

class AgentException(Exception):
    def __init__(self, message: str, context: str = ""):
        self.message = message
        self.context = context
        super().__init__(self.message)

class FunctionCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Any = None
    result_str: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def success(self):
        return self.error_message is None and self.result_str is not None

    def __str__(self):
        _str = f'Calling function: {self.name} with arguments: {self.arguments}\n'
        if self.success:
            _str += f'Return:\n---\n{self.result_str}\n---\n'
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
    return f'Return of calling function {function_call.name} with arguments {function_call.arguments}:\n---\n{result}\n---\n'

class Function(BaseModel):
    name: str
    description: str
    properties: Dict[str, Any]
    required: List[str] = Field(default_factory=list)
    additional_properties: bool = False
    strict: bool = True
    function: Optional[Callable] = None
    processor: Callable = default_function_call_processor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_tool(self, provider: Providers):
        # This logic might be moved to provider specific implementations later
        if provider == Providers.OPENAI:
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
        raise NotImplementedError(f"Provider {provider} not supported for tool conversion yet")

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

class MCP(BaseModel):
    server_label: str
    server_url: str
    require_approval: bool = False # 'never' in original, changed to bool for simplicity or keep str if needed. Original was 'never' (str) but typed as bool?
    # Original: require_approval: bool = 'never' -> This is weird. Let's assume it's a string or bool.
    # If it was 'never', it should be a string.
    allowed_tools: Optional[List[str]] = None

    def to_tool(self, provider: Providers):
        if provider == Providers.OPENAI:
             _tool = {
                "type": "mcp",
                "server_label": self.server_label,
                "server_url": self.server_url,
                "require_approval": self.require_approval,
            }
             if self.allowed_tools is not None:
                _tool["allowed_tools"] = self.allowed_tools
             return _tool
        return None

class Message(BaseModel):
    role: Roles
    content: Union[str, List[Dict[str, Any]]] # Content can be string or list of content parts (for images)
    creator: str
    raw_response: Any = None
    function_calls: List[FunctionCall] = Field(default_factory=list)
    modality: Modalities = Modalities.TEXT
    logprobs: List[float] = Field(default_factory=list)
    parsed: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None
    usage: Dict[str, float] = Field(default_factory=dict)
    model_args: Dict[str, Any] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)
    _errors: List[Exception] = Field(default_factory=list)
    _attempts: List['Message'] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def error_message(self):
        return '\n'.join([str(e) for e in self._errors])

    @property
    def cost(self) -> CompletionCost:
        # Placeholder for cost calculation
        return CompletionCost()

    @property
    def is_function_call(self) -> bool:
        return len(self.function_calls) > 0

    def to_dict(self):
        return self.model_dump(exclude={'raw_response', '_errors', '_attempts'})

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

class Prompt(BaseModel):
    path: str
    prompt: str
    _functions: List[Function] = Field(default_factory=list, alias='functions_list')
    _mcp_servers: List[MCP] = Field(default_factory=list, alias='mcp_servers_list')
    parser: Optional[Callable[[str], Dict[str, Any]]] = None
    exception_prompt: str = "Error: {error_message}. Please fix."
    interrupt_prompt: str = "Result: {call_results}. Continue?"
    format: Optional[Any] = None # Pydantic model class for structured output
    xml_tags: List[str] = Field(default_factory=list)
    md_tags: List[str] = Field(default_factory=list)
    signal_tags: List[str] = Field(default_factory=list)
    required_xml_tags: List[str] = Field(default_factory=list)
    required_md_tags: List[str] = Field(default_factory=list)
    allow_web_search: bool = False
    computer_use_config: Dict[str, Any] = Field(default_factory=dict)

    functions: Dict[str, Function] = Field(default_factory=dict, init=False)
    mcp_servers: Dict[str, MCP] = Field(default_factory=dict, init=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context):
        self.functions = {f.name: f for f in self._functions}
        self.mcp_servers = {m.server_label: m for m in self._mcp_servers}

    def link_function(self, name: str, function: Callable):
        if name in self.functions:
            self.functions[name].link_function(function)

    def __call__(self, **kwargs):
        if not kwargs:
            return self.prompt
        return self.prompt.format(**kwargs)

    @property
    def exception_handler(self):
        # Recursive prompt creation logic (simplified for now)
        return Prompt(
            path=f'__{self.path}_exception_handler',
            prompt=self.exception_prompt,
            parser=self.parser,
            functions_list=self._functions,
            mcp_servers_list=self._mcp_servers,
            exception_prompt=self.exception_prompt,
            interrupt_prompt=self.interrupt_prompt,
            format=self.format,
            xml_tags=self.xml_tags,
            md_tags=self.md_tags,
            signal_tags=self.signal_tags,
            required_xml_tags=self.required_xml_tags,
            required_md_tags=self.required_md_tags,
            allow_web_search=self.allow_web_search,
            computer_use_config=self.computer_use_config,
        )
    
    @property
    def interrupt_handler(self):
         return Prompt(
            path=f'__{self.path}_interrupt_handler',
            prompt=self.interrupt_prompt,
            parser=self.parser,
            functions_list=self._functions,
            mcp_servers_list=self._mcp_servers,
            exception_prompt=self.exception_prompt,
            interrupt_prompt=self.interrupt_prompt,
            format=self.format,
            xml_tags=self.xml_tags,
            md_tags=self.md_tags,
            signal_tags=self.signal_tags,
            required_xml_tags=self.required_xml_tags,
            required_md_tags=self.required_md_tags,
            allow_web_search=self.allow_web_search,
            computer_use_config=self.computer_use_config,
        )

